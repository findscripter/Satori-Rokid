package com.satori.glasses.capture

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.view.Surface
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Consent-backed display capture, video only. The first-person recorder remains the audio owner. */
class ScreenRecorder(context: Context, private val onEvent: (String, JSONObject) -> Unit) : Closeable {
    private val app = context.applicationContext
    private val main = Handler(context.mainLooper)
    private val thread = HandlerThread("screen-recorder").apply { start() }
    private val worker = Handler(thread.looper)
    private val closed = AtomicBoolean()
    private val scheduled = AtomicBoolean()
    private val session = AtomicReference<Session?>()
    private val lifecycle = Any()

    val active: Boolean get() = session.get()?.state?.get() in setOf("starting", "recording", "finalizing")

    private class Session(val file: File, val sourceWidth: Int, val sourceHeight: Int, val densityDpi: Int,
                          val metadata: JSONObject) {
        val size = ScreenCaptureSize.fromDisplay(sourceWidth, sourceHeight)
        val pending = File(file.parentFile, file.nameWithoutExtension + ".pending.mp4")
        val sidecar = File(file.parentFile, file.nameWithoutExtension + ".json")
        val sidecarPending = File(file.parentFile, file.nameWithoutExtension + ".pending.json")
        val state = AtomicReference("starting")
        val stopping = AtomicBoolean()
        val stoppedAt = AtomicLong()
        val requestedAt = SystemClock.elapsedRealtimeNanos()
        val timeline = ScreenRecordingTimeline()
        var projection: MediaProjection? = null
        var projectionCallback: MediaProjection.Callback? = null
        var display: VirtualDisplay? = null
        var codec: MediaCodec? = null
        var surface: Surface? = null
        var muxer: MediaMuxer? = null
        var output: FileOutputStream? = null
        var track = -1
        var muxerStarted = false
        var inputEos = false
        var outputEos = false
        var ownsPending = false
        val info = MediaCodec.BufferInfo()
        @Volatile var projectionBeforeNs = 0L
        @Volatile var projectionAfterNs = 0L
        @Volatile var firstEncodedObservedNs = 0L
        @Volatile var lastEncodedObservedNs = 0L
        @Volatile var reason = ""
        @Volatile var codecName = ""
        @Volatile var finalized = false
    }

    /** Call only after OS consent and promotion of the service to foreground mediaProjection type. */
    fun start(file: File, resultCode: Int, data: Intent, sourceWidth: Int, sourceHeight: Int, densityDpi: Int,
              sessionMetadata: JSONObject = JSONObject()) {
        val s: Session
        try {
            synchronized(lifecycle) {
                check(!closed.get() && !active) { "screen_recorder_busy_or_closed" }
                require(resultCode == Activity.RESULT_OK) { "screen_capture_not_authorized" }
                require(densityDpi in 40..1600) { "invalid_display_density" }
                val root = File(app.filesDir, "screen-recordings").canonicalFile
                val target = file.absoluteFile
                require(target.parentFile?.canonicalFile == root && target.name.matches(Regex("screen-[0-9]+\\.mp4"))) {
                    "invalid_screen_output_path"
                }
                val metadata = JSONObject()
                for (key in listOf("run_session_id", "boot_id", "data_run_id", "clock_domain", "capture_group_id")) {
                    val value = sessionMetadata.optString(key)
                    require(value.length <= 120) { "invalid_session_metadata" }
                    if (value.isNotBlank()) metadata.put(key, value)
                }
                s = Session(target, sourceWidth, sourceHeight, densityDpi, metadata)
                session.set(s)
            }
        } catch (_: Exception) {
            emit("screen_rejected", JSONObject().put("reason", "invalid_request_or_busy")
                .put("error", "录屏尚未就绪、授权无效或已有录屏，请稍后重试"))
            return
        }
        worker.post {
            try {
                if (s.stopping.get() || closed.get()) throw IllegalStateException("cancelled_before_capture")
                prepare(s, resultCode, data)
                wake()
            } catch (failure: Exception) { fail(s, "screen_start_failed", failure.javaClass.simpleName) }
        }
    }

    fun stop(reason: String = "user") {
        val s = session.get() ?: return
        if (!active || !s.stopping.compareAndSet(false, true)) return
        s.reason = reason.take(80)
        s.stoppedAt.set(SystemClock.elapsedRealtimeNanos())
        s.state.set("finalizing")
        wake()
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        stop("service_stop")
        wake()
    }

    fun snapshot(): JSONObject = session.get()?.let(::stats) ?: JSONObject().put("state", "idle")
        .put("path", "").put("with_audio", false).put("finalized", false).put("frames", 0)

    private fun prepare(s: Session, resultCode: Int, data: Intent) {
        val folder = checkNotNull(s.file.parentFile)
        check(folder.mkdirs() || folder.isDirectory)
        check(!s.file.exists() && !s.sidecar.exists() && !s.sidecarPending.exists()) { "output_exists" }
        check(s.pending.createNewFile()) { "pending_exists" }
        s.ownsPending = true
        s.output = FileOutputStream(s.pending)
        s.muxer = MediaMuxer(s.output!!.fd, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        s.codec = codec
        s.codecName = codec.name
        val caps = codec.codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).videoCapabilities
        check(caps.areSizeAndRateSupported(s.size.width, s.size.height, 15.0)) { "unsupported_screen_size" }
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, s.size.width, s.size.height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, maxOf(1_000_000, s.size.width * s.size.height * 4))
            setInteger(MediaFormat.KEY_FRAME_RATE, 15)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
        }
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        s.surface = codec.createInputSurface()
        codec.start()
        val manager = app.getSystemService(MediaProjectionManager::class.java)
        val projection = checkNotNull(manager.getMediaProjection(resultCode, data))
        s.projection = projection
        val callback = object : MediaProjection.Callback() {
            override fun onStop() { if (session.get() === s) stop("projection_revoked") }
        }
        s.projectionCallback = callback
        projection.registerCallback(callback, worker)
        s.projectionBeforeNs = SystemClock.elapsedRealtimeNanos()
        s.display = projection.createVirtualDisplay("Satori research screen", s.size.width, s.size.height,
            s.densityDpi, DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, s.surface, null, worker)
        s.projectionAfterNs = SystemClock.elapsedRealtimeNanos()
        check(s.display != null) { "virtual_display_unavailable" }
        emit("screen_requested", stats(s))
    }

    private fun wake(delayMs: Long = 0) {
        if (scheduled.compareAndSet(false, true)) worker.postDelayed(pump, delayMs)
    }

    private val pump = object : Runnable {
        override fun run() {
            scheduled.set(false)
            val s = session.get()
            if (s != null && active) {
                try {
                    if (s.file.usableSpace < 512L * 1024 * 1024 && !s.stopping.get()) stop("storage_low")
                    if (s.stopping.get() && !s.inputEos) {
                        releaseProjection(s)
                        s.codec?.signalEndOfInputStream()
                        s.inputEos = true
                    }
                    drain(s)
                    when {
                        s.outputEos -> finish(s)
                        s.stopping.get() && SystemClock.elapsedRealtimeNanos() - s.stoppedAt.get() > 4_000_000_000L ->
                            fail(s, "screen_eos_timeout")
                        s.timeline.frames == 0L && SystemClock.elapsedRealtimeNanos() - s.requestedAt > 12_000_000_000L ->
                            fail(s, "screen_no_frames")
                    }
                } catch (failure: Exception) { fail(s, "screen_codec_or_storage_failure", failure.javaClass.simpleName) }
            }
            if (active) wake(15) else if (closed.get()) thread.quitSafely()
        }
    }

    private fun drain(s: Session) {
        val codec = s.codec ?: return
        repeat(16) {
            val index = codec.dequeueOutputBuffer(s.info, 0)
            if (index == MediaCodec.INFO_TRY_AGAIN_LATER) return
            if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                check(!s.muxerStarted) { "screen_format_changed" }
                s.track = s.muxer!!.addTrack(codec.outputFormat)
                s.muxer!!.start(); s.muxerStarted = true
            } else if (index >= 0) try {
                val info = s.info
                if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                    check(s.muxerStarted) { "screen_muxer_not_ready" }
                    val bytes = checkNotNull(codec.getOutputBuffer(index))
                    val observed = SystemClock.elapsedRealtimeNanos()
                    val pts = s.timeline.append(info.presentationTimeUs)
                    val first = s.firstEncodedObservedNs == 0L
                    if (first) s.firstEncodedObservedNs = observed
                    s.lastEncodedObservedNs = observed
                    val normalized = MediaCodec.BufferInfo().apply { set(info.offset, info.size, pts, info.flags) }
                    s.muxer!!.writeSampleData(s.track, bytes, normalized)
                    if (first) {
                        if (!s.stopping.get()) s.state.set("recording")
                        emit("screen_started", stats(s))
                    }
                }
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) s.outputEos = true
            } finally { codec.releaseOutputBuffer(index, false) }
        }
    }

    private fun finish(s: Session) {
        check(s.muxerStarted && s.timeline.frames > 0) { "screen_missing_video" }
        s.muxer!!.stop(); s.muxer!!.release(); s.muxer = null; s.muxerStarted = false
        s.output?.fd?.sync(); s.output?.close(); s.output = null
        release(s)
        check(s.pending.length() > 0 && !s.file.exists() && !s.sidecar.exists()) { "screen_output_conflict" }
        val finalStats = stats(s).put("state", "saved").put("finalized", true)
        check(s.sidecarPending.createNewFile()) { "screen_sidecar_conflict" }
        FileOutputStream(s.sidecarPending).use { out ->
            out.write(finalStats.toString(2).toByteArray(Charsets.UTF_8)); out.fd.sync()
        }
        Files.move(s.pending.toPath(), s.file.toPath())
        Files.move(s.sidecarPending.toPath(), s.sidecar.toPath())
        s.finalized = true; s.state.set("saved")
        emit("screen_saved", stats(s))
    }

    private fun fail(s: Session, reason: String, exceptionType: String = "") {
        if (s.state.get() in setOf("saved", "error")) return
        s.reason = reason
        s.stopping.set(true); s.stoppedAt.compareAndSet(0, SystemClock.elapsedRealtimeNanos())
        release(s)
        s.state.set("error")
        val result = stats(s).put("exception_type", exceptionType)
            .put("error", "录屏未完成，已保留产生的文件；请检查授权、存储与设备编码能力")
        if (s.ownsPending && !s.sidecar.exists() && !s.sidecarPending.exists()) runCatching {
            if (s.sidecarPending.createNewFile()) {
                FileOutputStream(s.sidecarPending).use { out -> out.write(result.toString(2).toByteArray(Charsets.UTF_8)); out.fd.sync() }
                Files.move(s.sidecarPending.toPath(), s.sidecar.toPath())
            }
        }
        emit("screen_error", result)
        if (closed.get()) thread.quitSafely()
    }

    private fun releaseProjection(s: Session) {
        runCatching { s.display?.release() }; s.display = null
        s.projectionCallback?.let { callback -> runCatching { s.projection?.unregisterCallback(callback) } }
        s.projectionCallback = null
        runCatching { s.projection?.stop() }; s.projection = null
    }

    private fun release(s: Session) {
        releaseProjection(s)
        runCatching { s.codec?.stop() }; runCatching { s.codec?.release() }; s.codec = null
        runCatching { s.surface?.release() }; s.surface = null
        runCatching { s.muxer?.release() }; s.muxer = null; s.muxerStarted = false
        runCatching { s.output?.close() }; s.output = null
    }

    private fun stats(s: Session): JSONObject {
        val end = s.stoppedAt.get().takeIf { it > 0 } ?: SystemClock.elapsedRealtimeNanos()
        return JSONObject(s.metadata.toString()).put("schema", "satori.screen-recording/1")
            .put("media_role", "screen")
            .put("state", s.state.get()).put("path", s.file.absolutePath).put("pending_path", s.pending.absolutePath)
            .put("sidecar_path", s.sidecar.absolutePath).put("with_audio", false).put("finalized", s.finalized)
            .put("source", "android_media_projection").put("source_width", s.sourceWidth).put("source_height", s.sourceHeight)
            .put("width", s.size.width).put("height", s.size.height).put("density_dpi", s.densityDpi)
            .put("target_fps", 15).put("codec", s.codecName).put("frames", s.timeline.frames)
            .put("request_elapsed_ns", s.requestedAt).put("stop_requested_elapsed_ns", s.stoppedAt.get())
            .put("duration_ms", (end - s.requestedAt).coerceAtLeast(0) / 1_000_000)
            .put("media_duration_ms", s.timeline.durationUs() / 1000)
            .put("projection_start_before_elapsed_ns", s.projectionBeforeNs).put("projection_start_after_elapsed_ns", s.projectionAfterNs)
            .put("first_encoded_observed_elapsed_ns", s.firstEncodedObservedNs).put("last_encoded_observed_elapsed_ns", s.lastEncodedObservedNs)
            .put("encoder_pts_origin_us", s.timeline.originUs ?: JSONObject.NULL).put("last_encoder_pts_us", s.timeline.lastUs ?: JSONObject.NULL)
            .put("container_pts_origin_us", 0).put("capture_start_elapsed_ns", JSONObject.NULL)
            .put("timeline_mapping", "manual_sync_required").put("timeline_note", "encoder receipt time is not capture time; align visible SYNC markers")
            .put("reason", s.reason)
    }

    private fun emit(name: String, data: JSONObject) { main.post { runCatching { onEvent(name, data) } } }
}
