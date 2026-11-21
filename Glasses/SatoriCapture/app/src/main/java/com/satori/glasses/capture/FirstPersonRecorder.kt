package com.satori.glasses.capture

import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Owned, tightly packed unrotated I420. After offerVideo the caller must not modify/reuse i420. */
data class RecordingVideoFrame(val width: Int, val height: Int, val rotation: Int, val timestampNs: Long, val i420: ByteArray)

/**
 * One CameraX analysis stream -> H264 + shared 16 kHz PCM -> AAC -> MP4. There is no camera/microphone
 * ownership here. Offers do not call a codec or perform I/O; video holds at most three queued frames,
 * audio has its own bounded queue and sample clock. Every codec/muxer operation runs on one worker.
 */
class FirstPersonRecorder(context: Context, private val onEvent: (String, JSONObject) -> Unit) : Closeable {
    private val main = Handler(context.mainLooper)
    private val thread = HandlerThread("first-person-recorder").apply { start() }
    private val worker = Handler(thread.looper)
    private val scheduled = AtomicBoolean()
    private val closed = AtomicBoolean()
    private val session = AtomicReference<Session?>(null)
    private val lifecycle = Any()

    val active: Boolean get() = session.get()?.state?.get() in setOf("starting", "recording", "finalizing")
    val wantsFrames: Boolean get() = !closed.get() && session.get()?.let {
        !it.stopping.get() && it.state.get() in setOf("starting", "recording")
    } == true

    private class AudioPacket(val ptsUs: Long, val bytes: ByteArray) { var offset = 0 }
    private data class Encoded(val video: Boolean, val ptsUs: Long, val flags: Int, val bytes: ByteArray)
    private class RecorderFailure(val reason: String) : Exception()

    private class Session(val file: File, val withAudio: Boolean, val epochNs: Long) {
        val pending = File(file.parentFile, file.nameWithoutExtension + ".pending.mp4")
        val sidecar = File(file.parentFile, file.nameWithoutExtension + ".json")
        val sidecarPending = File(file.parentFile, file.nameWithoutExtension + ".pending.json")
        val lock = Any()
        val videos = ArrayDeque<RecordingVideoFrame>()
        val audio = ArrayDeque<AudioPacket>()
        val clock = RecordingAudioTimeline(epochNs)
        val state = AtomicReference("starting")
        val stopping = AtomicBoolean()
        val stoppedAtNs = AtomicLong()
        val failure = AtomicReference("")
        val droppedVideo = AtomicLong()
        val droppedAudio = AtomicLong()
        val droppedAudioSamples = AtomicLong()
        val audioOfferedSamples = AtomicLong()
        @Volatile var width = 0
        @Volatile var height = 0
        @Volatile var rotation = 0
        @Volatile var frames = 0L
        @Volatile var videoInputFrames = 0L
        @Volatile var audioInputSamples = 0L
        @Volatile var audioPackets = 0L
        @Volatile var firstVideoPts = -1L
        @Volatile var lastVideoPts = -1L
        @Volatile var lastAudioPts = -1L
        @Volatile var lastAudioEndUs = 0L
        @Volatile var finalized = false
        @Volatile var diagnosticStage = "starting"
        @Volatile var exceptionType = ""
        @Volatile var codecDiagnostic = ""
        @Volatile var videoCodecName = ""
        @Volatile var audioCodecName = ""
        @Volatile var videoSizeSupported = false
        var lastOfferedVideoNs = -1L
        var firstFrame: RecordingVideoFrame? = null
        var audioPrimingDropped = 0L
        var initialized = false
        var muxer: MediaMuxer? = null
        var videoCodec: MediaCodec? = null
        var audioCodec: MediaCodec? = null
        var color = 0
        var videoTrack = -1
        var audioTrack = -1
        var muxerStarted = false
        var videoInputEos = false
        var audioInputEos = false
        var videoOutputEos = false
        var audioOutputEos = false
        var encodedWaitingBytes = 0
        val encodedWaiting = ArrayList<Encoded>()
        val videoInfo = MediaCodec.BufferInfo()
        val audioInfo = MediaCodec.BufferInfo()
    }

    /** File must be a new caller-owned private MP4 path. State/wantsFrames change before return. */
    fun start(file: File, withAudio: Boolean = true) {
        synchronized(lifecycle) {
            if (closed.get() || active) {
                emit("video_rejected", JSONObject().put("reason", "recorder_busy_or_closed"))
                return
            }
            val target = file.absoluteFile
            if (target.parentFile == null || !target.name.endsWith(".mp4", true)) {
                emit("video_error", JSONObject().put("error", "录像路径无效").put("reason", "invalid_output_path"))
                return
            }
            session.set(Session(target, withAudio, SystemClock.elapsedRealtimeNanos()))
        }
        wake()
    }

    fun offerVideo(frame: RecordingVideoFrame) {
        val s = session.get() ?: return
        if (!wantsFrames) return
        try {
            require(frame.i420.size == RecordingMediaMath.i420Size(frame.width, frame.height))
            RecordingMediaMath.validateRotation(frame.rotation)
            require(frame.timestampNs >= 0)
        } catch (_: IllegalArgumentException) { s.failure.compareAndSet("", "invalid_video_frame"); wake(); return }
        synchronized(s.lock) {
            if (s.stopping.get() || session.get() !== s) return
            if (frame.timestampNs < s.epochNs || frame.timestampNs <= s.lastOfferedVideoNs) {
                s.droppedVideo.incrementAndGet()
                return
            }
            s.lastOfferedVideoNs = frame.timestampNs
            if (s.videos.size >= 3) { s.videos.removeFirst(); s.droppedVideo.incrementAndGet() }
            s.videos.addLast(frame)
        }
        wake()
    }

    /** timestampNs is the first sample's elapsedRealtimeNanos; PCM is copied because sources reuse buffers. */
    fun offerAudio(timestampNs: Long, pcm16Mono: ByteArray) {
        val s = session.get() ?: return
        if (!wantsFrames || !s.withAudio) return
        try { RecordingMediaMath.validatePcm(pcm16Mono) }
        catch (_: IllegalArgumentException) { s.failure.compareAndSet("", "invalid_audio_packet"); wake(); return }
        synchronized(s.lock) {
            if (s.stopping.get() || session.get() !== s) return
            val position = try { s.clock.position(timestampNs, pcm16Mono.size / 2) }
            catch (_: IllegalArgumentException) { s.failure.compareAndSet("", "invalid_audio_timestamp"); null }
            if (position != null) {
                s.audioOfferedSamples.addAndGet(position.samples.toLong())
                if (s.audio.size >= 64) {
                    s.droppedAudio.incrementAndGet()
                    s.droppedAudioSamples.addAndGet(position.samples.toLong())
                } else s.audio.addLast(AudioPacket(position.ptsUs, pcm16Mono.copyOfRange(position.skipSamples * 2, pcm16Mono.size)))
            }
        }
        wake()
    }

    /** Immediately stops accepting inputs. EOS/muxer finalization continue asynchronously. */
    fun stop() {
        val s = session.get() ?: return
        if (s.state.get() !in setOf("starting", "recording", "finalizing")) return
        if (s.stopping.compareAndSet(false, true)) {
            s.stoppedAtNs.set(SystemClock.elapsedRealtimeNanos())
            s.state.set("finalizing")
        }
        wake()
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        stop()
        wake()
    }

    fun snapshot(): JSONObject = session.get()?.let(::stats) ?: JSONObject().put("state", "idle")
        .put("path", "").put("duration_ms", 0).put("captured_duration_ms", 0).put("frames", 0).put("audio_samples", 0)
        .put("dropped_video_frames", 0).put("dropped_audio_packets", 0).put("finalized", false)

    private fun wake(delay: Long = 0) {
        if (scheduled.compareAndSet(false, true)) worker.postDelayed(pump, delay)
    }

    private val pump = object : Runnable {
        override fun run() {
            scheduled.set(false)
            val s = session.get()
            if (s != null && s.state.get() in setOf("starting", "recording", "finalizing")) {
                try { step(s) }
                catch (failure: RecorderFailure) { fail(s, failure.reason) }
                catch (failure: Exception) {
                    s.exceptionType = failure.javaClass.name
                    if (failure is MediaCodec.CodecException) s.codecDiagnostic = failure.diagnosticInfo.orEmpty()
                    Log.e("FirstPersonRecorder", "media failure at ${s.diagnosticStage}", failure)
                    fail(s, "codec_or_storage_failure")
                }
                catch (failure: LinkageError) {
                    s.exceptionType = failure.javaClass.name
                    Log.e("FirstPersonRecorder", "codec unavailable at ${s.diagnosticStage}", failure)
                    fail(s, "codec_unavailable")
                }
            }
            if (active) wake(10) else if (closed.get()) thread.quitSafely()
        }
    }

    private fun step(s: Session) {
        s.failure.get().takeIf { it.isNotEmpty() }?.let { throw RecorderFailure(it) }
        if (!s.initialized) {
            if (!s.file.parentFile!!.isDirectory && !s.file.parentFile!!.mkdirs()) throw RecorderFailure("cannot_create_directory")
            if (s.file.exists() || s.pending.exists() || s.sidecar.exists() || s.sidecarPending.exists()) throw RecorderFailure("output_already_exists")
            s.diagnosticStage = "muxer_create"
            s.muxer = MediaMuxer(s.pending.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            s.initialized = true
        }
        if (s.videoCodec == null) {
            val first = synchronized(s.lock) { s.videos.pollFirst() }
            if (first == null) {
                if (s.stopping.get() || SystemClock.elapsedRealtimeNanos() - s.epochNs >= 10_000_000_000L) throw RecorderFailure("no_video_frame")
                return
            }
            // Retain the first visible frame while the driver configures. Later offers can drop
            // queued frames, but cannot move the beginning of video away from the common clock.
            s.firstFrame = first
            configure(s, first)
        }
        drain(s, true)
        if (s.withAudio) drain(s, false)
        repeat(2) { queueVideo(s) }
        if (s.withAudio) repeat(4) { queueAudio(s) }
        drain(s, true)
        if (s.withAudio) drain(s, false)
        if (!s.stopping.get() && SystemClock.elapsedRealtimeNanos() - s.epochNs >= 10_000_000_000L &&
            (!s.muxerStarted || (s.withAudio && s.audioInputSamples == 0L))) throw RecorderFailure("audio_or_track_start_timeout")
        if (s.stopping.get()) {
            if (s.withAudio && s.audioInputSamples == 0L && synchronized(s.lock) { s.audio.isEmpty() }) throw RecorderFailure("no_audio_samples")
            if (s.stoppedAtNs.get() > 0 && SystemClock.elapsedRealtimeNanos() - s.stoppedAtNs.get() > 5_000_000_000L) throw RecorderFailure("eos_timeout")
            if (s.firstFrame == null && synchronized(s.lock) { s.videos.isEmpty() }) queueEos(s, true)
            if (s.withAudio && synchronized(s.lock) { s.audio.isEmpty() }) queueEos(s, false)
            drain(s, true)
            if (s.withAudio) drain(s, false)
            if (s.videoOutputEos && (!s.withAudio || s.audioOutputEos)) finish(s)
        }
    }

    private fun configure(s: Session, first: RecordingVideoFrame) {
        s.width = first.width; s.height = first.height; s.rotation = first.rotation
        s.muxer!!.setOrientationHint(first.rotation)
        s.diagnosticStage = "video_encoder_create"
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        s.videoCodec = codec
        s.videoCodecName = codec.name
        val capabilities = codec.codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC)
        s.videoSizeSupported = capabilities.videoCapabilities?.isSizeSupported(s.width, s.height) == true
        if (!s.videoSizeSupported) throw RecorderFailure("unsupported_video_size")
        if (capabilities.videoCapabilities?.areSizeAndRateSupported(s.width, s.height, 15.0) != true) throw RecorderFailure("unsupported_video_frame_rate")
        val colors = capabilities.colorFormats
        s.color = listOf(MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar).firstOrNull { it in colors }
            ?: throw RecorderFailure("unsupported_yuv_codec")
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, s.width, s.height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, s.color)
            setInteger(MediaFormat.KEY_BIT_RATE, (s.width.toLong() * s.height * 5).coerceIn(500_000, 8_000_000).toInt())
            setInteger(MediaFormat.KEY_FRAME_RATE, 15)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
        }
        s.diagnosticStage = "video_encoder_configure"
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        s.diagnosticStage = "video_encoder_start"
        codec.start()
        if (s.withAudio) {
            s.diagnosticStage = "audio_encoder_create"
            val audio = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
            s.audioCodec = audio
            s.audioCodecName = audio.name
            s.diagnosticStage = "audio_encoder_configure"
            audio.configure(MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, 16_000, 1).apply {
                setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
                setInteger(MediaFormat.KEY_BIT_RATE, 64_000)
                setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 32_768)
            }, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            s.diagnosticStage = "audio_encoder_start"
            audio.start()
        }
    }

    private fun queueVideo(s: Session) {
        if (s.videoInputEos || (s.firstFrame == null && synchronized(s.lock) { s.videos.isEmpty() })) return
        val codec = s.videoCodec ?: return
        s.diagnosticStage = "video_dequeue_input"
        val index = codec.dequeueInputBuffer(0)
        if (index < 0) return
        val frame = s.firstFrame?.also { s.firstFrame = null } ?: synchronized(s.lock) { s.videos.removeFirst() }
        if (frame.width != s.width || frame.height != s.height || frame.rotation != s.rotation) throw RecorderFailure("video_geometry_changed")
        s.diagnosticStage = "video_get_input_image"
        val image = if (s.color == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) codec.getInputImage(index)
            else runCatching { codec.getInputImage(index) }.getOrNull()
        if (image != null) {
            try {
                val crop = image.cropRect
                require(crop.width() >= s.width && crop.height() >= s.height)
                require(crop.left % 2 == 0 && crop.top % 2 == 0)
                val planes = image.planes.mapIndexed { planeIndex, plane ->
                    val divisor = if (planeIndex == 0) 1 else 2
                    RecordingPlane(plane.buffer, plane.rowStride, plane.pixelStride,
                        plane.buffer.position() + crop.top / divisor * plane.rowStride + crop.left / divisor * plane.pixelStride)
                }
                s.diagnosticStage = "video_copy_input_planes"
                RecordingMediaMath.copyI420ToPlanes(frame.i420, s.width, s.height, planes)
            } finally { image.close() }
        } else {
            if (s.color == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) throw RecorderFailure("flexible_input_image_unavailable")
            val input = codec.getInputBuffer(index) ?: throw RecorderFailure("video_input_unavailable")
            input.clear()
            RecordingMediaMath.copyPackedI420(frame.i420, s.width, s.height, input,
                s.color == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
        }
        val pts = RecordingMediaMath.videoPtsUs(frame.timestampNs, s.epochNs, s.lastVideoPts)
        s.diagnosticStage = "video_queue_input"
        codec.queueInputBuffer(index, 0, frame.i420.size, pts, 0)
        if (s.firstVideoPts < 0) s.firstVideoPts = pts
        s.lastVideoPts = pts
        s.videoInputFrames++
        if (!s.muxerStarted) maybeStartMuxer(s)
    }

    private fun queueAudio(s: Session) {
        if (s.audioInputEos || synchronized(s.lock) { s.audio.isEmpty() }) return
        val codec = s.audioCodec ?: return
        s.diagnosticStage = "audio_dequeue_input"
        val index = codec.dequeueInputBuffer(0)
        if (index < 0) return
        val packet = synchronized(s.lock) { s.audio.peekFirst() } ?: throw RecorderFailure("audio_queue_invariant")
        val input = codec.getInputBuffer(index) ?: throw RecorderFailure("audio_input_unavailable")
        input.clear()
        val count = minOf(input.remaining(), packet.bytes.size - packet.offset) and -2
        if (count <= 0) throw RecorderFailure("audio_input_too_small")
        input.put(packet.bytes, packet.offset, count)
        val pts = packet.ptsUs + packet.offset / 2 * 1_000_000L / 16_000
        s.diagnosticStage = "audio_queue_input"
        codec.queueInputBuffer(index, 0, count, pts, 0)
        packet.offset += count
        s.lastAudioPts = pts
        s.audioInputSamples += count / 2
        s.lastAudioEndUs = pts + count / 2 * 1_000_000L / 16_000
        if (packet.offset == packet.bytes.size) synchronized(s.lock) { s.audio.removeFirst() }
    }

    private fun queueEos(s: Session, video: Boolean) {
        if (if (video) s.videoInputEos else s.audioInputEos) return
        val codec = (if (video) s.videoCodec else s.audioCodec) ?: return
        s.diagnosticStage = if (video) "video_queue_eos" else "audio_queue_eos"
        val index = codec.dequeueInputBuffer(0)
        if (index < 0) return
        val pts = if (video) (s.lastVideoPts + 66_667).coerceAtLeast(0) else s.lastAudioEndUs.coerceAtLeast(0)
        codec.queueInputBuffer(index, 0, 0, pts, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        if (video) s.videoInputEos = true else s.audioInputEos = true
    }

    private fun drain(s: Session, video: Boolean) {
        val codec = (if (video) s.videoCodec else s.audioCodec) ?: return
        val info = if (video) s.videoInfo else s.audioInfo
        repeat(32) {
            s.diagnosticStage = if (video) "video_drain_output" else "audio_drain_output"
            val index = codec.dequeueOutputBuffer(info, 0)
            if (index == MediaCodec.INFO_TRY_AGAIN_LATER) return
            if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (s.muxerStarted) throw RecorderFailure("codec_format_changed_after_mux_start")
                val track = s.muxer!!.addTrack(codec.outputFormat)
                if (video) s.videoTrack = track else s.audioTrack = track
                maybeStartMuxer(s)
            } else if (index >= 0) {
                try {
                    if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                        val output = codec.getOutputBuffer(index) ?: throw RecorderFailure("codec_output_unavailable")
                        output.position(info.offset); output.limit(info.offset + info.size)
                        val packet = Encoded(video, info.presentationTimeUs, info.flags, ByteArray(info.size).also { output.get(it) })
                        if (s.muxerStarted) writeSample(s, packet) else {
                            s.encodedWaitingBytes += packet.bytes.size
                            if (s.encodedWaitingBytes > 4 * 1024 * 1024 || s.encodedWaiting.size >= 512) throw RecorderFailure("track_start_buffer_full")
                            s.encodedWaiting.add(packet)
                        }
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        if (video) s.videoOutputEos = true else s.audioOutputEos = true
                    }
                } finally { codec.releaseOutputBuffer(index, false) }
            }
        }
    }

    private fun maybeStartMuxer(s: Session) {
        if (s.muxerStarted || s.firstVideoPts < 0 || s.videoTrack < 0 || (s.withAudio && s.audioTrack < 0)) return
        s.diagnosticStage = "muxer_start"
        s.muxer!!.start()
        s.muxerStarted = true
        if (!s.stopping.get()) s.state.set("recording")
        s.encodedWaiting.sortedBy { it.ptsUs }.forEach { writeSample(s, it) }
        s.encodedWaiting.clear(); s.encodedWaitingBytes = 0
        emit("video_started", stats(s))
    }

    private fun writeSample(s: Session, packet: Encoded) {
        // Codec timestamps retain the shared elapsed clock. Only the MP4 is rebased to its first
        // real video input, avoiding a camera cold-start hole in the container timeline.
        if (s.firstVideoPts < 0) throw RecorderFailure("missing_container_pts_origin")
        val containerPts = packet.ptsUs - s.firstVideoPts
        // AAC priming preceding this zero point is not valid captured media. Do not shift later audio.
        if (!packet.video && containerPts < 0) { s.audioPrimingDropped++; return }
        if (containerPts < 0) throw RecorderFailure("negative_video_pts")
        val info = MediaCodec.BufferInfo().apply { set(0, packet.bytes.size, containerPts, packet.flags) }
        s.diagnosticStage = if (packet.video) "muxer_write_video" else "muxer_write_audio"
        s.muxer!!.writeSampleData(if (packet.video) s.videoTrack else s.audioTrack, ByteBuffer.wrap(packet.bytes), info)
        if (packet.video) s.frames++ else s.audioPackets++
    }

    private fun finish(s: Session) {
        if (!s.muxerStarted || s.frames == 0L || (s.withAudio && (s.audioPackets == 0L || s.audioInputSamples == 0L))) throw RecorderFailure("missing_media_track")
        s.diagnosticStage = "muxer_stop"
        s.muxer!!.stop()
        s.muxer!!.release(); s.muxer = null; s.muxerStarted = false
        releaseCodecs(s)
        if (s.pending.length() <= 0 || s.file.exists()) throw RecorderFailure("invalid_final_file")
        val success = stats(s).put("state", "saved").put("finalized", true)
        s.sidecarPending.writeText(success.toString(2))
        if (!s.pending.renameTo(s.file)) throw RecorderFailure("final_rename_failed")
        if (!s.sidecarPending.renameTo(s.sidecar)) {
            s.file.renameTo(s.pending)
            throw RecorderFailure("sidecar_rename_failed")
        }
        s.finalized = true
        s.state.set("saved")
        clearQueues(s)
        emit("video_saved", stats(s))
    }

    private fun fail(s: Session, reason: String) {
        s.failure.set(reason)
        s.stopping.set(true)
        s.stoppedAtNs.compareAndSet(0, SystemClock.elapsedRealtimeNanos())
        releaseCodecs(s)
        runCatching { s.muxer?.release() }; s.muxer = null
        s.state.set("error")
        clearQueues(s)
        runCatching {
            if (s.initialized) {
                s.sidecarPending.writeText(stats(s).toString(2))
                if (!s.sidecar.exists()) s.sidecarPending.renameTo(s.sidecar)
            }
        }
        emit("video_error", stats(s).put("error", failureMessage(reason)).put("reason", reason))
    }

    private fun releaseCodecs(s: Session) {
        listOfNotNull(s.videoCodec, s.audioCodec).forEach { codec -> runCatching { codec.stop() }; runCatching { codec.release() } }
        s.videoCodec = null; s.audioCodec = null
    }
    private fun clearQueues(s: Session) {
        synchronized(s.lock) { s.videos.clear(); s.audio.clear() }
        s.firstFrame = null
        s.encodedWaiting.clear(); s.encodedWaitingBytes = 0
    }

    private fun stats(s: Session): JSONObject {
        val end = s.stoppedAtNs.get().takeIf { it > 0 } ?: SystemClock.elapsedRealtimeNanos()
        val captureStart = if (s.firstVideoPts >= 0) s.epochNs + s.firstVideoPts * 1_000 else null
        val capturedDurationMs = captureStart?.let { ((end - it) / 1_000_000).coerceAtLeast(0) } ?: 0L
        val fps = if (s.frames > 1 && s.lastVideoPts > s.firstVideoPts) (s.frames - 1) * 1_000_000.0 / (s.lastVideoPts - s.firstVideoPts) else 0.0
        val counts = synchronized(s.lock) { s.videos.size to s.audio.size }
        return JSONObject().put("state", s.state.get()).put("path", s.file.absolutePath).put("pending_path", s.pending.absolutePath)
            .put("sidecar_path", s.sidecar.absolutePath).put("start_elapsed_ns", s.epochNs)
            .put("duration_ms", ((end - s.epochNs) / 1_000_000).coerceAtLeast(0))
            .put("capture_start_elapsed_ns", captureStart ?: JSONObject.NULL).put("captured_duration_ms", capturedDurationMs)
            .put("startup_duration_ms", captureStart?.let { (it - s.epochNs) / 1_000_000 } ?: JSONObject.NULL)
            .put("container_pts_origin_us", if (s.firstVideoPts >= 0) s.firstVideoPts else JSONObject.NULL)
            .put("media_duration_ms", if (s.firstVideoPts >= 0)
                (maxOf(s.lastVideoPts + 66_667, s.lastAudioEndUs) - s.firstVideoPts).coerceAtLeast(0) / 1_000 else 0)
            .put("width", s.width).put("height", s.height).put("rotation", s.rotation).put("target_fps", 15).put("fps", fps)
            .put("video_codec", s.videoCodecName).put("audio_codec", s.audioCodecName).put("color_format", s.color)
            .put("video_size_supported", s.videoSizeSupported).put("diagnostic_stage", s.diagnosticStage)
            .put("exception_type", s.exceptionType).put("codec_diagnostic", s.codecDiagnostic)
            .put("frames", s.frames).put("video_input_frames", s.videoInputFrames).put("with_audio", s.withAudio)
            .put("audio_samples", s.audioInputSamples).put("audio_encoded_packets", s.audioPackets)
            .put("audio_priming_packets_dropped", s.audioPrimingDropped)
            .put("audio_offered_samples", s.audioOfferedSamples.get()).put("audio_rate", 16_000).put("audio_channels", 1)
            .put("first_video_pts_us", s.firstVideoPts).put("last_video_pts_us", s.lastVideoPts).put("last_audio_pts_us", s.lastAudioPts)
            .put("dropped_video_frames", s.droppedVideo.get()).put("dropped_audio_packets", s.droppedAudio.get())
            .put("dropped_audio_samples", s.droppedAudioSamples.get()).put("queued_video_frames", counts.first).put("queued_audio_packets", counts.second)
            .put("finalized", s.finalized).put("reason", s.failure.get())
    }

    private fun failureMessage(reason: String): String = when (reason) {
        "no_video_frame" -> "未收到相机画面，录像未保存"
        "no_audio_samples", "audio_or_track_start_timeout" -> "未收到有效声音或音轨未就绪，录像未保存"
        "output_already_exists" -> "录像文件已经存在，请重新开始录像"
        "unsupported_video_size", "unsupported_video_frame_rate" -> "当前相机分辨率或帧率不支持录像"
        "eos_timeout" -> "录像编码结束超时，未完成文件已保留"
        else -> "录像保存失败，未完成文件已保留"
    }

    private fun emit(name: String, data: JSONObject) { main.post { runCatching { onEvent(name, data) } } }
}
