package com.satori.glasses.capture

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CaptureRequest
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraState
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/**
 * CameraX ImageAnalysis turned into JPEG at a throttled rate; no preview (the mono-green display is not a
 * viewfinder). Each frame keeps the sensor timestamp (timestampSource=REALTIME on this unit, i.e. the
 * elapsedRealtimeNanos base shared with SensorEvent) and the clockwise rotation that makes it upright.
 * Pixels are sent unrotated; the PC rotates them.
 */
class CameraStreamer(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val onFrame: (JSONObject, ByteArray) -> Unit,
    private val onEvent: (String, JSONObject) -> Unit,
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "camera-analysis") }
    private val jpegBuffer = ByteArrayOutputStream(128 * 1024)
    private val main = Handler(Looper.getMainLooper())
    private var provider: ProcessCameraProvider? = null
    private var boundSize: Size? = null
    private var requestedSize: Size? = null
    private var recoveryDelayMs = INITIAL_RECOVERY_MS
    private var wanted = false
    private var lastState: CameraState.Type? = null
    private var nv21: ByteArray? = null
    private var frameIndex = 0L
    private data class VideoTarget(val generation: Long, val sink: (RecordingVideoFrame) -> Unit)
    private val videoGeneration = AtomicLong()
    @Volatile private var videoTarget: VideoTarget? = null
    @Volatile private var analysisEnabled = true

    @Volatile private var intervalNs = 500_000_000L
    @Volatile private var quality = 80
    @Volatile private var lastSentNs = 0L
    @Volatile var framesEncoded = 0L
        private set

    /** Main thread only. */
    fun start(config: CaptureConfig) {
        wanted = true
        applyRate(config)
        val size = Size(config.width, config.height)
        requestedSize = size
        main.removeCallbacksAndMessages(null)
        provider?.let {
            bind(it, size)
            return
        }
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val p = future.get()
                provider = p
                if (wanted) bind(p, size)
            } catch (e: Exception) {
                onEvent("camera_error", JSONObject().put("stage", "provider").put("error", e.toString()))
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /** Main thread only. */
    fun stop() {
        wanted = false
        main.removeCallbacksAndMessages(null)
        provider?.unbindAll()
        boundSize = null
    }

    fun applyRate(config: CaptureConfig) {
        intervalNs = (1_000_000_000.0 / config.fps).toLong()
        quality = config.jpegQuality
    }

    fun setVideoSink(sink: ((RecordingVideoFrame) -> Unit)?): Long {
        val generation = videoGeneration.incrementAndGet()
        videoTarget = sink?.let { VideoTarget(generation, it) }
        return generation
    }
    fun setAnalysisEnabled(enabled: Boolean) { analysisEnabled = enabled }

    fun shutdown() {
        stop()
        executor.shutdown()
    }

    @androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun bind(p: ProcessCameraProvider, size: Size) {
        p.unbindAll()
        val selector = ResolutionSelector.Builder()
            .setResolutionStrategy(ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build()
        val builder = ImageAnalysis.Builder()
            .setResolutionSelector(selector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
        // A 15 fps sensor rate covers 1-10 fps sampling and saves power on the 210 mAh battery.
        Camera2Interop.Extender(builder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(15, 15))
        val analysis = builder.build()
        analysis.setAnalyzer(executor, ::analyze)
        try {
            val camera = p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            boundSize = size
            lastState = null
            camera.cameraInfo.cameraState.removeObservers(owner)
            camera.cameraInfo.cameraState.observe(owner) { state ->
                if (state.type == CameraState.Type.OPEN) recoveryDelayMs = INITIAL_RECOVERY_MS
                if (state.type == lastState && state.error == null) return@observe
                lastState = state.type
                onEvent(
                    "camera_state",
                    JSONObject().put("state", state.type.name).put("error", state.error?.code ?: JSONObject.NULL),
                )
                // CLOSED with an error is terminal for CameraX (recoverable errors retry by themselves, e.g. while
                // the Rokid assistant holds the camera); rebind with backoff so capture is not stuck forever.
                if (state.type == CameraState.Type.CLOSED && state.error != null && wanted) scheduleRecovery()
            }
            onEvent("camera_bound", JSONObject().put("width", size.width).put("height", size.height))
        } catch (e: Exception) {
            boundSize = null
            onEvent("camera_error", JSONObject().put("stage", "bind").put("error", e.toString()))
        }
    }

    private fun scheduleRecovery() {
        val delay = recoveryDelayMs
        recoveryDelayMs = (recoveryDelayMs * 2).coerceAtMost(MAX_RECOVERY_MS)
        main.removeCallbacksAndMessages(null)
        main.postDelayed({
            val p = provider
            val size = requestedSize
            if (wanted && p != null && size != null) {
                onEvent("camera_recovery", JSONObject().put("after_ms", delay))
                bind(p, size)
            }
        }, delay)
    }

    private fun analyze(image: ImageProxy) {
        try {
            val timestampNs = image.imageInfo.timestamp
            videoTarget?.let { target ->
                try {
                    val p = image.planes
                    val frame = RecordingVideoFrame(image.width, image.height, image.imageInfo.rotationDegrees,
                        timestampNs, RecordingYuv.i420(image.width, image.height,
                            RecordingYuv.Plane(p[0].buffer, p[0].rowStride, p[0].pixelStride),
                            RecordingYuv.Plane(p[1].buffer, p[1].rowStride, p[1].pixelStride),
                            RecordingYuv.Plane(p[2].buffer, p[2].rowStride, p[2].pixelStride)))
                    target.sink(frame)
                } catch (e: Exception) {
                    onEvent("video_frame_error", JSONObject().put("error", "YUV frame copy failed").put("type", e.javaClass.simpleName)
                        .put("video_generation", target.generation).put("ts_ns", timestampNs))
                }
            }
            if (!analysisEnabled) return
            if (lastSentNs != 0L && timestampNs - lastSentNs < intervalNs) return
            // Advance by whole intervals so the average rate matches fps despite the 15 fps sensor cadence;
            // resynchronize after a gap (pause, camera hiccup).
            lastSentNs = if (lastSentNs != 0L && timestampNs - lastSentNs < 2 * intervalNs) lastSentNs + intervalNs else timestampNs
            val started = SystemClock.elapsedRealtimeNanos()
            val jpeg = encodeJpeg(image)
            val header = JSONObject()
                .put("type", "frame")
                .put("ts_ns", timestampNs)
                .put("index", frameIndex++)
                .put("width", image.width)
                .put("height", image.height)
                .put("rotation", image.imageInfo.rotationDegrees)
                .put("quality", quality)
                .put("encode_ms", (SystemClock.elapsedRealtimeNanos() - started) / 1e6)
            framesEncoded++
            onFrame(header, jpeg)
        } catch (e: Exception) {
            onEvent("camera_error", JSONObject().put("stage", "encode").put("error", e.toString()))
        } finally {
            image.close()
        }
    }

    /** YUV_420_888 (any row/pixel stride) -> NV21 -> JPEG. */
    private fun encodeJpeg(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val needed = width * height * 3 / 2
        val out = nv21?.takeIf { it.size == needed } ?: ByteArray(needed).also { nv21 = it }
        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]
        val yBuffer = y.buffer
        var index = 0
        if (y.pixelStride == 1) {
            for (row in 0 until height) {
                yBuffer.position(row * y.rowStride)
                yBuffer.get(out, index, width)
                index += width
            }
        } else {
            for (row in 0 until height) {
                for (col in 0 until width) out[index++] = yBuffer.get(row * y.rowStride + col * y.pixelStride)
            }
        }
        val vBuffer = v.buffer
        val uBuffer = u.buffer
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                out[index++] = vBuffer.get(row * v.rowStride + col * v.pixelStride)
                out[index++] = uBuffer.get(row * u.rowStride + col * u.pixelStride)
            }
        }
        jpegBuffer.reset()
        YuvImage(out, ImageFormat.NV21, width, height, null)
            .compressToJpeg(Rect(0, 0, width, height), quality, jpegBuffer)
        return jpegBuffer.toByteArray()
    }

    companion object {
        private const val INITIAL_RECOVERY_MS = 2000L
        private const val MAX_RECOVERY_MS = 30_000L
    }
}
