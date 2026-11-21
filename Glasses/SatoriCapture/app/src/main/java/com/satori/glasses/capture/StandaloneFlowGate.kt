package com.satori.glasses.capture

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.io.Closeable
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.round
import kotlin.math.roundToInt

/**
 * Feed consecutive camera JPEGs on one background worker, before sparsely selecting model inputs.
 * Uses real LK forward/backward tracks + RANSAC camera motion, dense Farneback residuals, and
 * motion-compensated frame differences, as in the desktop flow_gate.py. Processing Mats never
 * exceed 160 x 120; bitmap decoding is subsampled before converting to a Mat.
 *
 * This class never starts capture or performs network I/O. Native initialization/processing failure
 * is explicit (state=unavailable), with one timed fallback keyframe every five seconds. reset()
 * retries initialization and forgets prior frames; close() releases native state permanently.
 */
class StandaloneFlowGate internal constructor(private val initialize: () -> Boolean) : Closeable {
    constructor() : this({ OpenCVLoader.initLocal() })

    private val policy = StandaloneFlowPolicy()
    private var previous: Mat? = null
    private var previousAt: Long? = null
    private var lastInputAt: Long? = null
    private var lastFallbackAt: Long? = null
    private var initializationAttempted = false
    private var ready = false
    private var nativeError = ""
    private var closed = false

    @Synchronized
    fun analyze(jpeg: ByteArray, atMs: Long): JSONObject {
        if (closed) return fallback(atMs, "closed")
        if (atMs < 0 || lastInputAt?.let { atMs <= it } == true) {
            return flowSample(atMs.coerceAtLeast(0), "unknown", null, null)
                .put("error", "non_monotonic_timestamp")
        }
        lastInputAt = atMs
        if (!initializationAttempted) {
            initializationAttempted = true
            try {
                ready = initialize()
                if (!ready) nativeError = "opencv_init_failed"
            } catch (error: Exception) {
                nativeError = describe(error)
            } catch (error: LinkageError) {
                nativeError = describe(error)
            }
        }
        if (!ready) return fallback(atMs, nativeError)
        val started = System.nanoTime()
        try {
            val current = decode(jpeg)
            val old = previous
            val oldAt = previousAt
            previous = null
            previousAt = null
            var retained = false
            try {
                val result = if (old == null || oldAt == null) {
                    policy.start(atMs)
                } else if (atMs - oldAt > 1_500 || old.size() != current.size()) {
                    policy.start(atMs, "gap")
                } else {
                    policy.update(atMs, measure(old, current, ((atMs - oldAt) / 1000.0).coerceAtLeast(.05)))
                }
                result.put("analysis_width", current.cols()).put("analysis_height", current.rows())
                    .put("elapsed_ms", (System.nanoTime() - started) / 1_000_000.0)
                previous = current
                previousAt = atMs
                retained = true
                return result
            } finally {
                old?.release()
                if (!retained) current.release()
            }
        } catch (_: InvalidImage) {
            clearPrevious()
            policy.reset()
            return flowSample(atMs, "unknown", null, null).put("error", "jpeg_decode_failed")
        } catch (error: Exception) {
            return disable(atMs, error)
        } catch (error: LinkageError) {
            return disable(atMs, error)
        }
    }

    @Synchronized
    fun reset() {
        clearPrevious()
        policy.reset()
        lastInputAt = null
        lastFallbackAt = null
        initializationAttempted = false
        ready = false
        nativeError = ""
    }

    @Synchronized
    override fun close() {
        if (closed) return
        reset()
        closed = true
    }

    private fun clearPrevious() {
        previous?.release()
        previous = null
        previousAt = null
    }

    private fun disable(atMs: Long, error: Throwable): JSONObject {
        clearPrevious()
        ready = false
        nativeError = describe(error)
        return fallback(atMs, nativeError)
    }

    private fun fallback(atMs: Long, error: String): JSONObject {
        val now = atMs.coerceAtLeast(0)
        val key = lastFallbackAt?.let { now - it >= 5_000 } ?: true
        if (key) lastFallbackAt = now
        return flowSample(now, "unavailable", if (key) "fallback" else null, null)
            .put("available", false).put("method", "timed_fallback").put("error", error)
    }

    private fun describe(error: Throwable) = "${error.javaClass.simpleName}: ${error.message.orEmpty()}".take(200)

    private class InvalidImage : Exception()

    private fun decode(jpeg: ByteArray): Mat {
        if (jpeg.isEmpty() || jpeg.size > 16 * 1024 * 1024) throw InvalidImage()
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0 || bounds.outWidth > 32_768 || bounds.outHeight > 32_768) {
            throw InvalidImage()
        }
        var sample = 1
        while (bounds.outWidth / (sample * 2) >= 160 || bounds.outHeight / (sample * 2) >= 120) sample *= 2
        val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }) ?: throw InvalidImage()
        var scaled = bitmap
        val gray = Mat()
        var success = false
        try {
            val scale = minOf(1.0, 160.0 / bitmap.width, 120.0 / bitmap.height)
            if (scale < 1) {
                scaled = Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).roundToInt().coerceAtLeast(1),
                    (bitmap.height * scale).roundToInt().coerceAtLeast(1), true)
            }
            NativeMats().use { mats ->
                val rgba = mats.own(Mat())
                Utils.bitmapToMat(scaled, rgba)
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            }
            success = true
            return gray
        } finally {
            if (!success) gray.release()
            if (scaled !== bitmap) scaled.recycle()
            bitmap.recycle()
        }
    }

    private fun measure(old: Mat, current: Mat, seconds: Double): StandaloneFlowMetrics? = NativeMats().use { mats ->
        val corners = mats.own(MatOfPoint())
        Imgproc.goodFeaturesToTrack(old, corners, 200, .01, 7.0, mats.own(Mat()), 7)
        if (corners.rows() < 12) return@use null
        val source = mats.own(MatOfPoint2f(*corners.toArray()))
        val ahead = mats.own(MatOfPoint2f())
        val back = mats.own(MatOfPoint2f())
        val forwardStatus = mats.own(MatOfByte())
        val backStatus = mats.own(MatOfByte())
        val criteria = TermCriteria(TermCriteria.EPS or TermCriteria.COUNT, 20, .03)
        Video.calcOpticalFlowPyrLK(old, current, source, ahead, forwardStatus, mats.own(MatOfFloat()), Size(21.0, 21.0), 3, criteria)
        if (ahead.empty()) return@use null
        Video.calcOpticalFlowPyrLK(current, old, ahead, back, backStatus, mats.own(MatOfFloat()), Size(21.0, 21.0), 3, criteria)
        if (back.empty()) return@use null
        val p = source.toArray()
        val q = ahead.toArray()
        val r = back.toArray()
        val forwardOk = forwardStatus.toArray()
        val backwardOk = backStatus.toArray()
        val from = ArrayList<Point>()
        val to = ArrayList<Point>()
        repeat(minOf(p.size, q.size, r.size, forwardOk.size, backwardOk.size)) { index ->
            if (forwardOk[index].toInt() != 0 && backwardOk[index].toInt() != 0 &&
                hypot(r[index].x - p[index].x, r[index].y - p[index].y) < 1.0) {
                from.add(p[index])
                to.add(q[index])
            }
        }
        if (from.size < 12) return@use null
        val fit = mats.own(Calib3d.estimateAffinePartial2D(mats.own(MatOfPoint2f(*from.toTypedArray())),
            mats.own(MatOfPoint2f(*to.toTypedArray())), mats.own(Mat()), Calib3d.RANSAC, 1.5))
        if (fit.empty()) return@use null
        val model = DoubleArray(6)
        fit.get(0, 0, model)
        if (model.any { !it.isFinite() }) return@use null
        val width = old.cols()
        val height = old.rows()
        val diagonal = hypot(width.toDouble(), height.toDouble())
        val probes = arrayOf(Point(0.0, 0.0), Point(width.toDouble(), 0.0), Point(0.0, height.toDouble()),
            Point(width.toDouble(), height.toDouble()), Point(width / 2.0, height / 2.0))
        val head = probes.sumOf { point ->
            hypot(model[0] * point.x + model[1] * point.y + model[2] - point.x,
                model[3] * point.x + model[4] * point.y + model[5] - point.y)
        } / probes.size / diagonal / seconds

        val flow = mats.own(Mat())
        Video.calcOpticalFlowFarneback(old, current, flow, .5, 3, 15, 3, 5, 1.2, 0)
        val vectors = FloatArray(width * height * 2)
        flow.get(0, 0, vectors)
        val rates = DoubleArray(width * height)
        var moving = 0
        for (y in 0 until height) for (x in 0 until width) {
            val index = y * width + x
            val predictedX = model[0] * x + model[1] * y + model[2] - x
            val predictedY = model[3] * x + model[4] * y + model[5] - y
            val rate = hypot(vectors[index * 2] - predictedX, vectors[index * 2 + 1] - predictedY) / diagonal / seconds
            rates[index] = rate
            if (rate >= .12) moving++
        }
        rates.sort()
        val percentileAt = .95 * (rates.size - 1)
        val low = floor(percentileAt).toInt()
        val high = (low + 1).coerceAtMost(rates.lastIndex)
        val local = rates[low] + (rates[high] - rates[low]) * (percentileAt - low)

        val size = Size(width.toDouble(), height.toDouble())
        val warped = mats.own(Mat())
        Imgproc.warpAffine(old, warped, fit, size, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar.all(0.0))
        val valid = mats.own(Mat())
        Imgproc.warpAffine(mats.own(Mat(height, width, CvType.CV_8UC1, Scalar.all(255.0))), valid, fit, size,
            Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar.all(0.0))
        val difference = mats.own(Mat())
        Core.absdiff(warped, current, difference)
        Imgproc.GaussianBlur(difference, difference, Size(3.0, 3.0), 0.0)
        val validBytes = ByteArray(width * height)
        val differenceBytes = ByteArray(width * height)
        valid.get(0, 0, validBytes)
        difference.get(0, 0, differenceBytes)
        var visible = 0
        var changed = 0
        validBytes.indices.forEach { index ->
            if ((validBytes[index].toInt() and 255) == 255) {
                visible++
                if ((differenceBytes[index].toInt() and 255) >= 30) changed++
            }
        }
        StandaloneFlowMetrics(head, local, moving.toDouble() / rates.size,
            changed.toDouble() / visible.coerceAtLeast(1), from.size)
    }

    private class NativeMats : Closeable {
        private val values = ArrayList<Mat>()
        fun <T : Mat> own(value: T): T = value.also { values.add(it) }
        override fun close() { values.asReversed().forEach { it.release() } }
    }
}

internal data class StandaloneFlowMetrics(
    val headRate: Double, val localRate: Double, val localArea: Double, val changed: Double, val tracks: Int,
)

/** Pure temporal policy: its inputs are measured optical flow, never guessed from elapsed time. */
internal class StandaloneFlowPolicy {
    private var state = "unknown"
    private var pending: String? = null
    private var stillSince: Long? = null
    private var unknownSince: Long? = null
    private var lastKeyAt: Long? = null

    fun reset() {
        state = "unknown"
        pending = null
        stillSince = null
        unknownSince = null
        lastKeyAt = null
    }

    fun start(atMs: Long, reason: String = "start"): JSONObject {
        reset()
        lastKeyAt = atMs
        return flowSample(atMs, state, reason, null)
    }

    fun update(atMs: Long, metrics: StandaloneFlowMetrics?): JSONObject {
        if (metrics == null) {
            state = "unknown"
            stillSince = null
            if (unknownSince == null) unknownSince = atMs
            else if (atMs - unknownSince!! >= 3_000) pending = null
            return flowSample(atMs, state, every("unclear", atMs, 2_000), null)
        }
        unknownSince = null
        val next = when {
            metrics.headRate >= if (state == "head") .04 else .10 -> "head"
            state == "hands" -> if (metrics.localRate >= .05 || metrics.changed >= .03) "hands" else "still"
            (metrics.localRate >= .12 && metrics.localArea >= .02) || metrics.changed >= .06 -> "hands"
            else -> "still"
        }
        val reason = when (next) {
            "hands" -> {
                stillSince = null
                if (pending != "hands") {
                    pending = "hands"
                    mark("hands_start", atMs)
                } else every("hands_ongoing", atMs, 2_000)
            }
            "head" -> {
                stillSince = null
                if (pending == null) pending = "head"
                null
            }
            else -> {
                if (stillSince == null) stillSince = atMs
                if (pending != null && atMs - stillSince!! >= 400) {
                    val settled = if (pending == "hands") "hands_settled" else "view_settled"
                    pending = null
                    mark(settled, atMs)
                } else if (pending == null) every("heartbeat", atMs, 5_000) else null
            }
        }
        state = next
        return flowSample(atMs, state, reason, metrics)
    }

    private fun mark(reason: String, atMs: Long): String { lastKeyAt = atMs; return reason }
    private fun every(reason: String, atMs: Long, interval: Long): String? =
        if (lastKeyAt?.let { atMs - it >= interval } != false) mark(reason, atMs) else null
}

private fun flowSample(atMs: Long, state: String, reason: String?, metrics: StandaloneFlowMetrics?): JSONObject =
    JSONObject().put("timestamp_ms", atMs).put("state", state).put("key", reason != null)
        .put("reason", reason ?: JSONObject.NULL).put("available", state != "unavailable")
        .put("method", "lk_ransac_farneback")
        .put("head_rate", round((metrics?.headRate ?: 0.0) * 10_000) / 10_000)
        .put("local_rate", round((metrics?.localRate ?: 0.0) * 10_000) / 10_000)
        .put("local_area", round((metrics?.localArea ?: 0.0) * 10_000) / 10_000)
        .put("changed", round((metrics?.changed ?: 0.0) * 10_000) / 10_000)
        .put("tracks", metrics?.tracks ?: 0)
