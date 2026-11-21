package com.satori.glasses.capture

import kotlin.math.roundToInt

/** Preserve the actual display aspect ratio while bounding simultaneous camera/screen codec load. */
data class ScreenCaptureSize(val width: Int, val height: Int) {
    companion object {
        fun fromDisplay(width: Int, height: Int): ScreenCaptureSize {
            require(width in 2..16384 && height in 2..16384) { "invalid display dimensions" }
            val scale = minOf(1.0, 1280.0 / maxOf(width, height))
            fun even(value: Int) = ((value * scale).roundToInt() / 2 * 2).coerceAtLeast(2)
            return ScreenCaptureSize(even(width), even(height))
        }
    }
}

/** Surface encoder PTS are media timestamps, not assumed to be Android's elapsedRealtime clock. */
class ScreenRecordingTimeline {
    @Volatile var originUs: Long? = null
        private set
    @Volatile var lastUs: Long? = null
        private set
    @Volatile var frames = 0L
        private set

    @Synchronized fun append(rawPtsUs: Long): Long {
        require(rawPtsUs >= 0) { "negative screen PTS" }
        require(lastUs == null || rawPtsUs > lastUs!!) { "non-increasing screen PTS" }
        if (originUs == null) originUs = rawPtsUs
        lastUs = rawPtsUs
        frames++
        return rawPtsUs - originUs!!
    }

    @Synchronized fun durationUs(): Long = if (frames == 0L) 0 else lastUs!! - originUs!!
}
