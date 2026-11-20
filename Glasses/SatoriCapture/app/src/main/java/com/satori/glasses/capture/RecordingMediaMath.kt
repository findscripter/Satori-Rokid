package com.satori.glasses.capture

import java.nio.ByteBuffer
import kotlin.math.max

internal data class RecordingPlane(val bytes: ByteBuffer, val rowStride: Int, val pixelStride: Int,
                                   val origin: Int = bytes.position())

/** Pure sample/time/layout calculations shared by the real codecs and JVM boundary fixtures. */
internal object RecordingMediaMath {
    const val AUDIO_RATE = 16_000
    fun i420Size(width: Int, height: Int): Int {
        require(width in 2..4096 && height in 2..4096 && width % 2 == 0 && height % 2 == 0)
        return Math.multiplyExact(width, height) * 3 / 2
    }
    fun validateRotation(rotation: Int) { require(rotation in listOf(0, 90, 180, 270)) }
    fun validatePcm(pcm: ByteArray) { require(pcm.isNotEmpty() && pcm.size % 2 == 0 && pcm.size <= AUDIO_RATE * 2) }
    fun videoPtsUs(timestampNs: Long, epochNs: Long, previous: Long): Long {
        require(timestampNs >= epochNs && epochNs >= 0)
        return max((timestampNs - epochNs) / 1_000, previous + 1)
    }

    fun copyI420ToPlanes(source: ByteArray, width: Int, height: Int, planes: List<RecordingPlane>) {
        require(source.size == i420Size(width, height) && planes.size == 3)
        var sourceOffset = 0
        planes.forEachIndexed { index, plane ->
            val columns = if (index == 0) width else width / 2
            val rows = if (index == 0) height else height / 2
            require(plane.rowStride >= (columns - 1) * plane.pixelStride + 1 && plane.pixelStride > 0 && plane.origin >= 0)
            val last = plane.origin.toLong() + (rows - 1L) * plane.rowStride + (columns - 1L) * plane.pixelStride
            require(last < plane.bytes.limit())
            if (plane.pixelStride == 1) {
                val row = plane.bytes.duplicate()
                for (y in 0 until rows) {
                    row.position(plane.origin + y * plane.rowStride)
                    row.put(source, sourceOffset, columns)
                    sourceOffset += columns
                }
            } else for (y in 0 until rows) for (x in 0 until columns) {
                plane.bytes.put(plane.origin + y * plane.rowStride + x * plane.pixelStride, source[sourceOffset++])
            }
        }
    }

    /** Used only for explicitly advertised planar-I420 or semiplanar-NV12 codec formats. */
    fun copyPackedI420(source: ByteArray, width: Int, height: Int, output: ByteBuffer, nv12: Boolean) {
        require(source.size == i420Size(width, height) && output.remaining() >= source.size)
        if (!nv12) { output.put(source); return }
        val luma = width * height
        val chroma = luma / 4
        output.put(source, 0, luma)
        for (index in 0 until chroma) { output.put(source[luma + index]); output.put(source[luma + chroma + index]) }
    }
}

internal data class RecordingAudioPosition(val skipSamples: Int, val samples: Int, val ptsUs: Long)

/** Stable 16 kHz sample clock; dropped queue packets still advance it, large capture gaps retain real time. */
internal class RecordingAudioTimeline(private val epochNs: Long) {
    private var originNs: Long? = null
    private var nextSample = 0L
    private var lastTimestamp = -1L
    init { require(epochNs >= 0) }
    fun position(timestampNs: Long, sampleCount: Int): RecordingAudioPosition? {
        require(timestampNs >= 0 && sampleCount > 0 && timestampNs > lastTimestamp)
        lastTimestamp = timestampNs
        val before = (epochNs - timestampNs).coerceAtLeast(0)
        val skip = if (before == 0L) 0 else if (before >= sampleCount * 1_000_000_000L / RecordingMediaMath.AUDIO_RATE) sampleCount
            else ((before * RecordingMediaMath.AUDIO_RATE + 999_999_999L) / 1_000_000_000L).toInt()
        if (skip == sampleCount) return null
        val effectiveNs = timestampNs + skip * 1_000_000_000L / RecordingMediaMath.AUDIO_RATE
        val origin = originNs ?: effectiveNs.also { originNs = it }
        val expectedNs = origin + nextSample * 1_000_000_000L / RecordingMediaMath.AUDIO_RATE
        if (effectiveNs - expectedNs > 50_000_000L) {
            val gap = effectiveNs - origin
            val fromTimestamp = gap / 1_000_000_000L * RecordingMediaMath.AUDIO_RATE +
                gap % 1_000_000_000L * RecordingMediaMath.AUDIO_RATE / 1_000_000_000L
            nextSample = max(nextSample, fromTimestamp)
        }
        val pts = (origin - epochNs) / 1_000 + nextSample * 1_000_000L / RecordingMediaMath.AUDIO_RATE
        nextSample += sampleCount - skip
        return RecordingAudioPosition(skip, sampleCount - skip, pts)
    }
}
