package com.satori.glasses.capture

import java.nio.ByteBuffer

/** Copy actual row/pixel strides, including a plane buffer's initial offset, without retaining an ImageProxy. */
object RecordingYuv {
    data class Plane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)

    fun i420(width: Int, height: Int, y: Plane, u: Plane, v: Plane): ByteArray {
        require(width in 2..4096 && height in 2..4096 && width % 2 == 0 && height % 2 == 0)
        val pixels = width * height
        val output = ByteArray(pixels + pixels / 2)
        fun copy(source: Plane, cols: Int, rows: Int, offset: Int) {
            require(source.rowStride > 0 && source.pixelStride > 0)
            val buffer = source.buffer.duplicate()
            val base = buffer.position()
            val last = base.toLong() + (rows - 1L) * source.rowStride + (cols - 1L) * source.pixelStride
            require(last < buffer.limit()) { "truncated YUV plane" }
            var index = offset
            for (row in 0 until rows) {
                if (source.pixelStride == 1) {
                    buffer.position(base + row * source.rowStride)
                    buffer.get(output, index, cols)
                    index += cols
                } else for (col in 0 until cols) {
                    output[index++] = buffer.get(base + row * source.rowStride + col * source.pixelStride)
                }
            }
        }
        copy(y, width, height, 0)
        copy(u, width / 2, height / 2, pixels)
        copy(v, width / 2, height / 2, pixels + pixels / 4)
        return output
    }
}

/** A manual recording outlives task waits, removal and loss of Internet; folding/storage exhaustion stop both. */
data class StandaloneCaptureNeeds(val analysis: Boolean, val video: Boolean) {
    val camera: Boolean get() = analysis || video
    companion object {
        fun calculate(taskRunnable: Boolean, network: Boolean, modelConfigured: Boolean, videoRequested: Boolean,
                      deviceOpen: Boolean, storageAvailable: Boolean): StandaloneCaptureNeeds {
            val permitted = deviceOpen && storageAvailable
            return StandaloneCaptureNeeds(permitted && taskRunnable && network && modelConfigured, permitted && videoRequested)
        }
    }
}
