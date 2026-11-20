package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer

class RecordingMediaMathTest {
    @Test fun videoPtsUsesElapsedSensorTimeAndStaysStrictlyIncreasing() {
        val epoch = 2_000_000_000L
        assertEquals(0L, RecordingMediaMath.videoPtsUs(epoch, epoch, -1))
        assertEquals(66_666L, RecordingMediaMath.videoPtsUs(epoch + 66_666_667, epoch, 0))
        assertEquals(66_667L, RecordingMediaMath.videoPtsUs(epoch + 66_666_668, epoch, 66_666))
        reject { RecordingMediaMath.videoPtsUs(epoch - 1, epoch, -1) }
    }

    @Test fun audioPtsCountsAllOfferedSamplesEvenWhenAnEncodedQueueDropsAPacket() {
        val epoch = 1_000_000_000L
        val timeline = RecordingAudioTimeline(epoch)
        assertEquals(100_000L, timeline.position(epoch + 100_000_000, 1600)!!.ptsUs)
        timeline.position(epoch + 200_000_000, 1600) // intentionally dropped by a full audio queue
        assertEquals(300_000L, timeline.position(epoch + 300_000_000, 1600)!!.ptsUs)
    }

    @Test fun audioClipsSamplesBeforeStartAndPreservesRealCaptureGaps() {
        val epoch = 1_000_000_000L
        val timeline = RecordingAudioTimeline(epoch)
        val first = timeline.position(epoch - 10_000_000, 320)!!
        assertEquals(160, first.skipSamples)
        assertEquals(160, first.samples)
        assertEquals(0L, first.ptsUs)
        assertEquals(500_000L, timeline.position(epoch + 500_000_000, 1600)!!.ptsUs)
        reject { timeline.position(epoch + 400_000_000, 1600) }
        assertNull(RecordingAudioTimeline(epoch).position(epoch - 100_000_000, 160))
    }

    @Test fun smallAudioTimestampJitterDoesNotChangeTheSampleClock() {
        val epoch = 1_000_000_000L
        val timeline = RecordingAudioTimeline(epoch)
        assertEquals(0L, timeline.position(epoch, 1600)!!.ptsUs)
        assertEquals(100_000L, timeline.position(epoch + 105_000_000, 1600)!!.ptsUs)
        assertEquals(200_000L, timeline.position(epoch + 198_000_000, 1600)!!.ptsUs)
    }

    @Test fun pcmAndI420BoundariesAreRejectedBeforeEncoding() {
        assertEquals(6, RecordingMediaMath.i420Size(2, 2))
        assertEquals(460800, RecordingMediaMath.i420Size(640, 480))
        reject { RecordingMediaMath.i420Size(3, 2) }
        reject { RecordingMediaMath.i420Size(0, 2) }
        reject { RecordingMediaMath.i420Size(10000, 10000) }
        RecordingMediaMath.validatePcm(ByteArray(3200))
        reject { RecordingMediaMath.validatePcm(ByteArray(0)) }
        reject { RecordingMediaMath.validatePcm(ByteArray(3)) }
        reject { RecordingMediaMath.validatePcm(ByteArray(32002)) }
        listOf(0, 90, 180, 270).forEach { RecordingMediaMath.validateRotation(it) }
        reject { RecordingMediaMath.validateRotation(45) }
    }

    @Test fun threePlaneCopyHonorsRowStridePixelStrideAndBufferPosition() {
        val source = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val y = ByteBuffer.allocate(16).apply { repeat(capacity()) { put(it, 99.toByte()) }; position(1) }
        val u = ByteBuffer.allocate(8).apply { repeat(capacity()) { put(it, 99.toByte()) }; position(1) }
        val v = ByteBuffer.allocate(8).apply { repeat(capacity()) { put(it, 99.toByte()) }; position(2) }
        RecordingMediaMath.copyI420ToPlanes(source, 4, 2, listOf(
            RecordingPlane(y, 6, 1), RecordingPlane(u, 6, 2), RecordingPlane(v, 6, 2)))
        assertEquals(1, y.get(1).toInt())
        assertEquals(4, y.get(4).toInt())
        assertEquals(5, y.get(7).toInt())
        assertEquals(8, y.get(10).toInt())
        assertEquals(99, y.get(5).toInt())
        assertEquals(9, u.get(1).toInt())
        assertEquals(10, u.get(3).toInt())
        assertEquals(11, v.get(2).toInt())
        assertEquals(12, v.get(4).toInt())
        assertEquals(99, u.get(2).toInt())
    }

    @Test fun planeBoundsAndPackedColorOrderAreExplicit() {
        val source = byteArrayOf(1, 2, 3, 4, 5, 6)
        reject { RecordingMediaMath.copyI420ToPlanes(source, 2, 2, listOf(
            RecordingPlane(ByteBuffer.allocate(3), 2, 1), RecordingPlane(ByteBuffer.allocate(1), 1, 1),
            RecordingPlane(ByteBuffer.allocate(1), 1, 1))) }
        val planar = ByteBuffer.allocate(6)
        RecordingMediaMath.copyPackedI420(source, 2, 2, planar, false)
        assertArrayEquals(source, planar.array())
        val nv12 = ByteBuffer.allocate(12)
        val wide = ByteArray(12) { it.toByte() }
        RecordingMediaMath.copyPackedI420(wide, 4, 2, nv12, true)
        assertArrayEquals(byteArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11), nv12.array())
    }

    private fun reject(action: () -> Unit) {
        try { action(); fail("invalid media input was accepted") } catch (_: IllegalArgumentException) { }
    }
}
