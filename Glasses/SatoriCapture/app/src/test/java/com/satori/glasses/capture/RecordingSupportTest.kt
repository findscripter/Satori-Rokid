package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer

class RecordingSupportTest {
    @Test fun paddedAndInterleavedPlanesBecomeI420WithoutChangingTheirPositions() {
        val y = ByteBuffer.wrap(byteArrayOf(99, 1, 2, 9, 9, 3, 4, 9, 9)).apply { position(1) }
        val u = ByteBuffer.wrap(byteArrayOf(99, 5, 9)).apply { position(1) }
        val v = ByteBuffer.wrap(byteArrayOf(99, 6, 9)).apply { position(1) }
        val result = RecordingYuv.i420(2, 2, RecordingYuv.Plane(y, 4, 1), RecordingYuv.Plane(u, 2, 2), RecordingYuv.Plane(v, 2, 2))
        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 5, 6), result)
        assertEquals(1, y.position()); assertEquals(1, u.position()); assertEquals(1, v.position())
    }

    @Test fun malformedPlanesAndOddDimensionsAreRejected() {
        val plane = RecordingYuv.Plane(ByteBuffer.allocate(1), 2, 1)
        for ((width, height) in listOf(2 to 2, 3 to 2, 2 to 3, 0 to 2)) {
            try { RecordingYuv.i420(width, height, plane, plane, plane); fail("accepted invalid frame") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun recordingKeepsCameraRunningWithoutNetworkOrModelConfiguration() {
        val needs = StandaloneCaptureNeeds.calculate(false, false, false, true, true, true)
        assertFalse(needs.analysis); assertTrue(needs.video); assertTrue(needs.camera)
    }

    @Test fun waitingForConfirmationAndPausedTasksDoNotPauseVideo() {
        val needs = StandaloneCaptureNeeds.calculate(false, true, true, true, true, true)
        assertFalse(needs.analysis); assertTrue(needs.camera)
        val normal = StandaloneCaptureNeeds.calculate(true, true, true, false, true, true)
        assertTrue(normal.analysis); assertFalse(normal.video); assertTrue(normal.camera)
    }

    @Test fun foldingOrStorageExhaustionStopsBothCameraConsumers() {
        for ((deviceOpen, storage) in listOf(false to true, true to false)) {
            val needs = StandaloneCaptureNeeds.calculate(true, true, true, true, deviceOpen, storage)
            assertFalse(needs.camera); assertFalse(needs.analysis); assertFalse(needs.video)
        }
    }
}
