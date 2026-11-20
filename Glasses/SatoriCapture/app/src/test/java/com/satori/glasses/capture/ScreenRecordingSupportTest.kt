package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test

class ScreenRecordingSupportTest {
    @Test fun displayDimensionsAreReadInputsAndScaleWithoutInventingARGeometry() {
        assertEquals(ScreenCaptureSize(640, 480), ScreenCaptureSize.fromDisplay(640, 480))
        assertEquals(ScreenCaptureSize(720, 1280), ScreenCaptureSize.fromDisplay(1080, 1920))
        assertEquals(ScreenCaptureSize(1280, 720), ScreenCaptureSize.fromDisplay(3840, 2160))
        val odd = ScreenCaptureSize.fromDisplay(641, 481)
        assertEquals(0, odd.width % 2); assertEquals(0, odd.height % 2)
        assertTrue(kotlin.math.abs(odd.width.toDouble() / odd.height - 641.0 / 481) < .01)
    }

    @Test fun malformedDisplaySizesAreRejected() {
        for ((w, h) in listOf(0 to 480, 640 to -1, 1 to 1, 16385 to 720)) {
            try { ScreenCaptureSize.fromDisplay(w, h); fail("invalid display dimensions accepted") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun containerTimelineUsesMediaPTSAndNeverInventsElapsedClockCalibration() {
        val timeline = ScreenRecordingTimeline()
        assertNull(timeline.originUs); assertEquals(0L, timeline.durationUs())
        assertEquals(0L, timeline.append(4_000_000))
        assertEquals(66_666L, timeline.append(4_066_666))
        assertEquals(2L, timeline.frames); assertEquals(66_666L, timeline.durationUs())
        for (invalid in listOf(-1L, 4_066_666L, 4_000_000L)) {
            try { timeline.append(invalid); fail("invalid screen PTS accepted") }
            catch (_: IllegalArgumentException) { }
        }
        assertEquals(2L, timeline.frames)
        assertEquals(133_333L, timeline.append(4_133_333))
    }
}
