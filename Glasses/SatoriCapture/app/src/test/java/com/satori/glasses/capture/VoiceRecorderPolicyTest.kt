package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test

class VoiceRecorderPolicyTest {
    @Test fun oldPttDefaultsEndAtTwentySecondsWhileContinuousSourceDoesNot() {
        assertFalse(VoiceRecorder.durationExpired(19_999_999_999L, false, 20000))
        assertTrue(VoiceRecorder.durationExpired(20_000_000_000L, false, 20000))
        assertFalse(VoiceRecorder.durationExpired(86_400_000_000_000L, true, 20000))
    }
    @Test fun configuredShortCaptureStillUsesTheExactMonotonicBoundary() {
        assertFalse(VoiceRecorder.durationExpired(999_999_999L, false, 1000))
        assertTrue(VoiceRecorder.durationExpired(1_000_000_000L, false, 1000))
    }
    @Test fun sharedCaptureKeepsProcessedZeroAndRawTwoWithoutASecondHardwareRead() {
        val frame = shortArrayOf(10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27)
        assertEquals(10, VoiceRecorder.channelSample(frame, 0, 8, false).toInt())
        assertEquals(12, VoiceRecorder.channelSample(frame, 0, 8, true).toInt())
        assertEquals(20, VoiceRecorder.channelSample(frame, 1, 8, false).toInt())
        assertEquals(22, VoiceRecorder.channelSample(frame, 1, 8, true).toInt())
        assertEquals(30, VoiceRecorder.channelSample(shortArrayOf(30), 0, 1, true).toInt())
    }
}
