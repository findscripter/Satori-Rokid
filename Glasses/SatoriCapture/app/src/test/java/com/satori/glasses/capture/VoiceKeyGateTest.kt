package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test

/** Timed routing only; these fixtures do not inject a hardware key or start a microphone. */
class VoiceKeyGateTest {
    @Test fun twoShortRemoteReleasesToggleRecordingOnceWithoutRequiringAHeldKey() {
        val gate = RecordingKeyGate()
        assertFalse(gate.press(1000))
        assertTrue(gate.press(1350))
        assertFalse(gate.press(1650))
    }

    @Test fun recordingGestureRejectsDuplicatesAndDistantSingleSyncPresses() {
        val gate = RecordingKeyGate()
        assertFalse(gate.press(0))
        assertFalse(gate.press(20))
        assertTrue(gate.press(300))
        assertFalse(gate.press(1000))
        assertFalse(gate.press(1800))
        gate.reset()
        assertFalse(gate.press(2000))
    }

    @Test fun firstSignalAtZeroIsAccepted() {
        val gate = VoiceKeyGate { 0L }
        assertTrue(gate.accept("broadcast:AI_START"))
    }

    @Test fun aiBroadcastAndProgBlueFromOneGestureProduceOneAcceptedAction() {
        var now = 1000L
        val gate = VoiceKeyGate { now }
        var actions = 0
        if (gate.accept("broadcast:AI_START")) actions++
        now += 20
        if (gate.accept("key:186")) actions++
        assertEquals(1, actions)
    }

    @Test fun keyThenBroadcastIsDeduplicatedInEitherDeliveryOrder() {
        var now = 1000L
        val gate = VoiceKeyGate { now }
        assertTrue(gate.accept("key:186"))
        now += 100
        assertFalse(gate.accept("broadcast:AI_START"))
    }

    @Test fun sameSourceDebouncesBelow150MillisecondsButAllowsTheBoundary() {
        var now = 1000L
        val gate = VoiceKeyGate { now }
        assertTrue(gate.accept("broadcast:AI_START"))
        now = 1149
        assertFalse(gate.accept("broadcast:AI_START"))
        now = 1150
        assertTrue(gate.accept("broadcast:AI_START"))
    }

    @Test fun differentSourcesRequire500MillisecondsWithoutExtendingOnRejectedEvents() {
        var now = 1000L
        val gate = VoiceKeyGate { now }
        assertTrue(gate.accept("broadcast:AI_START"))
        now = 1499
        assertFalse(gate.accept("key:186"))
        now = 1500
        assertTrue(gate.accept("key:186"))
    }

    @Test fun rejectedCrossSourceSignalDoesNotChangeTheSourceOfTheLastAcceptedGesture() {
        var now = 1000L
        val gate = VoiceKeyGate { now }
        assertTrue(gate.accept("broadcast:AI_START"))
        now = 1050
        assertFalse(gate.accept("key:186"))
        now = 1150
        assertTrue(gate.accept("broadcast:AI_START"))
    }

    @Test fun anotherPhysicalLongPressCanBeAcceptedAfterTheCrossSourceWindow() {
        var now = 1000L
        val gate = VoiceKeyGate { now }
        assertTrue(gate.accept("broadcast:AI_START"))
        now = 1020
        assertFalse(gate.accept("key:186"))
        now = 1500
        assertTrue(gate.accept("broadcast:LONG_PRESS"))
        now = 1520
        assertFalse(gate.accept("key:186"))
    }
}
