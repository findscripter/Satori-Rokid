package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Temporal fixtures follow the Python gate's observable motion/keyframe contract. */
class StandaloneFlowGateTest {
    private fun metrics(head: Double = 0.0, local: Double = 0.0, area: Double = 0.0, changed: Double = 0.0) =
        StandaloneFlowMetrics(head, local, area, changed, 40)

    private fun assertKey(sample: JSONObject, state: String, reason: String?) {
        assertEquals(state, sample.getString("state"))
        assertEquals(reason != null, sample.getBoolean("key"))
        if (reason == null) assertTrue(sample.isNull("reason")) else assertEquals(reason, sample.getString("reason"))
    }

    @Test fun firstFrameAndLongStillViewAreRetained() {
        val policy = StandaloneFlowPolicy()
        assertKey(policy.start(0), "unknown", "start")
        assertKey(policy.update(200, metrics()), "still", null)
        assertKey(policy.update(4_999, metrics()), "still", null)
        assertKey(policy.update(5_000, metrics()), "still", "heartbeat")
    }

    @Test fun handActivityIsRetainedAtStartDuringAndAfterSettling() {
        val policy = StandaloneFlowPolicy()
        policy.start(0)
        assertKey(policy.update(200, metrics(local = .14, area = .1)), "hands", "hands_start")
        assertKey(policy.update(1_000, metrics(local = .08)), "hands", null)
        assertKey(policy.update(2_200, metrics(changed = .04)), "hands", "hands_ongoing")
        assertKey(policy.update(2_400, metrics()), "still", null)
        assertKey(policy.update(2_799, metrics()), "still", null)
        assertKey(policy.update(2_800, metrics()), "still", "hands_settled")
    }

    @Test fun compensatedDifferenceCatchesFastSmallObjects() {
        val policy = StandaloneFlowPolicy()
        policy.start(0)
        assertKey(policy.update(200, metrics(local = .01, area = .001, changed = .07)), "hands", "hands_start")
    }

    @Test fun tinyResidualAreaAloneDoesNotCountAsHands() {
        val policy = StandaloneFlowPolicy()
        policy.start(0)
        assertKey(policy.update(200, metrics(local = .2, area = .001)), "still", null)
    }

    @Test fun cameraMotionTakesPrecedenceAndUsesHysteresis() {
        val policy = StandaloneFlowPolicy()
        policy.start(0)
        assertKey(policy.update(200, metrics(head = .11, local = .3, area = .3, changed = .3)), "head", null)
        assertKey(policy.update(400, metrics(head = .06)), "head", null)
        assertKey(policy.update(600, metrics(head = .03)), "still", null)
        assertKey(policy.update(1_000, metrics()), "still", "view_settled")
    }

    @Test fun headTurnDoesNotLosePendingHandSettlement() {
        val policy = StandaloneFlowPolicy()
        policy.start(0)
        policy.update(200, metrics(changed = .1))
        assertKey(policy.update(400, metrics(head = .2)), "head", null)
        policy.update(600, metrics())
        assertKey(policy.update(1_000, metrics()), "still", "hands_settled")
    }

    @Test fun shortTrackingLossKeepsPendingActivityButLongLossDropsIt() {
        val short = StandaloneFlowPolicy()
        short.start(0)
        short.update(200, metrics(changed = .1))
        assertKey(short.update(400, null), "unknown", null)
        short.update(600, metrics())
        assertKey(short.update(1_000, metrics()), "still", "hands_settled")

        val long = StandaloneFlowPolicy()
        long.start(0)
        long.update(200, metrics(changed = .1))
        long.update(400, null)
        assertKey(long.update(3_400, null), "unknown", "unclear")
        long.update(3_600, metrics())
        assertKey(long.update(4_000, metrics()), "still", null)
    }

    @Test fun gapResetsPendingActivityAndStartsNewView() {
        val policy = StandaloneFlowPolicy()
        policy.start(0)
        policy.update(200, metrics(changed = .1))
        assertKey(policy.start(2_000, "gap"), "unknown", "gap")
        policy.update(2_200, metrics())
        assertKey(policy.update(2_600, metrics()), "still", null)
    }

    @Test fun nativeInitializationFailureFallsBackWithoutDecodingOrThrowing() {
        var initializationAttempts = 0
        val gate = StandaloneFlowGate { initializationAttempts++; false }
        try {
            assertKey(gate.analyze(byteArrayOf(1), 0), "unavailable", "fallback")
            val waiting = gate.analyze(byteArrayOf(1), 4_999)
            assertKey(waiting, "unavailable", null)
            assertFalse(waiting.getBoolean("available"))
            assertTrue(waiting.getString("error").isNotBlank())
            assertKey(gate.analyze(byteArrayOf(1), 5_000), "unavailable", "fallback")
            assertEquals(1, initializationAttempts)
            gate.reset()
            assertKey(gate.analyze(byteArrayOf(1), 5_001), "unavailable", "fallback")
            assertEquals(2, initializationAttempts)
        } finally { gate.close() }
    }

    @Test fun missingNativeLibraryAlsoFallsBackAndCloseIsIdempotent() {
        val gate = StandaloneFlowGate { throw UnsatisfiedLinkError("test native library unavailable") }
        assertKey(gate.analyze(byteArrayOf(1), 10), "unavailable", "fallback")
        gate.close()
        gate.close()
        assertKey(gate.analyze(byteArrayOf(1), 20), "unavailable", "fallback")
        assertEquals("closed", gate.analyze(byteArrayOf(1), 21).getString("error"))
    }
}
