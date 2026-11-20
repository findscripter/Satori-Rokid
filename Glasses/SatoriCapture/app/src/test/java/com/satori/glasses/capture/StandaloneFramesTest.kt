package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test

class StandaloneFramesTest {
    private fun frame(at: Long, key: String? = null) = StandaloneFrame("f$at", at, byteArrayOf(1), key)

    @Test fun selectionKeepsNewestAndSpreadsAcrossWindow() {
        val buffer = StandaloneFrameBuffer()
        (1..100).forEach { buffer.add(frame(it * 1000L)) }
        val frames = buffer.select(10_000, 100_000, 12)
        assertEquals(12, frames.size)
        assertEquals(100_000L, frames.last().atMs)
        assertTrue(frames.first().atMs < 20_000)
        assertEquals(frames.map { it.atMs }.sorted(), frames.map { it.atMs })
    }

    @Test fun gapsClearAllPreviousEvidenceAndEnforceMonotonicIds() {
        val buffer = StandaloneFrameBuffer()
        buffer.add(frame(20))
        buffer.add(frame(10))
        assertEquals(listOf(20L), buffer.select(0, 30).map { it.atMs })
        buffer.clear()
        assertTrue(buffer.select(0, 30).isEmpty())
    }

    @Test fun noOldFramesAndKeyFramesGetPriority() {
        val buffer = StandaloneFrameBuffer()
        (1..100).forEach { buffer.add(frame(it * 1000L, if (it == 51) "hands_settled" else null)) }
        val frames = buffer.select(50_000, 100_000, 4)
        assertTrue(frames.all { it.atMs > 50_000 })
        assertTrue(frames.any { it.atMs == 51_000L })
        assertEquals(100_000L, frames.last().atMs)
    }

    @Test fun ringIsBoundedAndTimesAreNotFutureOrOlderThanNinetySeconds() {
        val buffer = StandaloneFrameBuffer(20)
        (1..1000).forEach { buffer.add(frame(it * 1000L)) }
        assertEquals(20, buffer.size)
        assertTrue(buffer.select(0, 999_000).all { it.atMs <= 999_000 && it.atMs >= 909_000 })
    }

    @Test fun coldCameraFrameAfterThreeSecondsIsAcceptedWithoutWaitingToDeadline() {
        val buffer = StandaloneFrameBuffer()
        var at = 1000L
        val frames = buffer.awaitFresh(999, 11_000, { at }, { false }) { delay ->
            at += delay
            if (at == 5000L) buffer.add(frame(at))
        }
        assertEquals(listOf(5000L), frames.map { it.atMs })
        assertEquals(5000L, at)
    }

    @Test fun snapshotTimeoutDoesNotReuseOldOrFutureFrames() {
        val buffer = StandaloneFrameBuffer()
        buffer.add(frame(900)); buffer.add(frame(20_000))
        var at = 1000L
        assertTrue(buffer.awaitFresh(999, 11_000, { at }, { false }) { at += it }.isEmpty())
        assertEquals(11_000L, at)
    }

    @Test fun cancellationWinsWhenAFrameArrivesAndStopsWaiting() {
        val buffer = StandaloneFrameBuffer()
        var at = 1000L
        var cancelled = false
        val frames = buffer.awaitFresh(999, 11_000, { at }, { cancelled }) {
            at += it
            buffer.add(frame(at))
            cancelled = true
        }
        assertTrue(frames.isEmpty())
        assertEquals(1050L, at)
    }

    @Test fun cancellationIsCheckedAgainBeforeReturningAnAvailableFrame() {
        val buffer = StandaloneFrameBuffer().apply { add(frame(1000)) }
        var checks = 0
        assertTrue(buffer.awaitFresh(999, 11_000, { 1000 }, { ++checks > 1 }) { fail("must not wait") }.isEmpty())
    }

    @Test fun waitingDoesNotLockOutTheCameraProducer() {
        val buffer = StandaloneFrameBuffer()
        var at = 1000L
        val frames = buffer.awaitFresh(999, 11_000, { at }, { false }) {
            at += it
            val producer = Thread { buffer.add(frame(at)) }
            producer.isDaemon = true
            producer.start()
            producer.join(1000)
            assertFalse("camera producer blocked by snapshot wait", producer.isAlive)
        }
        assertEquals(listOf(1050L), frames.map { it.atMs })
    }
}
