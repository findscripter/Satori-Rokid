package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class SessionJournalTest {
    @get:Rule val temporary = TemporaryFolder()

    private class Clock(var elapsedNs: Long = 5_000_000_000L, var unixMs: Long = 1_000_000L) {
        fun sample() = SessionClockSample(elapsedNs, unixMs)
        fun advance(ns: Long) { elapsedNs += ns; unixMs += ns / 1_000_000L }
    }

    private fun rows(directory: File, name: String): List<JSONObject> = File(directory, name)
        .readLines(Charsets.UTF_8).map { JSONObject(it) }

    private fun expectIo(block: () -> Unit) {
        try { block(); fail("Expected IOException") } catch (_: IOException) { }
    }

    @Test fun firstEventCreatesManifestAndClockAnchorAndKeepsLegacyFields() {
        val directory = temporary.newFolder()
        val clock = Clock()
        val journal = SessionJournal({ directory }, "test-version", 17, clock::sample)
        assertTrue(journal.append("confirmed", JSONObject().put("revision", 42)))
        val manifest = JSONObject(File(directory, "session.json").readText())
        val event = rows(directory, "events.jsonl").single()
        val anchor = rows(directory, "clock-anchors.jsonl").single()
        assertEquals("test-version", manifest.getString("app_version"))
        assertEquals("android_boot_17", manifest.getString("boot_id"))
        assertEquals(17, manifest.getInt("boot_count"))
        assertEquals(clock.elapsedNs, manifest.getLong("start_elapsed_ns"))
        assertEquals(clock.unixMs, manifest.getLong("start_unix_ms"))
        assertEquals("confirmed", event.getString("name"))
        assertEquals(42, event.getJSONObject("data").getInt("revision"))
        for (row in listOf(event, anchor)) {
            assertEquals(manifest.getString("session_id"), row.getString("session_id"))
            assertEquals("android.elapsedRealtimeNanos", row.getString("clock_domain"))
            assertEquals(clock.elapsedNs, row.getLong("elapsed_ns"))
            assertEquals(clock.unixMs, row.getLong("unix_ms"))
            assertEquals(clock.elapsedNs / 1_000_000L, row.getLong("at_ms"))
        }
        assertEquals("session_start", anchor.getString("reason"))
        assertEquals(manifest.getString("session_id"), journal.ensureSession("later").getString("session_id"))
        assertEquals(1, rows(directory, "clock-anchors.jsonl").size)
    }

    @Test fun payloadCannotReplaceIdentityOrClockAndInputsAndOutputsAreDetached() {
        val directory = temporary.newFolder()
        val clock = Clock()
        val journal = SessionJournal({ directory }, "test", 0, clock::sample)
        val manifest = journal.ensureSession("start")
        val sessionId = manifest.getString("session_id")
        manifest.put("session_id", "changed-by-caller")
        val source = JSONObject().put("session_id", "spoof").put("boot_id", "spoof")
            .put("clock_domain", "spoof").put("elapsed_ns", -1).put("unix_ms", -1).put("at_ms", -1)
            .put("nested", JSONObject().put("value", 1))
        val envelope = journal.envelope(source)
        source.getJSONObject("nested").put("value", 99)
        assertEquals(sessionId, envelope.getString("session_id"))
        assertEquals("android_boot_0", envelope.getString("boot_id"))
        assertEquals(SessionJournal.CLOCK_DOMAIN, envelope.getString("clock_domain"))
        assertEquals(clock.elapsedNs, envelope.getLong("elapsed_ns"))
        assertEquals(clock.elapsedNs / 1_000_000L, envelope.getLong("at_ms"))
        assertEquals(1, envelope.getJSONObject("nested").getInt("value"))
        assertEquals("spoof", source.getString("session_id"))
        assertEquals(sessionId, journal.ensureSession("again").getString("session_id"))
    }

    @Test fun statesDeduplicateByRevisionButRetainTheWholeChangedSnapshot() {
        val directory = temporary.newFolder()
        val clock = Clock()
        val journal = SessionJournal({ directory }, "test", 1, clock::sample)
        val first = JSONObject().put("revision", 4).put("phase", "executing")
            .put("goal", JSONObject().put("text", "收好书"))
        assertTrue(journal.state(first, "start"))
        clock.advance(30)
        assertFalse(journal.state(JSONObject(first.toString()).put("phase", "ignored-same-revision"), "poll"))
        first.put("revision", 5).put("phase", "paused")
        assertTrue(journal.state(first, "command"))
        first.getJSONObject("goal").put("text", "later mutation")
        val saved = rows(directory, "states.jsonl")
        assertEquals(2, saved.size)
        assertEquals(listOf(4L, 5L), saved.map { it.getLong("revision") })
        assertEquals("command", saved[1].getString("reason"))
        assertEquals("paused", saved[1].getJSONObject("state").getString("phase"))
        assertEquals("收好书", saved[1].getJSONObject("state").getJSONObject("goal").getString("text"))
        expectIo { journal.state(JSONObject().put("phase", "missing-revision"), "bad") }
    }

    @Test fun statusIsAtMostOncePerElapsedSecondEvenWhenWallClockJumps() {
        val directory = temporary.newFolder()
        val clock = Clock()
        val journal = SessionJournal({ directory }, "test", 1, clock::sample)
        assertTrue(journal.status(JSONObject().put("camera_active", true)))
        clock.advance(999_999_999)
        clock.unixMs += 3_600_000
        assertFalse(journal.status(JSONObject().put("camera_active", false)))
        clock.advance(1)
        assertTrue(journal.status(JSONObject().put("camera_active", false)))
        val saved = rows(directory, "status.jsonl")
        assertEquals(2, saved.size)
        assertTrue(saved[0].getBoolean("camera_active"))
        assertFalse(saved[1].getBoolean("camera_active"))
        assertEquals(1_000_000_000L, saved[1].getLong("elapsed_ns") - saved[0].getLong("elapsed_ns"))
        val anchor = journal.clockAnchor("wall_clock_check")
        assertEquals(clock.unixMs, anchor.getLong("unix_ms"))
        assertEquals(clock.elapsedNs, anchor.getLong("elapsed_ns"))
        assertEquals(2, rows(directory, "clock-anchors.jsonl").size)
    }

    @Test fun newDirectoryGetsNewIdentityAndResetsRevisionAndStatusDeduplication() {
        var directory = temporary.newFolder("first")
        val firstDirectory = directory
        val clock = Clock()
        val journal = SessionJournal({ directory }, "test", 3, clock::sample)
        journal.state(JSONObject().put("revision", 8), "start")
        journal.status(JSONObject().put("phase", "idle"))
        val first = journal.ensureSession("first")
        directory = temporary.newFolder("second")
        assertTrue(journal.state(JSONObject().put("revision", 8), "start"))
        assertTrue(journal.status(JSONObject().put("phase", "idle")))
        val second = journal.ensureSession("second")
        assertNotEquals(first.getString("session_id"), second.getString("session_id"))
        assertEquals(1, rows(firstDirectory, "states.jsonl").size)
        assertEquals(1, rows(directory, "states.jsonl").size)
        assertEquals(second.getString("session_id"), rows(directory, "status.jsonl").single().getString("session_id"))
    }

    @Test fun missingBootCountIsExplicitlyUnknownRatherThanInvented() {
        val directory = temporary.newFolder()
        val journal = SessionJournal({ directory }, "test", -1) { SessionClockSample(1, 2) }
        val identity = journal.ensureSession("test")
        assertEquals("unknown", identity.getString("boot_id"))
        assertEquals("unavailable_do_not_merge", identity.getString("boot_identity_kind"))
    }

    @Test fun writeFailuresAreVisibleAndDoNotConsumeStateOrStatusSlots() {
        val directory = temporary.newFolder()
        val journal = SessionJournal({ directory }, "test", 1) { SessionClockSample(1, 2) }
        journal.ensureSession("start")
        val blockedState = File(directory, "states.jsonl").apply { mkdir() }
        expectIo { journal.state(JSONObject().put("revision", 1), "blocked") }
        assertTrue(blockedState.delete())
        assertTrue(journal.state(JSONObject().put("revision", 1), "retry"))
        val blockedStatus = File(directory, "status.jsonl").apply { mkdir() }
        expectIo { journal.status(JSONObject()) }
        assertTrue(blockedStatus.delete())
        assertTrue(journal.status(JSONObject()))
        File(directory, "events.jsonl").mkdir()
        expectIo { journal.append("blocked", JSONObject()) }
    }

    @Test fun missingDirectoryAndForeignSessionFailWithoutOverwritingSavedData() {
        expectIo { SessionJournal({ null }, "test", 1) { SessionClockSample(1, 2) }.ensureSession("missing") }
        val file = temporary.newFile()
        expectIo { SessionJournal({ file }, "test", 1) { SessionClockSample(1, 2) }.append("bad", JSONObject()) }
        val directory = temporary.newFolder()
        val original = "{\"session_id\":\"keep-this\"}"
        File(directory, "session.json").writeText(original)
        expectIo { SessionJournal({ directory }, "test", 1) { SessionClockSample(1, 2) }.ensureSession("foreign") }
        assertEquals(original, File(directory, "session.json").readText())
    }

    @Test fun concurrentEventsKeepOneSessionAndCompleteJsonLines() {
        val directory = temporary.newFolder()
        val journal = SessionJournal({ directory }, "test", 1) { SessionClockSample(1, 2) }
        val pool = Executors.newFixedThreadPool(4)
        val ready = CountDownLatch(4)
        val start = CountDownLatch(1)
        val futures = (0 until 4).map { worker -> pool.submit {
            ready.countDown()
            assertTrue(start.await(5, TimeUnit.SECONDS))
            repeat(10) { index -> journal.append("event", JSONObject().put("worker", worker).put("index", index)) }
        } }
        try {
            assertTrue(ready.await(5, TimeUnit.SECONDS)); start.countDown()
            futures.forEach { it.get(10, TimeUnit.SECONDS) }
        } finally { pool.shutdownNow() }
        val saved = rows(directory, "events.jsonl")
        assertEquals(40, saved.size)
        assertEquals(1, saved.map { it.getString("session_id") }.distinct().size)
        assertEquals(40, saved.map { "${it.getJSONObject("data").getInt("worker")}:${it.getJSONObject("data").getInt("index")}" }.distinct().size)
        assertEquals(1, rows(directory, "clock-anchors.jsonl").size)
    }
}
