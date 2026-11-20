package com.satori.glasses.capture

import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

/** Both readings are taken on this device; Unix time is an index, never the video clock. */
data class SessionClockSample(val elapsedNs: Long, val unixMs: Long)

/**
 * Append-only experiment journal. The owner allocates a fresh run directory and supplies Android's
 * elapsedRealtimeNanos clock. Identity fields are written by this class, not trusted from payloads.
 * All public operations are serialized, including a directory change and its first clock anchor.
 */
class SessionJournal(
    private val directory: () -> File?,
    private val appVersion: String,
    private val bootCount: Int,
    private val clock: () -> SessionClockSample
) {
    private data class Context(val directory: File, val manifest: JSONObject) {
        var ready = false
        var lastRevision: Long? = null
        var lastStatusElapsedNs: Long? = null
    }

    private var context: Context? = null

    @Synchronized
    @Throws(IOException::class)
    fun ensureSession(reason: String): JSONObject {
        val sample = clock()
        return copy(current(reason, sample).manifest)
    }

    @Synchronized
    @Throws(IOException::class)
    fun clockAnchor(reason: String): JSONObject {
        val sample = clock()
        val active = current(reason, sample)
        val row = envelope(active, sample, JSONObject().put("reason", reason)
            .put("schema", "satori.clock-anchor/1"))
        appendLine(active, "clock-anchors.jsonl", row)
        return copy(row)
    }

    @Synchronized
    @Throws(IOException::class)
    fun envelope(data: JSONObject = JSONObject()): JSONObject {
        val sample = clock()
        return envelope(current("first_envelope", sample), sample, data)
    }

    @Synchronized
    @Throws(IOException::class)
    fun append(name: String, data: JSONObject): Boolean {
        val sample = clock()
        val active = current("first_event", sample)
        appendLine(active, "events.jsonl", envelope(active, sample, JSONObject()
            .put("schema", "satori.event/1").put("name", name).put("data", copy(data))))
        return true
    }

    @Synchronized
    @Throws(IOException::class)
    fun state(snapshot: JSONObject, reason: String): Boolean {
        if (!snapshot.has("revision") || snapshot.isNull("revision")) throw IOException("state revision missing")
        val revision = try { snapshot.getLong("revision") }
        catch (failure: Exception) { throw IOException("state revision invalid", failure) }
        val sample = clock()
        val active = current("first_state", sample)
        if (active.lastRevision == revision) return false
        appendLine(active, "states.jsonl", envelope(active, sample, JSONObject()
            .put("schema", "satori.state/1").put("reason", reason).put("revision", revision)
            .put("state", copy(snapshot))))
        active.lastRevision = revision
        return true
    }

    @Synchronized
    @Throws(IOException::class)
    fun status(compact: JSONObject): Boolean {
        val sample = clock()
        val active = current("first_status", sample)
        val previous = active.lastStatusElapsedNs
        if (previous != null && sample.elapsedNs - previous < STATUS_INTERVAL_NS) return false
        appendLine(active, "status.jsonl", envelope(active, sample, copy(compact).put("schema", "satori.status/1")))
        active.lastStatusElapsedNs = sample.elapsedNs
        return true
    }

    private fun current(reason: String, sample: SessionClockSample): Context {
        val target = try { directory()?.canonicalFile }
        catch (failure: Exception) { throw IOException("data directory unavailable", failure) }
            ?: throw IOException("data directory unavailable")
        var active = context
        if (active == null || active.directory != target) {
            if (!target.isDirectory && !target.mkdirs()) throw IOException("cannot create data directory")
            // A fresh directory is a session boundary. Never relabel someone else's saved records.
            if (File(target, "session.json").exists()) throw IOException("data session already exists")
            val manifest = JSONObject().put("schema", "satori.session/1")
                .put("session_id", UUID.randomUUID().toString())
                .put("boot_count", bootCount)
                .put("boot_id", if (bootCount >= 0) "android_boot_$bootCount" else "unknown")
                .put("boot_identity_kind", if (bootCount >= 0) "android.settings.global.BOOT_COUNT" else "unavailable_do_not_merge")
                .put("clock_domain", CLOCK_DOMAIN)
                .put("start_elapsed_ns", sample.elapsedNs).put("start_unix_ms", sample.unixMs)
                .put("app_version", appVersion).put("reason", reason)
            active = Context(target, manifest)
            context = active
        }
        if (!active.ready) {
            writeManifest(active)
            appendLine(active, "clock-anchors.jsonl", envelope(active, sample, JSONObject()
                .put("schema", "satori.clock-anchor/1").put("reason", "session_start")))
            active.ready = true
        }
        return active
    }

    private fun envelope(active: Context, sample: SessionClockSample, data: JSONObject): JSONObject = copy(data)
        .put("session_id", active.manifest.getString("session_id"))
        .put("boot_id", active.manifest.getString("boot_id"))
        .put("clock_domain", CLOCK_DOMAIN)
        // Keep the pre-journal event convention: at_ms is elapsedRealtime milliseconds.
        .put("elapsed_ns", sample.elapsedNs).put("unix_ms", sample.unixMs).put("at_ms", sample.elapsedNs / 1_000_000L)

    private fun writeManifest(active: Context) {
        val target = File(active.directory, "session.json")
        val temporary = File(active.directory, ".session.json.tmp")
        try {
            FileOutputStream(temporary).use { output ->
                output.write(active.manifest.toString(2).toByteArray(Charsets.UTF_8))
                output.flush()
                output.fd.sync()
            }
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING)
        } catch (failure: Exception) {
            temporary.delete()
            if (failure is IOException) throw failure
            throw IOException("cannot save session manifest", failure)
        }
    }

    private fun appendLine(active: Context, filename: String, row: JSONObject) {
        try {
            FileOutputStream(File(active.directory, filename), true).use { output ->
                output.write((row.toString() + "\n").toByteArray(Charsets.UTF_8))
                output.flush()
                output.fd.sync()
            }
        } catch (failure: Exception) {
            if (failure is IOException) throw failure
            throw IOException("cannot append data journal", failure)
        }
    }

    private fun copy(value: JSONObject) = JSONObject(value.toString())

    companion object {
        const val CLOCK_DOMAIN = "android.elapsedRealtimeNanos"
        private const val STATUS_INTERVAL_NS = 1_000_000_000L
    }
}
