package com.satori.glasses.capture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Rect
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.view.KeyEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.ExecutionException
import java.util.concurrent.FutureTask

/** Real attached-window onDraw + journal acceptance, driven by synthetic Left key events.
 * No phone video/PTS, screen recorder, camera, microphone, model API or credential edit is used.
 * Existing user state is restored byte-for-byte; the test-created journal run remains as scoped evidence.
 */
@RunWith(AndroidJUnit4::class)
class SyncMarkerDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test(timeout = 60_000) fun leftKeyRecordsOnlyTheActualMonotonicDrawAndPreservesTaskState() {
        stage("started")
        assumeTrue("preserve an existing live session", StandaloneService.instance == null && CaptureService.instance == null)
        assumeTrue("preserve a private pending import", !File(target.filesDir, "standalone-import.json").exists())
        assumeTrue("camera permission must be pregranted", target.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        assumeTrue("microphone permission must be pregranted", target.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        val stateFile = File(target.filesDir, "standalone-state.json")
        val stateFiles = listOf(stateFile, File(stateFile.path + ".bak"), File(stateFile.path + ".new"))
        val backups = stateFiles.associateWith { if (it.isFile) it.readBytes() else null }
        var hud: StandaloneActivity? = null
        var service: StandaloneService? = null
        var savedFields: Map<String, Any?> = emptyMap()
        val proof = JSONObject().put("scope", "synthetic key and real HUD onDraw/journal only; phone video PTS is not verified")
        var directory: File? = null
        try {
            stateFiles.drop(1).forEach { check(!it.exists() || it.delete()) }
            stateFile.writeText(StandaloneEngine().snapshot(SystemClock.elapsedRealtime()).toString())
            hud = instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as StandaloneActivity
            await("idle service did not start") { StandaloneService.instance != null }
            service = StandaloneService.instance!!
            val live = requireNotNull(service)
            val panel = requireNotNull(hud)
            savedFields = onMain { listOf("api", "voice", "speech").associateWith { get<Any?>(live, it) } }
            onMain { set(live, "api", null); set(live, "voice", null); set(live, "speech", null) }
            val engine = get<StandaloneEngine>(live, "engine")
            val before = onMain { engine.snapshot() }
            val requestedAfter = SystemClock.elapsedRealtimeNanos()
            onMain {
                assertTrue(StandaloneActivity.isPanelVisible)
                assertEquals(0, panel.window.attributes.flags and WindowManager.LayoutParams.FLAG_SECURE)
                key(panel, KeyEvent.ACTION_DOWN)
                key(panel, KeyEvent.ACTION_DOWN, repeat = 1)
                assertTrue("Left DOWN/repeat must not request synchronization", live.snapshot().getJSONObject("sync_marker").optString("id").isBlank())
                key(panel, KeyEvent.ACTION_UP)
                val id = live.snapshot().getJSONObject("sync_marker").getString("id")
                key(panel, KeyEvent.ACTION_UP)
                assertEquals("duplicate Left UP must be ignored", id, live.snapshot().getJSONObject("sync_marker").getString("id"))
            }
            await("marker never reached a real attached-window onDraw") {
                onMain { live.snapshot().getJSONObject("sync_marker").let { it.has("first_draw_elapsed_ns") && !it.isNull("first_draw_elapsed_ns") } }
            }
            val marker = onMain { live.snapshot().getJSONObject("sync_marker") }
            stage("first_actual_draw_received")
            val requested = marker.getLong("requested_elapsed_ns")
            val firstDraw = marker.getLong("first_draw_elapsed_ns")
            val deadline = marker.getLong("visible_until_elapsed_ns")
            assertTrue(requested >= requestedAfter)
            assertTrue("draw must use elapsedRealtimeNanos and occur after the request", firstDraw >= requested)
            assertTrue("first draw must be within the marker's visible interval", firstDraw <= deadline)
            assertTrue(firstDraw <= SystemClock.elapsedRealtimeNanos())
            onMain {
                val view = panel.window.decorView.findViewWithTag<TextView>("hud_sync_marker")
                assertNotNull(view)
                assertTrue("visible marker must be attached/shown", view.isAttachedToWindow && view.isShown)
                assertEquals(Color.WHITE, view.currentTextColor)
                assertFalse(view is Button || view is EditText || view.isClickable || view.hasOnClickListeners())
                assertEquals("SYNC ${marker.getString("id")} ·${marker.getString("session_tag")}", view.text.toString())
                assertTrue("marker must be within the visible viewport", view.getLocalVisibleRect(Rect()))
                val layout = requireNotNull(view.layout)
                assertEquals("marker must remain one complete line", 1, layout.lineCount)
                assertEquals("marker identifier/session tag must not be ellipsized", 0, layout.getEllipsisCount(0))
                assertTrue("adaptive font must fit the line", layout.getLineWidth(0) <= view.width - view.totalPaddingLeft - view.totalPaddingRight + 1f)
                repeat(3) { view.invalidate(); panel.window.decorView.invalidate() }
            }
            instrumentation.waitForIdleSync()
            SystemClock.sleep(100)
            directory = onMain { get<File?>(live, "directory") }
            val run = requireNotNull(directory)
            val markerFile = File(run, "sync-markers.jsonl")
            val matching = markerRows(markerFile).filter { it.optString("id") == marker.getString("id") }
            assertEquals("repeat drawing must persist exactly one first-draw row", 1, matching.size)
            val row = matching.single()
            assertEquals(marker.getString("session_id"), row.getString("session_id"))
            assertEquals(marker.getString("id"), row.getString("marker_id"))
            assertEquals(requested, row.getLong("requested_elapsed_ns"))
            assertEquals(firstDraw, row.getLong("first_draw_elapsed_ns"))
            assertTrue(row.getString("boot_id").isNotBlank())
            assertTrue(row.getLong("unix_ms") > 0)
            assertEquals("ui_draw_submission_not_final_video_presentation", row.getString("draw_time_kind"))
            assertEquals(firstDraw, onMain { live.snapshot().getJSONObject("sync_marker").getLong("first_draw_elapsed_ns") })
            onMain {
                val after = engine.snapshot()
                assertEquals("Left must not change task revision", before.getLong("revision"), after.getLong("revision"))
                assertEquals(before.getString("phase"), after.getString("phase"))
                assertEquals(before.opt("goal"), after.opt("goal"))
                assertFalse(live.snapshot().getBoolean("recording")); assertFalse(live.snapshot().getBoolean("voice_busy"))
                assertFalse(live.snapshot().getBoolean("capturing")); assertFalse(live.snapshot().getBoolean("busy"))
                assertEquals(0L, live.snapshot().getJSONObject("microphone").getLong("source_starts"))
            }
            proof.put("first_marker", marker).put("first_marker_row", row)
            stage("first_draw_journal_and_task_invariance_verified")
            await("expired marker must return to GONE without an extra user action", timeoutMs = 3500) {
                onMain { panel.window.decorView.findViewWithTag<View>("hud_sync_marker").visibility == View.GONE }
            }
            // A non-key command exercises the public immediate-refresh API as well.
            onMain { live.command("mark_sync") }
            await("new marker must have its own actual draw timestamp") {
                onMain { live.snapshot().getJSONObject("sync_marker").let {
                    it.optString("id") != marker.getString("id") && it.has("first_draw_elapsed_ns") && !it.isNull("first_draw_elapsed_ns")
                } }
            }
            val next = onMain { live.snapshot().getJSONObject("sync_marker") }
            val all = markerRows(markerFile)
            assertEquals(1, all.count { it.optString("id") == marker.getString("id") })
            assertEquals(1, all.count { it.optString("id") == next.getString("id") })
            assertTrue(next.getLong("first_draw_elapsed_ns") > firstDraw)
            proof.put("second_marker", next).put("complete", true)
            stage("second_marker_verified")
        } finally {
            try {
                if (directory == null) directory = service?.let { live -> onMain { get<File?>(live, "directory") } }
                directory?.let { File(it, "sync-marker-component-proof.json").writeText(proof.toString(2)) }
            } finally {
                try {
                    onMain {
                        service?.let { live ->
                            get<StandaloneEngine>(live, "engine").pause(SystemClock.elapsedRealtime())
                            savedFields.forEach { (name, value) -> set(live, name, value) }
                        }
                        target.stopService(Intent(target, StandaloneService::class.java))
                        hud?.finish()
                    }
                    await("test service must stop") { StandaloneService.instance == null }
                    await("HUD visibility must clear") { !StandaloneActivity.isPanelVisible }
                } finally {
                    stateFiles.forEach { file -> backups[file]?.let { file.writeBytes(it) } ?: check(!file.exists() || file.delete()) }
                }
            }
        }
        stage("complete")
    }

    private fun markerRows(file: File): List<JSONObject> = file.readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
    private fun stage(name: String) { instrumentation.sendStatus(0, Bundle().apply { putString("sync_marker_checkpoint", name) }) }
    private fun key(panel: StandaloneActivity, action: Int, repeat: Int = 0) {
        val now = SystemClock.uptimeMillis()
        assertTrue(panel.dispatchKeyEvent(KeyEvent(now, now, action, KeyEvent.KEYCODE_DPAD_LEFT, repeat)))
    }
    private fun <T> get(owner: Any, name: String): T {
        @Suppress("UNCHECKED_CAST")
        return owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(owner) as T
    }
    private fun set(owner: Any, name: String, value: Any?) { owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(owner, value) }
    private fun <T> onMain(action: () -> T): T {
        if (Looper.myLooper() == Looper.getMainLooper()) return action()
        val task = FutureTask<T> { action() }
        instrumentation.runOnMainSync(task)
        return try { task.get() } catch (failure: ExecutionException) { throw (failure.cause ?: failure) }
    }
    private fun await(message: String, timeoutMs: Long = 5000, condition: () -> Boolean) {
        val end = SystemClock.elapsedRealtime() + timeoutMs
        while (!condition() && SystemClock.elapsedRealtime() < end) SystemClock.sleep(30)
        assertTrue(message, condition())
    }
}
