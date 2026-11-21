package com.satori.glasses.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.SystemClock
import android.os.Bundle
import android.os.Looper
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.InputDevice
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.json.JSONArray
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.FutureTask
import java.util.concurrent.ExecutionException

/** Component acceptance for the read-only HUD and injected phone-key protocol.
 * No real remote, live microphone, camera, model request, screen recording or credential edit is exercised.
 * The idle service's API/speech/voice references are isolated and restored before it is stopped.
 */
@RunWith(AndroidJUnit4::class)
class StandalonePanelDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext
    private val checkpoints = JSONArray()
    private val checkpointFile by lazy {
        File(target.filesDir, "standalone-panel-checkpoints").apply { check(mkdirs() || isDirectory) }
            .let { File(it, "panel-${System.currentTimeMillis()}.json") }
    }

    @Test(timeout = 60_000) fun readonlyHudRoutesPhoneKeysWithoutVisibleButtonsOrNetwork() {
        checkpoint("started")
        guard("preserve an existing live service", StandaloneService.instance == null && CaptureService.instance == null)
        guard("preserve a pending private import", !File(target.filesDir, "standalone-import.json").exists())
        guard("camera permission must be pregranted", target.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        guard("microphone permission must be pregranted", target.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        var uiFailurePropagated = false
        try { onMain<Unit> { throw AssertionError("panel_ui_assertion_propagation_probe") } }
        catch (failure: AssertionError) { uiFailurePropagated = failure.message == "panel_ui_assertion_propagation_probe" }
        assertTrue("UI assertions must reach the instrumentation thread", uiFailurePropagated)
        checkpoint("ui_assertion_propagation_verified")
        val state = File(target.filesDir, "standalone-state.json")
        val files = listOf(state, File(state.path + ".bak"), File(state.path + ".new"))
        val original = files.associateWith { if (it.exists()) it.readBytes() else null }
        var activity: StandaloneActivity? = null
        var service: StandaloneService? = null
        var fields: Map<String, Any?> = emptyMap()
        try {
            files.drop(1).forEach { check(!it.exists() || it.delete()) }
            state.writeText(StandaloneEngine().snapshot(SystemClock.elapsedRealtime()).toString())
            activity = instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as StandaloneActivity
            await { StandaloneService.instance != null }
            service = StandaloneService.instance!!
            val live = requireNotNull(service)
            val hud = requireNotNull(activity)
            fields = onMain { listOf("api", "voice", "speech", "worn", "legsOpen", "lastPresence", "lastFolded").associateWith { get<Any?>(live, it) } }
            onMain {
                set(live, "api", null); set(live, "voice", null); set(live, "speech", null)
                set(live, "worn", true); set(live, "legsOpen", true); set(live, "lastPresence", true)
                set(live, "lastFolded", false)
            }
            checkpoint("idle_hud_opened_and_runtime_isolated")
            val engine = get<StandaloneEngine>(live, "engine")
            onMain {
                assertTrue("native key interception is only enabled while HUD is foreground", StandaloneActivity.isPanelVisible)
                assertEquals("official screen recording must not be blocked", 0, hud.window.attributes.flags and WindowManager.LayoutParams.FLAG_SECURE)
                val root = hud.window.decorView.findViewWithTag<View>("satori_readonly_hud")
                assertNotNull(root)
                val views = descendants(root)
                assertFalse("glasses HUD must have no buttons", views.any { it is Button })
                assertFalse("glasses HUD must have no editable credential/goal fields", views.any { it is EditText })
                assertFalse("HUD must have no clickable controls", views.any { it.isClickable || it.hasOnClickListeners() })
                val text = views.filterIsInstance<TextView>().joinToString("\n") { it.text.toString() }
                assertFalse(text.contains("API 密钥")); assertFalse(text.contains("接口设置"))
                assertFalse(text.contains("HTTPS")); assertFalse(text.contains("帧："))
            }
            checkpoint("readonly_hud_no_controls_or_secure_flag")

            for (key in listOf(KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_NUMPAD_ENTER)) {
                SystemClock.sleep(520) // Distinct legitimate phone gestures, outside the cross-source gate.
                onMain {
                    readyPending(engine, SystemClock.elapsedRealtime())
                    val before = engine.snapshot().getLong("revision")
                    sendKey(hud, KeyEvent.ACTION_DOWN, key)
                    assertEquals("confirmation acts on UP only", "awaiting_plan_confirmation", engine.snapshot().getString("phase"))
                    sendKey(hud, KeyEvent.ACTION_DOWN, key, repeat = 1)
                    sendKey(hud, KeyEvent.ACTION_UP, key)
                    val after = engine.snapshot()
                    assertEquals("planning", after.getString("phase"))
                    // One production confirm performs cancelRequest + confirm, each touching revision once.
                    assertEquals("remote gesture must submit exactly one confirmation", before + 2, after.getLong("revision"))
                    sendKey(hud, KeyEvent.ACTION_UP, key)
                    assertEquals("duplicate UP must be ignored", after.getLong("revision"), engine.snapshot().getLong("revision"))
                }
                checkpoint("confirm_key_${key}_up_only_once")
            }

            SystemClock.sleep(520)
            onMain {
                val at = SystemClock.elapsedRealtime()
                readyPending(engine, at)
                set(live, "error", "")
                touch(hud, MotionEvent.ACTION_DOWN, 80f, 100f)
                touch(hud, MotionEvent.ACTION_UP, 80f, 100f)
                assertEquals("coordinate tap confirms the same pending", "planning", engine.snapshot().getString("phase"))
                sendKey(hud, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER)
                sendKey(hud, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER)
                assertEquals("duplicate key after tap must not start voice", "", get<String>(live, "error"))
            }
            checkpoint("touch_confirm_and_duplicate_key_suppressed")
            SystemClock.sleep(520)
            onMain {
                val at = SystemClock.elapsedRealtime()
                readyPending(engine, at)
                touch(hud, MotionEvent.ACTION_DOWN, 80f, 100f)
                touch(hud, MotionEvent.ACTION_MOVE, 80f, 250f)
                touch(hud, MotionEvent.ACTION_UP, 80f, 260f)
                assertEquals("scrolling must not confirm", "awaiting_plan_confirmation", engine.snapshot().getString("phase"))
            }
            checkpoint("scroll_does_not_confirm")
            SystemClock.sleep(520)
            onMain {
                val at = SystemClock.elapsedRealtime()
                readyPending(engine, at)
                touch(hud, MotionEvent.ACTION_DOWN, 80f, 100f)
                touch(hud, MotionEvent.ACTION_POINTER_DOWN, 80f, 100f)
                touch(hud, MotionEvent.ACTION_POINTER_UP, 80f, 100f)
                touch(hud, MotionEvent.ACTION_UP, 80f, 100f)
                assertEquals("multitouch signal must not confirm", "awaiting_plan_confirmation", engine.snapshot().getString("phase"))
                touch(hud, MotionEvent.ACTION_DOWN, 80f, 100f)
                readyPending(engine, at + 1000)
                val revision = engine.snapshot().getLong("revision")
                touch(hud, MotionEvent.ACTION_UP, 80f, 100f)
                assertEquals("tap UP must not confirm a replacement pending", revision, engine.snapshot().getLong("revision"))
            }
            checkpoint("multitouch_and_replaced_pending_touch_guard")

            onMain {
                val at = SystemClock.elapsedRealtime()
                readyPending(engine, at)
                sendKey(hud, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER)
                readyPending(engine, at + 1000)
                val revision = engine.snapshot().getLong("revision")
                sendKey(hud, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER)
                assertEquals("UP from the previous pending cannot confirm a new one", "awaiting_plan_confirmation", engine.snapshot().getString("phase"))
                assertEquals(revision, engine.snapshot().getLong("revision"))

                // Keep a real, unfinished planning state, but no API/voice/camera can start.
                engine.confirm(at + 1003); engine.resume(at + 1004)
                assertEquals("Back begins in an unfinished planning state", "planning", engine.snapshot().getString("phase"))
                assertTrue("Back begins unpaused through public resume", engine.snapshot().isNull("paused_reason"))
                sendKey(hud, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_BACK)
                assertTrue(engine.snapshot().isNull("paused_reason"))
                sendKey(hud, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_BACK)
                assertEquals("user", engine.snapshot().getString("paused_reason"))
                sendKey(hud, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_BACK)
                assertEquals("duplicate Back UP must not resume", "user", engine.snapshot().getString("paused_reason"))
                sendKey(hud, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_BACK)
                sendKey(hud, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_BACK)
                assertTrue("second complete Back gesture resumes", engine.snapshot().isNull("paused_reason"))

                set(live, "error", "")
                sendKey(hud, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_DPAD_RIGHT)
                assertEquals("right-key DOWN must not start voice", "", get<String>(live, "error"))
                sendKey(hud, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_DPAD_RIGHT)
                assertTrue("right-key UP must reach the real voice entry (voice deliberately absent)", get<String>(live, "error").contains("语音"))
                assertFalse(live.snapshot().getBoolean("recording"))
                assertFalse(live.snapshot().getBoolean("capturing"))
            }
            checkpoint("replaced_key_pending_back_pause_resume_and_right_voice")
        } catch (failure: Throwable) {
            checkpoint("failed", JSONObject().put("failure_type", failure.javaClass.simpleName))
            throw failure
        } finally {
            try {
                onMain {
                    service?.let { live ->
                        get<StandaloneEngine>(live, "engine").pause(SystemClock.elapsedRealtime())
                        fields.forEach { (name, value) -> set(live, name, value) }
                    }
                    target.stopService(Intent(target, StandaloneService::class.java))
                    activity?.finish()
                }
                await { StandaloneService.instance == null }
                await { !StandaloneActivity.isPanelVisible }
            } finally {
                files.forEach { file -> original[file]?.let { file.writeBytes(it) } ?: check(!file.exists() || file.delete()) }
            }
        }
        checkpoint("complete", complete = true)
    }

    private fun readyPending(engine: StandaloneEngine, at: Long) {
        engine.restore(pending(at), at)
        engine.resume(at + 2)
        assertTrue("restored fixture must be explicitly resumed", engine.snapshot().isNull("paused_reason"))
        assertEquals("awaiting_plan_confirmation", engine.snapshot().getString("phase"))
    }

    private fun pending(at: Long): JSONObject {
        val engine = StandaloneEngine()
        engine.setGoal("只读面板组件验收", nowMs = at)
        val request = engine.nextRequest(at, listOf("fixture-f1"))!!
        engine.accept(request, JSONObject("""{"goal_summary":"面板组件验收","horizon":"short","desires":[{"desire_id":"d1","title":"验证确认","success_criteria":"确认完成"}],"clarifying_question":null,"confidence":0.9,"evidence_frame_ids":["fixture-f1"]}"""), at + 1)
        return engine.snapshot(at + 1)
    }
    private fun descendants(view: View): List<View> = listOf(view) + if (view is ViewGroup)
        (0 until view.childCount).flatMap { descendants(view.getChildAt(it)) } else emptyList()
    private fun sendKey(activity: Activity, action: Int, code: Int, repeat: Int = 0) {
        val now = SystemClock.uptimeMillis()
        assertTrue(activity.dispatchKeyEvent(KeyEvent(now, now, action, code, repeat)))
    }
    private fun touch(activity: Activity, action: Int, x: Float, y: Float) {
        val now = SystemClock.uptimeMillis()
        val event = if (action in listOf(MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_POINTER_UP)) {
            val properties = Array(2) { index -> MotionEvent.PointerProperties().apply { id = index; toolType = MotionEvent.TOOL_TYPE_FINGER } }
            val coordinates = Array(2) { index -> MotionEvent.PointerCoords().apply { this.x = x + index * 20; this.y = y; pressure = 1f; size = 1f } }
            MotionEvent.obtain(now, now, action or (1 shl MotionEvent.ACTION_POINTER_INDEX_SHIFT), 2, properties, coordinates,
                0, 0, 1f, 1f, 0, 0, InputDevice.SOURCE_TOUCHSCREEN, 0)
        } else MotionEvent.obtain(now, now, action, x, y, 0)
        try { activity.dispatchTouchEvent(event) } finally { event.recycle() }
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
    private fun guard(reason: String, allowed: Boolean) {
        if (!allowed) checkpoint("skipped", JSONObject().put("reason", reason))
        assumeTrue(reason, allowed)
    }
    private fun checkpoint(name: String, details: JSONObject = JSONObject(), complete: Boolean = false) {
        checkpoints.put(JSONObject().put("name", name).put("at_ms", SystemClock.elapsedRealtime()).put("details", details))
        checkpointFile.writeText(JSONObject().put("schema", "satori.panel-component-checkpoints/1")
            .put("complete", complete).put("scope", "synthetic key/touch component checks; not actual phone or hardware-key acceptance")
            .put("checkpoints", checkpoints).toString(2))
        instrumentation.sendStatus(0, Bundle().apply {
            putString("panel_checkpoint", name)
            putString("panel_checkpoint_file", "standalone-panel-checkpoints/${checkpointFile.name}")
            putString("panel_checkpoint_complete", complete.toString())
        })
    }
    private fun await(condition: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 10000
        while (!condition() && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(50)
        assertTrue(condition())
    }
}
