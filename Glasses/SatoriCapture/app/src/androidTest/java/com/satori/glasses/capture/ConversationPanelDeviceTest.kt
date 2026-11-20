package com.satori.glasses.capture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.lang.reflect.InvocationTargetException
import java.util.concurrent.ExecutionException
import java.util.concurrent.FutureTask

/** Explicit UI fixture only: no Service field, task, profile, voice or model is changed. */
@RunWith(AndroidJUnit4::class)
class ConversationPanelDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test(timeout = 30_000) fun chatPanelKeepsTwoTurnsAlignedAndUpdatesOnlyThePendingReply() {
        assumeTrue("use -e satoriConversationHud true for the visible UI fixture",
            InstrumentationRegistry.getArguments().getString("satoriConversationHud") == "true")
        assumeTrue("camera permission must already be granted", target.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        assumeTrue("microphone permission must already be granted", target.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        for (name in listOf("standalone-import.json", "standalone-tools-import.json"))
            assumeTrue("preserve pending configuration imports", !File(target.filesDir, name).exists())
        val persisted = File(target.filesDir, "standalone-state.json")
        assumeTrue("an existing user-paused task is required", persisted.isFile && JSONObject(persisted.readText()).optString("paused_reason") == "user")
        onMain { StandaloneService.instance?.snapshot() }?.let { assumeIdle(it) }
        var panel: StandaloneActivity? = null
        var refreshHandler: Handler? = null
        var refresh: Runnable? = null
        try {
            val activity = instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as StandaloneActivity
            panel = activity
            await { onMain { StandaloneService.instance != null } }
            val service = onMain { StandaloneService.instance!! }
            val before = onMain { service.snapshot() }
            assumeIdle(before)
            onMain {
                refreshHandler = field(activity, "main") as Handler
                refresh = field(activity, "refresh") as Runnable
                refreshHandler!!.removeCallbacks(refresh!!)
            }
            val root = onMain { activity.window.decorView }
            fun text(tag: String) = root.findViewWithTag<TextView>(tag)
            val fixture = JSONObject().put("goal", JSONObject().put("text", "PRESET_GOAL_DO_NOT_SHOW"))
                .put("phase", "executing").put("paused_reason", "user").put("manual_id", "PRESET_MANUAL_DO_NOT_SHOW")
                .put("last_spoken", "OLD_SPOKEN_DO_NOT_SHOW").put("last_answer", JSONObject().put("text", "OLD_ANSWER_DO_NOT_SHOW"))
                .put("last_user_text", "OLD_USER_DO_NOT_SHOW").put("last_reply_text", "OLD_REPLY_DO_NOT_SHOW").put("device_error", "")
                .put("chat_messages", JSONArray())
                .put("interaction_feedback", JSONObject()).put("video", JSONObject().put("state", "idle"))
                .put("screen", JSONObject().put("state", "idle"))
            onMain {
                render(activity, fixture)
                for (tag in listOf("hud_goal", "hud_step", "hud_manual", "hud_voice", "hud_remote_hint", "hud_title"))
                    assertNull("preset view must be removed: $tag", root.findViewWithTag<View>(tag))
                assertTrue("idle panel must not prefill features or sample prompts", visibleText(root).isBlank())
            }
            fun row(turn: String, role: String, body: String, state: String = "complete") = JSONObject()
                .put("id", "$turn:$role").put("role", role).put("text", body).put("turn_id", turn).put("state", state)
            // Synthetic fixture provenance belongs in the test report, never in production bubble labels.
            val question = "消息会留在这里吗？"
            val longReply = "会保留当前会话。" + "你可以向上滚动查看前面的对话，新的答复会放在对应问题下面。".repeat(9) + "内容到这里结束。"
            assertTrue(longReply.length > 150)
            val messages = JSONArray().put(row("t1", "user", question)).put(row("t1", "assistant", longReply))
            fixture.put("chat_messages", messages)
            onMain {
                render(activity, fixture)
                assertEquals(question, text("chat_user_t1:user").text.toString())
                assertEquals(longReply, text("chat_assistant_t1:assistant").text.toString())
                assertEquals("long replies must not be clipped by a line limit", Int.MAX_VALUE, text("chat_assistant_t1:assistant").maxLines)
                assertTrue(text("chat_user_t1:user").contentDescription.toString().let { it.contains("你") || it.contains("用户") })
                assertTrue(text("chat_assistant_t1:assistant").contentDescription.toString().contains("助手"))
                assertFalse(visibleText(root).contains("DO_NOT_SHOW"))
            }
            val firstReply = "会保留当前会话，可以向上查看。"
            val secondQuestion = "那我可以接着问吗？"
            messages.getJSONObject(1).put("text", firstReply)
            messages.put(row("t2", "user", secondQuestion)).put(row("t2", "assistant", "", "pending"))
            fixture.put("dialogue_busy", true).put("interaction_feedback", JSONObject()
                .put("request_id", "ui-fixture-only").put("turn_id", "t2").put("stage", "model").put("elapsed_ms", 16_000)
                .put("question_preview", secondQuestion))
            onMain {
                render(activity, fixture)
                assertEquals(question, text("chat_user_t1:user").text.toString())
                assertEquals(firstReply, text("chat_assistant_t1:assistant").text.toString())
                assertEquals(secondQuestion, text("chat_user_t2:user").text.toString())
                assertTrue(text("hud_interaction").text.contains("仍在等待答复"))
                assertFalse(text("hud_interaction").text.contains(secondQuestion))
                assertEquals(1, Regex(Regex.escape(question)).findAll(visibleText(root)).count())
                assertEquals(1, Regex(Regex.escape(secondQuestion)).findAll(visibleText(root)).count())
            }
            val secondReply = "可以，直接说就好。"
            messages.getJSONObject(3).put("text", secondReply).put("state", "complete")
            fixture.put("dialogue_busy", false).put("interaction_feedback", JSONObject())
            onMain { render(activity, fixture) }
            instrumentation.waitForIdleSync()
            SystemClock.sleep(150) // Allow the attached window to submit the fixture frame before capture.
            onMain {
                assertTrue("capture only the intended foreground panel", activity.hasWindowFocus())
                assertEquals(question, text("chat_user_t1:user").text.toString())
                assertEquals(firstReply, text("chat_assistant_t1:assistant").text.toString())
                assertEquals(secondQuestion, text("chat_user_t2:user").text.toString())
                assertEquals(secondReply, text("chat_assistant_t2:assistant").text.toString())
                assertTrue(root.findViewWithTag<View>("hud_interaction")?.visibility != View.VISIBLE)
                assertLeftRight(text("chat_user_t1:user"), text("chat_assistant_t1:assistant"))
                assertLeftRight(text("chat_user_t2:user"), text("chat_assistant_t2:assistant"))
                assertFalse(visibleText(root).contains("DO_NOT_SHOW"))
                assertFalse(visibleText(root).contains("界面测试"))
                assertFalse(visibleText(root).contains("测试回复"))
                val after = service.snapshot()
                assertEquals(before.getString("task_instance_id"), after.getString("task_instance_id"))
                assertEquals(before.opt("goal").toString(), after.opt("goal").toString())
                assertEquals("user", after.optString("paused_reason"))
                assumeIdle(after)
            }
            val screenshot = requireNotNull(instrumentation.uiAutomation.takeScreenshot()) { "UI screenshot unavailable" }
            val folder = File(target.filesDir, "diagnostics").apply { check(mkdirs() || isDirectory) }
            val output = File(folder, "chat-panel-fixture-${System.currentTimeMillis()}-${SystemClock.elapsedRealtimeNanos()}.png")
            try {
                check(output.createNewFile())
                output.outputStream().use { check(screenshot.compress(Bitmap.CompressFormat.PNG, 100, it)) }
            } finally { screenshot.recycle() }
            instrumentation.sendStatus(0, Bundle().apply {
                putString("conversation_hud_fixture_screenshot", "diagnostics/${output.name}")
                putString("scope", "Synthetic UI text; not a model answer, live speech test or runtime screen recording")
            })
        } finally {
            onMain {
                panel?.let { activity ->
                    refresh?.let { refreshHandler?.removeCallbacks(it) }
                    try { activity.javaClass.getDeclaredMethod("render").apply { isAccessible = true }.invoke(activity) }
                    finally {
                        refresh?.let { refreshHandler?.post(it) }
                        activity.finish()
                    }
                }
            }
        }
    }

    private fun assertLeftRight(user: TextView, assistant: TextView) {
        val userPosition = IntArray(2)
        val assistantPosition = IntArray(2)
        user.getLocationOnScreen(userPosition)
        assistant.getLocationOnScreen(assistantPosition)
        assertTrue("message bubbles must be laid out", user.width > 0 && assistant.width > 0)
        assertTrue("user bubbles must begin to the right of assistant bubbles", userPosition[0] > assistantPosition[0])
        assertTrue("user bubbles must align toward the right edge", userPosition[0] + user.width >= assistantPosition[0] + assistant.width)
    }

    private fun assumeIdle(state: JSONObject) {
        assumeTrue("real task must remain user-paused and idle", state.optString("paused_reason") == "user" &&
            !state.optBoolean("dialogue_busy") && !state.optBoolean("busy") && !state.optBoolean("speech_busy") &&
            !state.optBoolean("recording") && !state.optBoolean("voice_busy"))
        for (name in listOf("video", "screen")) assumeTrue("preserve active recordings",
            state.optJSONObject(name)?.optString("state") !in setOf("starting", "recording", "finalizing"))
    }
    private fun render(panel: StandaloneActivity, state: JSONObject) {
        try { panel.javaClass.getDeclaredMethod("renderState", JSONObject::class.java).apply { isAccessible = true }.invoke(panel, state) }
        catch (failure: InvocationTargetException) { throw failure.targetException }
    }
    private fun field(owner: Any, name: String): Any? = owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(owner)
    private fun visibleText(view: View): String {
        if (view.visibility != View.VISIBLE) return ""
        return if (view is TextView) view.text.toString() else if (view is ViewGroup)
            (0 until view.childCount).joinToString("\n") { visibleText(view.getChildAt(it)) } else ""
    }
    private fun await(check: () -> Boolean) {
        val until = SystemClock.elapsedRealtime() + 5000
        while (!check() && SystemClock.elapsedRealtime() < until) SystemClock.sleep(25)
        assertTrue(check())
    }
    private fun <T> onMain(action: () -> T): T {
        if (Looper.myLooper() == Looper.getMainLooper()) return action()
        val task = FutureTask<T> { action() }
        instrumentation.runOnMainSync(task)
        return try { task.get() } catch (failure: ExecutionException) { throw (failure.cause ?: failure) }
    }
}
