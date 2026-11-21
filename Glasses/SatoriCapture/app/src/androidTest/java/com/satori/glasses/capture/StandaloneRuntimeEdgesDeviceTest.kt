package com.satori.glasses.capture

import android.content.Context
import android.content.ContextWrapper
import android.os.Handler
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicInteger

/**
 * Runs the real service pulse against an isolated engine and the real voice busy flag.
 * Service.onCreate is deliberately not called: no configuration/credential is loaded, no camera is
 * created, no receiver is registered and no production service or file is changed. There is no fake
 * wear event. lastSpoken proves passage through the service's speech submission gate only; with
 * speech=null this test does not claim that synthesis or audible playback occurred.
 */
@RunWith(AndroidJUnit4::class)
class StandaloneRuntimeEdgesDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test fun voiceBusyDoesNotConsumeTheOnlyPendingConfirmationReminder() {
        val stateFile = File(target.filesDir, "standalone-state.json")
        val stateFiles = listOf(stateFile, File(stateFile.path + ".bak"), File(stateFile.path + ".new"))
        val savedBytes = stateFiles.associateWith { if (it.exists()) it.readBytes() else null }
        val safeRunningState = onMain {
            val running = StandaloneService.instance
            running == null || running.snapshot().isNull("goal")
        }
        assumeTrue("a running user goal must be preserved", safeRunningState)
        savedBytes.forEach { (_, bytes) ->
            if (bytes != null) {
                val saved = runCatching { JSONObject(String(bytes, Charsets.UTF_8)) }.getOrNull()
                assumeTrue("unrecognized persisted state must be preserved", saved != null)
                assumeTrue("a saved user goal must be preserved", saved!!.isNull("goal"))
            }
        }

        val networkCalls = AtomicInteger()
        val transport = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "runtime-test-only-key", "planner", "monitor")) {
            networkCalls.incrementAndGet()
            throw AssertionError("an isolated pending-confirmation pulse must never use the network")
        }
        var service: StandaloneService? = null
        var voice: StandaloneVoice? = null
        var engine: StandaloneEngine? = null
        var originalEngineState: JSONObject? = null
        try {
            onMain {
                val isolated = StandaloneService()
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", Context::class.java)
                    .apply { isAccessible = true }.invoke(isolated, target)
                service = isolated
                val localEngine = field(isolated, "engine") as StandaloneEngine
                engine = localEngine
                originalEngineState = JSONObject((field(localEngine, "state") as JSONObject).toString())
                assertNull("the isolated runtime must not have a visual transport", field(isolated, "api"))
                assertNull("this test must not synthesize or play speech", field(isolated, "speech"))
                val localVoice = StandaloneVoice(target, transport, { fail("voice was never started") }, { fail("voice was never started") })
                voice = localVoice
                setField(isolated, "voice", localVoice)
                val now = SystemClock.elapsedRealtime()
                localEngine.setGoal("设备本地提醒边缘测试：不会实际执行", true, now)
                val request = localEngine.nextRequest(now + 1, listOf("synthetic-runtime-1"))!!
                val answer = JSONObject().put("goal_summary", "本地提醒边缘测试").put("horizon", "short")
                    .put("desires", JSONArray().put(JSONObject().put("desire_id", "d1").put("title", "确认本地测试")
                        .put("success_criteria", "本地测试已确认")))
                    .put("clarifying_question", JSONObject.NULL).put("first_plan", JSONObject.NULL)
                    .put("confidence", .9).put("evidence_frame_ids", JSONArray().put("synthetic-runtime-1"))
                localEngine.accept(request, answer, now + 2)
                assertEquals("awaiting_plan_confirmation", localEngine.snapshot().getString("phase"))
                val state = field(localEngine, "state") as JSONObject
                state.getJSONObject("pending").put("prompted_ms", now - 61_000).put("reminded", false)
                setField(localVoice, "busy", true)
                pulse(isolated)
                assertFalse("voice recording/recognition consumed the only confirmation reminder",
                    localEngine.snapshot().getJSONObject("pending").getBoolean("reminded"))
                assertEquals("a busy microphone must suppress speech submission", "", field(isolated, "lastSpoken"))
                assertFalse("pending confirmation must not start the camera", field(isolated, "capturing") as Boolean)

                setField(localVoice, "busy", false)
                pulse(isolated)
                assertTrue("the deferred reminder should be consumed after speech input finishes",
                    localEngine.snapshot().getJSONObject("pending").getBoolean("reminded"))
                assertTrue("the real service speech gate must receive the deferred reminder",
                    (field(isolated, "lastSpoken") as String).contains("还在等你确认"))
                assertFalse(field(isolated, "capturing") as Boolean)
                assertFalse(field(isolated, "inFlight") as Boolean)
                assertNull(field(isolated, "api"))

                val speak = StandaloneService::class.java.getDeclaredMethod("speak",
                    List::class.java, String::class.java, String::class.java, String::class.java).apply { isAccessible = true }
                val beforeReply = field(isolated, "lastSpoken")
                setField(localVoice, "busy", true)
                speak.invoke(isolated, listOf("本轮答复"), "问题", null, "")
                assertEquals("本轮答复", field(isolated, "lastReplyText"))
                assertEquals(beforeReply, field(isolated, "lastSpoken"))
                setField(localVoice, "busy", false)
                pulse(isolated)
                assertEquals("本轮答复", field(isolated, "lastSpoken"))
                setField(isolated, "lastSpoken", "")
                pulse(isolated)
                assertEquals("a deferred reply must only be submitted once", "", field(isolated, "lastSpoken"))

                setField(localVoice, "busy", true)
                speak.invoke(isolated, listOf("待取消的旧答复"), "问题", null, "")
                StandaloneService::class.java.getDeclaredMethod("invalidate", Boolean::class.javaPrimitiveType)
                    .apply { isAccessible = true }.invoke(isolated, false)
                pulse(isolated)
                assertEquals("invalidating a turn must discard its pending speech", "", field(isolated, "lastSpoken"))
            }
            assertEquals("this runtime boundary test must remain entirely local", 0, networkCalls.get())
        } finally {
            onMain {
                voice?.close()
                service?.let {
                    setField(it, "voice", null)
                    setField(it, "closed", true)
                    (field(it, "main") as Handler).removeCallbacksAndMessages(null)
                    (field(it, "worker") as ExecutorService).shutdownNow()
                }
                originalEngineState?.let { original -> engine?.let { setField(it, "state", original) } }
            }
            transport.close()
            // No production file was changed, so restoration means verifying the exact original bytes.
            savedBytes.forEach { (file, original) ->
                if (original == null) assertFalse("runtime test created a production state file", file.exists())
                else {
                    assertTrue("runtime test removed a production state file", file.isFile)
                    assertArrayEquals("runtime test changed production state bytes", original, file.readBytes())
                }
            }
        }
    }

    /** Internal simulation of the production broadcast handler; this is not a real wear-sensor test. */
    @Test fun offHeadRetainsFramesWhileFoldingPausesAndUserPauseSurvives() {
        withIsolatedPersistingService { service, engine ->
            val at = SystemClock.elapsedRealtime()
            setField(service, "worn", true)
            setField(service, "legsOpen", true)
            setField(service, "lastPresence", null)
            applyPresence(service)
            engine.setGoal("内部模拟摘戴验收：不得拍摄或执行网络请求", true, at)
            val goal = engine.snapshot().getJSONObject("goal").getString("text")
            val buffer = field(service, "frames") as StandaloneFrameBuffer
            // Only the RAM ring boundary is exercised; these bytes never enter image processing or a model.
            buffer.add(StandaloneFrame("presence-synthetic-1", at + 1, byteArrayOf(1)))
            assertEquals(1, buffer.size)
            val pendingRequest = engine.nextRequest(at + 2, listOf("presence-synthetic-1"))!!

            setField(service, "worn", false)
            applyPresence(service)
            val removed = engine.snapshot()
            assertEquals(goal, removed.getJSONObject("goal").getString("text"))
            assertTrue(removed.isNull("paused_reason"))
            assertEquals("removed", removed.getString("presence"))
            assertEquals(pendingRequest.revision, removed.getLong("revision"))
            assertEquals(1, buffer.size)
            engine.cancelRequest(at + 3)

            setField(service, "legsOpen", false)
            applyPresence(service)
            assertEquals("folded", engine.snapshot().getString("paused_reason"))
            assertEquals(0, buffer.size)
            assertFalse(field(service, "capturing") as Boolean)
            assertFalse(field(service, "inFlight") as Boolean)
            assertNull(engine.nextRequest(at + 2, listOf("presence-synthetic-1")))

            setField(service, "legsOpen", null)
            applyPresence(service)
            assertEquals("an unknown hinge reading cannot undo known folding", "folded", engine.snapshot().getString("paused_reason"))
            setField(service, "legsOpen", true)
            applyPresence(service)
            val returned = engine.snapshot()
            assertEquals(goal, returned.getJSONObject("goal").getString("text"))
            assertEquals("removed", returned.getString("presence"))
            assertTrue("returning from a device-only pause must allow progress", returned.isNull("paused_reason"))
            assertNotNull("the engine must be eligible for a fresh request after simulated return",
                engine.nextRequest(SystemClock.elapsedRealtime(), listOf("presence-synthetic-2")))
            engine.cancelRequest(SystemClock.elapsedRealtime())
            assertFalse("api=null keeps the real service camera off", field(service, "capturing") as Boolean)

            service.command("pause")
            buffer.add(StandaloneFrame("presence-synthetic-3", at + 3, byteArrayOf(2)))
            setField(service, "worn", true)
            applyPresence(service)
            assertEquals("user", engine.snapshot().getString("paused_reason"))
            assertEquals(1, buffer.size)
            setField(service, "worn", false)
            applyPresence(service)
            assertEquals("a simulated wear event must not override a deliberate user pause", "user",
                engine.snapshot().getString("paused_reason"))
            assertEquals(goal, engine.snapshot().getJSONObject("goal").getString("text"))
            assertNull(engine.nextRequest(SystemClock.elapsedRealtime(), listOf("presence-synthetic-4")))
            assertFalse(field(service, "capturing") as Boolean)
            assertNull(field(service, "api"))
            assertNull(field(service, "speech"))
        }
    }

    /** Real AtomicFile persistence is backed up/restored; no production service is stopped or replaced. */
    private fun withIsolatedPersistingService(block: (StandaloneService, StandaloneEngine) -> Unit) {
        assumeTrue("do not race a production service's state writer", onMain { StandaloneService.instance } == null)
        assumeTrue("do not interrupt legacy PC capture", onMain { CaptureService.instance } == null)
        val state = File(target.filesDir, "standalone-state.json")
        val files = listOf(state, File(state.path + ".bak"), File(state.path + ".new"))
        val backups = files.associateWith { if (it.exists()) it.readBytes() else null }
        backups.forEach { (_, bytes) ->
            if (bytes != null) {
                val saved = runCatching { JSONObject(String(bytes, Charsets.UTF_8)) }.getOrNull()
                assumeTrue("unrecognized persisted state must be preserved", saved != null)
                assumeTrue("a saved user goal must be preserved", saved!!.isNull("goal"))
            }
        }
        var service: StandaloneService? = null
        try {
            // Prevent AtomicFile from promoting an old auxiliary file during this isolated component test.
            files.drop(1).forEach { check(!it.exists() || it.delete()) }
            onMain {
                val isolated = StandaloneService()
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", Context::class.java)
                    .apply { isAccessible = true }.invoke(isolated, target)
                service = isolated
                assertNull(field(isolated, "api"))
                assertNull(field(isolated, "speech"))
                block(isolated, field(isolated, "engine") as StandaloneEngine)
            }
        } finally {
            try {
                onMain {
                    service?.let {
                        setField(it, "closed", true)
                        (field(it, "main") as Handler).removeCallbacksAndMessages(null)
                        (field(it, "worker") as ExecutorService).shutdownNow()
                    }
                }
            } finally {
                backups.forEach { (file, bytes) ->
                    if (bytes == null) check(!file.exists() || file.delete()) else file.writeBytes(bytes)
                }
            }
        }
    }

    private fun pulse(service: StandaloneService) = StandaloneService::class.java.getDeclaredMethod("pulse")
        .apply { isAccessible = true }.invoke(service)

    private fun applyPresence(service: StandaloneService) = StandaloneService::class.java.getDeclaredMethod("applyPresence")
        .apply { isAccessible = true }.invoke(service)

    private fun field(receiver: Any, name: String): Any? = receiver.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.get(receiver)

    private fun setField(receiver: Any, name: String, value: Any?) = receiver.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.set(receiver, value)

    private fun <T> onMain(action: () -> T): T {
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() }
        catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }
}
