package com.satori.glasses.capture

import android.content.ContextWrapper
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.lang.reflect.InvocationTargetException
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Synthetic 16 kHz SAPI WAV -> the REAL VoiceRecorder PCM sink -> production Vosk worker/finalResult
 * -> StandaloneVoice.deliver -> REAL StandaloneService.onText/command -> task state.
 *
 * Reflection is confined to this test. It constructs the existing recorder with active=false, so
 * AudioRecord is never opened, and injects PCM through its existing production closures. No cloud,
 * user credential, vendor speech, actual camera, hardware key, or live microphone is used.
 * A context-attached production service instance exercises the real methods without starting its
 * lifecycle loop. Its api remains null, preventing capture/HTTP even when "continue" is recognized.
 * Existing user state bytes are restored; the newly created fixture run remains as explicit evidence.
 * This is a command/ASR pipeline check, NOT microphone, audibility, temple-key or lifecycle acceptance.
 */
@RunWith(AndroidJUnit4::class)
class StandaloneVoiceCommandDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test(timeout = 240_000) fun offlineAudioCommandsReachRealProductionTaskControls() {
        assumeTrue("an existing standalone service must not be interrupted", StandaloneService.instance == null)
        assumeTrue("legacy capture must not be interrupted", CaptureService.instance == null)
        assumeTrue("pending private profile import must remain untouched", !File(target.filesDir, "standalone-import.json").exists())
        val state = File(target.filesDir, "standalone-state.json")
        val stateFiles = listOf(state, File(state.path + ".bak"), File(state.path + ".new"))
        val originalState = stateFiles.associateWith { if (it.isFile) it.readBytes() else null }
        val networkAttempts = AtomicInteger()
        val dummyProfile = StandaloneProfile("https://example.invalid/v1", "fixture-only-not-an-api-key", "fixture-planner", "fixture-monitor")
        val api = StandaloneApi(dummyProfile) { networkAttempts.incrementAndGet(); throw AssertionError("fixture commands must never use HTTP") }
        val delivered = AtomicReference<String?>()
        val recognition = AtomicReference<JSONObject?>()
        val failure = AtomicReference<String?>()
        val callback = AtomicReference(CountDownLatch(0))
        val service = onMain {
            StandaloneService().also { instance ->
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", android.content.Context::class.java)
                    .apply { isAccessible = true }.invoke(instance, target)
                set(instance, "profile", dummyProfile)
                set(instance, "api", null)
                set(instance, "worn", true)
                set(instance, "legsOpen", true)
                set(instance, "lastPresence", true)
                set(instance, "lastFolded", false)
                set(instance, "camera", CameraStreamer(target, instance,
                    { _, _ -> throw AssertionError("fixture commands must never open camera") }, { _, _ -> }))
            }
        }
        val parse = StandaloneService::class.java.getDeclaredMethod("onText", String::class.java).apply { isAccessible = true }
        val voice = StandaloneVoice(target, api, { text ->
            delivered.set(text)
            try { parse.invoke(service, text) }
            catch (e: InvocationTargetException) { failure.set("production command failed: ${e.targetException.javaClass.simpleName}") }
            catch (e: Exception) { failure.set("production command failed: ${e.javaClass.simpleName}") }
            finally { callback.get().countDown() }
        }, { message -> failure.set(message); callback.get().countDown() }, { audit ->
            recognition.set(JSONObject(audit.toString()))
        })
        val voiceWorker = get<ExecutorService>(voice, "worker")
        val engine = get<StandaloneEngine>(service, "engine")
        val proof = JSONArray()
        var run: File? = null
        try {
            onMain { set(service, "voice", voice); engine.presence(true, SystemClock.elapsedRealtime()) }
            // Native model is loaded once and owned/released by the production Voice instance's worker.
            voiceWorker.submit {
                set(voice, "offlineModel", Model(StandaloneOfflineModel.prepare(target).absolutePath))
                set(voice, "offlineAttempted", true)
            }.get(60, TimeUnit.SECONDS)

            fun recognize(id: String, expected: String): JSONObject {
                delivered.set(null); recognition.set(null); failure.set(null); callback.set(CountDownLatch(1))
                val before = onMain { engine.snapshot(SystemClock.elapsedRealtime()) }
                val injectedRecorder = injectIntoProductionRecorder(voice, voiceWorker, fixture(id))
                assertTrue("Vosk command callback timed out: $id", callback.get().await(35, TimeUnit.SECONDS))
                val after = onMain { engine.snapshot(SystemClock.elapsedRealtime()) }
                if (run == null) run = get<File?>(service, "directory")
                proof.put(JSONObject().put("fixture", id).put("expected_spoken_text", expected)
                    .put("actual_asr_text", delivered.get() ?: JSONObject.NULL).put("error", failure.get() ?: JSONObject.NULL)
                    .put("recognition_trace", recognition.get() ?: JSONObject.NULL)
                    .put("before", before).put("after", after))
                failure.get()?.let { fail("$id failed: $it") }
                assertEquals("ASR fixture text mismatch; expected text must not be substituted", expected, delivered.get())
                assertNotNull("production recognition trace must precede text delivery", recognition.get())
                assertEquals(delivered.get(), recognition.get()!!.getString("selected"))
                assertEquals("vosk_offline", recognition.get()!!.getString("provider"))
                assertFalse("Voice busy must clear before command delivery", voice.busy)
                assertFalse("synthetic injection must never start real recorder", injectedRecorder.active)
                assertEquals("fixture pipeline attempted network", 0, networkAttempts.get())
                return after
            }

            val goal = recognize("goal", "我要整理书桌")
            assertEquals("整理书桌", goal.getJSONObject("goal").getString("text"))
            assertEquals("decomposing", goal.getString("phase"))
            run = get<File?>(service, "directory")
            assertNotNull("real goal command must create its run directory", run)

            // Deterministic model-stage fixture only prepares a pending plan. No completion/command
            // is forged: the following confirmation still has to arrive through real Vosk/onText.
            onMain {
                val at = SystemClock.elapsedRealtime()
                val request = engine.nextRequest(at, listOf("fixture-f1"))!!
                assertEquals("decompose", request.stage)
                engine.accept(request, decomposition(), at + 1)
                assertEquals("awaiting_plan_confirmation", engine.snapshot().getString("phase"))
            }
            assertEquals("executing", recognize("confirm", "确认").getString("phase"))
            assertEquals("user", recognize("pause", "暂停").getString("paused_reason"))
            assertTrue("recognized continue must lift user pause", recognize("resume", "继续").isNull("paused_reason"))
            val repeated = recognize("repeat", "重复")
            assertEquals("executing", repeated.getString("phase"))
            assertEquals("recognized repeat must reach the real service speech request", "请把书放好", get<String>(service, "lastSpoken"))
            assertEquals("stopped", recognize("stop", "结束任务").getString("phase"))
            assertEquals(0, networkAttempts.get())
            val recorded = File(run, "events.jsonl").readLines().map { JSONObject(it) }
                .filter { it.optString("name") == "voice_text" }.map { it.getJSONObject("data").getString("text") }
            // The first goal may create the run AFTER onText emits voice_text. The independent proof
            // above retains that actual ASR result; subsequent commands must all reach the real event log.
            assertEquals("real service must record subsequent actual recognized commands",
                listOf("确认", "暂停", "继续", "重复", "结束任务"), recorded.takeLast(5))
        } finally {
            try {
                run?.let {
                    File(it, "voice-command-fixture-proof.json").writeText(JSONObject()
                        .put("schema", "satori.voice-command-fixture-proof/1")
                        .put("scope", "synthetic PCM through production sink, Vosk, deliver and Service parser; no microphone, key or playback acceptance")
                    .put("recognition_policy", "existing supported goal-prefix candidates within a raw Vosk score gap of 2.0; no literal word correction")
                        .put("network_attempts", networkAttempts.get()).put("commands", proof).toString(2))
                }
            } finally {
                try {
                    onMain {
                        voice.close()
                        get<CameraStreamer>(service, "camera").shutdown()
                        get<StandaloneFlowGate>(service, "gate").close()
                        get<Handler>(service, "main").removeCallbacksAndMessages(null)
                        get<ExecutorService>(service, "worker").shutdownNow()
                    }
                    voiceWorker.awaitTermination(10, TimeUnit.SECONDS)
                    api.close()
                } finally {
                    for ((file, bytes) in originalState) {
                        if (bytes == null) check(!file.exists() || file.delete()) else file.writeBytes(bytes)
                    }
                }
            }
        }
    }

    private fun injectIntoProductionRecorder(voice: StandaloneVoice, worker: ExecutorService, wav: ByteArray): VoiceRecorder {
        val parsed = WavData.parse(wav)
        assertEquals(16000, parsed.sampleRate)
        assertEquals(1, parsed.channels)
        onMain { voice.cancel() }
        val ticket = get<AtomicInteger>(voice, "generation").incrementAndGet()
        worker.submit {
            val model = get<Model>(voice, "offlineModel")
            val recognizer = StandaloneVoice::class.java.getDeclaredMethod("newOfflineRecognizer", Model::class.java)
                .apply { isAccessible = true }.invoke(voice, model) as Recognizer
            val sessionClass = Class.forName("com.satori.glasses.capture.StandaloneVoice\$OfflineSession")
            val session = sessionClass.getDeclaredConstructor(Int::class.javaPrimitiveType, Recognizer::class.java)
                .apply { isAccessible = true }.newInstance(ticket, recognizer)
            set(voice, "offlineSession", session)
        }.get(10, TimeUnit.SECONDS)
        onMain {
            set(voice, "busy", true)
            set(voice, "active", false) // Critical: startRecorder constructs closures but cannot open AudioRecord.
            StandaloneVoice::class.java.getDeclaredMethod("startRecorder", Int::class.javaPrimitiveType, Boolean::class.javaPrimitiveType)
                .apply { isAccessible = true }.invoke(voice, ticket, true)
        }
        val recorder = get<VoiceRecorder>(voice, "recorder")
        assertFalse(recorder.active)
        val sink = get<(JSONObject, ByteArray) -> Unit>(recorder, "sink")
        val event = get<(String, JSONObject) -> Unit>(recorder, "onEvent")
        var offset = parsed.offset
        val end = parsed.offset + parsed.length
        var chunk = 0
        while (offset < end) {
            val size = minOf(3200, end - offset)
            sink(JSONObject().put("chunk", chunk++).put("sample_rate", 16000).put("channels", 1).put("format", "s16le"),
                wav.copyOfRange(offset, offset + size))
            offset += size
        }
        // Explicit fixture silence ends the utterance without modifying its recognized/expected text.
        repeat(5) { sink(JSONObject().put("chunk", chunk++), ByteArray(3200)) }
        event("ptt_stop", JSONObject().put("reason", "synthetic_fixture_end"))
        return recorder
    }

    private fun fixture(id: String): ByteArray {
        val manifest = instrumentation.context.assets.open("voice-commands/source-manifest.json")
            .bufferedReader().use { JSONObject(it.readText()) }.getJSONArray("fixtures")
        val entry = (0 until manifest.length()).map { manifest.getJSONObject(it) }.single { it.getString("id") == id }
        val bytes = instrumentation.context.assets.open("voice-commands/${entry.getString("file")}").use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
        assertEquals("fixture source hash mismatch", entry.getString("sha256"), hash)
        return bytes
    }

    private fun decomposition() = JSONObject("""{"goal_summary":"整理书桌","horizon":"short","desires":[{"desire_id":"d1","title":"收好书","success_criteria":"书已放好"}],"clarifying_question":null,"confidence":0.9,"evidence_frame_ids":["fixture-f1"],"first_plan":{"steps":[{"step_id":"s1","instruction":"把书放好","spoken_hint":"请把书放好","required_objects":["书"],"checkpoints":[{"checkpoint_id":"c1","question":"书已放好吗"}]}]}}""")

    private fun <T> get(owner: Any, name: String): T {
        @Suppress("UNCHECKED_CAST")
        return owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(owner) as T
    }

    private fun set(owner: Any, name: String, value: Any?) {
        owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(owner, value)
    }

    private fun <T> onMain(action: () -> T): T {
        if (Looper.myLooper() == Looper.getMainLooper()) return action()
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() }
        catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }
}
