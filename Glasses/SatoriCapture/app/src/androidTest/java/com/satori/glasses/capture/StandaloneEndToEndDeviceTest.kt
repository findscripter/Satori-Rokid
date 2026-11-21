package com.satori.glasses.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
import java.io.ByteArrayOutputStream
import java.io.File
import java.lang.reflect.Method
import java.net.URI
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicInteger
import javax.net.ssl.HttpsURLConnection

/**
 * Explicit opt-in CONTROLLED integration acceptance. Pixels and presence are simulated; Service,
 * Engine, prompts, HTTPS responses, text command routing and local speech are production code.
 * This is not a camera/ASR/wearing test. No model result, checkpoint, completion streak or phase is
 * injected. At most 12 real chat requests are permitted; cloud speech/ASR is blocked before transport.
 */
@RunWith(AndroidJUnit4::class)
class StandaloneEndToEndDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext
    private val clock get() = SystemClock.elapsedRealtime()
    private val requestCount = AtomicInteger()
    private val forbiddenAudioRequests = AtomicInteger()
    private val unexpectedCameraFrames = AtomicInteger()
    private val unexpectedImuBatches = AtomicInteger()
    private val speechEvents = ArrayList<Pair<String, JSONObject>>()
    private val evidenceLock = Any()
    private lateinit var evidence: File
    private lateinit var service: StandaloneService
    private lateinit var camera: CameraStreamer
    private lateinit var imu: ImuStreamer
    private lateinit var pulse: Method
    private lateinit var onFrame: Method
    private lateinit var onText: Method
    private lateinit var event: Method
    private var frameIndex = 0L
    private var deadline = 0L
    private var scene = 0
    private var phaseStarted = 0L
    private var activePhase = ""

    @Test(timeout = 720_000) fun syntheticTwoObjectTaskUsesRealCoordinatorModelAndLocalSpeechToComplete() {
        requireExplicitSelection()
        val originals = preserveOnlyAnIdleInstallation()
        val wasRunning = main { StandaloneService.instance != null }
        val profile = StandaloneSettings(target).load().also { it.validate() }
        verifyOptionalCredentialDigest(profile)
        evidence = File(target.filesDir, "standalone-e2e/e2e-${System.currentTimeMillis()}")
        check(evidence.mkdirs())
        deadline = clock + 690_000 // reserve 30 seconds of the JUnit limit for cleanup/restore
        val fixtures = (0..2).map { drawScene(it) }
        fixtures.forEachIndexed { index, bytes -> File(evidence, "scene-$index.jpg").writeBytes(bytes) }
        val metadata = JSONObject().put("kind", "controlled_service_integration")
            .put("real_wearing_test", false).put("camera_input", "synthetic_onFrame_injection")
            .put("presence", "test_internal_simulation").put("commands", "production_onText_no_microphone")
            .put("model", "real_provider_responses").put("speech", "production_local_tts_no_cloud")
            .put("http_request_limit", 12).put("duration_limit_ms", 720_000)
        var activity: Activity? = null
        var completed: JSONObject? = null
        var multistep = false
        var passed = false
        try {
            main { target.stopService(Intent(target, StandaloneService::class.java)) }
            await("idle service did not stop", 10_000) { main { StandaloneService.instance == null } }
            writeFreshState(originals.keys.toList())
            activity = instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            await("production service did not start", 10_000) { main { StandaloneService.instance != null } }
            service = main { StandaloneService.instance!! }
            prepareControlledService(profile)
            note("isolation_ready", JSONObject().put("real_camera_callback", "drop_and_fail")
                .put("real_imu_callback", "drop_and_fail").put("automatic_tick", false))
            main { write(service, "directory", evidence) }
            textCommand("我要先把红色方块完整移入左侧蓝色框，再把黄色圆形完整移入右侧绿色框。" +
                "严格按顺序分成两步，每次只移动一个物体；两个都放好后才完成。")
            // The production goal command creates a run. Route all subsequent production artifacts
            // into the clearly marked test directory, retaining its original empty run directory.
            controlledMain {
                val allocated = read<File?>(service, "directory")
                metadata.put("allocated_production_run", allocated?.name ?: "")
                write(service, "directory", evidence)
                method(service, "persist").invoke(service)
            }
            File(evidence, "metadata.json").writeText(metadata.toString(2))

            var nextFrame = clock
            var nextPulse = clock
            var nextSnapshot = clock
            var pausedAndResumed = false
            var replanRequested = false
            var awaitingResumeAt = 0L
            var pausedRequestCount = 0
            var firstSignature: String? = null
            var stageOneObserveAtEntry = 0
            var actionAfter = clock
            var initialObservationSeen = false
            val confirmedUtterances = linkedSetOf<String>()
            val stepHintUtterances = linkedMapOf<String, String>()
            val hintWaitStarted = linkedMapOf<String, Long>()
            while (clock < deadline) {
                val state = main { service.snapshot() }
                assertHealthy(state)
                val phase = state.getString("phase")
                if (phase != activePhase) { activePhase = phase; phaseStarted = clock; note("phase", JSONObject().put("phase", phase)) }
                val events = productionEvents()
                val observeResponses = events.count { it.optString("name") == "response" && it.optJSONObject("data")?.optString("stage") == "observe" }
                val planRequested = events.any { it.optString("name") == "request_start" && it.optJSONObject("data")?.optString("stage") == "plan" }
                if (clock >= nextSnapshot) {
                    note("snapshot", JSONObject(state.toString()).put("scene", scene).put("http_requests", requestCount.get())
                        .put("synthetic_frames", frameIndex).put("real_camera_frames", camera.framesEncoded))
                    nextSnapshot = clock + 1_000
                }
                if (phase == "complete") {
                    assertEquals("stage 0/1 must never complete the two-object goal", 2, scene)
                    completed = state
                    break
                }
                if (awaitingResumeAt != 0L) {
                    assertEquals("pause must be represented by the production engine", "user", state.optString("paused_reason"))
                    assertFalse("paused coordinator must not capture", state.optBoolean("capturing"))
                    assertFalse("paused coordinator must not request a model", state.optBoolean("busy"))
                    assertEquals("pause must not issue HTTP", pausedRequestCount, requestCount.get())
                    if (clock >= awaitingResumeAt && spoken(state.optString("last_spoken"))) {
                        textCommand("继续")
                        awaitingResumeAt = 0L
                        pausedAndResumed = true
                        actionAfter = clock + 300
                    }
                } else if (!state.optBoolean("busy") && clock >= actionAfter) {
                    when (phase) {
                        "awaiting_plan_confirmation", "awaiting_step_confirmation" -> {
                            val played = completedSpeechId(state.optString("last_spoken"), confirmedUtterances)
                            if (played != null) {
                                confirmedUtterances.add(played)
                                note("confirmation_prompt_played", JSONObject().put("phase", phase).put("utterance_id", played))
                                textCommand("确认")
                                actionAfter = clock + 300
                            } else assertTrue("the pending prompt never completed local playback", clock - phaseStarted < 45_000)
                        }
                        "awaiting_clarification" -> {
                            // This is a real user-style answer, not a model reply or engine state write.
                            textCommand("这是测试示意图。先把红色方块放入左边蓝框，再把黄色圆形放入右边绿框，" +
                                "每步只移动一个物体，完整在对应框内才算完成。")
                            actionAfter = clock + 300
                        }
                        "executing" -> {
                            val stepSignature = signature(state)
                            val hint = stepHint(state)
                            assertTrue("active production step must provide a spoken hint", hint.isNotBlank())
                            val hintId = completedSpeechId(hint, stepHintUtterances.values.toSet())
                            if (stepSignature !in stepHintUtterances && hintId != null) {
                                stepHintUtterances[stepSignature] = hintId
                                note("step_hint_played", JSONObject().put("signature", stepSignature).put("utterance_id", hintId).put("text", hint))
                            }
                            val hintReady = stepSignature in stepHintUtterances
                            if (!hintReady) {
                                val began = hintWaitStarted.getOrPut(stepSignature) { clock }
                                assertTrue("active step hint never completed local playback", clock - began < 45_000)
                            } else if (!planRequested && !replanRequested) {
                                // Exercise the actual independent plan stage within the same 12-call budget.
                                textCommand("重新规划")
                                replanRequested = true
                                actionAfter = clock + 300
                            } else if (!pausedAndResumed) {
                                textCommand("暂停")
                                pausedRequestCount = requestCount.get()
                                awaitingResumeAt = clock + 3_000
                                note("pause_window_started", JSONObject().put("http_requests", pausedRequestCount))
                            } else {
                                val signature = signature(state)
                                if (firstSignature == null) firstSignature = signature
                                if (scene == 0 && observeResponses > 0) {
                                    initialObservationSeen = true
                                    scene = 1
                                    stageOneObserveAtEntry = observeResponses
                                    note("scene_changed", JSONObject().put("scene", scene).put("meaning", "red_inside_blue_yellow_outside_green"))
                                } else if (scene == 1) {
                                    val progressed = signature != firstSignature || doneSteps(state) > 0
                                    if (progressed) {
                                        multistep = true
                                        scene = 2
                                        note("scene_changed", JSONObject().put("scene", scene).put("meaning", "both_objects_inside_targets")
                                            .put("first_signature", firstSignature).put("next_signature", signature))
                                    } else if (observeResponses - stageOneObserveAtEntry >= 3) {
                                        val plannedSteps = plannedSteps(state)
                                        assertTrue("a multi-step plan never advanced after the first object was placed", plannedSteps < 2)
                                        // A combined model step still must not complete at stage 1. Retain this
                                        // limitation explicitly; do not describe this run as multi-step coverage.
                                        scene = 2
                                        note("scene_changed", JSONObject().put("scene", scene).put("multistep_coverage", false)
                                            .put("reason", "model_combined_both_objects_into_one_step"))
                                    }
                                }
                            }
                        }
                    }
                }
                if (clock >= nextFrame) {
                    // Also attempt an injected frame while paused: production onFrame must reject it.
                    val before = frameLines()
                    inject(fixtures[scene])
                    if (awaitingResumeAt != 0L) assertEquals("paused onFrame must discard input", before, frameLines())
                    nextFrame = clock + 200
                }
                if (clock >= nextPulse) {
                    val latest = main { service.snapshot() }
                    if (requestCount.get() >= 12 && !latest.optBoolean("busy") &&
                        latest.optString("phase") in listOf("decomposing", "planning", "executing") && awaitingResumeAt == 0L) {
                        fail("12 real requests exhausted before the task completed")
                    }
                    controlledMain { pulse.invoke(service) }
                    nextPulse = clock + 1_000
                }
                SystemClock.sleep(50)
            }
            assertNotNull("controlled coordinator did not reach complete within the time budget", completed)
            assertTrue("initial not-yet-complete scene was not observed", initialObservationSeen)
            assertTrue("production pause/resume was not exercised", pausedAndResumed)
            assertTrue("production independent planning stage was not exercised", productionEvents().any {
                it.optString("name") == "response" && it.optJSONObject("data")?.optString("stage") == "plan"
            })
            assertTrue("at least one real step hint must finish local playback", stepHintUtterances.isNotEmpty())
            if (multistep) assertTrue("next-step hint must have its own completed utterance", stepHintUtterances.size >= 2)
            metadata.put("step_hint_utterances", JSONObject(stepHintUtterances as Map<*, *>))
            val completionText = completed!!.optString("last_spoken")
            assertTrue("completion must request the actual final message", completionText.contains("目标已经全部完成"))
            val voiceDeadline = minOf(deadline, clock + 45_000)
            var nextVoiceSnapshot = clock
            while (!spoken(completionText) && clock < voiceDeadline) {
                if (clock >= nextVoiceSnapshot) {
                    note("snapshot", main { service.snapshot() }.put("scene", scene).put("http_requests", requestCount.get())
                        .put("waiting_for_completion_utterance", true))
                    nextVoiceSnapshot = clock + 1_000
                }
                SystemClock.sleep(50)
            }
            assertTrue("completion message did not finish matching local playback", spoken(completionText))
            metadata.put("completion_utterance_verified", true).put("completion_text", completionText)
            assertTrue("no earlier task hint/confirmation prompt completed local playback", earlierSpeechCompleted(completionText))
            assertEquals("cloud speech/ASR must never be contacted", 0, forbiddenAudioRequests.get())
            assertHardwareStopped()
            passed = true
        } catch (failure: Throwable) {
            if (::evidence.isInitialized) note("failure", JSONObject().put("type", failure.javaClass.simpleName)
                .put("message", if (failure is AssertionError) failure.message.orEmpty().take(300) else "controlled integration failed"))
            throw failure
        } finally {
            metadata.put("passed", passed).put("http_requests", requestCount.get()).put("multistep_progress_observed", multistep)
                .put("real_camera_frames", if (::camera.isInitialized) camera.framesEncoded else 0)
                .put("unexpected_camera_callbacks", unexpectedCameraFrames.get()).put("unexpected_imu_callbacks", unexpectedImuBatches.get())
                .put("forbidden_audio_http_attempts", forbiddenAudioRequests.get())
            completed?.let { metadata.put("final_state", it) }
            File(evidence, "metadata.json").writeText(metadata.toString(2))
            try {
                main { target.stopService(Intent(target, StandaloneService::class.java)) }
                await("controlled service did not stop", 10_000) { main { StandaloneService.instance == null } }
            } finally {
                originals.forEach { (file, bytes) -> if (bytes == null) check(!file.exists() || file.delete()) else file.writeBytes(bytes) }
                if (wasRunning) {
                    main { target.startForegroundService(Intent(target, StandaloneService::class.java)) }
                    await("original idle service did not restart", 10_000) { main { StandaloneService.instance != null } }
                }
                main { activity?.finish() }
            }
        }
    }

    private fun prepareControlledService(profile: StandaloneProfile) {
        pulse = method(service, "pulse")
        onFrame = method(service, "onFrame", JSONObject::class.java, ByteArray::class.java)
        onText = method(service, "onText", String::class.java)
        event = method(service, "event", String::class.java, JSONObject::class.java)
        main {
            val handler = read<Handler>(service, "main")
            handler.removeCallbacks(read(service, "tick"))
            read<GlassesInputReceiver?>(service, "receiver")?.let { service.unregisterReceiver(it) }
            write(service, "receiver", null)
            read<StandaloneVoice?>(service, "voice")?.close()
            write(service, "voice", null)
            camera = read(service, "camera")
            imu = read(service, "imu")
            camera.stop(); imu.stop()
            val discardCamera: (JSONObject, ByteArray) -> Unit = { _, _ -> unexpectedCameraFrames.incrementAndGet() }
            val discardImu: (JSONObject, ByteArray) -> Unit = { _, _ -> unexpectedImuBatches.incrementAndGet() }
            write(camera, "onFrame", discardCamera)
            write(imu, "onBatch", discardImu)
            write(service, "worn", true)
            write(service, "legsOpen", true)
            write(service, "lastPresence", null)
            write(service, "lastFolded", false)
            method(service, "applyPresence").invoke(service)
            read<StandaloneApi?>(service, "api")?.close()
            val counted = StandaloneApi(profile) { address ->
                synchronized(requestCount) {
                    if (clock >= deadline || requestCount.get() >= 12) throw ApiFailure(message = "controlled request budget exhausted")
                    check(URI(address).path.endsWith("/chat/completions"))
                    requestCount.incrementAndGet()
                }
                URI(address).toURL().openConnection() as HttpsURLConnection
            }
            write(service, "api", counted)
            read<StandaloneSpeech?>(service, "speech")?.close()
            read<StandaloneApi?>(service, "speechApi")?.close()
            val noCloud = StandaloneApi(profile) {
                forbiddenAudioRequests.incrementAndGet()
                throw ApiFailure(message = "controlled integration requires local speech")
            }
            write(service, "speechApi", noCloud)
            write(service, "speech", StandaloneSpeech(service, noCloud) { name, data ->
                synchronized(speechEvents) { speechEvents.add(name to JSONObject(data.toString())) }
                event.invoke(service, name, data)
            })
            handler.removeCallbacks(read(service, "tick"))
        }
        await("local voice service is not ready", 8_000) {
            main {
                val speech = read<StandaloneSpeech>(service, "speech")
                read<StandaloneVendorSpeech>(speech, "vendor").ready || read<Boolean>(speech, "local")
            }
        }
        assertHardwareStopped()
    }

    /** Pre-mark simulated acquisition so updateCapture can only retain it or stop; never start sensors. */
    private fun controlledMain(action: () -> Unit) = main {
        val handler = read<Handler>(service, "main")
        handler.removeCallbacks(read(service, "tick"))
        camera.stop(); imu.stop()
        write(service, "capturing", true)
        try { action() } finally {
            handler.removeCallbacks(read(service, "tick"))
            camera.stop(); imu.stop()
        }
        assertHardwareStopped()
    }

    private fun textCommand(text: String) {
        note("text_command", JSONObject().put("text", text).put("source", "controlled_onText_no_microphone"))
        controlledMain { onText.invoke(service, text) }
    }

    private fun inject(jpeg: ByteArray) {
        val at = SystemClock.elapsedRealtimeNanos()
        val header = JSONObject().put("index", ++frameIndex).put("ts_ns", at).put("rotation", 0)
            .put("width", 640).put("height", 480).put("controlled_test", true).put("synthetic_scene", scene)
        onFrame.invoke(service, header, jpeg)
    }

    private fun assertHealthy(state: JSONObject) {
        assertEquals("HTTP budget exceeded", true, requestCount.get() <= 12)
        assertEquals("production stage response was rejected", 0, state.optInt("failures"))
        assertEquals("production transport failed", 0, state.optInt("transient_failures"))
        assertTrue("coordinator entered error state", state.isNull("error"))
        assertTrue("coordinator reported an operational error", state.optString("device_error").isBlank())
        assertEquals("local speech attempted cloud transport", 0, forbiddenAudioRequests.get())
        assertHardwareStopped()
    }

    private fun assertHardwareStopped() {
        assertEquals("real camera frames must never be generated", 0L, camera.framesEncoded)
        assertEquals("unexpected real camera callback was discarded", 0, unexpectedCameraFrames.get())
        assertEquals("unexpected real IMU callback was discarded", 0, unexpectedImuBatches.get())
        assertFalse("real camera must remain stopped", read(camera, "wanted"))
        assertNull("real IMU must remain stopped", read<Any?>(imu, "thread"))
    }

    private fun spoken(text: String): Boolean = synchronized(speechEvents) {
        completedSpeechId(text) != null
    }

    private fun completedSpeechId(text: String, excluded: Set<String> = emptySet()): String? = synchronized(speechEvents) {
        val ids = speechEvents.filter { it.first == "speech_requested" && it.second.optString("text") == text }
            .map { it.second.optString("id") }.filter { it.isNotBlank() && it !in excluded }.toSet()
        ids.lastOrNull { id ->
            val began = speechEvents.any { it.first == "audio_started" && it.second.optString("id") == id }
            val ended = speechEvents.any { it.first == "audio_done" && it.second.optString("id") == id &&
                (it.second.optString("provider") == "rokid_local" || it.second.optString("source") == "local_tts") }
            began && ended
        }
    }

    private fun earlierSpeechCompleted(completion: String): Boolean = synchronized(speechEvents) {
        speechEvents.filter { it.first == "speech_requested" }.map { it.second.optString("text") }
            .any { it.isNotBlank() && it != completion && spoken(it) }
    }

    private fun signature(state: JSONObject) = state.optString("active_desire_id") + "/" + state.optString("active_step_id")
    private fun stepHint(state: JSONObject): String {
        val desire = objects(state.optJSONArray("desires")).firstOrNull { it.optString("desire_id") == state.optString("active_desire_id") }
        return objects(desire?.optJSONObject("plan")?.optJSONArray("steps"))
            .firstOrNull { it.optString("step_id") == state.optString("active_step_id") }?.optString("spoken_hint").orEmpty()
    }
    private fun doneSteps(state: JSONObject): Int = objects(state.optJSONArray("desires")).sumOf { desire ->
        objects(desire.optJSONObject("plan")?.optJSONArray("steps")).count { it.optString("status") in listOf("done", "skipped") }
    }
    private fun plannedSteps(state: JSONObject): Int = objects(state.optJSONArray("desires")).sumOf { desire ->
        desire.optJSONObject("plan")?.optJSONArray("steps")?.length() ?: 1
    }
    private fun objects(array: JSONArray?) = if (array == null) emptyList() else (0 until array.length()).map { array.getJSONObject(it) }
    private fun productionEvents(): List<JSONObject> = File(evidence, "events.jsonl").takeIf { it.isFile }?.readLines()
        ?.mapNotNull { runCatching { JSONObject(it) }.getOrNull() } ?: emptyList()
    private fun frameLines() = File(evidence, "frames.jsonl").takeIf { it.isFile }?.useLines { it.count() } ?: 0

    private fun note(name: String, data: JSONObject) {
        synchronized(evidenceLock) {
            File(evidence, "e2e-control.jsonl").appendText(JSONObject().put("at_ms", clock).put("name", name).put("data", data).toString() + "\n")
        }
    }

    private fun preserveOnlyAnIdleInstallation(): Map<File, ByteArray?> {
        assumeTrue("do not interrupt legacy capture", main { CaptureService.instance == null })
        val existing = main { StandaloneService.instance?.snapshot() }
        assumeTrue("existing user goal must be preserved", existing == null || (existing.optString("phase") == "idle" && existing.isNull("goal")))
        assumeTrue("do not consume a pending private import", !File(target.filesDir, "standalone-import.json").exists())
        assumeTrue("install harness must grant camera permission", target.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        assumeTrue("install harness must grant microphone permission", target.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        val base = File(target.filesDir, "standalone-state.json")
        val files = listOf(base, File(base.path + ".bak"), File(base.path + ".new"))
        val backups = files.associateWith { if (it.exists()) it.readBytes() else null }
        backups.values.filterNotNull().forEach { bytes ->
            val saved = runCatching { JSONObject(String(bytes, Charsets.UTF_8)) }.getOrNull()
            assumeTrue("unreadable original state must be preserved", saved != null)
            assumeTrue("saved user goal must be preserved", saved!!.isNull("goal"))
        }
        return backups
    }

    private fun writeFreshState(files: List<File>) {
        files.drop(1).forEach { check(!it.exists() || it.delete()) }
        files.first().writeText(StandaloneEngine().snapshot(clock).toString())
    }

    private fun verifyOptionalCredentialDigest(profile: StandaloneProfile) {
        val expected = InstrumentationRegistry.getArguments().getString("expected_key_sha256").orEmpty()
        if (expected.isNotBlank()) {
            val actual = MessageDigest.getInstance("SHA-256").digest(profile.apiKey.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it.toInt() and 255) }
            assertTrue("device credential differs from the approved source", actual.equals(expected, true))
        }
    }

    private fun requireExplicitSelection() {
        val selected = InstrumentationRegistry.getArguments().getString("class").orEmpty().split(',').map { it.trim().substringBefore('#') }
        assumeTrue("controlled live integration requires explicit class/method selection", javaClass.name in selected)
    }

    private fun await(message: String, timeout: Long, condition: () -> Boolean) {
        val end = minOf(if (deadline == 0L) Long.MAX_VALUE else deadline + 20_000, clock + timeout.coerceAtLeast(0))
        while (!condition() && clock < end) SystemClock.sleep(50)
        assertTrue(message, condition())
    }

    private fun <T> main(action: () -> T): T {
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() }
        catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }

    private fun method(owner: Any, name: String, vararg types: Class<*>): Method =
        owner.javaClass.getDeclaredMethod(name, *types).apply { isAccessible = true }
    private fun write(owner: Any, name: String, value: Any?) = owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(owner, value)
    private fun <T> read(owner: Any, name: String): T {
        @Suppress("UNCHECKED_CAST")
        return owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(owner) as T
    }

    private fun drawScene(stage: Int): ByteArray {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.WHITE)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 21f; color = Color.DKGRAY }
            canvas.drawText("SYNTHETIC / CONTROLLED INTEGRATION", 30f, 30f, paint)
            paint.style = Paint.Style.STROKE; paint.strokeWidth = 8f; paint.color = Color.BLUE
            canvas.drawRect(40f, 160f, 290f, 400f, paint)
            paint.color = Color.rgb(0, 145, 0)
            canvas.drawRect(350f, 160f, 600f, 400f, paint)
            paint.style = Paint.Style.FILL; paint.color = Color.BLUE
            canvas.drawText("BLUE TARGET", 85f, 440f, paint)
            paint.color = Color.rgb(0, 125, 0)
            canvas.drawText("GREEN TARGET", 395f, 440f, paint)
            val redY = if (stage >= 1) 245f else 75f
            paint.color = Color.RED
            canvas.drawRect(125f, redY, 205f, redY + 80f, paint)
            val yellowY = if (stage >= 2) 285f else 110f
            paint.color = Color.rgb(255, 210, 0)
            canvas.drawCircle(475f, yellowY, 40f, paint)
            paint.style = Paint.Style.STROKE; paint.strokeWidth = 2f; paint.color = Color.BLACK
            canvas.drawCircle(475f, yellowY, 40f, paint)
            return ByteArrayOutputStream().use { stream ->
                check(bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream))
                stream.toByteArray()
            }
        } finally { bitmap.recycle() }
    }
}
