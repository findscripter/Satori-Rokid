package com.satori.glasses.capture

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.vosk.Model
import org.vosk.Recognizer
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Opt-in controlled device checks. Ordinary instrumentation discovery skips every test in this class.
 * Explicitly select this class (or one method) with `am instrument -e class ...StandaloneLiveDeviceTest`.
 * At most one synthetic-image vision request and one short TTS request are made; each API call has
 * a 45 s absolute budget. No camera, microphone, cloud ASR, user goal, or PC gateway is used.
 */
@RunWith(AndroidJUnit4::class)
class StandaloneLiveDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Before fun requireExplicitSelectionAndInactiveService() {
        val selected = InstrumentationRegistry.getArguments().getString("class").orEmpty()
            .split(',').map { it.trim().substringBefore('#') }
        assumeTrue("live checks require explicit class/method selection", javaClass.name in selected)
        var safe = false
        instrumentation.runOnMainSync {
            val state = StandaloneService.instance?.snapshot()
            safe = CaptureService.instance == null && (state == null ||
                (!state.optBoolean("capturing") && !state.optBoolean("busy") && !state.optBoolean("recording") &&
                    (state.optString("phase") in listOf("idle", "stopped", "complete") || !state.isNull("paused_reason"))))
        }
        assumeTrue("pause the service and finish voice activity before controlled live checks", safe)
    }

    @Test(timeout = 60_000) fun syntheticFramePassesRealBdiDecompositionContract() {
        val profile = configuredProfile()
        val engine = StandaloneEngine()
        val at = SystemClock.elapsedRealtime()
        engine.setGoal("描述并整理示意图中的桌面", confirmMode = true, nowMs = at)
        val request = engine.nextRequest(at + 1, listOf("synthetic-1"))
            ?: throw AssertionError("local BDI engine did not issue decomposition request")
        assertEquals("decompose", request.stage)
        // Match the real service's prompt and frame-ID envelope rather than a connectivity-only schema.
        val prompts = target.assets.open("standalone-prompts.json").bufferedReader().use { JSONObject(it.readText()) }
        request.payload.put("frames_note", JSONArray().put(JSONObject().put("frame_id", "synthetic-1")
            .put("timestamp_ms", at).put("key", "start")))
        StandaloneApi(profile).use { api ->
            val result = api.complete(
                model = profile.planner,
                prompt = prompts.getString(request.stage),
                payload = request.payload,
                frames = listOf(syntheticJpeg()),
                timeoutMs = 45_000,
            )
            assertTrue("vision response must be a nonempty JSON object", result.length() > 0)
            java.io.File(target.filesDir, "standalone-synthetic-contract.json").writeText(result.toString())
            engine.accept(request, result, SystemClock.elapsedRealtime())
            val accepted = engine.snapshot(SystemClock.elapsedRealtime())
            assertEquals("real provider reply did not satisfy the BDI stage contract", 0, accepted.getInt("failures"))
            assertTrue("decomposition must wait for confirmation or clarification",
                accepted.getString("phase") in listOf("awaiting_plan_confirmation", "awaiting_clarification"))
        }
    }

    @Test(timeout = 80_000) fun productionTtsReturnsValidWavAndAudioTrackCompletesPlayback() {
        // The override is in memory only. No saved provider setting or credential is changed.
        val profile = configuredProfile().copy(ttsModel = "gpt-4o-mini-tts")
        val wav = StandaloneApi(profile).use { api -> api.synthesize("你好，我要整理书桌。") }
        StandaloneApi.validateWav(wav)
        val pcm = WavData.parse(wav)
        assertTrue("TTS returned empty PCM", pcm.length > 0)
        val durationMs = pcm.length * 1000L / (pcm.channels * 2L * pcm.sampleRate)
        assertTrue("short TTS fixture exceeded the playback acceptance budget", durationMs in 100..15_000)
        val started = AtomicBoolean(false)
        val outcome = AtomicReference("")
        val completed = CountDownLatch(1)
        val player = PromptPlayer(target) { name, data ->
            if (data.optString("id") == "standalone-live-tts") {
                if (name == "audio_started") started.set(true)
                if (name in listOf("audio_done", "audio_error", "audio_incomplete", "audio_stopped", "audio_skipped")) {
                    outcome.set(name)
                    completed.countDown()
                }
            }
        }
        try {
            player.play("standalone-live-tts", wav, immediate = true)
            assertTrue("AudioTrack completion event was not received", completed.await(20, TimeUnit.SECONDS))
            assertTrue("AudioTrack never reported playback start", started.get())
            assertEquals("AudioTrack must consume the full prompt", "audio_done", outcome.get())
            // audio_done proves buffer playback completion; human audibility/pronunciation is not asserted.
        } finally { player.shutdown() }
    }

    @Test(timeout = 60_000) fun bundledGreetingIsRecognizedLocallyWithoutCloudAsr() {
        val wav = instrumentation.context.assets.open("standalone-test-hello-16k.wav").use { it.readBytes() }
        StandaloneApi.validateWav(wav)
        val pcm = WavData.parse(wav)
        assertEquals("offline recognizer fixture must be 16 kHz", 16_000, pcm.sampleRate)
        assertEquals("offline recognizer fixture must be mono", 1, pcm.channels)
        val directory = StandaloneOfflineModel.prepare(target)
        val model = Model(directory.absolutePath)
        try {
            val recognizer = Recognizer(model, 16_000f)
            try {
                val phrases = ArrayList<String>()
                var offset = pcm.offset
                val end = pcm.offset + pcm.length
                while (offset < end) {
                    val count = minOf(4_096, end - offset)
                    val chunk = wav.copyOfRange(offset, offset + count)
                    if (recognizer.acceptWaveForm(chunk, chunk.size)) {
                        phrases.add(JSONObject(recognizer.result).optString("text"))
                    }
                    offset += count
                }
                phrases.add(JSONObject(recognizer.finalResult).optString("text"))
                val text = phrases.joinToString("").replace(Regex("\\s+"), "")
                assertTrue("local greeting recognition returned no text", text.isNotBlank())
                assertTrue("local fixture was not recognized as a plausible Chinese greeting",
                    text.contains("你好") || text.contains('你') || text.contains("您好") || text.contains('您'))
            } finally { recognizer.close() }
        } finally { model.close() }
    }

    private fun configuredProfile(): StandaloneProfile {
        // Validation errors are sanitized by production code. Never log/interpolate the profile or key.
        val profile = StandaloneSettings(target).load()
        profile.validate()
        val expected = InstrumentationRegistry.getArguments().getString("expected_key_sha256")
        if (!expected.isNullOrBlank()) {
            val actual = java.security.MessageDigest.getInstance("SHA-256").digest(profile.apiKey.toByteArray())
                .joinToString("") { "%02x".format(it.toInt() and 255) }
            assertTrue("device credential does not match the provisioned source", actual == expected)
        }
        return profile
    }

    private fun syntheticJpeg(): ByteArray {
        val bitmap = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888)
        try {
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.WHITE)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG)
            paint.color = Color.RED
            canvas.drawRect(30f, 40f, 130f, 170f, paint)
            paint.color = Color.BLUE
            canvas.drawCircle(230f, 110f, 55f, paint)
            paint.color = Color.GREEN
            canvas.drawRect(150f, 190f, 290f, 215f, paint)
            return ByteArrayOutputStream().use { stream ->
                check(bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream))
                stream.toByteArray()
            }
        } finally { bitmap.recycle() }
    }
}
