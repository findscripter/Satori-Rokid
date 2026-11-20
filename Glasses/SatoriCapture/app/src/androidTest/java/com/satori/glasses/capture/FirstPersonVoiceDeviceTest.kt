package com.satori.glasses.capture

import android.content.ContextWrapper
import android.os.Handler
import android.os.Looper
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
import java.security.MessageDigest
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** Synthetic PCM -> production decoder factory/appendOffline -> exact text -> isolated real Service router.
 * This never starts recording, the service lifecycle, microphone, camera, vendor playback or cloud calls.
 * The null videoRecorder intentionally validates routing with "录像服务尚未就绪" instead of claiming MP4 acceptance.
 */
@RunWith(AndroidJUnit4::class)
class FirstPersonVoiceDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test(timeout = 120_000) fun syntheticRecordingCommandsUseProductionDecoderAndDoNotBecomeGoals() {
        assumeTrue("do not disturb a live standalone session", StandaloneService.instance == null)
        assumeTrue("do not disturb legacy capture", CaptureService.instance == null)
        val state = File(target.filesDir, "standalone-state.json")
        val stateFiles = listOf(state, File(state.path + ".bak"), File(state.path + ".new"))
        val saved = stateFiles.associateWith { if (it.exists()) it.readBytes() else null }
        val network = AtomicInteger()
        val api = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "fixture-only-not-an-api-key", "fixture", "fixture")) {
            network.incrementAndGet(); throw AssertionError("recording command fixture must never use HTTP")
        }
        val voice = StandaloneVoice(target, api, { fail("test never starts Voice delivery/microphone") }, { fail("unexpected Voice callback: $it") })
        val service = onMain {
            StandaloneService().also { instance ->
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", android.content.Context::class.java)
                    .apply { isAccessible = true }.invoke(instance, target)
            }
        }
        val engine = get<StandaloneEngine>(service, "engine")
        val parse = StandaloneService::class.java.getDeclaredMethod("onText", String::class.java).apply { isAccessible = true }
        val model = Model(StandaloneOfflineModel.prepare(target).absolutePath)
        val proof = JSONArray()
        val folder = File(target.filesDir, "standalone-voice-diagnostics").apply { check(mkdirs() || isDirectory) }
        val report = File(folder, "recording-command-fixtures-${System.currentTimeMillis()}.json")
        try {
            val manifest = instrumentation.context.assets.open("recording-commands/source-manifest.json")
                .bufferedReader().use { JSONObject(it.readText()) }.getJSONArray("fixtures")
            for (index in 0 until manifest.length()) {
                val source = manifest.getJSONObject(index)
                val wav = instrumentation.context.assets.open("recording-commands/${source.getString("file")}").use { it.readBytes() }
                val hash = MessageDigest.getInstance("SHA-256").digest(wav).joinToString("") { "%02x".format(it.toInt() and 255) }
                assertEquals(source.getString("sha256"), hash)
                val pcm = WavData.parse(wav)
                assertEquals(16000, pcm.sampleRate); assertEquals(1, pcm.channels)
                val native = StandaloneVoice::class.java.getDeclaredMethod("newOfflineRecognizer", Model::class.java)
                    .apply { isAccessible = true }.invoke(voice, model) as Recognizer
                val sessionType = Class.forName("com.satori.glasses.capture.StandaloneVoice\$OfflineSession")
                val session = sessionType.getDeclaredConstructor(Int::class.javaPrimitiveType, Recognizer::class.java)
                    .apply { isAccessible = true }.newInstance(1, native)
                val append = StandaloneVoice::class.java.getDeclaredMethod("appendOffline", sessionType, String::class.java)
                    .apply { isAccessible = true }
                try {
                    fun feed(bytes: ByteArray) {
                        if (native.acceptWaveForm(bytes, bytes.size)) append.invoke(voice, session, native.result)
                    }
                    var offset = pcm.offset
                    val end = pcm.offset + pcm.length
                    while (offset < end) { val count = minOf(3200, end - offset); feed(wav.copyOfRange(offset, offset + count)); offset += count }
                    repeat(5) { feed(ByteArray(3200)) }
                    append.invoke(voice, session, native.finalResult)
                    val raw = StandaloneVoice.normalizeOfflineText(get<ArrayList<String>>(session, "rawSegments").joinToString(" "))
                    val selected = StandaloneVoice.normalizeOfflineText(get<ArrayList<String>>(session, "segments").joinToString(" "))
                    val record = JSONObject().put("source", source).put("raw_top1", raw).put("selected", selected)
                        .put("selection_trace", JSONArray(get<ArrayList<JSONObject>>(session, "selections")))
                    proof.put(record)
                    assertEquals("ASR mismatch must not be replaced with fixture text", source.getString("spoken_text"), raw)
                    assertEquals("recording command top1 must remain the selected command", raw, selected)
                    onMain {
                        assertNull("isolated route must have no recorder", get<FirstPersonRecorder?>(service, "videoRecorder"))
                        set(service, "error", "")
                        parse.invoke(service, selected)
                        val error = get<String>(service, "error")
                        record.put("route_error", error).put("state_after", engine.snapshot())
                        if (source.getString("id") == "start") assertEquals("录像服务尚未就绪", error)
                        else assertEquals("stop recording must route as a recording command", "", error)
                        assertTrue("recording speech must not set a task goal", engine.snapshot().isNull("goal"))
                        assertEquals("idle", engine.snapshot().getString("phase"))
                    }
                } finally { native.close() }
            }
            assertEquals(2, proof.length())
            assertEquals(0, network.get())
        } finally {
            try {
                report.writeText(JSONObject().put("scope", "ASR and isolated null-recorder routing; no microphone, hardware key or MP4 acceptance")
                    .put("network_attempts", network.get()).put("fixtures", proof).toString(2))
            } finally {
                model.close()
                onMain { voice.close(); get<StandaloneFlowGate>(service, "gate").close(); get<Handler>(service, "main").removeCallbacksAndMessages(null)
                    get<ExecutorService>(service, "worker").shutdownNow(); get<ExecutorService>(service, "videoPublisher").shutdownNow() }
                get<ExecutorService>(voice, "worker").awaitTermination(5, TimeUnit.SECONDS)
                api.close()
                for ((file, bytes) in saved) if (bytes == null) check(!file.exists() || file.delete()) else file.writeBytes(bytes)
            }
        }
    }

    private fun <T> get(owner: Any, name: String): T {
        @Suppress("UNCHECKED_CAST")
        return owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(owner) as T
    }
    private fun set(owner: Any, name: String, value: Any?) { owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(owner, value) }
    private fun <T> onMain(action: () -> T): T {
        if (Looper.myLooper() == Looper.getMainLooper()) return action()
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() }
        catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }
}
