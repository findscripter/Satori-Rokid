package com.satori.glasses.capture

import android.os.Looper
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/** Existing synthetic WAVs through the real streaming Vosk decoder; no hardware microphone or network. */
@RunWith(AndroidJUnit4::class)
class ContinuousVoiceDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext

    @Test fun automaticInputRoutesWithoutInventingGoalsOrConfirmingQuestions() = onMain {
        val service = StandaloneService()
        fun field(name: String) = service.javaClass.getDeclaredField(name).apply { isAccessible = true }
        val parse = service.javaClass.getDeclaredMethod("onText", String::class.java).apply { isAccessible = true }
        val engine = field("engine").get(service) as StandaloneEngine
        fun say(text: String) {
            field("pendingRecognitionTrace").set(service, JSONObject().put("automatic", true))
            parse.invoke(service, text)
        }
        try {
            say("今天天气不错")
            assertTrue(engine.snapshot().isNull("goal"))
            engine.setGoal("本地测试目标", true, SystemClock.elapsedRealtime())
            val before = engine.snapshot().getLong("revision")
            field("inFlight").set(service, true)
            say("任务完成了吧？")
            assertEquals(before, engine.snapshot().getLong("revision"))
            assertTrue("a status question must not cancel visual inference", field("inFlight").get(service) as Boolean)
            assertTrue((field("lastSpoken").get(service) as String).contains("还未确认"))
            say("停止监听")
            assertFalse(field("listeningEnabled").get(service) as Boolean)
            assertTrue(field("inFlight").get(service) as Boolean)
        } finally {
            (field("main").get(service) as android.os.Handler).removeCallbacksAndMessages(null)
            (field("worker").get(service) as java.util.concurrent.ExecutorService).shutdownNow()
            (field("videoPublisher").get(service) as java.util.concurrent.ExecutorService).shutdownNow()
        }
    }

    @Test(timeout = 180_000) fun twoUtterancesAutoSubmitWithoutStopAndShareOneCapture() {
        val pcm = AtomicReference<(JSONObject, ByteArray) -> Unit>()
        val starts = AtomicInteger()
        val network = AtomicInteger()
        val texts = Collections.synchronizedList(ArrayList<String>())
        val failures = Collections.synchronizedList(ArrayList<String>())
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { sink, _, _ ->
            pcm.set(sink)
            object : SharedMicrophone.Source {
                override fun start() = starts.incrementAndGet()
                override fun stop(reason: String) = Unit
            }
        })
        val api = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "fixture", "none", "none")) {
            network.incrementAndGet(); throw AssertionError("continuous speech must remain local")
        }
        val voice = onMain { StandaloneVoice(context, api, { texts.add(it) }, { failures.add(it) }, {}, microphone) }
        try {
            onMain { voice.startContinuous() }
            await { microphone.subscriberCount == 1 }
            assertTrue(voice.continuous); assertFalse("listening must not block visual inference", voice.busy)
            fun feed(id: String, count: Int) {
                val manifest = JSONObject(instrumentation.context.assets.open("voice-commands/source-manifest.json").bufferedReader().use { it.readText() }).getJSONArray("fixtures")
                val row = (0 until manifest.length()).map { manifest.getJSONObject(it) }.single { it.getString("id") == id }
                val wav = instrumentation.context.assets.open("voice-commands/${row.getString("file")}").use { it.readBytes() }
                val data = WavData.parse(wav)
                var at = data.offset
                while (at < data.offset + data.length) {
                    val end = minOf(at + 3200, data.offset + data.length)
                    pcm.get().invoke(JSONObject().put("utterance", starts.get()).put("ts_ns", SystemClock.elapsedRealtimeNanos()), wav.copyOfRange(at, end))
                    at = end; SystemClock.sleep(100)
                }
                repeat(25) {
                    pcm.get().invoke(JSONObject().put("utterance", starts.get()).put("ts_ns", SystemClock.elapsedRealtimeNanos()), ByteArray(3200))
                    SystemClock.sleep(100)
                }
                await { texts.size >= count || failures.isNotEmpty() }
                assertTrue(failures.toString(), failures.isEmpty())
            }
            feed("goal", 1)
            feed("pause", 2)
            assertEquals(listOf("我要整理书桌", "暂停"), texts.toList())
            assertTrue(voice.active); assertFalse(voice.busy)
            assertEquals(1, starts.get())
            assertEquals(0, network.get())
            onMain { voice.cancel() }
            await { microphone.subscriberCount == 0 }
            assertFalse(voice.active)
            onMain { voice.startContinuous() }
            await { microphone.subscriberCount == 1 }
            feed("resume", 3)
            assertEquals("继续", texts.last())
            assertEquals(2, starts.get())
        } finally { onMain { voice.close() }; microphone.close(); api.close() }
    }

    @Test fun unavailableLocalModelNeverFallsBackToBackgroundCloudUploads() {
        val network = AtomicInteger()
        val failures = Collections.synchronizedList(ArrayList<String>())
        val api = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "fixture", "none", "none")) {
            network.incrementAndGet(); throw AssertionError("automatic fallback uploaded audio")
        }
        val voice = onMain { StandaloneVoice(context, api, { fail("no audio") }, { failures.add(it) }) }
        try {
            voice.javaClass.getDeclaredField("offlineAttempted").apply { isAccessible = true }.set(voice, true)
            onMain { voice.startContinuous() }
            await { failures.isNotEmpty() }
            assertTrue(voice.automaticUnavailable)
            assertFalse(voice.active); assertFalse(voice.busy)
            assertEquals(0, network.get())
        } finally { onMain { voice.close() }; api.close() }
    }

    private fun await(check: () -> Boolean) {
        val end = SystemClock.elapsedRealtime() + 60_000
        while (!check() && SystemClock.elapsedRealtime() < end) SystemClock.sleep(50)
        assertTrue("timed out", check())
    }

    private fun <T> onMain(block: () -> T): T {
        if (Looper.myLooper() == Looper.getMainLooper()) return block()
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { block() })
        instrumentation.runOnMainSync(task)
        return try { task.get() } catch (e: java.util.concurrent.ExecutionException) { throw (e.cause ?: e) }
    }
}
