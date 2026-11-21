package com.satori.glasses.capture

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/** The actual speech coordinator must use device-local synthesis, with a transport that forbids network I/O. */
@RunWith(AndroidJUnit4::class)
class StandaloneSpeechDeviceTest {
    private fun selected() {
        val classes = InstrumentationRegistry.getArguments().getString("class").orEmpty().split(',').map { it.substringBefore('#') }
        assumeTrue("explicit local speech acceptance only", javaClass.name in classes)
    }

    @Test fun productionSpeechCompletesWithoutAnyCloudRequest() {
        selected()
        val test = InstrumentationRegistry.getInstrumentation()
        val network = AtomicInteger()
        val api = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "device-test-only-key", "planner", "monitor")) {
            network.incrementAndGet(); throw AssertionError("speech attempted network I/O")
        }
        val done = CountDownLatch(1)
        val outcome = AtomicReference("")
        val requested = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
        val completed = AtomicReference("")
        val speech = StandaloneSpeech(test.targetContext, api) { name, data ->
            if (name == "speech_requested" && data.optString("text") == "请确认当前步骤。") requested.add(data.getString("id"))
            if (name == "audio_done") completed.set(data.getString("id"))
            if (name in listOf("audio_done", "audio_error", "audio_incomplete")) { outcome.set(name); done.countDown() }
        }
        try {
            speech.speak("请确认当前步骤。")
            assertTrue("local speech did not finish", done.await(25, TimeUnit.SECONDS))
            assertEquals("audio_done", outcome.get())
            assertTrue("completed audio must identify the exact requested sentence", completed.get() in requested)
            assertEquals("local speech must not depend on a cloud endpoint", 0, network.get())
        } finally { speech.close(); api.close() }
    }

    @Test fun stopSuppressesLateCompletionWithoutCloudFallback() {
        selected()
        val test = InstrumentationRegistry.getInstrumentation()
        val network = AtomicInteger()
        val api = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "device-test-only-key", "planner", "monitor")) {
            network.incrementAndGet(); throw AssertionError("cancelled local speech attempted network I/O")
        }
        val started = CountDownLatch(1)
        val lateDone = AtomicInteger()
        val speech = StandaloneSpeech(test.targetContext, api) { name, _ ->
            if (name == "audio_started") started.countDown()
            if (name == "audio_done") lateDone.incrementAndGet()
        }
        try {
            speech.speak("这是用于检查暂停的较长本机语音。收到停止指令之后，助手应该停止当前播报，并且不会继续播放旧提示。")
            assertTrue("local speech did not start", started.await(15, TimeUnit.SECONDS))
            speech.stop()
            Thread.sleep(1500)
            assertEquals(0, lateDone.get())
            assertEquals(0, network.get())
        } finally { speech.close(); api.close() }
    }

    @Test fun queuedFailureFromPreviousTaskIsDiscardedAfterStop() {
        selected()
        val test = InstrumentationRegistry.getInstrumentation()
        val errors = AtomicInteger()
        val api = StandaloneApi(StandaloneProfile("https://example.invalid/v1", "device-test-only-key", "planner", "monitor")) {
            throw AssertionError("this event-ordering test must not use the network")
        }
        val speech = StandaloneSpeech(test.targetContext, api) { name, _ -> if (name == "audio_error") errors.incrementAndGet() }
        try {
            // Reproduce a worker failure already queued to main immediately before the user changes tasks.
            // Reflection injects only that asynchronous boundary; assertions concern the observable callback.
            val emit = StandaloneSpeech::class.java.declaredMethods.single { it.name == "emit" }.apply { isAccessible = true }
            test.runOnMainSync {
                val data = JSONObject().put("error", "接口额度不足，请充值或更换接口")
                if (emit.parameterCount == 2) emit.invoke(speech, "audio_error", data)
                else emit.invoke(speech, "audio_error", data, 0)
                speech.stop()
            }
            test.waitForIdleSync()
            assertEquals("an old failure could pause the newly selected task", 0, errors.get())
        } finally { speech.close(); api.close() }
    }
}
