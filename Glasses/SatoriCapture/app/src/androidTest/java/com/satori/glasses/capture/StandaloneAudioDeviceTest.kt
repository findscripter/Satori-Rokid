package com.satori.glasses.capture

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Hardware checks use only a bundled WAV and short RAM-only recording, never a network endpoint. */
@RunWith(AndroidJUnit4::class)
class StandaloneAudioDeviceTest {
    @Test fun bundledPromptFinishesPlaybackWithoutNetwork() {
        val test = InstrumentationRegistry.getInstrumentation()
        val wav = test.context.assets.open("standalone-test-hello-16k.wav").use { it.readBytes() }
        val done = CountDownLatch(1)
        val outcome = AtomicReference("")
        val player = PromptPlayer(test.targetContext) { name, _ ->
            if (name in listOf("audio_done", "audio_error", "audio_incomplete", "audio_stopped")) { outcome.set(name); done.countDown() }
        }
        try {
            player.play("local-hardware-test", wav, true)
            assertTrue(done.await(15, TimeUnit.SECONDS))
            assertEquals("audio_done", outcome.get())
        } finally { player.shutdown() }
    }

    @Test fun microphoneProducesMonoPcmWithoutPcGatewayOrUpload() {
        val test = InstrumentationRegistry.getInstrumentation()
        val target = test.targetContext
        val activity = test.startActivitySync(Intent(target, StandaloneActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        var safe = false
        test.runOnMainSync { safe = StandaloneService.instance?.snapshot()?.let { !it.optBoolean("recording") && !it.optBoolean("busy") && !it.optBoolean("capturing") } == true }
        assumeTrue("preserve a running user task", safe)
        val chunks = CountDownLatch(3)
        val stopped = CountDownLatch(1)
        val error = AtomicReference("")
        val voice = VoiceRecorder({ header, bytes ->
            if (header.optInt("channels") != 1 || header.optInt("sample_rate") != 16000 || bytes.isEmpty()) error.set("invalid_pcm")
            chunks.countDown()
        }, { name, _ -> if (name == "ptt_error") { error.set("ptt_error"); chunks.countDown() }; if (name == "ptt_stop") stopped.countDown() })
        try {
            test.runOnMainSync { voice.start("local_hardware_test") }
            assertTrue("microphone produced no PCM", chunks.await(8, TimeUnit.SECONDS))
            voice.stop("local_test_done")
            assertTrue(stopped.await(5, TimeUnit.SECONDS))
            assertEquals("", error.get())
            assertNull(CaptureService.instance)
        } finally { voice.stop("test_finally"); test.runOnMainSync { activity.finish() } }
    }
}
