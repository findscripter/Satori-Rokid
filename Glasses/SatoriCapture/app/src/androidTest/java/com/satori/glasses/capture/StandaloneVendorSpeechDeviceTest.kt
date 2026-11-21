package com.satori.glasses.capture

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.atomic.AtomicBoolean

/** Explicit opt-in, one local phrase; no credential, network, microphone or camera is used. */
@RunWith(AndroidJUnit4::class)
class StandaloneVendorSpeechDeviceTest {
    @Test(timeout = 55_000) fun exportedVendorServiceStartsAndFinishesOneChinesePrompt() {
        val selected = InstrumentationRegistry.getArguments().getString("class").orEmpty()
            .split(',').map { it.trim().substringBefore('#') }
        assumeTrue("vendor playback acceptance requires explicit class/method selection", javaClass.name in selected)
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var quiet = false
        instrumentation.runOnMainSync {
            val state = StandaloneService.instance?.snapshot()
            quiet = CaptureService.instance == null && (state == null ||
                (!state.optBoolean("capturing") && !state.optBoolean("busy") && !state.optBoolean("recording") &&
                    (state.optString("phase") in listOf("idle", "stopped", "complete") || !state.isNull("paused_reason"))))
        }
        assumeTrue("pause existing work before the single local voice check", quiet)
        val started = AtomicBoolean()
        val finished = AtomicBoolean()
        val adapter = StandaloneVendorSpeech(instrumentation.targetContext) { name, data ->
            if (data.optString("id") == "vendor-device-test") {
                if (name == "audio_started") started.set(true)
                if (name == "audio_done") finished.set(true)
            }
        }
        try {
            val result = adapter.speakBlocking("你好，Satori本机语音测试", "vendor-device-test") { true }
            assertTrue("vendor service did not complete the local prompt", result)
            assertTrue("vendor service must bind without a signature permission", adapter.ready)
            assertTrue("vendor callback did not report synthesis/playback start", started.get())
            assertTrue("vendor callback did not report playback stop/completion", finished.get())
            // Callback completion is verified; audibility and pronunciation require a human check.
        } finally { adapter.close() }
    }
}
