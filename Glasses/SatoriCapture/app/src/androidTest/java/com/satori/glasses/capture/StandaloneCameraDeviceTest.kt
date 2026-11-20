package com.satori.glasses.capture

import android.content.Intent
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Real camera + native image processing, with no image upload or image retained on disk. */
@RunWith(AndroidJUnit4::class)
class StandaloneCameraDeviceTest {
    @Test fun cameraFeedsNativeFlowWhilePcGatewayIsAbsent() {
        val test = InstrumentationRegistry.getInstrumentation()
        val context = test.targetContext
        val activity = test.startActivitySync(Intent(context, StandaloneActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        val deadline = SystemClock.elapsedRealtime() + 5000
        while (StandaloneService.instance == null && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(100)
        val service = StandaloneService.instance ?: throw AssertionError("standalone service did not start")
        var safe = false
        test.runOnMainSync {
            val state = service.snapshot()
            safe = state.optString("phase") == "idle" && !state.optBoolean("capturing") && CaptureService.instance == null
        }
        assumeTrue("preserve any existing user task", safe)
        val frames = CountDownLatch(3)
        val error = AtomicReference("")
        val gate = StandaloneFlowGate()
        val camera = CameraStreamer(context, service, { header, jpeg ->
            try {
                assertTrue(jpeg.size > 1024)
                val sample = gate.analyze(jpeg, header.optLong("ts_ns") / 1_000_000)
                assertTrue("native gate unavailable", sample.optBoolean("available"))
                frames.countDown()
            } catch (e: Throwable) { error.set(e.javaClass.simpleName + ":" + (e.message ?: "")); while (frames.count > 0) frames.countDown() }
        }, { name, _ -> if (name == "camera_error") error.set("camera_error") })
        try {
            test.runOnMainSync { camera.start(CaptureConfig(fps = 2.0)) }
            assertTrue("camera did not deliver three frames", frames.await(20, TimeUnit.SECONDS))
            assertEquals("", error.get())
            assertNull("legacy PC capture path must remain unused", CaptureService.instance)
        } finally {
            test.runOnMainSync { camera.shutdown(); activity.finish() }
            gate.close()
        }
    }
}
