package com.satori.glasses.capture

import android.app.Activity
import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Rejection boundaries only: no projection permission prompt, recording, camera, mic or network. */
@RunWith(AndroidJUnit4::class)
class ScreenRecorderDeviceTest {
    @Test fun deniedConsentCannotCreateMediaOrStartAProjection() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val events = AtomicReference<JSONObject>()
        val done = CountDownLatch(1)
        val recorder = ScreenRecorder(context) { name, data ->
            if (name == "screen_rejected") { events.set(data); done.countDown() }
        }
        val file = File(context.filesDir, "screen-recordings/screen-${System.currentTimeMillis()}.mp4")
        try {
            recorder.start(file, Activity.RESULT_CANCELED, Intent(), 640, 480, 160)
            assertTrue(done.await(5, TimeUnit.SECONDS))
            assertEquals("invalid_request_or_busy", events.get().getString("reason"))
            assertFalse(recorder.active)
            assertFalse(file.exists())
            assertFalse(File(file.parentFile, file.nameWithoutExtension + ".pending.mp4").exists())
            assertEquals("idle", recorder.snapshot().getString("state"))
        } finally { recorder.close() }
    }

    @Test fun existingOutputIsPreservedWithoutObtainingProjection() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val folder = File(context.filesDir, "screen-recordings").apply { check(mkdirs() || isDirectory) }
        val file = File(folder, "screen-${System.nanoTime()}.mp4")
        check(file.createNewFile())
        val fixture = byteArrayOf(9, 8, 7, 6)
        file.writeBytes(fixture)
        val done = CountDownLatch(1)
        val error = AtomicReference<JSONObject>()
        val recorder = ScreenRecorder(context) { name, data ->
            if (name == "screen_error") { error.set(data); done.countDown() }
        }
        try {
            recorder.start(file, Activity.RESULT_OK, Intent(), 640, 480, 160)
            assertTrue(done.await(5, TimeUnit.SECONDS))
            assertEquals("screen_start_failed", error.get().getString("reason"))
            assertArrayEquals(fixture, file.readBytes())
            assertFalse(File(folder, file.nameWithoutExtension + ".pending.mp4").exists())
            assertFalse(File(folder, file.nameWithoutExtension + ".json").exists())
            assertFalse(recorder.active)
        } finally { recorder.close(); check(file.delete()) }
    }
}
