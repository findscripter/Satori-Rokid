package com.satori.glasses.capture

import android.content.Context
import android.content.ContextWrapper
import android.os.Handler
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.Closeable
import java.io.File
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicInteger

/**
 * Component-only reproduction of delayed recorder callbacks. The actual private onVideoEvent method
 * runs on an isolated attached Service, without onCreate, codecs, microphone, camera, speech or API.
 * No file is created and the old saved path cannot trigger MediaStore. This proves callback routing,
 * not real recording, sensor input, synthesis or playback.
 */
@RunWith(AndroidJUnit4::class)
class FirstPersonEventRaceDeviceTest {
    @Test fun delayedPreviousSessionEventsCannotDetachOrChangeTheCurrentSession() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        var service: StandaloneService? = null
        val closedSubscriptions = AtomicInteger()
        val lease = Closeable { closedSubscriptions.incrementAndGet() }
        val suffix = UUID.randomUUID().toString()
        val previous = File(target.cacheDir, "recording-race-A-$suffix.mp4").absolutePath
        val current = File(target.cacheDir, "recording-race-B-$suffix.mp4").absolutePath
        assertFalse("old callback path must not publish or read an actual file", File(previous).exists())
        val runningService = StandaloneService.instance
        try {
            instrumentation.runOnMainSync {
                val isolated = StandaloneService()
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", Context::class.java)
                    .apply { isAccessible = true }.invoke(isolated, target)
                service = isolated
                assertNull(field(isolated, "api"))
                assertNull(field(isolated, "speech"))
                assertNull(field(isolated, "videoRecorder"))
                setField(isolated, "videoMicrophone", lease)
                setField(isolated, "videoSessionPath", current)
                setField(isolated, "error", "current-session-error-marker")
                setField(isolated, "lastSpoken", "current-session-speech-marker")
                setField(isolated, "lastVideoName", "current-last-video-marker")

                listOf("video_started", "video_error", "video_saved").forEach { event ->
                    invokeEvent(isolated, event, JSONObject().put("path", previous)
                        .put("error", "stale-previous-session-error"))
                    assertEquals("$event closed the new session's microphone lease", 0, closedSubscriptions.get())
                    assertSame("$event replaced the current microphone lease", lease, field(isolated, "videoMicrophone"))
                    assertEquals("$event cleared/replaced the current session path", current, field(isolated, "videoSessionPath"))
                    assertEquals("$event replaced the current error", "current-session-error-marker", field(isolated, "error"))
                    assertEquals("$event submitted stale speech", "current-session-speech-marker", field(isolated, "lastSpoken"))
                    assertEquals("a nonexistent old saved callback changed the last video", "current-last-video-marker", field(isolated, "lastVideoName"))
                }

                // A matching terminal callback must still release its own lease: the fix cannot
                // merely ignore every callback. Speech remains null, so lastSpoken is intent only.
                invokeEvent(isolated, "video_error", JSONObject().put("path", current).put("error", "current-session-failure"))
                assertEquals(1, closedSubscriptions.get())
                assertNull(field(isolated, "videoMicrophone"))
                assertNull(field(isolated, "videoSessionPath"))
                assertEquals("current-session-failure", field(isolated, "error"))
                assertFalse(field(isolated, "capturing") as Boolean)
                assertSame("isolated callback checks must not replace the production service", runningService, StandaloneService.instance)
            }
        } finally {
            instrumentation.runOnMainSync {
                service?.let {
                    setField(it, "closed", true)
                    setField(it, "videoMicrophone", null)
                    (field(it, "main") as Handler).removeCallbacksAndMessages(null)
                    (field(it, "worker") as ExecutorService).shutdownNow()
                    (field(it, "videoPublisher") as ExecutorService).shutdownNow()
                }
            }
        }
    }

    private fun invokeEvent(receiver: StandaloneService, name: String, data: JSONObject) =
        StandaloneService::class.java.getDeclaredMethod("onVideoEvent", String::class.java, JSONObject::class.java)
            .apply { isAccessible = true }.invoke(receiver, name, data)

    private fun field(receiver: Any, name: String): Any? = receiver.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.get(receiver)

    private fun setField(receiver: Any, name: String, value: Any?) = receiver.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.set(receiver, value)
}
