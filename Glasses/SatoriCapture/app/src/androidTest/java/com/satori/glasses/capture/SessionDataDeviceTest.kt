package com.satori.glasses.capture

import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.FutureTask
import java.util.concurrent.atomic.AtomicBoolean

/** Local component fixtures only: no actual sensors, recording, user state writes or network clients. */
@RunWith(AndroidJUnit4::class)
class SessionDataDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test fun sampledImuTailIsSavedEvenAfterCaptureStops() = isolated { service, dir ->
        val at = SystemClock.elapsedRealtimeNanos()
        val payload = ByteBuffer.allocate(25).order(ByteOrder.BIG_ENDIAN).put(1.toByte()).putLong(at)
            .putFloat(1f).putFloat(2f).putFloat(3f).putFloat(0f).array()
        assertFalse(field(service, "capturing") as Boolean)
        repeat(2) { invoke(service, "onImu", arrayOf(JSONObject::class.java, ByteArray::class.java),
            JSONObject().put("count", 1).put("record_bytes", 25), payload) }
        assertArrayEquals(payload + payload, File(dir, "imu.bin").readBytes())
        val rows = File(dir, "imu-batches.jsonl").readLines().map(::JSONObject)
        assertEquals(2, rows.size)
        assertEquals(0L, rows[0].getLong("file_offset"))
        assertEquals(25L, rows[1].getLong("file_offset"))
        rows.forEach { assertEquals(25, it.getInt("byte_count")); assertIdentity(dir, it) }
    }

    @Test fun uprightFrameMetadataMatchesSavedPixelsAndKeepsSensorTimestamp() = isolated { service, dir ->
        val bitmap = Bitmap.createBitmap(64, 32, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(android.graphics.Color.GRAY)
        val output = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, output)
        bitmap.recycle()
        val at = SystemClock.elapsedRealtimeNanos()
        set(service, "capturing", true)
        invoke(service, "onFrame", arrayOf(JSONObject::class.java, ByteArray::class.java),
            JSONObject().put("index", 0).put("ts_ns", at).put("width", 64).put("height", 32).put("rotation", 90), output.toByteArray())
        set(service, "capturing", false)
        val row = JSONObject(File(dir, "frames.jsonl").readLines().single())
        val saved = BitmapFactory.decodeFile(File(dir, row.getString("frame_id") + ".jpg").absolutePath)
        try {
            assertEquals(32, saved.width); assertEquals(64, saved.height)
            assertEquals(saved.width, row.getInt("width")); assertEquals(saved.height, row.getInt("height"))
            assertEquals(64, row.getInt("sensor_width")); assertEquals(32, row.getInt("sensor_height"))
            assertEquals(90, row.getInt("sensor_rotation")); assertEquals(0, row.getInt("rotation"))
            assertEquals(at, row.getLong("ts_ns")); assertEquals(at / 1_000_000, row.getLong("timestamp_ms"))
            assertTrue(row.getLong("elapsed_ns") >= at)
            assertIdentity(dir, row)
        } finally { saved.recycle() }
    }

    @Test fun savedVideoSidecarLinksToTheExactDataSession() = isolated { service, dir ->
        val clip = File(dir, "recording-fixture.mp4")
        val sidecar = File(dir, "recording-fixture.json")
        val at = SystemClock.elapsedRealtimeNanos()
        sidecar.writeText(JSONObject().put("capture_start_elapsed_ns", at).put("finalized", true).toString())
        invoke(service, "associateVideoData", arrayOf(File::class.java), clip)
        val linked = JSONObject(sidecar.readText())
        val identity = JSONObject(File(dir, "session.json").readText())
        assertEquals(identity.getString("session_id"), linked.getString("run_session_id"))
        assertEquals(identity.getString("boot_id"), linked.getString("boot_id"))
        assertEquals(dir.name, linked.getString("data_run_id"))
        assertEquals("android.elapsedRealtimeNanos", linked.getString("clock_domain"))
        assertEquals(at, linked.getLong("capture_start_elapsed_ns"))
        val event = File(dir, "events.jsonl").readLines().map(::JSONObject).single { it.getString("name") == "video_data_link" }
        assertIdentity(dir, event)
    }

    @Test fun diskFailureRemainsVisibleAndPreventsRecordingWithoutData() {
        isolated { service, dir ->
            val blocked = File(dir, "blocked-directory").apply { writeText("fixture") }
            set(service, "directory", blocked)
            service.command("mark_sync")
            assertTrue((field(service, "dataLoggingFailed") as AtomicBoolean).get())
            set(service, "error", "later audio failure")
            service.command("record_start")
            val snapshot = service.snapshot()
            assertFalse(snapshot.getBoolean("data_logging_ok"))
            assertTrue(snapshot.getString("device_error").contains("实时数据保存失败"))
            assertFalse(snapshot.getBoolean("capturing"))
            assertNull(field(service, "videoRecorder"))
        }
    }

    private fun assertIdentity(dir: File, row: JSONObject) {
        val session = JSONObject(File(dir, "session.json").readText())
        assertEquals(session.getString("session_id"), row.getString("session_id"))
        assertEquals(session.getString("boot_id"), row.getString("boot_id"))
        assertEquals("android.elapsedRealtimeNanos", row.getString("clock_domain"))
        assertTrue(row.getLong("elapsed_ns") > 0); assertTrue(row.getLong("unix_ms") > 0)
    }

    private fun isolated(block: (StandaloneService, File) -> Unit) {
        val directory = File(target.filesDir, "session-data-tests/fixture-${UUID.randomUUID()}").apply { check(mkdirs()) }
        var isolated: StandaloneService? = null
        val original = StandaloneService.instance
        try {
            main {
                val service = StandaloneService()
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", Context::class.java)
                    .apply { isAccessible = true }.invoke(service, target)
                isolated = service
                set(service, "directory", directory)
                set(service, "camera", CameraStreamer(target, service, { _, _ -> }, { _, _ -> }))
                set(service, "imu", ImuStreamer(target) { _, _ -> })
                assertNull(field(service, "api")); assertNull(field(service, "voice")); assertNull(field(service, "speech"))
                block(service, directory)
                assertSame(original, StandaloneService.instance)
            }
        } finally {
            main { isolated?.let { service ->
                set(service, "closed", true)
                set(service, "capturing", false)
                (field(service, "main") as Handler).removeCallbacksAndMessages(null)
                (field(service, "camera") as CameraStreamer).shutdown()
                (field(service, "imu") as ImuStreamer).stop()
                (field(service, "gate") as StandaloneFlowGate).close()
                (field(service, "worker") as ExecutorService).shutdownNow()
                (field(service, "videoPublisher") as ExecutorService).shutdownNow()
            } }
        }
    }

    private fun field(receiver: Any, name: String): Any? = receiver.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(receiver)
    private fun set(receiver: Any, name: String, value: Any?) = receiver.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(receiver, value)
    private fun invoke(receiver: Any, name: String, types: Array<Class<*>>, vararg values: Any?) =
        receiver.javaClass.getDeclaredMethod(name, *types).apply { isAccessible = true }.invoke(receiver, *values)
    private fun <T> main(action: () -> T): T {
        val task = FutureTask(Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() } catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }
}
