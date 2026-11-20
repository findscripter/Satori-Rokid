package com.satori.glasses.capture

import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Component race fixtures, plus one explicitly guarded 800 ms real IMU callback smoke. */
@RunWith(AndroidJUnit4::class)
class ImuStopDeviceTest {
    private val target get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun realSensorCallbacksUseTheActiveLooperAndStopDrainsTheirTail() {
        assumeTrue("Protect active production capture", StandaloneService.instance == null && CaptureService.instance == null)
        val manager = target.getSystemService(SensorManager::class.java)
        val available = setOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_GAME_ROTATION_VECTOR)
            .filter { manager.getDefaultSensor(it) != null }.toSet()
        assumeTrue("Device has no supported IMU sensors", available.isNotEmpty())
        val batches = Collections.synchronizedList(mutableListOf<Pair<JSONObject, ByteArray>>())
        val streamer = ImuStreamer(target) { header, payload -> batches += JSONObject(header.toString()) to payload.copyOf() }
        val before = streamer.stats().getLong("ignored_old_generation")
        try {
            // This public start is the production SensorManager registration path, with no service,
            // camera, microphone, model client or writes to a user's experiment directory.
            streamer.start(50)
            available.forEach { type -> assertTrue("Available sensor type $type did not register",
                streamer.sensors.any { it.endsWith("(type $type)") }) }
            SystemClock.sleep(800)
        } finally { streamer.stop() }
        val deliveryDeadline = SystemClock.elapsedRealtime() + 1000
        while (streamer.stats().getLong("pending_delivery_records") != 0L && SystemClock.elapsedRealtime() < deliveryDeadline) {
            SystemClock.sleep(10)
        }
        val after = streamer.stats()
        assertFalse(after.getBoolean("receiving"))
        assertEquals(0, after.getInt("buffered_records"))
        assertEquals(0L, after.getLong("pending_delivery_records"))
        assertEquals(before, after.getLong("ignored_old_generation"))
        val counts = available.associateWith { 0 }.toMutableMap()
        val snapshot = synchronized(batches) { batches.toList() }
        assertTrue("No real SensorManager samples reached the batch consumer", snapshot.isNotEmpty())
        var totalBytes = 0L
        var minTimestamp = Long.MAX_VALUE
        var maxTimestamp = 0L
        for ((header, payload) in snapshot) {
            assertEquals(25, header.getInt("record_bytes"))
            assertEquals(0, payload.size % 25)
            assertEquals(header.getInt("count") * 25, payload.size)
            totalBytes += payload.size
            val buffer = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
            while (buffer.hasRemaining()) {
                val type = buffer.get().toInt() and 0xff
                val at = buffer.long
                assertTrue("Unexpected sensor type $type", type in available)
                assertTrue("Invalid sample time", at > 0)
                repeat(4) { assertTrue("Non-finite IMU value", buffer.float.isFinite()) }
                counts[type] = counts.getValue(type) + 1
                minTimestamp = minOf(minTimestamp, at); maxTimestamp = maxOf(maxTimestamp, at)
            }
        }
        available.forEach { type -> assertTrue("Registered sensor type $type produced no real samples", counts.getValue(type) > 0) }
        assertTrue(maxTimestamp <= SystemClock.elapsedRealtimeNanos())
        Log.i("SatoriImuSmoke", JSONObject().put("sample_counts", JSONObject(counts.mapKeys { it.key.toString() }))
            .put("bytes", totalBytes).put("sample_min_elapsed_ns", minTimestamp).put("sample_max_elapsed_ns", maxTimestamp)
            .put("stats_after_stop", after).toString())
    }

    @Test fun stopFlushesBufferedTailAndRejectsAnAlreadyRunningLateCallback() {
        val batches = Collections.synchronizedList(mutableListOf<ByteArray>())
        val streamer = ImuStreamer(target) { _, payload -> batches += payload }
        val old = install(streamer, "imu-stop-fixture")
        val oldLooper = old.looper
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val done = CountDownLatch(1)
        try {
            sample(streamer, oldLooper, 100)
            Handler(oldLooper).post {
                entered.countDown()
                try { if (release.await(5, TimeUnit.SECONDS)) sample(streamer, oldLooper, 200) }
                finally { done.countDown() }
            }
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            streamer.stop()
            assertEquals(1, batches.size)
            assertEquals(100L, timestamp(batches.single()))
            assertFalse(streamer.stats().getBoolean("receiving"))
            release.countDown()
            assertTrue(done.await(3, TimeUnit.SECONDS))
            assertEquals(1L, streamer.stats().getLong("ignored_after_stop"))
            assertEquals(0, streamer.stats().getInt("buffered_records"))
            assertEquals(0L, streamer.stats().getLong("pending_delivery_records"))
            streamer.stop()
            assertEquals(1, batches.size)
        } finally { release.countDown(); old.quitSafely(); old.join(3000); streamer.stop() }
    }

    @Test fun previousLooperCannotAppendToOrFlushTheNextGeneration() {
        val batches = Collections.synchronizedList(mutableListOf<ByteArray>())
        val streamer = ImuStreamer(target) { _, payload -> batches += payload }
        val old = install(streamer, "imu-old-fixture")
        val oldLooper = old.looper
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val done = CountDownLatch(1)
        var next: HandlerThread? = null
        try {
            Handler(oldLooper).post {
                entered.countDown()
                try { if (release.await(5, TimeUnit.SECONDS)) sample(streamer, oldLooper, 200) }
                finally { done.countDown() }
            }
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            streamer.stop()
            next = install(streamer, "imu-new-fixture")
            sample(streamer, next.looper, 300)
            release.countDown()
            assertTrue(done.await(3, TimeUnit.SECONDS))
            flush(streamer, oldLooper)
            assertTrue(batches.isEmpty())
            assertEquals(1, streamer.stats().getInt("buffered_records"))
            assertEquals(1L, streamer.stats().getLong("ignored_old_generation"))
            streamer.stop()
            assertEquals(1, batches.size)
            assertEquals(300L, timestamp(batches.single()))
        } finally {
            release.countDown(); streamer.stop()
            old.quitSafely(); old.join(3000)
            next?.quitSafely(); next?.join(3000)
        }
    }

    @Test fun stopDoesNotWaitForAnInFlightConsumerHoldingAnotherServiceLock() {
        val consumerLock = Any()
        val entered = CountDownLatch(1)
        val delivered = CountDownLatch(1)
        val streamer = ImuStreamer(target) { _, _ ->
            entered.countDown()
            synchronized(consumerLock) { delivered.countDown() }
        }
        val source = install(streamer, "imu-delivery-fixture")
        try {
            sample(streamer, source.looper, 100)
            synchronized(consumerLock) {
                Handler(source.looper).post { flush(streamer, source.looper) }
                assertTrue(entered.await(3, TimeUnit.SECONDS))
                streamer.stop()
                assertEquals(1L, streamer.stats().getLong("pending_delivery_records"))
                assertEquals(0, streamer.stats().getInt("buffered_records"))
            }
            assertTrue(delivered.await(3, TimeUnit.SECONDS))
            source.join(3000)
            assertEquals(0L, streamer.stats().getLong("pending_delivery_records"))
        } finally { streamer.stop(); source.quitSafely(); source.join(3000) }
    }

    private fun install(streamer: ImuStreamer, name: String): HandlerThread {
        val source = HandlerThread(name).also { it.start() }
        synchronized(field(streamer, "lock")!!) {
            set(streamer, "thread", source); set(streamer, "handler", Handler(source.looper))
            set(streamer, "receiving", true)
            set(streamer, "generation", (field(streamer, "generation") as Long) + 1)
        }
        return source
    }

    private fun sample(streamer: ImuStreamer, source: Looper, atNs: Long) {
        ImuStreamer::class.java.getDeclaredMethod("appendSample", Int::class.javaPrimitiveType,
            Long::class.javaPrimitiveType, FloatArray::class.java, Looper::class.java).apply { isAccessible = true }
            .invoke(streamer, 1, atNs, floatArrayOf(1f, 2f, 3f), source)
    }

    private fun flush(streamer: ImuStreamer, source: Looper) = ImuStreamer::class.java
        .getDeclaredMethod("flush", Looper::class.java).apply { isAccessible = true }.invoke(streamer, source)
    private fun timestamp(payload: ByteArray): Long = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN).apply { get() }.long
    private fun field(receiver: Any, name: String): Any? = receiver.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(receiver)
    private fun set(receiver: Any, name: String, value: Any?) = receiver.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(receiver, value)
}
