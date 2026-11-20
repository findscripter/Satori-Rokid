package com.satori.glasses.capture

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Accelerometer (m/s^2, gravity included), gyroscope (rad/s) and game rotation vector (unit quaternion x, y, z, w)
 * in the device frame documented by Rokid: +X right, +Y up, +Z towards the wearer.
 * Every 100 ms one binary batch is emitted; each record is big-endian `u8 type, i64 timestampNs, f32 x, y, z, w`
 * (25 bytes; type 1 = accel, 4 = gyro, 15 = game rotation vector; w = 0 for accel/gyro).
 * Timestamps are SensorEvent.timestamp (elapsedRealtimeNanos base).
 */
class ImuStreamer(context: Context, private val onBatch: (JSONObject, ByteArray) -> Unit) : SensorEventListener {
    private val manager = context.getSystemService(SensorManager::class.java)
    private val lock = Any()
    private val buffer: ByteBuffer = ByteBuffer.allocate(RECORD_BYTES * MAX_RECORDS).order(ByteOrder.BIG_ENDIAN)
    private val totals = LongArray(3)
    private var count = 0
    private var dropped = 0L
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var receiving = false
    private var generation = 0L
    private var ignoredAfterStop = 0L
    private var ignoredOldGeneration = 0L
    private var pendingDeliveryRecords = 0L

    @Volatile var sensors: List<String> = emptyList()
        private set
    @Volatile var periodUs = 0
        private set

    fun start(hz: Int) {
        stop()
        val t = HandlerThread("imu").also { it.start() }
        val h = Handler(t.looper)
        synchronized(lock) {
            thread = t
            handler = h
            receiving = true
            generation++
            periodUs = 1_000_000 / hz
        }
        val found = mutableListOf<String>()
        for (type in TYPES) {
            val sensor = manager.getDefaultSensor(type) ?: continue
            if (manager.registerListener(this, sensor, periodUs, 0, h)) found += "${sensor.name} (type $type)"
        }
        sensors = found
        h.postDelayed(object : Runnable {
            override fun run() {
                flush(h.looper)
                h.postDelayed(this, FLUSH_MS)
            }
        }, FLUSH_MS)
    }

    fun stop() {
        // Closing admission and taking the tail are one operation. A listener already executing on
        // the old looper must either have appended before this lock or be counted as ignored later.
        val stopped = synchronized(lock) {
            receiving = false
            Triple(handler, thread, takeBatchLocked()).also { handler = null; thread = null }
        }
        manager.unregisterListener(this)
        stopped.first?.removeCallbacksAndMessages(null)
        deliver(stopped.third)
        stopped.second?.quitSafely()
    }

    fun stats(): JSONObject = synchronized(lock) {
        JSONObject()
            .put("accel", totals[0])
            .put("gyro", totals[1])
            .put("rotation", totals[2])
            .put("dropped", dropped)
            .put("period_us", periodUs)
            .put("sensors", JSONArray(sensors))
            .put("receiving", receiving).put("generation", generation)
            .put("ignored_after_stop", ignoredAfterStop).put("ignored_old_generation", ignoredOldGeneration)
            .put("buffered_records", count).put("pending_delivery_records", pendingDeliveryRecords)
    }

    override fun onSensorChanged(event: SensorEvent) {
        appendSample(event.sensor.type, event.timestamp, event.values, Looper.myLooper())
    }

    private fun appendSample(type: Int, timestampNs: Long, values: FloatArray, sourceLooper: Looper?) {
        val slot = when (type) {
            Sensor.TYPE_ACCELEROMETER -> 0
            Sensor.TYPE_GYROSCOPE -> 1
            Sensor.TYPE_GAME_ROTATION_VECTOR -> 2
            else -> return
        }
        synchronized(lock) {
            if (!receiving) { ignoredAfterStop++; return }
            if (sourceLooper == null || sourceLooper !== handler?.looper) { ignoredOldGeneration++; return }
            if (count >= MAX_RECORDS) {
                dropped++
                return
            }
            val w = when {
                slot != 2 -> 0f
                values.size > 3 -> values[3]
                else -> sqrt(max(0f, 1f - values[0] * values[0] - values[1] * values[1] - values[2] * values[2]))
            }
            buffer.put(TYPES[slot].toByte()).putLong(timestampNs)
                .putFloat(values[0]).putFloat(values[1]).putFloat(values[2]).putFloat(w)
            count++
            totals[slot]++
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private data class Batch(val records: Int, val payload: ByteArray)

    private fun flush(sourceLooper: Looper) {
        val batch = synchronized(lock) {
            if (!receiving || sourceLooper !== handler?.looper) return
            takeBatchLocked()
        }
        deliver(batch)
    }

    /** Caller holds lock; delivery is kept outside it because consumers may acquire service locks. */
    private fun takeBatchLocked(): Batch? {
        if (count == 0) return null
        val records = count
        val payload = ByteArray(buffer.position())
        buffer.flip()
        buffer.get(payload)
        buffer.clear()
        count = 0
        pendingDeliveryRecords += records
        return Batch(records, payload)
    }

    private fun deliver(batch: Batch?) {
        if (batch == null) return
        try { onBatch(JSONObject().put("type", "imu").put("count", batch.records).put("record_bytes", RECORD_BYTES), batch.payload) }
        finally { synchronized(lock) { pendingDeliveryRecords -= batch.records } }
    }

    companion object {
        const val RECORD_BYTES = 25
        private const val MAX_RECORDS = 4096
        private const val FLUSH_MS = 100L
        private val TYPES = intArrayOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_GAME_ROTATION_VECTOR)
    }
}
