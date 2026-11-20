package com.satori.glasses.capture

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Push-to-talk capture. Rokid documents an 8-channel 16 kHz PCM16 layout (mask 0x6000FC): channels 0/1 are the
 * system-processed signal, 2-5 the raw microphones, 6/7 the hardware echo reference. Only channel 0 is sent,
 * as 100 ms mono s16le chunks. Falls back to plain mono if the 8-channel layout cannot be opened.
 * Each utterance owns an id; a recorder thread runs only while `current` holds its id, so a quick
 * stop/start never revives an old thread or lets a finishing thread clear a newer utterance.
 */
class VoiceRecorder(
    private val sink: (JSONObject, ByteArray) -> Unit,
    private val onEvent: (String, JSONObject) -> Unit,
    private val continuous: Boolean = false,
    private val maxDurationMs: Long = 20_000,
    private val rawSink: ((JSONObject, ByteArray) -> Unit)? = null,
) {
    init { require(maxDurationMs in 1..86_400_000) }
    private val current = AtomicInteger(0) // id of the utterance being recorded, 0 when idle
    private val stopReasons = ConcurrentHashMap<Int, String>()
    private var utterance = 0
    private var thread: Thread? = null
    private val resourceOwner = AtomicReference<Thread?>()

    val active: Boolean get() = current.get() != 0

    /** Main thread only. */
    fun start(reason: String): Int {
        val id = utterance + 1
        if (!current.compareAndSet(0, id)) return current.get()
        utterance = id
        val previous = thread
        thread = Thread({
            // Retain the actual hardware owner even when an intervening restart timed out.
            val previousOwner = resourceOwner.get() ?: previous
            previousOwner?.join(1000)
            if (previousOwner?.isAlive == true) {
                onEvent("ptt_error", JSONObject().put("utterance", id).put("error", "previous microphone is still closing"))
                current.compareAndSet(id, 0)
                stopReasons.remove(id)
                return@Thread
            }
            run(id, reason)
        }, "ptt-recorder").apply { start() }
        return id
    }

    /** Any thread. Only clears the utterance it observed, never a newer one started in between. */
    fun stop(reason: String) {
        val id = current.get()
        if (id == 0) return
        stopReasons[id] = reason
        if (!current.compareAndSet(id, 0)) stopReasons.remove(id)
    }

    private fun run(id: Int, reason: String) {
        var record: AudioRecord? = null
        var channels = 8
        var mask = ROKID_8CH_MASK
        var outcome = ""
        var ownsHardware = false
        try {
            if (current.get() != id) return
            if (!resourceOwner.compareAndSet(null, Thread.currentThread())) {
                onEvent("ptt_error", JSONObject().put("utterance", id).put("error", "microphone resource is still owned"))
                return
            }
            ownsHardware = true
            record = open(mask)
            if (record == null) {
                channels = 1
                mask = AudioFormat.CHANNEL_IN_MONO
                record = open(mask)
            }
            if (record == null) {
                onEvent("ptt_error", JSONObject().put("utterance", id).put("error", "AudioRecord could not be opened"))
                return
            }
            record.startRecording()
            val startedNs = SystemClock.elapsedRealtimeNanos()
            onEvent(
                "ptt_start",
                JSONObject().put("utterance", id).put("reason", reason).put("channels", channels)
                    .put("mask", mask).put("sample_rate", SAMPLE_RATE)
                    .put("recording_state", record.recordingState),
            )
            val frame = ShortArray(FRAME_SAMPLES * channels)
            val chunk = ByteBuffer.allocate(FRAME_SAMPLES * 2 * FRAMES_PER_CHUNK).order(ByteOrder.LITTLE_ENDIAN)
            val rawChunk = rawSink?.let { ByteBuffer.allocate(FRAME_SAMPLES * 2 * FRAMES_PER_CHUNK).order(ByteOrder.LITTLE_ENDIAN) }
            var chunkIndex = 0
            var chunkStartNs = 0L
            var peak = 0
            while (current.get() == id) {
                if (durationExpired(SystemClock.elapsedRealtimeNanos() - startedNs, continuous, maxDurationMs)) {
                    outcome = "max_duration"
                    break
                }
                val read = record.read(frame, 0, frame.size)
                if (read <= 0) {
                    onEvent("ptt_error", JSONObject().put("utterance", id).put("error", "read returned $read"))
                    outcome = "read_error"
                    break
                }
                val samples = read / channels
                if (chunk.position() == 0) {
                    // Approximate capture time of the first sample in this chunk.
                    chunkStartNs = SystemClock.elapsedRealtimeNanos() - samples * 1_000_000_000L / SAMPLE_RATE
                }
                for (i in 0 until samples) {
                    val value = channelSample(frame, i, channels, false)
                    chunk.putShort(value)
                    rawChunk?.putShort(channelSample(frame, i, channels, true))
                    val magnitude = if (value < 0) -value else value.toInt()
                    if (magnitude > peak) peak = magnitude
                }
                if (chunk.remaining() < FRAME_SAMPLES * 2) {
                    send(id, chunkIndex++, chunkStartNs, chunk, rawChunk, channels, peak)
                    peak = 0
                }
            }
            if (chunk.position() > 0) send(id, chunkIndex++, chunkStartNs, chunk, rawChunk, channels, peak)
            onEvent(
                "ptt_stop",
                JSONObject().put("utterance", id)
                    .put("reason", outcome.ifEmpty { stopReasons[id] ?: "stopped" })
                    .put("duration_ms", (SystemClock.elapsedRealtimeNanos() - startedNs) / 1e6)
                    .put("chunks", chunkIndex),
            )
        } catch (e: Exception) {
            onEvent("ptt_error", JSONObject().put("utterance", id).put("error", e.toString()))
        } finally {
            record?.let {
                runCatching { it.stop() }
                it.release()
            }
            stopReasons.remove(id)
            current.compareAndSet(id, 0)
            if (ownsHardware) resourceOwner.compareAndSet(Thread.currentThread(), null)
        }
    }

    private fun send(id: Int, index: Int, startNs: Long, chunk: ByteBuffer, rawChunk: ByteBuffer?, sourceChannels: Int, peak: Int) {
        val bytes = ByteArray(chunk.position())
        chunk.flip()
        chunk.get(bytes)
        chunk.clear()
        val header = JSONObject().put("type", "audio").put("utterance", id).put("chunk", index).put("ts_ns", startNs)
                .put("sample_rate", SAMPLE_RATE).put("channels", 1).put("format", "s16le")
                .put("source_channels", sourceChannels).put("peak", peak)
        sink(header, bytes)
        if (rawChunk != null && rawSink != null) {
            val rawBytes = ByteArray(rawChunk.position())
            rawChunk.flip(); rawChunk.get(rawBytes); rawChunk.clear()
            rawSink.invoke(JSONObject(header.toString()).put("channel_index", if (sourceChannels >= 3) 2 else 0)
                .put("raw_available", sourceChannels >= 3), rawBytes)
        }
    }

    @SuppressLint("MissingPermission")
    private fun open(mask: Int): AudioRecord? = try {
        val format = AudioFormat.Builder()
            .setSampleRate(SAMPLE_RATE)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(mask)
            .build()
        val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, mask, AudioFormat.ENCODING_PCM_16BIT)
        val size = maxOf(minimum, FRAME_SAMPLES * 2 * 8 * 16)
        val r = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.MIC)
            .setAudioFormat(format)
            .setBufferSizeInBytes(size)
            .build()
        if (r.state == AudioRecord.STATE_INITIALIZED) {
            r
        } else {
            r.release()
            null
        }
    } catch (e: Exception) {
        null
    }

    companion object {
        const val ROKID_8CH_MASK = 0x6000FC
        const val SAMPLE_RATE = 16000
        private const val FRAME_SAMPLES = 320 // 20 ms
        private const val FRAMES_PER_CHUNK = 5 // 100 ms
        internal fun durationExpired(elapsedNs: Long, continuous: Boolean, maxDurationMs: Long): Boolean =
            !continuous && elapsedNs >= maxDurationMs * 1_000_000L

        internal fun channelSample(frame: ShortArray, sample: Int, channels: Int, raw: Boolean): Short =
            frame[sample * channels + if (raw && channels >= 3) 2 else 0]
    }
}
