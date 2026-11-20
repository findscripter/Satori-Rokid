package com.satori.glasses.capture

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.SystemClock
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/** Plays PCM16 WAV prompts pushed by the PC on the glasses speaker, one at a time, and reports each outcome. */
class PromptPlayer(context: Context, private val onEvent: (String, JSONObject) -> Unit) {
    private val audio = context.getSystemService(AudioManager::class.java)
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "prompt-player") }
    private val generation = AtomicInteger()
    private val pending = AtomicInteger()
    val busy: Boolean get() = pending.get() > 0
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANT)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    @Volatile private var track: AudioTrack? = null
    @Volatile var playing = false
        private set

    /** `immediate` cancels the current and queued prompts first; otherwise prompts play in arrival order. */
    fun play(id: String, wav: ByteArray, immediate: Boolean) {
        if (immediate) stop()
        val ticket = generation.get()
        val queuedNs = SystemClock.elapsedRealtimeNanos()
        pending.incrementAndGet()
        try { executor.execute { try { playBlocking(id, wav, ticket, queuedNs) } finally { pending.decrementAndGet() } } }
        catch (failure: java.util.concurrent.RejectedExecutionException) { pending.decrementAndGet(); throw failure }
    }

    /** Stops the current prompt and drops queued ones. */
    fun stop() {
        generation.incrementAndGet()
        track?.let {
            runCatching {
                it.pause()
                it.flush()
            }
        }
    }

    fun shutdown() {
        stop()
        executor.shutdownNow()
    }

    private fun playBlocking(id: String, wav: ByteArray, ticket: Int, queuedNs: Long) {
        if (ticket != generation.get()) {
            onEvent("audio_skipped", JSONObject().put("id", id))
            return
        }
        val pcm = try {
            WavData.parse(wav)
        } catch (e: Exception) {
            onEvent("audio_error", JSONObject().put("id", id).put("error", e.message ?: e.toString()))
            return
        }
        val focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(attributes)
            .build()
        val focusGranted = audio.requestAudioFocus(focus) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        val channelMask = if (pcm.channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
        val frameBytes = 2 * pcm.channels
        val totalFrames = pcm.length / frameBytes
        val minBuffer = AudioTrack.getMinBufferSize(pcm.sampleRate, channelMask, AudioFormat.ENCODING_PCM_16BIT)
        var outcome = "audio_done"
        var error = ""
        val startedNs = SystemClock.elapsedRealtimeNanos()
        var created: AudioTrack? = null
        try {
            val t = AudioTrack.Builder()
                .setAudioAttributes(attributes)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(pcm.sampleRate)
                        .setChannelMask(channelMask)
                        .build(),
                )
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(maxOf(minBuffer, pcm.sampleRate * frameBytes / 5))
                .build()
            created = t
            track = t
            playing = true
            t.play()
            onEvent(
                "audio_started",
                JSONObject().put("id", id).put("queue_ms", (startedNs - queuedNs) / 1e6).put("focus", focusGranted)
                    .put("sample_rate", pcm.sampleRate).put("channels", pcm.channels)
                    .put("audio_ms", totalFrames * 1000.0 / pcm.sampleRate),
            )
            var offset = pcm.offset
            val end = pcm.offset + totalFrames * frameBytes
            while (offset < end) {
                if (ticket != generation.get()) {
                    outcome = "audio_stopped"
                    break
                }
                val written = t.write(wav, offset, minOf(8192, end - offset))
                if (written < 0) {
                    outcome = "audio_error"
                    error = "AudioTrack.write returned $written"
                    break
                }
                offset += written
            }
            if (outcome == "audio_done") {
                val deadline = SystemClock.elapsedRealtime() + totalFrames * 1000L / pcm.sampleRate + 2000L
                while (t.playbackHeadPosition < totalFrames && ticket == generation.get() &&
                    SystemClock.elapsedRealtime() < deadline
                ) {
                    Thread.sleep(15)
                }
                outcome = when {
                    ticket != generation.get() -> "audio_stopped"
                    t.playbackHeadPosition < totalFrames -> "audio_incomplete" // deadline hit before the end
                    else -> "audio_done"
                }
            }
        } catch (e: Exception) {
            outcome = "audio_error"
            error = e.toString()
        } finally {
            created?.let {
                runCatching { it.stop() }
                it.release()
            }
            track = null
            playing = false
            audio.abandonAudioFocusRequest(focus)
        }
        val result = JSONObject().put("id", id).put("play_ms", (SystemClock.elapsedRealtimeNanos() - startedNs) / 1e6)
            .put("frames", totalFrames)
        if (error.isNotEmpty()) result.put("error", error)
        onEvent(outcome, result)
    }
}

/** Location and format of the PCM samples inside a RIFF/WAVE byte array. */
data class WavData(val sampleRate: Int, val channels: Int, val offset: Int, val length: Int) {
    companion object {
        fun parse(bytes: ByteArray): WavData {
            require(bytes.size >= 44) { "WAV too short" }
            require(ascii(bytes, 0) == "RIFF" && ascii(bytes, 8) == "WAVE") { "not a RIFF/WAVE file" }
            val view = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            var format = -1
            var channels = 0
            var rate = 0
            var bits = 0
            var position = 12
            while (position + 8 <= bytes.size) {
                val id = ascii(bytes, position)
                val size = view.getInt(position + 4)
                val body = position + 8
                if (id == "data") {
                    val length = if (size < 0 || body + size > bytes.size) bytes.size - body else size
                    require(format == 1 || format == 0xFFFE) { "only PCM WAV is supported (format $format)" }
                    require(bits == 16) { "only 16-bit samples are supported ($bits)" }
                    require(channels == 1 || channels == 2) { "only mono or stereo is supported ($channels)" }
                    require(rate in 8000..48000) { "unsupported sample rate $rate" }
                    return WavData(rate, channels, body, length - length % (2 * channels))
                }
                require(size >= 0 && body + size <= bytes.size) { "truncated chunk $id" }
                if (id == "fmt ") {
                    require(size >= 16) { "fmt chunk too short" }
                    format = view.getShort(body).toInt() and 0xFFFF
                    channels = view.getShort(body + 2).toInt() and 0xFFFF
                    rate = view.getInt(body + 4)
                    bits = view.getShort(body + 14).toInt() and 0xFFFF
                }
                position = body + size + (size and 1)
            }
            throw IllegalArgumentException("no data chunk")
        }

        private fun ascii(bytes: ByteArray, offset: Int) = String(bytes, offset, 4, StandardCharsets.US_ASCII)
    }
}
