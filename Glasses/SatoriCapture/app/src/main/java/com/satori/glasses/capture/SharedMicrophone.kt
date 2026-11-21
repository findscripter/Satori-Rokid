package com.satori.glasses.capture

import org.json.JSONObject
import java.io.Closeable
import java.util.ArrayDeque
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** One 8-channel hardware recorder, independent bounded subscriptions, mono PCM16LE at 16 kHz.
 * Channel 0 serves ASR; optional channel 2 preserves raw environment audio for a video subscription.
 * Native recording is started/stopped only on first/last subscription. Consumer callbacks never run
 * on the hardware read thread. Backpressure terminates the affected subscription with ptt_error.
 */
class SharedMicrophone internal constructor(
    private val factory: SourceFactory,
    private val callbackCapacity: Int = 8,
) : Closeable {
    interface Source {
        fun start(): Int
        fun stop(reason: String)
    }
    fun interface SourceFactory {
        fun create(pcm: (JSONObject, ByteArray) -> Unit, raw: (JSONObject, ByteArray) -> Unit,
            event: (String, JSONObject) -> Unit): Source
    }

    constructor() : this(SourceFactory { pcm, raw, event ->
        val recorder = VoiceRecorder(pcm, event, continuous = true, rawSink = raw)
        object : Source {
            override fun start() = recorder.start("shared_microphone")
            override fun stop(reason: String) = recorder.stop(reason)
        }
    })

    private val lock = Any()
    private val subscribers = LinkedHashSet<Subscription>()
    // A cancelled consumer keeps its lease until an already running callback actually returns.
    // Repeated subscribe/cancel cannot accumulate stuck callback threads or silently reject work.
    private val callbackLeases = LinkedHashSet<Subscription>()
    private var source: Source? = null
    private var running = false
    private var sourceId = 0
    private var closed = false
    private val starts = AtomicLong()
    private val samples = AtomicLong()
    private val overflows = AtomicLong()
    @Volatile private var channels = 0
    private val callbacks = ThreadPoolExecutor(4, 4, 30, TimeUnit.SECONDS, ArrayBlockingQueue<Runnable>(4),
        { task -> Thread(task, "shared-microphone-callback").apply { isDaemon = true } }, ThreadPoolExecutor.AbortPolicy())
        .apply { allowCoreThreadTimeOut(true) }

    init { require(callbackCapacity in 1..64) }
    val sourceStarts: Long get() = starts.get()
    val subscriberCount: Int get() = synchronized(lock) { subscribers.size }
    val pcmSamples: Long get() = samples.get()

    fun stats(): JSONObject = JSONObject().put("source_starts", sourceStarts).put("subscriber_count", subscriberCount)
        .put("pcm_samples", pcmSamples).put("source_channels", channels).put("raw_available", channels >= 3)
        .put("audio_mode", when { channels >= 3 -> "rokid_8ch_ch0_ch2"; channels == 1 -> "mono_fallback"; else -> "starting" })
        .put("callback_overflows", overflows.get())

    fun subscribe(onPcm: (timestampNs: Long, ByteArray) -> Unit, onEvent: (String, JSONObject) -> Unit,
        raw: Boolean = false): Closeable {
        val subscription = Subscription(raw, onPcm, onEvent)
        synchronized(lock) {
            check(!closed) { "shared microphone is closed" }
            check(callbackLeases.size < 4) { "shared microphone callbacks are busy; retry after prior subscriptions finish" }
            callbackLeases.add(subscription)
            subscribers.add(subscription)
            if (!running) {
                try {
                    val hardware = source ?: factory.create({ header, bytes -> pcm(header, bytes, false) },
                        { header, bytes -> pcm(header, bytes, true) }, ::sourceEvent).also { source = it }
                    running = true
                    starts.incrementAndGet()
                    sourceId = hardware.start()
                    check(sourceId > 0)
                } catch (_: Exception) {
                    failAll("start_failed")
                }
            }
        }
        return subscription
    }

    private fun pcm(header: JSONObject, bytes: ByteArray, raw: Boolean) {
        val targets = synchronized(lock) {
            if (!running || header.optInt("utterance") != sourceId) return
            if (bytes.isEmpty() || bytes.size % 2 != 0 || bytes.size > 6400 || header.optLong("ts_ns", -1) < 0) {
                failAll("invalid_pcm"); return
            }
            if (!raw) { channels = header.optInt("source_channels", 1); samples.addAndGet(bytes.size / 2L) }
            subscribers.filter { it.raw == raw }
        }
        val timestamp = header.optLong("ts_ns")
        targets.forEach { it.offer(Delivery.Pcm(timestamp, bytes.copyOf())) }
    }

    private fun sourceEvent(name: String, data: JSONObject) {
        synchronized(lock) {
            if (!running || data.optInt("utterance") != sourceId) return
            if (name == "ptt_error" || name == "ptt_stop") { failAll(if (name == "ptt_error") "source_error" else "source_stopped"); return }
            subscribers.toList().forEach { it.offer(Delivery.Event(name, JSONObject(data.toString()))) }
        }
    }

    private fun failAll(reason: String) {
        val targets = subscribers.toList()
        targets.forEach { it.fail(reason) }
        stopIfEmpty()
    }

    private fun detach(subscription: Subscription) {
        synchronized(lock) { subscribers.remove(subscription); stopIfEmpty() }
    }

    private fun stopIfEmpty() {
        if (subscribers.isEmpty() && running) {
            running = false; sourceId = 0
            runCatching { source?.stop("no_subscribers") }
        }
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            callbackLeases.toList().forEach { it.close() }
            running = false; sourceId = 0
        }
        callbacks.shutdownNow()
    }

    private sealed class Delivery {
        class Pcm(val timestamp: Long, val bytes: ByteArray) : Delivery()
        class Event(val name: String, val data: JSONObject) : Delivery()
    }

    private inner class Subscription(val raw: Boolean, val onPcm: (Long, ByteArray) -> Unit,
        val onEvent: (String, JSONObject) -> Unit) : Closeable {
        private val state = Any()
        private val queue = ArrayDeque<Delivery>()
        private var active = true
        private var draining = false
        private var terminal: Delivery.Event? = null
        private val drain = Runnable {
            while (true) {
                val item = synchronized(state) {
                    if (terminal != null) terminal.also { terminal = null }
                    else if (active && queue.isNotEmpty()) queue.removeFirst()
                    else { draining = false; null }
                }
                if (item == null) { releaseLeaseIfIdle(); return@Runnable }
                try {
                    when (item) {
                        is Delivery.Pcm -> onPcm(item.timestamp, item.bytes)
                        is Delivery.Event -> onEvent(item.name, item.data)
                    }
                } catch (_: Exception) { fail("callback_failed") }
            }
        }

        fun offer(delivery: Delivery) {
            var overflow = false
            var rejected = false
            synchronized(state) {
                if (!active) return
                if (queue.size >= callbackCapacity) overflow = true else { queue.addLast(delivery); rejected = !schedule() }
            }
            if (overflow) { overflows.incrementAndGet(); fail("backpressure") }
            if (rejected) { detach(this); releaseLeaseIfIdle() }
        }

        private fun schedule(): Boolean {
            if (draining) return true
            draining = true
            return try { callbacks.execute(drain); true }
            catch (_: RejectedExecutionException) {
                draining = false
                // No extra thread is spawned; the source and other consumers remain bounded.
                active = false; queue.clear()
                false
            }
        }

        fun fail(reason: String) {
            synchronized(state) {
                if (!active) return
                active = false; queue.clear()
                terminal = Delivery.Event("ptt_error", JSONObject().put("reason", reason)
                    .put("error", "共享麦克风订阅已停止：$reason"))
                schedule()
            }
            detach(this)
        }

        override fun close() {
            synchronized(state) { active = false; queue.clear(); terminal = null }
            if (callbacks.remove(drain)) synchronized(state) { draining = false }
            detach(this)
            releaseLeaseIfIdle()
        }

        private fun releaseLeaseIfIdle() {
            val release = synchronized(state) { !active && !draining }
            if (release) synchronized(lock) { callbackLeases.remove(this) }
        }
    }
}
