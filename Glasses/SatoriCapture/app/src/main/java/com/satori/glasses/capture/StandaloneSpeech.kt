package com.satori.glasses.capture

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import org.json.JSONObject
import java.io.Closeable
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Future
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** Chinese speech through the installed Android engine or Rokid local service, then optional cloud WAV fallback. */
class StandaloneSpeech(
    context: Context,
    private val api: StandaloneApi,
    private val onEvent: (String, JSONObject) -> Unit,
) : Closeable {
    private val main = Handler(Looper.getMainLooper())
    private val worker = ThreadPoolExecutor(1, 1, 30, TimeUnit.SECONDS, ArrayBlockingQueue<Runnable>(8),
        { task -> Thread(task, "standalone-speech").apply { isDaemon = true } }, ThreadPoolExecutor.AbortPolicy())
    private val generation = AtomicInteger()
    private val sequence = AtomicInteger()
    private val pendingDispatch = AtomicInteger()
    val busy: Boolean get() = pendingDispatch.get() > 0 || queued.isNotEmpty() || localJobs.isNotEmpty() ||
        synthesizing.any { !it.isDone } || player.busy
    private val vendor = StandaloneVendorSpeech(context.applicationContext) { name, data ->
        val ticket = data.optString("id").split('-').getOrNull(1)?.toIntOrNull()
        main.post { if (!closed && ticket == generation.get()) onEvent(name, data) }
    }
    private val queued = ArrayDeque<Pair<String, Int>>()
    private val localJobs = ConcurrentHashMap<String, Pair<String, Int>>()
    private val player = PromptPlayer(context.applicationContext) { name, data ->
        val ticket = data.optString("id").split('-').getOrNull(1)?.toIntOrNull()
        main.post { if (!closed && ticket == generation.get()) onEvent(name, data) }
    }
    private var tts: TextToSpeech? = null
    private var initialized = false
    private var local = false
    @Volatile private var closed = false
    private val synthesizing = ConcurrentHashMap.newKeySet<Future<*>>()

    init {
        main.post {
            if (!closed) {
                try {
                    tts = TextToSpeech(context.applicationContext) { status -> main.post { finishInit(status) } }
                    main.postDelayed({ if (!initialized && !closed) finishInit(TextToSpeech.ERROR) }, 3000)
                } catch (_: Exception) { finishInit(TextToSpeech.ERROR) }
            }
        }
    }

    fun speak(text: String, immediate: Boolean = false) {
        if (text.isBlank() || closed) return
        if (immediate) stop()
        val ticket = generation.get()
        pendingDispatch.incrementAndGet()
        main.post {
            try {
                if (closed || ticket != generation.get()) return@post
                if (!initialized) {
                    if (queued.size >= 8) queued.removeFirst()
                    queued.addLast(text.take(2000) to ticket)
                } else dispatch(text.take(2000), ticket)
            } finally { pendingDispatch.decrementAndGet() }
        }
    }

    /** Cancels queued and in-flight synthesis, local utterances and playback; never cancels visual requests. */
    fun stop() {
        generation.incrementAndGet()
        vendor.stop()
        synthesizing.forEach { it.cancel(true) }
        synthesizing.clear()
        worker.purge()
        player.stop()
        main.post { queued.clear(); localJobs.clear(); tts?.stop() }
    }

    override fun close() {
        if (closed) return
        closed = true
        stop()
        worker.shutdownNow()
        vendor.close()
        player.shutdown()
        main.post { tts?.shutdown(); tts = null }
    }

    private fun finishInit(status: Int) {
        if (closed || initialized) return
        initialized = true
        val engine = tts
        local = status == TextToSpeech.SUCCESS && engine != null && runCatching {
            val installed = engine.voices?.firstOrNull { it.locale.language == "zh" && !it.isNetworkConnectionRequired }
            installed != null && engine.setVoice(installed) == TextToSpeech.SUCCESS
        }.getOrDefault(false)
        if (local) {
            engine?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String) {
                    val job = localJobs[utteranceId] ?: return
                    if (job.second == generation.get()) emit("audio_started", JSONObject().put("id", utteranceId).put("source", "local_tts"), job.second)
                }
                override fun onDone(utteranceId: String) {
                    val job = localJobs.remove(utteranceId) ?: return
                    if (job.second == generation.get()) emit("audio_done", JSONObject().put("id", utteranceId).put("source", "local_tts"), job.second)
                }
                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String) {
                    val job = localJobs.remove(utteranceId) ?: return
                    main.post { if (!closed && job.second == generation.get()) cloud(job.first, job.second) }
                }
            })
        } else { engine?.shutdown(); tts = null }
        emit("speech_ready", JSONObject().put("source", if (local) "local_tts" else "rokid_local_with_cloud_fallback"))
        while (queued.isNotEmpty()) {
            val (text, ticket) = queued.removeFirst()
            if (ticket == generation.get()) dispatch(text, ticket)
        }
    }

    private fun dispatch(text: String, ticket: Int) {
        if (closed || ticket != generation.get()) return
        if (local) {
            val id = "local-${sequence.incrementAndGet()}"
            emit("speech_requested", JSONObject().put("id", id).put("text", text).put("provider", "android_local"), ticket)
            localJobs[id] = text to ticket
            if (tts?.speak(text, TextToSpeech.QUEUE_ADD, null, id) != TextToSpeech.SUCCESS) {
                localJobs.remove(id)
                cloud(text, ticket)
            }
        } else cloud(text, ticket)
    }

    private fun cloud(text: String, ticket: Int) {
        if (closed || ticket != generation.get()) return
        synthesizing.removeIf { it.isDone }
        try {
            val future = worker.submit {
                if (closed || ticket != generation.get()) return@submit
                try {
                    val id = "vendor-$ticket-${sequence.incrementAndGet()}"
                    emit("speech_requested", JSONObject().put("id", id).put("text", text).put("provider", "rokid_local"), ticket)
                    if (vendor.speakBlocking(text, id) { !closed && ticket == generation.get() }) return@submit
                    if (closed || ticket != generation.get() || Thread.currentThread().isInterrupted) return@submit
                    val wav = api.synthesize(text)
                    pendingDispatch.incrementAndGet()
                    main.post {
                        try { if (!closed && ticket == generation.get()) {
                                val cloudId = "cloud-$ticket-${sequence.incrementAndGet()}"
                                onEvent("speech_requested", JSONObject().put("id", cloudId).put("text", text).put("provider", "cloud"))
                                player.play(cloudId, wav, false)
                            }
                        } finally { pendingDispatch.decrementAndGet() }
                    }
                } catch (e: Exception) {
                    if (!closed && ticket == generation.get()) emit("audio_error", JSONObject()
                        .put("error", if (e is ApiFailure) e.message else "语音合成失败，请检查语音模型配置"), ticket)
                }
            }
            synthesizing.add(future)
            if (closed || ticket != generation.get()) future.cancel(true)
        } catch (_: java.util.concurrent.RejectedExecutionException) {
            emit("audio_error", JSONObject().put("error", "语音队列已满，请稍后重试"), ticket)
        }
    }

    private fun emit(name: String, data: JSONObject, ticket: Int = generation.get()) {
        main.post { if (!closed && ticket == generation.get()) onEvent(name, data) }
    }
}
