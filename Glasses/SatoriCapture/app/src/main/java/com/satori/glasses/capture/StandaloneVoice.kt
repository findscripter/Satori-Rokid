package com.satori.glasses.capture

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import org.vosk.Model
import org.vosk.Recognizer

/** Continuous speech stays in bundled local Vosk; manual input can fall back to system/cloud ASR. */
class StandaloneVoice(
    private val context: Context,
    private val api: StandaloneApi,
    private val onText: (String) -> Unit,
    private val onError: (String) -> Unit,
    private val onRecognition: (JSONObject) -> Unit = {},
    private val sharedMicrophone: SharedMicrophone? = null,
) : Closeable {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { Thread(it, "standalone-asr").apply { isDaemon = true } }
    private val generation = AtomicInteger()
    @Volatile var active: Boolean = false
        private set
    /** Manual recording/recognition blocks visual requests; continuous listening does not. */
    @Volatile var busy: Boolean = false
        private set
    @Volatile var continuous: Boolean = false
        private set
    @Volatile var automaticUnavailable: Boolean = false
        private set
    private val queuedAudio = AtomicInteger()
    @Volatile private var closed = false
    @Volatile private var recorder: VoiceRecorder? = null
    @Volatile private var transcribing: Future<*>? = null
    private var recognizer: SpeechRecognizer? = null
    private val sharedCapture = AtomicReference<SharedCapture?>()
    val listening: Boolean get() = continuous && active && sharedCapture.get()?.subscription?.get() != null
    private class SharedCapture(val ticket: Int) {
        val finished = AtomicBoolean()
        val subscription = AtomicReference<Closeable?>()
        var timeout: Runnable? = null
        var onStop: ((String) -> Unit)? = null
    }
    // Native objects are used/closed on one worker, never concurrently with acceptWaveForm.
    private var offlineModel: Model? = null
    private var offlineAttempted = false
    private var offlineSession: OfflineSession? = null
    private class OfflineSession(val ticket: Int, val recognizer: Recognizer) {
        val segments = ArrayList<String>()
        val rawSegments = ArrayList<String>()
        val selections = ArrayList<JSONObject>()
        var failed = false
        var sampleBytes = 0
    }

    fun start() = startMode(false)

    fun startContinuous() {
        if (!automaticUnavailable) startMode(true)
    }

    private fun startMode(automatic: Boolean) {
        if (closed || active) return
        cancel()
        val ticket = generation.incrementAndGet()
        active = true
        continuous = automatic
        busy = !automatic
        enqueue {
            if (!valid(ticket)) return@enqueue
            try {
                if (!offlineAttempted) {
                    offlineAttempted = true
                    offlineModel = Model(StandaloneOfflineModel.prepare(context).absolutePath)
                }
                if (!valid(ticket)) return@enqueue
                val model = offlineModel
                if (model != null) {
                    if (!active) { report(ticket, "录音太短，请重新按住说话"); return@enqueue }
                    closeOfflineSession()
                    offlineSession = OfflineSession(ticket, newOfflineRecognizer(model))
                    main.post {
                        if (valid(ticket)) {
                            if (active) {
                                if (automatic) startContinuousRecorder(ticket) else startRecorder(ticket, true)
                            } else {
                                enqueue { if (offlineSession?.ticket == ticket) closeOfflineSession() }
                                report(ticket, "录音太短，请重新按住说话")
                            }
                        }
                    }
                    return@enqueue
                }
            } catch (_: Exception) { closeOfflineSession() }
            catch (_: LinkageError) { closeOfflineSession() }
            main.post {
                if (!valid(ticket)) return@post
                if (automatic) {
                    automaticUnavailable = true
                    active = false
                    report(ticket, "本地持续语音不可用，请重新打开助手")
                    return@post
                }
                if (!active) { report(ticket, "录音太短，请重新按住说话"); return@post }
                onError("离线语音模型不可用，改用系统或云端识别")
                if (sharedMicrophone == null && SpeechRecognizer.isRecognitionAvailable(context)) startRecognizer(ticket)
                else startRecorder(ticket, false)
            }
        }
    }

    /** Release the push-to-talk key. The captured utterance is now eligible for recognition/upload. */
    fun stop() {
        if (!active) return
        val ticket = generation.get()
        active = false
        busy = true
        main.post { if (valid(ticket)) {
            recognizer?.stopListening()
            if (!finishShared(ticket, "user_released", true)) recorder?.stop("user_released")
        } }
    }

    fun cancel() {
        val oldTicket = generation.get()
        generation.incrementAndGet()
        active = false
        busy = false
        continuous = false
        finishShared(oldTicket, "cancelled", false)
        recorder?.stop("cancelled")
        recorder = null
        transcribing?.cancel(true)
        main.post { recognizer?.cancel(); recognizer?.destroy(); recognizer = null }
        enqueue { if (offlineSession?.ticket == oldTicket) closeOfflineSession() }
    }

    override fun close() {
        if (closed) return
        closed = true
        cancel()
        // Native cleanup is ordered after the current native operation; never free its pointers on the main thread.
        enqueue { closeOfflineSession(); offlineModel?.close(); offlineModel = null }
        worker.shutdown()
    }

    private fun valid(ticket: Int) = !closed && ticket == generation.get()

    private fun startRecognizer(ticket: Int) {
        try {
            val current = SpeechRecognizer.createSpeechRecognizer(context.applicationContext)
            recognizer = current
            current.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) = Unit
                override fun onBeginningOfSpeech() = Unit
                override fun onRmsChanged(rmsdB: Float) = Unit
                override fun onBufferReceived(buffer: ByteArray?) = Unit
                override fun onEndOfSpeech() { if (valid(ticket)) active = false }
                override fun onError(error: Int) {
                    if (!valid(ticket)) return
                    active = false
                    releaseRecognizer(current)
                    report(ticket, "语音识别失败（$error），请重新按住说话")
                }
                override fun onResults(results: Bundle?) {
                    if (!valid(ticket)) return
                    active = false
                    releaseRecognizer(current)
                    val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()
                    if (text.isNotBlank() && text.length <= 2000) deliver(ticket, text) else report(ticket, "没有听清，请重新按住说话")
                }
                override fun onPartialResults(partialResults: Bundle?) = Unit
                override fun onEvent(eventType: Int, params: Bundle?) = Unit
            })
            current.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN").putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false))
            main.postDelayed({
                if (valid(ticket) && recognizer === current) {
                    active = false
                    current.cancel(); releaseRecognizer(current)
                    report(ticket, "语音识别超时，请重试")
                }
            }, 30000)
        } catch (_: Exception) {
            recognizer?.destroy(); recognizer = null
            if (valid(ticket)) startRecorder(ticket, false)
        }
    }

    private fun releaseRecognizer(current: SpeechRecognizer) {
        current.destroy()
        if (recognizer === current) recognizer = null
    }

    private fun startRecorder(ticket: Int, offline: Boolean) {
        val pcm = ByteArrayOutputStream()
        val lock = Any()
        var failed = false
        val sink: (JSONObject, ByteArray) -> Unit = { _: JSONObject, bytes: ByteArray ->
                var reachedLimit = false
                synchronized(lock) {
                    if (valid(ticket) && !failed) {
                        val accepted = if (sharedMicrophone != null && pcm.size() + bytes.size > StandaloneApi.MAX_PCM_BYTES)
                            bytes.copyOf((StandaloneApi.MAX_PCM_BYTES - pcm.size()).coerceAtLeast(0)) else bytes
                        if (pcm.size() + accepted.size <= StandaloneApi.MAX_PCM_BYTES) pcm.write(accepted)
                        else failed = true
                        if (offline && !failed && accepted.isNotEmpty()) enqueue {
                            val session = offlineSession
                            if (valid(ticket) && session?.ticket == ticket && !session.failed) {
                                try {
                                    if (session.recognizer.acceptWaveForm(accepted, accepted.size))
                                        appendOffline(session, session.recognizer.result)
                                } catch (_: Exception) { session.failed = true }
                                catch (_: LinkageError) { session.failed = true }
                            }
                        }
                        reachedLimit = sharedMicrophone != null && !failed && pcm.size() >= StandaloneApi.MAX_PCM_BYTES
                    }
                }
                if (reachedLimit) { active = false; finishShared(ticket, "max_duration", true) }
            }
        val events: (String, JSONObject) -> Unit = { name, _ ->
                when (name) {
                    "ptt_error" -> {
                        synchronized(lock) { failed = true }
                        finishShared(ticket, "source_error", false)
                        if (offline) enqueue { if (offlineSession?.ticket == ticket) closeOfflineSession() }
                        if (valid(ticket)) { active = false; report(ticket, "无法打开麦克风，请检查权限或其他应用是否占用") }
                    }
                    "ptt_stop" -> {
                        if (valid(ticket)) {
                            active = false
                            val bytes = synchronized(lock) { if (failed) null else pcm.toByteArray() }
                            if (bytes == null || bytes.size < 3200) {
                                if (offline) enqueue { if (offlineSession?.ticket == ticket) closeOfflineSession() }
                                report(ticket, "录音太短或录音失败，请重新按住说话")
                            }
                            else if (offline) finishOffline(ticket) else upload(ticket, bytes)
                        }
                    }
                }
            }
        val current = VoiceRecorder(sink, events)
        recorder = current
        if (valid(ticket) && active) {
            if (sharedMicrophone != null) startShared(ticket, sink, events) else current.start("standalone_user")
        }
    }

    private fun startContinuousRecorder(ticket: Int) {
        val sink: (JSONObject, ByteArray) -> Unit = { _, bytes ->
            if (valid(ticket)) {
                if (queuedAudio.incrementAndGet() > 64) {
                    queuedAudio.decrementAndGet()
                    failContinuous(ticket, "语音处理跟不上，稍后自动重试")
                } else try {
                    worker.execute {
                        try {
                            val session = offlineSession
                            if (valid(ticket) && session?.ticket == ticket) {
                                session.sampleBytes += bytes.size
                                if (session.recognizer.acceptWaveForm(bytes, bytes.size)) {
                                    appendOffline(session, session.recognizer.result)
                                    deliverOffline(session, true)
                                    session.sampleBytes = 0
                                } else if (session.sampleBytes >= StandaloneApi.MAX_PCM_BYTES) {
                                    // Bound uninterrupted speech/silence without retaining a growing PCM buffer.
                                    appendOffline(session, session.recognizer.finalResult)
                                    deliverOffline(session, true)
                                    closeOfflineSession()
                                    offlineSession = OfflineSession(ticket, newOfflineRecognizer(checkNotNull(offlineModel)))
                                }
                            }
                        } catch (_: Exception) { failContinuous(ticket, "本地语音识别失败，稍后自动重试") }
                        catch (_: LinkageError) { failContinuous(ticket, "本地语音组件异常，请重新打开助手") }
                        finally { queuedAudio.decrementAndGet() }
                    }
                } catch (_: java.util.concurrent.RejectedExecutionException) { queuedAudio.decrementAndGet() }
            }
        }
        val events: (String, JSONObject) -> Unit = { name, _ ->
            if (valid(ticket)) when (name) {
                "ptt_error" -> failContinuous(ticket, "麦克风暂时不可用，稍后自动重试")
                "ptt_stop" -> { active = false; finishOffline(ticket) }
            }
        }
        recorder = VoiceRecorder(sink, events, continuous = true)
        if (sharedMicrophone != null) startShared(ticket, sink, events, automatic = true)
        else recorder?.start("continuous_local")
    }

    private fun failContinuous(ticket: Int, message: String) {
        main.post { if (valid(ticket)) { cancel(); onError(message) } }
    }

    private fun startShared(ticket: Int, sink: (JSONObject, ByteArray) -> Unit, events: (String, JSONObject) -> Unit,
                            automatic: Boolean = false) {
        val microphone = sharedMicrophone ?: return
        val capture = SharedCapture(ticket)
        val started = SystemClock.elapsedRealtime()
        capture.onStop = { reason -> events("ptt_stop", JSONObject().put("reason", reason)
            .put("duration_ms", SystemClock.elapsedRealtime() - started).put("shared_microphone", true)) }
        sharedCapture.set(capture)
        try {
            val subscription = microphone.subscribe({ timestamp, bytes ->
                sink(JSONObject().put("type", "audio").put("ts_ns", timestamp).put("sample_rate", 16000)
                    .put("channels", 1).put("format", "s16le"), bytes)
            }, { name, data -> if (name == "ptt_error" && valid(ticket)) events(name, data) })
            capture.subscription.set(subscription)
            if (capture.finished.get()) capture.subscription.getAndSet(null)?.close()
            val timeout = Runnable { if (valid(ticket)) { active = false; finishShared(ticket, "max_duration", true) } }
            capture.timeout = timeout
            if (!automatic && !capture.finished.get()) main.postDelayed(timeout, 20_000)
        } catch (_: Exception) {
            finishShared(ticket, "subscribe_failed", false)
            events("ptt_error", JSONObject().put("error", "shared microphone could not be subscribed"))
        }
    }

    /** Finalizes only this PTT subscription; a video subscription retains the same hardware source. */
    private fun finishShared(ticket: Int, reason: String, finalize: Boolean): Boolean {
        val capture = sharedCapture.get() ?: return false
        if (capture.ticket != ticket || !sharedCapture.compareAndSet(capture, null)) return false
        capture.finished.set(true)
        capture.timeout?.let { main.removeCallbacks(it) }
        capture.subscription.getAndSet(null)?.close()
        if (finalize && valid(ticket)) capture.onStop?.invoke(reason)
        return true
    }

    private fun upload(ticket: Int, pcm: ByteArray) {
        if (!valid(ticket)) return
        transcribing = worker.submit {
            if (!valid(ticket)) return@submit
            try {
                val text = api.transcribe(StandaloneApi.wavFromPcm(pcm))
                deliver(ticket, text)
            } catch (e: Exception) {
                report(ticket, if (e is ApiFailure) e.message ?: "语音识别失败" else "语音识别失败，请检查语音模型配置")
            }
        }
    }

    private fun deliver(ticket: Int, text: String, audit: JSONObject? = null) {
        main.post {
            if (!valid(ticket)) return@post
            busy = false
            // Both callbacks are delivered in order on this same valid main turn. The service may
            // retain this trace until onText creates the initial run directory.
            if (audit != null) runCatching { onRecognition(audit) }
            if (valid(ticket)) onText(text)
        }
    }

    private fun report(ticket: Int, message: String) {
        main.post {
            if (!valid(ticket)) return@post
            try { onError(message) } finally { if (valid(ticket)) busy = false }
        }
    }

    private fun finishOffline(ticket: Int) {
        enqueue {
            if (!valid(ticket)) return@enqueue
            val session = offlineSession
            if (session?.ticket != ticket) { report(ticket, "离线识别已结束，请重试"); return@enqueue }
            try {
                if (session.failed) throw IllegalStateException()
                appendOffline(session, session.recognizer.finalResult)
                deliverOffline(session, continuous)
            } catch (_: Exception) { report(ticket, "离线语音识别失败，请重新说话或输入文字目标") }
            catch (_: LinkageError) { report(ticket, "离线语音组件不可用，请重新启动助手") }
            finally { closeOfflineSession() }
        }
    }

    private fun deliverOffline(session: OfflineSession, automatic: Boolean) {
        val text = normalizeOfflineText(session.segments.joinToString(" "))
        val raw = normalizeOfflineText(session.rawSegments.joinToString(" "))
        val audit = JSONObject().put("provider", "vosk_offline").put("raw_top1", raw).put("selected", text)
            .put("automatic", automatic)
            .put("reason", if (session.selections.any { it.optString("reason") == "goal_prefix_candidate_within_score_gap" })
                "goal_prefix_candidate_within_score_gap" else "top1_kept")
            .put("candidate_count", session.selections.sumOf { it.optInt("candidate_count") })
            .put("candidate_count_scope", "sum_of_nonempty_segment_candidates")
            .put("max_goal_prefix_score_gap", 2.0).put("segments", org.json.JSONArray(session.selections))
        session.segments.clear(); session.rawSegments.clear(); session.selections.clear()
        if (text.isNotBlank() && text.length <= 2000) deliver(session.ticket, text, audit)
        else if (!automatic) report(session.ticket, "没有听清，请重新说话")
        else main.post { if (valid(session.ticket)) busy = false }
    }

    private fun appendOffline(session: OfflineSession, result: String) {
        val selection = selectOfflineCandidate(JSONObject(result))
        val text = selection.getString("selected")
        if (text.isNotEmpty() && session.segments.sumOf { it.length } + text.length <= 2000) {
            session.segments += text
            session.rawSegments += selection.getString("raw_top1")
            session.selections += selection
        }
    }

    /** Production and the PCM-fixture test share exactly this decoder configuration. */
    private fun newOfflineRecognizer(model: Model): Recognizer = Recognizer(model, 16000f).apply {
        setMaxAlternatives(5)
    }

    private fun closeOfflineSession() {
        runCatching { offlineSession?.recognizer?.close() }
        offlineSession = null
    }

    private fun enqueue(task: () -> Unit) {
        try { worker.execute(task) } catch (_: java.util.concurrent.RejectedExecutionException) { }
    }

    companion object {
        internal fun selectOfflineCandidate(result: JSONObject): JSONObject {
            data class Candidate(val text: String, val score: Double?)
            val alternatives = result.optJSONArray("alternatives")
            val choices = if (alternatives != null) (0 until alternatives.length()).mapNotNull { index ->
                alternatives.optJSONObject(index)?.let { item ->
                    val text = item.opt("text") as? String ?: return@let null
                    val score = (item.opt("confidence") as? Number)?.toDouble()?.takeIf { it.isFinite() }
                    Candidate(normalizeOfflineText(text), score)
                }
            } else listOf(Candidate(normalizeOfflineText(result.opt("text") as? String ?: ""), null))
            val best = choices.firstOrNull() ?: Candidate("", null)
            var selected = best
            val reason = when {
                best.text.isBlank() -> "empty_result"
                supportedCommand(best.text) -> "best_is_supported_command"
                alternatives == null || choices.size <= 1 -> "single_result"
                best.score == null -> "missing_score_keep_top1"
                else -> {
                    val eligible = choices.drop(1).filter { candidate ->
                        goalPrefix(candidate.text) && candidate.score != null &&
                            kotlin.math.abs(best.score - candidate.score) <= 2.0
                    }.maxByOrNull { it.score!! }
                    if (eligible == null) "no_eligible_goal_candidate" else {
                        selected = eligible
                        "goal_prefix_candidate_within_score_gap"
                    }
                }
            }
            return JSONObject().put("raw_top1", best.text).put("selected", selected.text).put("reason", reason)
                .put("candidate_count", choices.count { it.text.isNotBlank() })
                .put("raw_top1_score", best.score ?: JSONObject.NULL).put("selected_score", selected.score ?: JSONObject.NULL)
                .put("score_kind", "vosk_raw_score_not_probability")
        }

        // Match only goal prefixes/control phrases already understood by StandaloneService.onText.
        private val GOAL_PREFIXES = listOf("目标改为", "新的目标是", "设置目标", "我的目标是", "我要", "我想")
        private val CONTROL_PHRASES = setOf("确认", "完成", "下一步", "确定", "暂停", "停一下", "继续", "恢复",
            "停止", "结束任务", "取消任务", "重复", "再说一遍", "重做", "重新做", "跳过", "重新规划",
            "同步标记", "打同步标记", "停止监听", "关闭语音")
        private val RECORDING_COMMAND = Regex(
            """(?:(?:嗯|哈罗|哈喽|哈啰|你好|您好|hello|请帮我|帮我|请|你|现在|直接|马上)[\s，,]*){0,5}(?:(?:用眼镜)?(开始|停止|结束)(?:第一视角)?(?:录像|录制)|用眼镜录像)(?:吧)?[。！!.]*""",
            RegexOption.IGNORE_CASE
        )
        internal fun recordingAction(transcript: String): String? {
            val verb = RECORDING_COMMAND.matchEntire(transcript.trim())?.groupValues?.get(1) ?: return null
            return if (verb == "停止" || verb == "结束") "record_stop" else "record_start"
        }
        private val RECORDING_QUESTION = Regex(
            """(?:你|眼镜)?(?:现在)?(?:可以|能不能|能|支持|怎么|如何|正在|在|还在|已经|有没有|是不是在)?(?:用眼镜)?(?:开始|停止|结束)?(?:第一视角)?(?:录像|录制)(?:了没有|了吗|了没|没有|吗|没)?"""
        )
        private val UNCLEAR_RECORDING = Regex(
            """(?:[一二三四五六七八九十百零两0-9]+楼(?:层)?(?:开始|停止|结束)(?:录像|录制)|(?:开始|停止|结束)录像用[\p{IsHan}]{1,12}|(?:你)?用眼镜的录像)"""
        )
        internal fun needsRecordingFeedback(transcript: String): Boolean {
            val words = commandText(transcript)
            return RECORDING_QUESTION.matches(words) || UNCLEAR_RECORDING.matches(words)
        }
        private val PROGRESS_QUERIES = setOf("完成了吗", "做完了吗", "任务完成了吗", "任务完成了吧", "现在完成了吗", "现在做到哪了", "现在什么进度")
        internal fun isImmediateControl(text: String) = recordingAction(text) != null || commandText(text) in CONTROL_PHRASES - setOf("确认", "确定")
        internal fun isExplicitGoal(text: String) = goalPrefix(text) &&
            !Regex("^我(?:想|要)(?:问|知道|了解|确认|弄明白)").containsMatchIn(commandText(text))
        internal fun isProgressQuery(text: String) = commandText(text) in PROGRESS_QUERIES
        internal fun automaticInputAllowed(text: String, awaitingClarification: Boolean): Boolean {
            val value = commandText(text)
            return value.isNotBlank() && value.length <= 300 &&
                (supportedCommand(text) || isProgressQuery(value) || awaitingClarification)
        }
        private fun commandText(text: String) = text.trim().trimEnd('。', '！', '!', '.', '？', '?')
        private fun goalPrefix(text: String): Boolean {
            val command = commandText(text)
            return GOAL_PREFIXES.any { command.startsWith(it) && command.removePrefix(it).isNotBlank() }
        }
        private fun supportedCommand(text: String) = recordingAction(text) != null || needsRecordingFeedback(text) || commandText(text) in CONTROL_PHRASES || goalPrefix(text)

        fun normalizeOfflineText(text: String): String = text.trim()
            .replace(Regex("(?<=\\p{IsHan})\\s+(?=\\p{IsHan})"), "")
    }
}

/** Copies the pinned assets on a worker, validating every file before native loading. */
object StandaloneOfflineModel {
    @Synchronized fun prepare(context: Context): File {
        val assets = context.applicationContext.assets
        val manifestBytes = assets.open("vosk-model-small-cn/source-manifest.json").use { it.readBytes() }
        require(manifestBytes.size <= 128 * 1024)
        val manifestHash = hex(MessageDigest.getInstance("SHA-256").digest(manifestBytes))
        val destination = File(context.applicationContext.noBackupFilesDir, "vosk-model-small-cn-0.22-v1")
        require(destination.mkdirs() || destination.isDirectory)
        val ready = File(destination, ".ready")
        if (ready.isFile && ready.readText() == manifestHash && File(destination, "am/final.mdl").isFile &&
            File(destination, "conf/model.conf").isFile && File(destination, "graph/Gr.fst").isFile) return destination
        val files = JSONObject(String(manifestBytes, Charsets.UTF_8)).getJSONArray("files")
        require(files.length() in 1..100)
        for (i in 0 until files.length()) {
            val entry = files.getJSONObject(i)
            val path = entry.getString("path")
            require(path.matches(Regex("[A-Za-z0-9_./-]{1,160}")) && !path.startsWith('/') && ".." !in path)
            val target = File(destination, path)
            require(target.canonicalPath.startsWith(destination.canonicalPath + File.separator))
            require(target.parentFile!!.mkdirs() || target.parentFile!!.isDirectory)
            val expected = entry.getLong("bytes")
            val temporary = File(target.parentFile, target.name + ".part")
            assets.open("vosk-model-small-cn/$path").use { input ->
                FileOutputStream(temporary).use { out ->
                    copyVerified(input, out, expected, entry.getString("sha256"))
                    out.fd.sync()
                }
            }
            check(temporary.renameTo(target)) { "离线语音模型安装失败" }
        }
        val marker = File(destination, ".ready.part").apply { writeText(manifestHash) }
        check(marker.renameTo(ready))
        return destination
    }

    /** Official models may intentionally include empty config files; both their size and hash still matter. */
    internal fun copyVerified(input: java.io.InputStream, output: java.io.OutputStream, expected: Long, sha256: String) {
        require(expected in 0..100_000_000)
        require(sha256.matches(Regex("[0-9a-f]{64}")))
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        var count = 0L
        while (true) {
            val size = input.read(buffer)
            if (size < 0) break
            count += size
            require(count <= expected) { "离线语音模型大小校验失败" }
            digest.update(buffer, 0, size)
            output.write(buffer, 0, size)
        }
        require(count == expected && hex(digest.digest()) == sha256) { "离线语音模型校验失败" }
    }
    private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it.toInt() and 255) }
}
