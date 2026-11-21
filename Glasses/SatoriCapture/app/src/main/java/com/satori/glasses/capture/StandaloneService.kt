package com.satori.glasses.capture

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.AudioManager
import android.media.ToneGenerator
import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import android.provider.Settings
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.AtomicFile
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicBoolean
import java.io.Closeable

/** The complete live loop on the glasses. No gateway, ADB mapping or PC process is used. */
class StandaloneService : LifecycleService() {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { Thread(it, "standalone-loop") }
    private val engine = StandaloneEngine()
    private val frames = StandaloneFrameBuffer()
    private val epoch = AtomicLong()
    private val voiceKeyGate = VoiceKeyGate { now() }
    private val speechRequested = AtomicLong()
    private val speechStarted = AtomicLong()
    private val speechDone = AtomicLong()
    private val speechErrors = AtomicLong()
    private val gate = StandaloneFlowGate()
    private lateinit var camera: CameraStreamer
    private lateinit var imu: ImuStreamer
    private var receiver: GlassesInputReceiver? = null
    private var api: StandaloneApi? = null
    private var speechApi: StandaloneApi? = null
    private var voiceApi: StandaloneApi? = null
    private var dialogueApi: StandaloneApi? = null
    private var dialogueBusy = false
    @Volatile private var visualQueryUntil = 0L
    private var interactionFeedback = JSONObject()
    private var receiveTone: ToneGenerator? = null
    private var manualEvidence = JSONObject()
    private val webTaskEvidence = java.util.concurrent.atomic.AtomicReference<Pair<String, String>?>(null)
    private var lastAnswer = JSONObject()
    private val manualConversation = ManualConversation()
    private var experimentLabels = ExperimentContext.defaults()
    private var restoredTaskId = ""
    private var journaledTask = ""
    private var journaledPhase = ""
    private var speech: StandaloneSpeech? = null
    private var voice: StandaloneVoice? = null
    private var sharedMicrophone: SharedMicrophone? = null
    private var videoRecorder: FirstPersonRecorder? = null
    private var screenRecorder: ScreenRecorder? = null
    private var projectionForeground = false
    private var captureGroupId = ""
    private var pairedVideoPath: String? = null
    private var pairedScreenPath: String? = null
    private var endingDual = false
    private val videoCaptureMetadata = HashMap<String, JSONObject>()
    private var videoMicrophone: Closeable? = null
    @Volatile private var videoSessionPath: String? = null
    private var videoSinkGeneration = 0L
    private val videoPublisher = Executors.newSingleThreadExecutor { Thread(it, "video-publisher") }
    private var lastVideoUri: String? = null
    private var lastVideoName = ""
    private var analyzing = false
    private var profile: StandaloneProfile? = null
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var capturing = false
    @Volatile private var closed = false
    @Volatile private var worn: Boolean? = null
    @Volatile private var legsOpen: Boolean? = null
    @Volatile private var directory: File? = null
    private var inFlight = false
    private var observedAt = 0L
    private var notBefore = 0L
    private var lastPresence: Boolean? = null
    private var lastFolded: Boolean? = null
    private var prompt = JSONObject()
    private var error = ""
    private var lastSpoken = ""
    private var lastUserText = ""
    private var lastReplyText = ""
    private val chatTranscript = ChatTranscript()
    private var inputTurnId: String? = null
    private var dialogueTurnId: String? = null
    private var listeningEnabled = true
    private var listenAfter = 0L
    private var speechWasBusy = false
    private val pendingSpeech = ArrayDeque<String>()
    private var pendingRecognitionTrace: JSONObject? = null
    private var lastInput = JSONObject()
    private var journal: SessionJournal? = null
    private val dataLoggingFailed = AtomicBoolean()
    private var syncSequence = 0
    private var syncMarker = JSONObject()
    private val modelSequence = AtomicLong()
    private val logLock = Any()

    private val tick = object : Runnable {
        override fun run() {
            if (closed) return
            runCatching { pulse() }.onFailure { error = "运行异常，请暂停后重试"; Log.e(TAG, "loop ${it.javaClass.simpleName}") }
            main.postDelayed(this, 1000)
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        stopService(Intent(this, CaptureService::class.java))
        foreground()
        prompt = assets.open("standalone-prompts.json").bufferedReader().use { JSONObject(it.readText()) }
        camera = CameraStreamer(this, this, ::onFrame, ::event)
        sharedMicrophone = SharedMicrophone()
        videoRecorder = FirstPersonRecorder(this) { name, data -> main.post { onVideoEvent(name, data) } }
        screenRecorder = ScreenRecorder(this) { name, data -> main.post { onScreenEvent(name, data) } }
        val videoPrefs = getSharedPreferences("video_recordings", MODE_PRIVATE)
        lastVideoUri = videoPrefs.getString("last_uri", null)
        lastVideoName = videoPrefs.getString("last_name", "") ?: ""
        imu = ImuStreamer(this, ::onImu)
        var restoredFrom = ""
        val snapshotFile = File(filesDir, "standalone-state.json")
        runCatching { if (snapshotFile.isFile) {
            val saved = JSONObject(snapshotFile.readText())
            engine.restore(saved, now())
            restoredTaskId = saved.optString("task_instance_id")
            restoredFrom = saved.optString("local_run")
        } }
            .onFailure { error = "上次状态无法恢复，请重新设置目标" }
        try {
            createDataSession("service_start")
            event("session_started", JSONObject().put("restored_from_run", restoredFrom).put("app_version", BuildConfig.VERSION_NAME))
        } catch (failure: Exception) { dataFailure("session", failure) }
        configure()
        registerInputs()
        worn = systemFlag("vendor.rkd.glasses.is_take_on")
        legsOpen = systemFlag("vendor.rkd.glasses.is_spread")
        applyPresence()
        persist()
        main.post(tick)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        // Restart restores paused state; explicit user resume is required before new uploads.
        return START_STICKY
    }

    fun command(action: String, text: String = "", confirmMode: Boolean = true) {
        if (Looper.myLooper() != Looper.getMainLooper()) { main.post { command(action, text, confirmMode) }; return }
        if (closed) return
        if (action == "goal" && directory == null) {
            try { createDataSession("first_goal") } catch (failure: Exception) { dataFailure("session", failure); return }
        }
        if (action == "mark_sync") { markSync(); return }
        event("command", JSONObject().put("action", action).put("text", text).put("confirm_mode", confirmMode)
            .put("revision_before", engine.snapshot(now()).optLong("revision")))
        if (dataLoggingFailed.get()) {
            if (action == "record_stop") stopVideoRecording("data_log_failure")
            if (action == "record_start") speak(listOf(DATA_ERROR))
            return
        }
        if (action == "record_start") {
            error = ""
            startVideoRecording()
            if (error.isNotEmpty()) speak(listOf(error))
            return
        }
        if (action == "record_stop") { stopVideoRecording("user"); return }
        if (action == "voice") { toggleVoice(); return }
        if (action == "repeat") { speak(engine.repeat()); return }
        invalidate(resetEvidence = action != "confirm")
        error = ""
        val lines = runCatching {
            when (action) {
                "goal" -> {
                    require(text.isNotBlank() && text.length <= 300) { "请输入目标（最多300字）" }
                    profile?.validate() ?: throw IllegalArgumentException("请先设置模型接口和密钥")
                    val acknowledgement = engine.setGoal(text.trim(), confirmMode, now())
                    manualConversation.reset(); manualEvidence = JSONObject(); lastAnswer = JSONObject()
                    val resumed = engine.resume(now())
                    if (profile?.assistanceMode == "reactive") listOf("目标已设置，需要帮助时直接向我提问。") else acknowledgement + resumed
                }
                "pause" -> engine.pause(now())
                "resume" -> engine.resume(now())
                "confirm" -> engine.confirm(now())
                "redo" -> engine.redo(now())
                "skip" -> engine.skip(now())
                "replan" -> engine.replan(now())
                "answer" -> engine.answerClarification(text, now())
                "stop" -> { manualConversation.reset(); manualEvidence = JSONObject(); lastAnswer = JSONObject(); engine.stop(now()) }
                "configure" -> { configure(); emptyList() }
                else -> emptyList()
            }
        }.getOrElse { error = it.message ?: "操作失败"; emptyList() }
        applyPresence()
        speak(lines)
        persist()
        updateCapture()
    }

    fun snapshot(): JSONObject = engine.snapshot(now()).apply {
        put("mode", "standalone").put("worn", worn ?: JSONObject.NULL).put("legs_open", legsOpen ?: JSONObject.NULL)
        put("capturing", capturing).put("frames", camera.framesEncoded).put("network", networkReady())
        put("busy", inFlight).put("recording", voice?.active == true && voice?.continuous != true).put("last_spoken", lastSpoken)
        put("last_user_text", lastUserText).put("last_reply_text", lastReplyText)
        put("chat_messages", chatTranscript.snapshot())
        put("listening", voice?.listening == true).put("listening_enabled", listeningEnabled)
        put("speech_busy", speech?.busy == true).put("automatic_voice_unavailable", voice?.automaticUnavailable == true)
        put("voice_busy", voice?.busy == true).put("last_input", JSONObject(lastInput.toString()))
        put("dialogue_busy", dialogueBusy).put("manual_id", profile?.manualId.orEmpty())
        put("interaction_feedback", feedbackSnapshot())
        put("manual_source", profile?.manualSource ?: "remote_api")
        put("assistance_mode", profile?.assistanceMode ?: "guided").put("last_answer", JSONObject(lastAnswer.toString()))
        put("manual_evidence", JSONObject(manualEvidence.toString()))
        put("device_error", displayError()).put("storage_free_mb", filesDir.usableSpace / 1024 / 1024)
        put("version", BuildConfig.VERSION_NAME).put("run", directory?.name ?: "")
        put("speech_requested_count", speechRequested.get()).put("speech_started_count", speechStarted.get())
        put("speech_done_count", speechDone.get()).put("speech_error_count", speechErrors.get())
        put("video", videoRecorder?.snapshot() ?: JSONObject().put("state", "idle"))
        put("screen", screenRecorder?.snapshot() ?: JSONObject().put("state", "idle"))
        put("capture_group_id", captureGroupId)
        put("video_last_uri", lastVideoUri ?: JSONObject.NULL).put("video_last_name", lastVideoName)
        put("microphone", sharedMicrophone?.stats() ?: JSONObject())
        put("sync_marker", JSONObject(syncMarker.toString())).put("data_logging_ok", !dataLoggingFailed.get())
    }

    private fun configure() {
        webTaskEvidence.set(null)
        manualConversation.reset(); manualEvidence = JSONObject(); lastAnswer = JSONObject()
        api?.close(); speech?.close(); speechApi?.close(); voice?.close(); voiceApi?.close(); dialogueApi?.close()
        api = null; speech = null; voice = null; speechApi = null; voiceApi = null
        val loaded = runCatching { StandaloneSettings(this).load() }.getOrNull()
        profile = loaded
        if (loaded == null || runCatching { loaded.validate() }.isFailure) {
            error = "请先配置 HTTPS 模型接口、密钥和语音模型"
            return
        }
        val readManual: (String) -> ByteArray = { name -> assets.open("manuals/$name").use { it.readBytes() } }
        api = StandaloneApi(loaded, manualAsset = readManual, onTransportEvent = ::event)
        speechApi = StandaloneApi(loaded)
        voiceApi = StandaloneApi(loaded)
        dialogueApi = StandaloneApi(loaded, manualAsset = readManual, onTransportEvent = ::event)
        event("assistance_configuration", JSONObject().put("manual_id", loaded.manualId)
            .put("assistance_mode", loaded.assistanceMode).put("manual_source", loaded.manualSource)
            .put("tool_transport", if (loaded.manualSource == "device") "device_asset_search_with_https_source_check"
                else if (loaded.manualSource == "web") "mcp_streamable_http"
                else if (loaded.toolsBaseUrl.isBlank()) "disabled" else "http_json_api")
            .put("planner_model", loaded.planner).put("monitor_model", loaded.monitor))
        writeExperimentConfiguration(loaded)
        speech = StandaloneSpeech(this, speechApi!!, ::event)
        voice = StandaloneVoice(this, voiceApi!!, { onText(it) }, { message ->
            listenAfter = now() + 5000
            error = message; event("voice_error", JSONObject().put("message", message)); persist()
        }, { trace ->
            pendingRecognitionTrace = JSONObject(trace.toString())
            event("voice_recognition", trace)
        }, sharedMicrophone)
    }

    private fun pulse() {
        playPendingSpeech()
        // A confirmation reminder is consumed by heartbeat; defer it while recording/recognizing speech.
        if (voice?.busy != true) speak(engine.heartbeat(now()))
        updateCapture()
        ensureListening()
        recordStatus()
        if (!capturing || inFlight || dialogueBusy || profile?.assistanceMode == "reactive" || now() < notBefore || voice?.busy == true || !networkReady()) return
        val selected = frames.select(observedAt, now())
        if (selected.isEmpty()) return
        val request = engine.nextRequest(now(), selected.map { it.id }) ?: return
        request.payload.put("frames_note", JSONArray(selected.map { frame -> JSONObject().put("frame_id", frame.id)
            .put("timestamp_ms", frame.atMs).put("key", frame.key ?: JSONObject.NULL) }))
        val version = epoch.get()
        val client = api ?: return
        val config = profile ?: return
        val userGoalState = engine.snapshot(now())
        val actualReports = userGoalState.optJSONArray("user_feedback") ?: JSONArray()
        val userGoalText = userGoalState.optJSONObject("goal")?.optString("text").orEmpty().take(400) +
            (0 until actualReports.length()).map { actualReports.getJSONObject(it) }
                .filter { it.optString("task_instance_id") == userGoalState.optString("task_instance_id") }
                .takeLast(3).joinToString("", prefix = "\n用户补充：") { it.optString("text") + "；" }.takeLast(600)
        error = ""
        inFlight = true
        val requestId = "q${modelSequence.incrementAndGet()}"
        beginFeedback(requestId, request.stage)
        event("request_start", JSONObject().put("stage", request.stage).put("frames", selected.size).put("request_id", requestId)
            .put("request_revision", request.revision)
            .put("frame_ids", JSONArray(selected.map { it.id })).put("frame_times_ms", JSONArray(selected.map { it.atMs }))
            .put("payload", JSONObject(request.payload.toString()))
            .put("requested_model", if (request.stage == "observe") config.monitor else config.planner))
        if (dataLoggingFailed.get()) { inFlight = false; finishFeedback(requestId, "data_log_failure"); engine.cancelRequest(now()); return }
        worker.execute {
            val result = runCatching {
                if (config.manualId.isNotBlank()) {
                    advanceFeedback(requestId, version, "manual_lookup")
                    val query = if (config.manualSource == "web") userGoalText.take(500) else
                        (request.payload.optString("goal") + " " + request.payload.optJSONObject("active_step")?.optString("instruction").orEmpty()).take(500)
                    val evidence = retrieveManual(client, query, requestId, version, config.manualId, request.payload.optString("task_instance_id"), request.revision,
                        selected.takeLast(2), userGoalText, request.stage in setOf("observe", "remind"))
                    if (!evidence.optBoolean("matched") && config.manualSource != "web") throw ApiFailure(422, message = "手册没有匹配依据，请核对型号、手册或提出具体问题")
                    if (evidence.optString("transport") == "not_requested") request.payload.put("research_decision", evidence.getJSONObject("search_plan"))
                    else request.payload.put("manual_context", evidence)
                    main.post { if (!closed && version == epoch.get()) manualEvidence = evidence }
                }
                if (closed || dataLoggingFailed.get() || version != epoch.get()) throw ApiFailure(message = "请求已取消")
                advanceFeedback(requestId, version, request.stage)
                val model = if (request.stage == "observe") config.monitor else config.planner
                val stagePrompt = if (config.manualSource == "web") {
                    val basePrompt = prompt.getString(request.stage)
                    basePrompt + if (request.payload.has("manual_context")) "\n输出增加source_sections数组，仅列出本次实际引用的citation_id；未使用资料则为空。" else ""
                } else if (config.manualId.isNotBlank()) StandaloneManuals.stagePrompt(prompt.getString(request.stage)) else prompt.getString(request.stage)
                client.complete(model, stagePrompt + (if (config.manualSource == "web") WebResearchPolicy.WEB_RULES else ""), request.payload, selected.map { it.jpeg },
                    if (request.stage == "observe") 75_000 else 120_000)
            }
            main.post {
                val runtimeChanged = version != epoch.get()
                val revisionChanged = request.revision != engine.snapshot(now()).optLong("revision")
                if (closed || runtimeChanged || revisionChanged || dataLoggingFailed.get()) {
                    finishFeedback(requestId, "discarded")
                    if (!closed && !runtimeChanged) inFlight = false
                    event("response_discarded", JSONObject().put("request_id", requestId).put("stage", request.stage)
                        .put("request_revision", request.revision).put("current_revision", engine.snapshot(now()).optLong("revision"))
                        .put("reason", if (closed) "service_closed" else if (dataLoggingFailed.get()) "data_log_failure"
                            else if (runtimeChanged) "runtime_changed" else "revision_changed")
                        .apply { result.getOrNull()?.let { put("result", it) } })
                    return@post
                }
                inFlight = false
                finishFeedback(requestId, if (result.isSuccess) "completed" else "error")
                result.onSuccess { response ->
                    if (config.manualId.isNotBlank()) manualEvidence.put("used_sections", response.optJSONArray("source_sections") ?: JSONArray())
                    val lines = runCatching {
                        engine.accept(request, response, now())
                    }.getOrElse {
                        engine.fail(request, null, "模型返回格式无效", now())
                    }
                    if (request.stage == "observe" && engine.snapshot(now()).optInt("failures") == 0) observedAt = selected.last().atMs
                    speak(lines, replyTo = "")
                    event("response", JSONObject().put("stage", request.stage).put("result", response).put("request_id", requestId)
                        .put("accepted", engine.snapshot(now()).optInt("failures") == 0))
                    notBefore = now() + 3000
                }.onFailure { failure ->
                    val apiFailure = failure as? ApiFailure
                    val message = if (apiFailure != null) apiFailure.message ?: "模型调用失败" else "模型调用失败，请检查配置"
                    error = message
                    if (apiFailure?.diagnostic in QUOTA_ERRORS) {
                        engine.pause(now())
                        speak(listOf("接口额度不足，任务已暂停，请补充额度后继续。"), replyTo = "")
                        updateCapture()
                    } else speak(engine.fail(request, apiFailure?.status, message, now()), replyTo = "")
                    notBefore = now() + if (apiFailure?.status == 429) maxOf(60_000, apiFailure.retryAfterMs) else 5000
                    event("request_error", JSONObject().put("stage", request.stage).put("status", apiFailure?.status ?: 0)
                        .put("diagnostic", apiFailure?.diagnostic ?: "").put("request_id", requestId))
                }
                persist()
            }
        }
    }

    private fun invalidate(resetEvidence: Boolean = true) {
        epoch.incrementAndGet()
        pendingSpeech.clear()
        visualQueryUntil = 0L
        dialogueTurnId?.let { chatTranscript.finish(it, "cancelled") }
        dialogueTurnId = null
        finishFeedback(interactionFeedback.optString("request_id"), "cancelled")
        if (resetEvidence) { engine.gap(now()); gate.reset() } else engine.cancelRequest(now())
        api?.cancel(); dialogueApi?.cancel(); speech?.stop(); voice?.cancel()
        dialogueBusy = false
        inFlight = false
        frames.clear()
        observedAt = now()
        notBefore = maxOf(notBefore, now() + 2000)
    }

    private fun feedbackSnapshot(): JSONObject = JSONObject(interactionFeedback.toString()).apply {
        if (optString("request_id").isNotBlank()) put("elapsed_ms", (now() - optLong("started_at_ms")).coerceAtLeast(0))
    }

    private fun beginFeedback(requestId: String, stage: String, question: String = "") {
        interactionFeedback = JSONObject().put("request_id", requestId).put("stage", stage)
            .put("started_at_ms", now()).put("question_preview", question.take(120))
        if (requestId.startsWith("d")) dialogueTurnId?.let { interactionFeedback.put("turn_id", it) }
        event("interaction_feedback", JSONObject(interactionFeedback.toString()).put("transition", "started"))
        if (instance === this) StandaloneActivity.refreshVisiblePanel()
    }

    private fun advanceFeedback(requestId: String, ticket: Long, stage: String) {
        main.post {
            if (closed || dataLoggingFailed.get() || ticket != epoch.get() || interactionFeedback.optString("request_id") != requestId) return@post
            if (interactionFeedback.optString("stage") == stage) return@post
            interactionFeedback.put("stage", stage)
            event("interaction_feedback", feedbackSnapshot().put("transition", "stage"))
            if (instance === this) StandaloneActivity.refreshVisiblePanel()
        }
    }

    private fun finishFeedback(requestId: String, reason: String) {
        if (requestId.isBlank() || interactionFeedback.optString("request_id") != requestId) return
        val finished = feedbackSnapshot().put("transition", "finished").put("reason", reason)
        interactionFeedback = JSONObject()
        runCatching { receiveTone?.stopTone() }
        event("interaction_feedback", finished)
        if (instance === this) StandaloneActivity.refreshVisiblePanel()
    }

    private fun acknowledgeQuestion(requestId: String) {
        if (instance !== this || closed || lastFolded == true || dataLoggingFailed.get() ||
            interactionFeedback.optString("request_id") != requestId) return
        // The request already stopped ASR; keep a short guard against recording our own cue.
        listenAfter = maxOf(listenAfter, now() + 300)
        val played = runCatching {
            val tone = receiveTone ?: ToneGenerator(AudioManager.STREAM_MUSIC, 35).also { receiveTone = it }
            tone.startTone(ToneGenerator.TONE_PROP_ACK, 120)
        }.getOrDefault(false)
        event(if (played) "feedback_cue_requested" else "feedback_cue_skipped", JSONObject()
            .put("request_id", requestId).put("kind", "local_tone").put("duration_ms", 120))
    }

    private fun onText(raw: String) {
        val text = raw.trim().trimEnd('。', '！', '!', '.', '？', '?')
        val automatic = pendingRecognitionTrace?.optBoolean("automatic") == true
        val stateBefore = engine.snapshot(now())
        val awaitingManualReply = manualConversation.awaitingReply(dialogueScope(stateBefore), now())
        val awaitingTaskReply = stateBefore.optString("phase") == "awaiting_clarification"
        val openDialogue = profile?.manualSource == "web"
        val manualTalk = openDialogue || profile?.manualId?.isNotBlank() == true &&
            !stateBefore.isNull("goal") && stateBefore.optString("phase") !in setOf("idle", "complete", "stopped") &&
            (StandaloneManuals.isTaskUtterance(text) || awaitingManualReply && text.length in 1..500)
        val ignored = if (openDialogue) {
            val minimum = if (!automatic || awaitingManualReply || awaitingTaskReply) 1 else 2
            text.length > 500 || text.count(Char::isLetterOrDigit) < minimum ||
                text.none { it.isLetterOrDigit() && it !in "嗯啊呃额唔哦噢哎" }
        } else automatic && !manualTalk && !StandaloneVoice.automaticInputAllowed(raw, awaitingTaskReply)
        if (ignored) {
            event("voice_ignored", JSONObject().put("text", text.take(500))
                .put("reason", if (openDialogue) "empty_or_invalid_utterance" else "not_addressed_command"))
            pendingRecognitionTrace = null
            return
        }
        if (text.isBlank()) { pendingRecognitionTrace = null; return }
        lastUserText = text.take(500)
        lastReplyText = ""
        val turnId = chatTranscript.beginUser(lastUserText)
        inputTurnId = turnId
        event("voice_text", JSONObject().put("text", text).put("chat_turn_id", turnId))
        if (instance === this) StandaloneActivity.refreshVisiblePanel()
        try {
            val recording = StandaloneVoice.recordingAction(raw)
            if (recording != null) command(recording)
            else if (StandaloneVoice.needsRecordingFeedback(raw)) speak(listOf(recordingFeedback()))
            else handleAcceptedText(text, automatic, manualTalk, awaitingManualReply)
        }
        finally {
            if (dialogueTurnId != turnId) chatTranscript.finish(turnId, "complete")
            inputTurnId = null
            pendingRecognitionTrace = null
        }
    }

    private fun handleAcceptedText(text: String, automatic: Boolean, manualTalk: Boolean, awaitingManualReply: Boolean) {
        val current = engine.snapshot(now())
        if (current.optString("phase") == "awaiting_clarification" && current.optJSONObject("pending")?.optBoolean("manual_identity") == true &&
            !StandaloneVoice.isImmediateControl(text) && !StandaloneVoice.isExplicitGoal(text)) {
            manualConversation.reset()
            command("answer", text)
            return
        }
        if (awaitingManualReply && manualTalk && !StandaloneVoice.isImmediateControl(text) &&
            !StandaloneVoice.isExplicitGoal(text)) {
            askManual(text)
            pendingRecognitionTrace = null
            return
        }
        if (StandaloneVoice.isProgressQuery(text)) {
            val state = engine.snapshot(now())
            val status = when {
                state.optString("phase") == "complete" -> "系统记录的目标已完成。"
                state.optString("phase") == "stopped" -> "任务已停止。"
                state.isNull("goal") -> "还没有设置目标，可以说我想做什么。"
                !state.isNull("paused_reason") -> "任务目前已暂停，可以说继续。"
                state.optString("phase") == "awaiting_step_confirmation" -> "当前步骤还在等待你的确认，尚未记为完成。"
                else -> "目前还未确认目标全部完成。"
            }
            speak(listOf(status) + if (state.isNull("paused_reason") && state.optString("phase") !in setOf("complete", "stopped", "idle")) engine.repeat() else emptyList())
            pendingRecognitionTrace = null
            return
        }
        when (text) {
            "停止监听", "关闭语音" -> {
                listeningEnabled = false
                voice?.cancel()
                speak(listOf("已关闭语音待听，按遥控器右键可以重新开启。"))
                foreground()
            }
            "确认", "完成", "下一步", "确定" -> command("confirm")
            "暂停", "停一下" -> command("pause")
            "继续", "恢复" -> command("resume")
            "停止", "结束任务", "取消任务" -> command("stop")
            "重复", "再说一遍" -> command("repeat")
            "重做", "重新做" -> command("redo")
            "跳过" -> command("skip")
            "重新规划" -> command("replan")
            "同步标记", "打同步标记" -> command("mark_sync")
            else -> {
                val prefix = listOf("目标改为", "新的目标是", "设置目标", "我的目标是", "我要", "我想").firstOrNull { text.startsWith(it) }
                val state = engine.snapshot(now())
                if (manualTalk && prefix != null && !StandaloneVoice.isExplicitGoal(text)) askManual(text)
                else if (prefix != null) setSpokenGoal(text.removePrefix(prefix).trim(), text)
                else if (manualTalk && profile?.manualSource == "web" && state.optString("phase") != "awaiting_clarification") askManual(text)
                else if (state.optString("goal").isBlank() || state.optString("phase") in listOf("idle", "complete", "stopped")) setSpokenGoal(text, text)
                else if (state.optString("phase") == "awaiting_clarification") command("answer", text)
                else if (manualTalk || !automatic && profile?.manualId?.isNotBlank() == true) askManual(text)
                else { error = "设置新目标请说“我要……”，操作请说“确认、暂停、继续”"; speak(listOf(error)) }
            }
        }
        pendingRecognitionTrace = null
    }

    private fun retrieveManual(client: StandaloneApi, query: String, requestId: String, ticket: Long,
                               manualId: String, taskId: String, revision: Long,
                               observedFrames: List<StandaloneFrame> = emptyList(), taskText: String = "", reuseWeb: Boolean = false,
                               observationTimeMs: Long = now(), dialogueHistory: JSONArray? = null): JSONObject {
        if (closed || dataLoggingFailed.get() || ticket != epoch.get()) throw ApiFailure(message = "请求已取消")
        val toolId = "$requestId-manual"
        event("tool_request", JSONObject().put("tool_call_id", toolId).put("parent_request_id", requestId)
            .put("task_instance_id", taskId).put("request_revision", revision)
            .put("tool", if (profile?.manualSource == "web") "prepare_research_context" else "search_manual")
            .put("arguments", JSONObject().put("manual_id", manualId).put("query", query)))
        val start = SystemClock.elapsedRealtimeNanos()
        var retrievalStage = "manual_lookup"
        try {
            if (closed || dataLoggingFailed.get() || ticket != epoch.get()) throw ApiFailure(message = "请求已取消")
            val config = profile ?: throw ApiFailure(message = "配置不可用")
            val evidence = if (config.manualSource != "web") client.searchManual(query) else {
                val cached = webTaskEvidence.get()?.takeIf { it.first == taskId }
                if (reuseWeb && cached != null) JSONObject(cached.second).put("cache_hit", true) else {
                    val fresh = observedFrames.filter { observationTimeMs - it.atMs in 0..15000 }.takeLast(2)
                    val inputText = (taskText.takeLast(1000) + "\n当前问题：" + query.takeLast(500)).takeLast(1600)
                    retrievalStage = "search_planning"
                    advanceFeedback(requestId, ticket, "search_planning")
                    event("web_search_plan_request", JSONObject().put("parent_request_id", requestId)
                        .put("userText", inputText).put("frame_ids", JSONArray(fresh.map { it.id })))
                    val rawPlan = client.complete(config.planner, WebResearchPolicy.PLAN_PROMPT,
                        JSONObject().put("userText", inputText).putOpt("dialogue_history", dialogueHistory).put("frames_note", JSONArray(fresh.map {
                            JSONObject().put("frame_id", it.id).put("timestamp_ms", it.atMs)
                        })), fresh.map { it.jpeg }, 30000)
                    val plan = WebResearchPolicy.plan(rawPlan)
                    if (closed || ticket != epoch.get()) throw ApiFailure(message = "请求已取消")
                    event("web_search_plan", JSONObject().put("parent_request_id", requestId).put("plan", plan))
                    if (plan.getBoolean("needs_search")) retrievalStage = "web_search"
                    advanceFeedback(requestId, ticket, if (plan.getBoolean("needs_search")) "web_search" else "model")
                    client.researchWeb(plan) { name, data ->
                        event(name, data.put("parent_request_id", requestId).put("task_instance_id", taskId))
                    }.put("cache_hit", false).also {
                        if (requestId.startsWith("q") && !closed && ticket == epoch.get() && !dataLoggingFailed.get()) webTaskEvidence.set(taskId to it.toString())
                    }
                }
            }
            val valid = !closed && !dataLoggingFailed.get() && ticket == epoch.get()
            event("tool_result", JSONObject().put("tool_call_id", toolId).put("parent_request_id", requestId)
                .put("latency_ms", (SystemClock.elapsedRealtimeNanos() - start) / 1e6).put("accepted", valid).put("result", evidence))
            if (!valid || dataLoggingFailed.get()) throw ApiFailure(message = "请求已取消")
            return evidence
        } catch (failure: Exception) {
            val apiFailure = failure as? ApiFailure
            event("tool_error", JSONObject().put("tool_call_id", toolId).put("parent_request_id", requestId)
                .put("latency_ms", (SystemClock.elapsedRealtimeNanos() - start) / 1e6).put("error_kind", failure.javaClass.simpleName).put("stage", retrievalStage)
                .put("http_status", apiFailure?.status ?: 0).put("diagnostic", apiFailure?.diagnostic.orEmpty()))
            throw StandaloneManuals.lookupFailure(retrievalStage, failure)
        }
    }

    private fun askManual(text: String) {
        val client = dialogueApi ?: return
        val config = profile ?: return
        if (text.isBlank() || text.length > 500 || config.manualId.isBlank() || lastFolded == true || dataLoggingFailed.get()) return
        val snapshotAt = now()
        var selected = frames.select(snapshotAt - 2001, snapshotAt, 1)
        invalidate(resetEvidence = false)
        val state = engine.snapshot(now())
        if (config.manualSource != "web" && (state.isNull("goal") || state.optString("phase") in setOf("idle", "complete", "stopped"))) { error = "请先设置任务，再向我提问"; return }
        val taskActive = !state.isNull("goal") && state.isNull("paused_reason") && state.optString("phase") !in setOf("idle", "complete", "stopped")
        val scope = dialogueScope(state)
        val taskContext = if (config.manualSource == "web" && !taskActive) JSONObject().put("goal", JSONObject.NULL).put("scope", "current_dialogue_only") else state
        val ticket = epoch.get()
        val requestId = "d${modelSequence.incrementAndGet()}"
        val turnId = inputTurnId
        dialogueTurnId = turnId
        val conversation = manualConversation.begin(scope, text, now())
        dialogueBusy = true
        val needCurrentView = selected.isEmpty() && ::camera.isInitialized && ::imu.isInitialized && networkReady()
        val snapshotDeadline = snapshotAt + 10_000
        if (needCurrentView) visualQueryUntil = snapshotDeadline + 500
        error = ""
        lastAnswer = JSONObject()
        beginFeedback(requestId, "received", text)
        event("dialogue_request", JSONObject().put("request_id", requestId).put("text", text)
            .put("chat_turn_id", turnId ?: JSONObject.NULL)
            .put("task_instance_id", state.optString("task_instance_id")).put("revision", state.optLong("revision")))
        acknowledgeQuestion(requestId)
        persist(); updateCapture()
        worker.execute {
            var answerSource = "unresolved"
            var stage = "manual_lookup"
            var modelCalled = false
            val startedNs = SystemClock.elapsedRealtimeNanos()
            val result = runCatching {
                fun lookupManual(): JSONObject {
                    stage = "manual_lookup"
                    advanceFeedback(requestId, ticket, stage)
                    val query = if (config.manualSource == "web") text else (0 until conversation.length()).map { conversation.getJSONObject(it) }
                        .filter { it.optString("role") == "user" }.takeLast(3).joinToString(" ") { it.getString("text") }.takeLast(500)
                    return retrieveManual(client, query, requestId, ticket, config.manualId, state.optString("task_instance_id"),
                        state.optLong("revision"), selected, taskContext.optJSONObject("goal")?.optString("text").orEmpty(),
                        observationTimeMs = maxOf(snapshotAt, selected.lastOrNull()?.atMs ?: snapshotAt), dialogueHistory = conversation)
                }
                val document = if (config.manualSource != "web") lookupManual() else null
                val metadataAnswer = document?.let { StandaloneManuals.metadataAnswer(text, it) }
                try {
                    if (needCurrentView && metadataAnswer == null) {
                        stage = "current_view"
                        advanceFeedback(requestId, ticket, stage)
                        selected = frames.awaitFresh(snapshotAt - 1, snapshotDeadline, ::now,
                            { closed || dataLoggingFailed.get() || ticket != epoch.get() }).takeLast(1)
                        if (closed || dataLoggingFailed.get() || ticket != epoch.get()) throw ApiFailure(message = "请求已取消")
                    }
                } finally {
                    if (needCurrentView) main.post { if (ticket == epoch.get()) { visualQueryUntil = 0L; updateCapture() } }
                }
                val snapshotOutcome = when {
                    metadataAnswer != null -> "local_document_answer"
                    needCurrentView && selected.isNotEmpty() -> "captured"
                    needCurrentView -> "timeout"
                    selected.isNotEmpty() -> "recent"
                    else -> "unavailable"
                }
                event("dialogue_visual_snapshot", JSONObject().put("request_id", requestId).put("frame_ids", JSONArray(selected.map { it.id }))
                    .put("available", selected.isNotEmpty()).put("wait_ms", now() - snapshotAt).put("outcome", snapshotOutcome))
                if (selected.isEmpty() && metadataAnswer == null && Regex("看看|看一下|眼前|视野|画面").containsMatchIn(text)) {
                    stage = "current_view"
                    throw ApiFailure(message = "这次没有取得眼镜画面，请保持镜腿展开、网络连接后再问一次。",
                        diagnostic = "visual_snapshot_$snapshotOutcome")
                }
                val evidence = document ?: lookupManual()
                if (closed || dataLoggingFailed.get() || ticket != epoch.get()) throw ApiFailure(message = "请求已取消")
                val direct = metadataAnswer
                answerSource = if (direct == null) "model" else "verified_manual_metadata"
                stage = if (direct == null) "model" else "metadata_answer"
                advanceFeedback(requestId, ticket, stage)
                event("dialogue_route", JSONObject().put("request_id", requestId).put("answer_source", answerSource)
                    .put("model_call_planned", direct == null).put("frame_count", if (direct == null) selected.size else 0))
                if (dataLoggingFailed.get()) throw ApiFailure(message = "请求已取消")
                val payload = JSONObject().put("user_utterance", text).put("task", taskContext).put("manual_context", evidence)
                    .put("dialogue_history", conversation)
                    .put("current_view", if (selected.isEmpty()) "本轮没有眼镜图像，仅根据文字回答。" else "本轮提问时的第一视角图像。")
                    .put("frames_note", JSONArray(selected.map { JSONObject().put("frame_id", it.id).put("timestamp_ms", it.atMs) }))
                val response = direct ?: run {
                    modelCalled = true
                    client.complete(config.planner, dialoguePrompt(config), payload, selected.map { it.jpeg }, 75000)
                }
                val answer = StandaloneManuals.answer(response, evidence)
                if (selected.isEmpty() && metadataAnswer == null) answer.put("text", "这次没取到眼镜画面。" + answer.getString("text"))
                answer to evidence
            }
            main.post {
                if (closed || dataLoggingFailed.get() || ticket != epoch.get() || engine.snapshot(now()).optString("task_instance_id") != state.optString("task_instance_id")) {
                    turnId?.let { chatTranscript.finish(it, "cancelled") }
                    event("dialogue_discarded", JSONObject().put("request_id", requestId).put("reason", "task_or_command_changed"))
                    return@post
                }
                dialogueBusy = false
                finishFeedback(requestId, if (result.isSuccess) "completed" else "error")
                result.onSuccess { (answer, evidence) ->
                    manualEvidence = JSONObject(evidence.toString()).put("used_sections", answer.getJSONArray("source_sections"))
                    lastAnswer = JSONObject(answer.toString()).put("request_id", requestId).put("at_ms", now())
                        .put("answer_source", answerSource).put("model_called", modelCalled)
                        .put("task_instance_id", state.optString("task_instance_id")).put("manual_id", config.manualId)
                    manualConversation.finish(scope, answer, now())
                    event("dialogue_response", JSONObject().put("request_id", requestId).put("answer", answer).put("evidence", evidence)
                        .put("answer_source", answerSource).put("model_called", modelCalled))
                    if (dataLoggingFailed.get()) return@onSuccess
                    if (answer.optString("kind") != "unrelated") {
                        if (answer.optString("kind") == "correction" && (config.manualSource != "web" || taskActive)) {
                            engine.addUserFeedback(text, now())
                            webTaskEvidence.set(null)
                        }
                        val replanned = if (answer.optBoolean("suggest_replan") && (config.manualSource != "web" || taskActive)) engine.replan(now()) else emptyList()
                        speak(listOf(answer.getString("text")) + replanned, replyTo = text, chatTurn = turnId,
                            displaySuffix = if (config.manualSource == "web") WebEvidence.sourceNote(evidence, answer) else "")
                    } else turnId?.let { chatTranscript.finish(it, "ignored") }
                    error = ""
                }.onFailure { failure ->
                    error = (failure as? ApiFailure)?.message ?: "手册答复未通过校验，请重说或查看资料"
                    val apiFailure = failure as? ApiFailure
                    event("dialogue_error", JSONObject().put("request_id", requestId).put("error_kind", failure.javaClass.simpleName)
                        .put("stage", stage).put("answer_source", answerSource).put("model_called", modelCalled)
                        .put("elapsed_ms", (SystemClock.elapsedRealtimeNanos() - startedNs) / 1e6)
                        .put("http_status", apiFailure?.status ?: 0).put("diagnostic", apiFailure?.diagnostic.orEmpty()))
                    speak(listOf(error), replyTo = text, chatTurn = turnId)
                }
                if (dialogueTurnId == turnId) dialogueTurnId = null
                persist(); updateCapture()
            }
        }
    }

    private fun dialogueScope(state: JSONObject): String = if (profile?.manualSource == "web" &&
        (state.isNull("goal") || !state.isNull("paused_reason") || state.optString("phase") in setOf("idle", "complete", "stopped")))
        "dialogue-${directory?.name ?: "session"}" else state.optString("task_instance_id")

    private fun setSpokenGoal(goal: String, transcript: String) {
        command("goal", goal)
        // The first transcript predates creation of the run; retain it in the newly created run as well.
        if (engine.snapshot(now()).optJSONObject("goal")?.optString("text") == goal.trim()) {
            event("voice_text", JSONObject().put("text", transcript).put("applied_goal", true))
            pendingRecognitionTrace?.let { event("voice_recognition", it) }
        }
    }

    private fun toggleVoice() {
        if (voice?.active == true && voice?.continuous != true) { event("voice_stop_requested", JSONObject()); voice?.stop(); return }
        if (dataLoggingFailed.get()) { error = DATA_ERROR; return }
        if (lastFolded == true) { error = "请展开镜腿再说话"; return }
        if (voice == null) { error = "请先配置语音识别接口"; return }
        listeningEnabled = true
        invalidate(resetEvidence = engine.snapshot(now()).optString("phase") != "awaiting_step_confirmation")
        event("voice_start_requested", JSONObject())
        voice?.start()
        foreground()
    }

    /** Foreground HUD key routes and the vendor broadcasts share one debounced voice entry point. */
    fun onVoiceKey(source: String) {
        if (Looper.myLooper() != Looper.getMainLooper()) { main.post { onVoiceKey(source) }; return }
        if (closed) return
        val allowed = voiceKeyGate.accept(source)
        val recognizing = voice?.busy == true && voice?.active != true
        inputEvent(JSONObject().put("source", source).put("accepted", allowed && !recognizing)
            .put("duplicate", !allowed).put("recognizing", recognizing).put("recording", voice?.active == true))
        if (allowed && !recognizing) toggleVoice()
    }

    private fun inputEvent(data: JSONObject) {
        lastInput = JSONObject(data.toString()).put("at_ms", now())
        Log.i(TAG, "input $lastInput")
        event("input_signal", lastInput)
        runCatching {
            val folder = File(filesDir, "input-events").apply { mkdirs() }
            File(folder, "input-${android.os.Process.myPid()}.jsonl").appendText(lastInput.toString() + "\n")
        }
    }

    private fun speak(lines: List<String>, replyTo: String? = null, chatTurn: String? = inputTurnId, displaySuffix: String = "") {
        // Generated text may be displayed even when playback is unavailable; audio events remain the playback evidence.
        val spoken = lines.filter(String::isNotBlank)
        if (spoken.isEmpty()) return
        if (replyTo != null) lastUserText = replyTo.take(500)
        lastReplyText = (spoken.joinToString("\n").trim() + displaySuffix).take(4000)
        if (!chatTranscript.answer(chatTurn, lastReplyText) && chatTurn != null && chatTurn == inputTurnId)
            chatTranscript.answer(null, lastReplyText)
        if (closed || lastFolded == true) return
        pendingSpeech.addAll(spoken)
        playPendingSpeech()
    }

    private fun playPendingSpeech() {
        if (pendingSpeech.isEmpty() || closed || lastFolded == true || voice?.busy == true) return
        // shortcut: half-duplex avoids self-recognition; add acoustic echo cancellation before enabling barge-in.
        if (voice?.continuous == true) voice?.cancel()
        speechWasBusy = true
        listenAfter = maxOf(listenAfter, now() + 500)
        while (pendingSpeech.isNotEmpty()) {
            lastSpoken = pendingSpeech.removeFirst()
            speech?.speak(lastSpoken)
        }
    }

    private fun ensureListening() {
        if (instance !== this) return
        val input = voice ?: return
        if (closed || !listeningEnabled || lastFolded == true || dataLoggingFailed.get() || filesDir.usableSpace <= 512L * 1024 * 1024) {
            if (input.continuous) input.cancel()
            return
        }
        if (speech?.busy == true) {
            if (input.continuous) input.cancel()
            speechWasBusy = true
            return
        }
        if (speechWasBusy) {
            speechWasBusy = false
            listenAfter = maxOf(listenAfter, now() + 500)
        }
        if (!input.active && !input.busy && !input.automaticUnavailable && now() >= listenAfter) input.startContinuous()
    }

    private fun applyPresence() {
        val folded = legsOpen?.not() ?: lastFolded ?: false
        val foldChanged = folded != lastFolded
        if (worn == lastPresence && !foldChanged) return
        lastPresence = worn
        lastFolded = folded
        event("presence_changed", JSONObject().put("present", worn ?: JSONObject.NULL).put("worn", worn ?: JSONObject.NULL)
            .put("folded", folded).put("off_head_continues", true)
            .put("legs_open", legsOpen ?: JSONObject.NULL).put("time_kind", "received_or_polled_state"))
        if (foldChanged) invalidate()
        speak(engine.presence(worn, now(), folded))
        persist()
        updateCapture()
    }

    private fun updateCapture() {
        val state = engine.snapshot(now())
        val phase = state.optString("phase")
        val network = networkReady()
        val storage = filesDir.usableSpace > 512L * 1024 * 1024
        val deviceOpen = lastFolded != true
        if ((!deviceOpen || !storage) && screenRecorder?.active == true) screenRecorder?.stop(if (!deviceOpen) "folded" else "storage_low")
        if ((!deviceOpen || !storage) && videoRecorder?.wantsFrames == true) {
            stopVideoRecording(if (!deviceOpen) "folded" else "storage_low", update = false)
        }
        val awaitingWebLabel = profile?.manualSource == "web" && phase == "awaiting_clarification" && state.isNull("paused_reason")
        val needs = StandaloneCaptureNeeds.calculate(!closed && !dataLoggingFailed.get() && (engine.needsVisualInput() || awaitingWebLabel || now() < visualQueryUntil), network, api != null,
            !closed && videoRecorder?.wantsFrames == true, deviceOpen, storage)
        if (analyzing && !needs.analysis) {
            frames.clear(); gate.reset()
            if (!network || !storage) invalidate()
        }
        if (analyzing != needs.analysis || capturing != needs.camera) {
            event("capture_state", JSONObject().put("camera_active", needs.camera).put("analysis_active", needs.analysis)
                .put("video_requested", needs.video).put("network", network).put("present", worn ?: JSONObject.NULL)
                .put("device_open", deviceOpen).put("storage_ok", storage)
                .put("phase", phase).put("paused_reason", state.opt("paused_reason") ?: JSONObject.NULL))
        }
        analyzing = needs.analysis
        if (::camera.isInitialized) camera.setAnalysisEnabled(needs.analysis)
        val allowed = needs.camera
        if (allowed == capturing) return
        capturing = allowed
        if (allowed) {
            camera.start(CaptureConfig(fps = 2.0, imuHz = 50))
            imu.start(50)
            wakeLock = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Satori:standalone")
                .apply { setReferenceCounted(false); acquire(4 * 60 * 60 * 1000L) }
        } else {
            camera.stop(); imu.stop(); frames.clear()
            event("imu_capture_stopped", imu.stats())
            wakeLock?.let { if (it.isHeld) it.release() }; wakeLock = null
            if (!storage) error = "存储剩余不足512MB，请导出数据后继续"
        }
    }

    private fun onImu(header: JSONObject, payload: ByteArray) {
        // stop() flushes already sampled records after capturing has become false.
        if (dataLoggingFailed.get()) return
        synchronized(logLock) {
            directory?.let { dir ->
                try {
                    val file = File(dir, "imu.bin")
                    val offset = file.length()
                    file.appendBytes(payload)
                    val entry = dataJournal().envelope(JSONObject(header.toString()).put("file_offset", offset).put("byte_count", payload.size))
                    File(dir, "imu-batches.jsonl").appendText(entry.toString() + "\n")
                } catch (failure: Exception) { dataFailure("imu", failure) }
            }
        }
    }

    private fun onFrame(header: JSONObject, bytes: ByteArray) {
        if (!capturing || closed) return
        val version = epoch.get()
        try {
            val rotation = header.optInt("rotation")
            var uprightWidth = header.optInt("width")
            var uprightHeight = header.optInt("height")
            val jpeg = if (rotation % 360 == 0) bytes else {
                val source = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return
                val transformed = Bitmap.createBitmap(source, 0, 0, source.width, source.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
                uprightWidth = transformed.width
                uprightHeight = transformed.height
                val out = ByteArrayOutputStream()
                transformed.compress(Bitmap.CompressFormat.JPEG, 80, out)
                if (source !== transformed) source.recycle()
                transformed.recycle()
                out.toByteArray()
            }
            val at = header.optLong("ts_ns") / 1_000_000
            val id = "f${header.optLong("index")}-${at}"
            val flow = gate.analyze(jpeg, at)
            synchronized(logLock) {
                if (!capturing || version != epoch.get()) return
                directory?.let { dir ->
                    File(dir, "$id.jpg").writeBytes(jpeg)
                    val entry = dataJournal().envelope(JSONObject(header.toString()).put("frame_id", id)
                        .put("timestamp_ms", at).put("sensor_width", header.optInt("width"))
                        .put("sensor_height", header.optInt("height")).put("sensor_rotation", rotation)
                        .put("width", uprightWidth).put("height", uprightHeight).put("rotation", 0).put("flow", flow))
                    File(dir, "frames.jsonl").appendText(entry.toString() + "\n")
                }
                frames.add(StandaloneFrame(id, at, jpeg, if (flow.optBoolean("key")) flow.optString("reason") else null))
            }
        } catch (failure: Exception) { dataFailure("frames", failure) }
    }

    private fun event(name: String, data: JSONObject) {
        // Profile/key are never included in events. Provider exception bodies are never logged.
        when (name) {
            "speech_requested" -> speechRequested.incrementAndGet()
            "audio_started" -> speechStarted.incrementAndGet()
            "audio_done" -> speechDone.incrementAndGet()
            "audio_error", "speech_error", "audio_incomplete" -> speechErrors.incrementAndGet()
        }
        synchronized(logLock) {
            if (directory != null) try { dataJournal().append(name, data) }
            catch (failure: Exception) { dataFailure("events", failure) }
        }
        if (name in listOf("speech_error", "audio_error", "camera_error", "ptt_error", "video_frame_error")) main.post {
            if (name == "video_frame_error" && (data.optLong("video_generation") != videoSinkGeneration || videoRecorder?.wantsFrames != true)) return@post
            error = data.optString("message", data.optString("error", name)).take(160)
            if (error.contains("额度不足")) { invalidate(); engine.pause(now()); persist(); updateCapture() }
            if (name == "video_frame_error") stopVideoRecording("frame_error")
        }
    }

    private fun persist() {
        val state = engine.snapshot(now()).put("local_run", directory?.name ?: "")
        runCatching {
            recordTrialBoundary(state)
            if (directory != null) dataJournal().state(state, "persist")
            val file = AtomicFile(File(filesDir, "standalone-state.json"))
            val stream = file.startWrite()
            try { stream.write(state.toString().toByteArray()); file.finishWrite(stream) }
            catch (e: Exception) { file.failWrite(stream); throw e }
        }.onFailure { dataFailure("state", it) }
    }

    private fun dialoguePrompt(config: StandaloneProfile) = if (config.manualSource == "web")
        WebResearchPolicy.DIALOGUE_PROMPT + WebResearchPolicy.WEB_RULES
    else StandaloneManuals.DIALOGUE_PROMPT + StandaloneManuals.GROUNDED_RULES

    private fun writeExperimentConfiguration(config: StandaloneProfile) {
        val labels = File(filesDir, "experiment-context.json")
        val parsed = runCatching {
            if (labels.isFile) {
                require(labels.length() in 1..4096)
                ExperimentContext.validate(StandaloneApi.decodeObject(labels.readText()))
            } else ExperimentContext.defaults()
        }
        experimentLabels = parsed.getOrElse { ExperimentContext.defaults() }
        val metadata = JSONObject(experimentLabels.toString()).put("schema", "satori.experiment/1")
            .put("metadata_valid", parsed.isSuccess).put("condition", "demo-${config.assistanceMode}")
            .put("baseline_claim", "guided_vs_reactive_demo_only_not_equal_capability_mechanism_baseline")
            .put("app_version", BuildConfig.VERSION_NAME).put("app_version_code", BuildConfig.VERSION_CODE)
            .put("device_model", android.os.Build.MODEL).put("android_api", android.os.Build.VERSION.SDK_INT)
            .put("planner_model", config.planner).put("monitor_model", config.monitor).put("asr", "vosk-small-cn-0.22")
            .put("speech_turns", "half_duplex").put("manual_id", config.manualId)
            .put("manual_source", config.manualSource)
            .put("model_backup_configured", config.backupApiKey.isNotEmpty())
            .put("manual_hash_source", "tool_result.result.passages[].sha256")
            .put("spatial_capability", "hud_only_no_verified_world_anchor")
        runCatching { directory?.let { dir ->
            File(dir, "experiment.json").writeText(metadata.toString(2))
            File(dir, "dialogue-prompt.txt").writeText(dialoguePrompt(config))
            File(dir, "manual-guidance-rules.txt").writeText(if (config.manualSource == "web") WebResearchPolicy.WEB_RULES else StandaloneManuals.GROUNDED_RULES)
            if (config.manualSource == "web") File(dir, "search-plan-prompt.txt").writeText(WebResearchPolicy.PLAN_PROMPT)
        } }
            .onFailure { dataFailure("experiment_configuration", it) }
        event("experiment_configuration", metadata)
    }

    private fun recordTrialBoundary(state: JSONObject) {
        if (directory == null || dataLoggingFailed.get() || state.isNull("goal")) return
        val task = state.optString("task_instance_id")
        val phase = state.optString("phase")
        if (journaledTask.isEmpty() && task == restoredTaskId && phase in setOf("complete", "stopped")) {
            dataJournal().append("trial_terminal_snapshot", JSONObject().put("trial_id", task).put("task_instance_id", task)
                .put("phase", phase).put("reason", "restored_already_ended_no_new_end_time"))
            journaledTask = task; journaledPhase = phase
            return
        }
        if (task != journaledTask) {
            if (journaledTask.isNotEmpty() && journaledPhase !in setOf("stopped", "complete"))
                dataJournal().append("trial_ended", JSONObject().put("trial_id", journaledTask).put("task_instance_id", journaledTask)
                    .put("reason", "replaced_by_new_goal").put("independent_success", JSONObject.NULL))
            dataJournal().append(if (task == restoredTaskId) "trial_resumed" else "trial_started", JSONObject(experimentLabels.toString())
                .put("trial_id", task).put("task_instance_id", task).put("condition", "demo-${profile?.assistanceMode ?: "guided"}")
                .put("goal", state.getJSONObject("goal").optString("text")).put("manual_id", profile?.manualId.orEmpty())
                .put("independent_success", JSONObject.NULL))
            journaledTask = task
        }
        if (phase in setOf("stopped", "complete") && phase != journaledPhase)
            dataJournal().append("trial_ended", JSONObject().put("trial_id", task).put("task_instance_id", task)
                .put("reason", if (phase == "complete") "system_reported_complete" else "user_stopped")
                .put("independent_success", JSONObject.NULL))
        journaledPhase = phase
    }

    @Synchronized private fun dataJournal(): SessionJournal {
        return journal ?: SessionJournal({ directory }, BuildConfig.VERSION_NAME,
            Settings.Global.getInt(contentResolver, Settings.Global.BOOT_COUNT, -1)) {
            SessionClockSample(SystemClock.elapsedRealtimeNanos(), System.currentTimeMillis())
        }.also { journal = it }
    }

    private fun createDataSession(reason: String) {
        if (directory == null) {
            val parent = File(filesDir, "standalone-runs")
            check(parent.mkdirs() || parent.isDirectory)
            var serial = System.currentTimeMillis()
            var created: File? = null
            repeat(100) {
                if (created == null) {
                    val candidate = File(parent, "run-${serial++}")
                    if (candidate.mkdir()) created = candidate
                    else check(candidate.exists()) { "cannot create data directory" }
                }
            }
            directory = checkNotNull(created) { "cannot allocate data session" }
        }
        val dir = checkNotNull(directory)
        dataJournal().ensureSession(reason)
        File(dir, "stream-schema.json").writeText(JSONObject().put("schema", "satori.live-stream-schema/1")
            .put("clock_domain", "android.elapsedRealtimeNanos")
            .put("frames", "ts_ns=camera sample; elapsed_ns=log receipt; JPEG is upright")
            .put("imu", "25-byte big-endian: u8 type, i64 sample elapsed_ns, f32 x/y/z/w; type 1 accel m/s2, 4 gyro rad/s, 15 game rotation quaternion")
            .put("imu_requested_hz", 50).put("analysis_requested_fps", 2)
            .put("capture_gaps", "task pause, folded temples, decomposition/clarification wait and network loss can stop analysis; off-head and step-confirmation continue; video may retain camera")
            .put("screen", "MediaProjection MP4 video only; pair by capture_group_id; encoder PTS requires visible SYNC marker calibration")
            .put("interaction", "events.jsonl stores voice recognition, commands, dialogue request/response and tool provenance; received time is not speech onset")
            .put("raw_audio_saved_separately", false).toString(2))
        if (!File(dir, "prompt-contract.json").exists()) {
            assets.open("standalone-prompts.json").use { input -> File(dir, "prompt-contract.json").outputStream().use { input.copyTo(it) } }
        }
        profile?.let(::writeExperimentConfiguration)
        dataJournal().state(engine.snapshot(now()), "session_start")
    }

    private fun dataFailure(kind: String, failure: Throwable) {
        Log.e(TAG, "data write failed: $kind (${failure.javaClass.simpleName})")
        if (!dataLoggingFailed.compareAndSet(false, true) || closed) return
        main.post {
            error = DATA_ERROR
            invalidate()
            engine.pause(now())
            stopVideoRecording("data_log_failure", update = false)
            screenRecorder?.stop("data_log_failure")
            updateCapture()
        }
    }

    private fun displayError() = if (dataLoggingFailed.get()) DATA_ERROR else error

    private var lastClockAnchorNs = 0L
    private fun recordStatus() {
        if (directory == null || dataLoggingFailed.get()) return
        try {
            val at = SystemClock.elapsedRealtimeNanos()
            val journal = dataJournal()
            if (lastClockAnchorNs == 0L || at - lastClockAnchorNs >= 30_000_000_000L) {
                journal.clockAnchor("periodic")
                lastClockAnchorNs = at
            }
            val state = engine.snapshot(now())
            journal.state(state, "heartbeat")
            journal.status(JSONObject().put("revision", state.optLong("revision")).put("phase", state.optString("phase"))
                .put("active_desire_id", state.opt("active_desire_id") ?: JSONObject.NULL)
                .put("active_step_id", state.opt("active_step_id") ?: JSONObject.NULL)
                .put("paused_reason", state.opt("paused_reason") ?: JSONObject.NULL)
                .put("camera_active", capturing).put("analysis_active", analyzing).put("network", networkReady())
                .put("worn", worn ?: JSONObject.NULL).put("legs_open", legsOpen ?: JSONObject.NULL)
                .put("frames_encoded", if (::camera.isInitialized) camera.framesEncoded else 0)
                .put("imu", if (::imu.isInitialized) imu.stats() else JSONObject())
                .put("voice_active", voice?.active == true).put("voice_busy", voice?.busy == true).put("model_busy", inFlight)
                .put("screen", screenRecorder?.snapshot() ?: JSONObject().put("state", "idle"))
                .put("capture_group_id", captureGroupId).put("dual_stopping", endingDual)
                .put("listening", voice?.listening == true).put("listening_enabled", listeningEnabled)
                .put("dialogue_busy", dialogueBusy).put("manual_id", profile?.manualId.orEmpty()).put("assistance_mode", profile?.assistanceMode ?: "guided")
                .put("interaction_feedback", feedbackSnapshot())
                .put("speech_busy", speech?.busy == true).put("microphone", sharedMicrophone?.stats() ?: JSONObject())
                .put("last_spoken", lastSpoken).put("device_error", displayError())
                .put("video", videoRecorder?.snapshot() ?: JSONObject().put("state", "idle")))
        } catch (failure: Exception) { dataFailure("status", failure) }
    }

    private fun markSync() {
        if (dataLoggingFailed.get()) return
        if (syncSequence >= 9999) { error = "同步标记已满，请结束本次会话后重新打开应用"; return }
        try {
            if (directory == null) createDataSession("first_sync_marker")
            val identity = dataJournal().ensureSession("sync_marker")
            val at = SystemClock.elapsedRealtimeNanos()
            val id = "S${(++syncSequence).toString().padStart(4, '0')}"
            syncMarker = JSONObject().put("id", id).put("marker_id", id)
                .put("session_id", identity.getString("session_id")).put("session_tag", identity.getString("session_id").take(6).uppercase())
                .put("requested_elapsed_ns", at).put("visible_until_elapsed_ns", at + 1_500_000_000L)
                .put("first_draw_elapsed_ns", JSONObject.NULL)
            event("sync_marker_requested", JSONObject(syncMarker.toString()))
            if (!dataLoggingFailed.get()) StandaloneActivity.refreshVisiblePanel()
            else syncMarker = JSONObject()
        } catch (failure: Exception) { dataFailure("sync_marker", failure) }
    }

    fun onSyncMarkerDrawn(id: String, drawElapsedNs: Long) {
        if (Looper.myLooper() != Looper.getMainLooper()) { main.post { onSyncMarkerDrawn(id, drawElapsedNs) }; return }
        if (closed || dataLoggingFailed.get() || !id.matches(Regex("S[0-9]{4}")) || syncMarker.optString("id") != id || !syncMarker.isNull("first_draw_elapsed_ns")) return
        val requested = syncMarker.optLong("requested_elapsed_ns")
        if (drawElapsedNs < requested || drawElapsedNs > syncMarker.optLong("visible_until_elapsed_ns")) return
        try {
            if (dataJournal().ensureSession("sync_marker_draw").getString("session_id") != syncMarker.optString("session_id")) {
                syncMarker = JSONObject()
                StandaloneActivity.refreshVisiblePanel()
                return
            }
            val row = dataJournal().envelope(JSONObject(syncMarker.toString()).put("first_draw_elapsed_ns", drawElapsedNs)
                .put("draw_time_kind", "ui_draw_submission_not_final_video_presentation"))
            synchronized(logLock) { File(checkNotNull(directory), "sync-markers.jsonl").appendText(row.toString() + "\n") }
            syncMarker.put("first_draw_elapsed_ns", drawElapsedNs)
            event("sync_marker_drawn", row)
        } catch (failure: Exception) { dataFailure("sync_marker", failure) }
    }

    private fun associateVideoData(file: File) {
        try {
            if (directory == null) return
            val identity = dataJournal().ensureSession("video_association")
            val sidecar = File(file.parentFile, file.nameWithoutExtension + ".json")
            val data = JSONObject(sidecar.readText()).put("run_session_id", identity.getString("session_id"))
                .put("boot_id", identity.getString("boot_id")).put("data_run_id", directory!!.name)
                .put("clock_domain", "android.elapsedRealtimeNanos")
                .put("media_role", "first_person")
            videoCaptureMetadata.remove(file.canonicalPath)?.let { metadata ->
                metadata.keys().asSequence().forEach { key -> data.put(key, metadata.get(key)) }
            }
            val atomic = AtomicFile(sidecar)
            val output = atomic.startWrite()
            try { output.write(data.toString(2).toByteArray()); atomic.finishWrite(output) }
            catch (failure: Exception) { atomic.failWrite(output); throw failure }
            event("video_data_link", JSONObject().put("file", file.name).put("data_run_id", directory!!.name)
                .put("capture_start_elapsed_ns", data.optLong("capture_start_elapsed_ns")))
        } catch (failure: Exception) { dataFailure("video_association", failure) }
    }

    private fun networkReady(): Boolean {
        val manager = getSystemService(ConnectivityManager::class.java)
        val capabilities = manager.getNetworkCapabilities(manager.activeNetwork) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    private fun registerInputs() {
        val input = GlassesInputReceiver({ !closed && StandaloneActivity.isPanelVisible }, { _, aborted ->
            if (aborted) main.post { onVoiceKey("broadcast:legacy") }
        },
            { value -> main.post { worn = value; applyPresence() } },
            { value -> main.post { legsOpen = value; applyPresence() } },
            { action -> main.post { if (StandaloneActivity.isPanelVisible && !closed) when (action) {
                "com.android.action.ACTION_TWO_FINGER_DOUBLE_TAP" -> command("confirm")
                "com.android.action.ACTION_TWO_FINGER_SWIPE_BACK" -> command("repeat")
                "com.android.action.ACTION_SPRITE_BUTTON_DOUBLE_CLICK" -> command("voice")
            } } }, capturePhysicalLongPress = true, onAiSignal = { action, ordered, aborted -> main.post {
                inputEvent(JSONObject().put("source", "broadcast:$action").put("ordered", ordered).put("aborted", aborted)
                    .put("panel_visible", StandaloneActivity.isPanelVisible))
                if (aborted) onVoiceKey("broadcast:$action")
            } })
        ContextCompat.registerReceiver(this, input, IntentFilter().apply {
            priority = IntentFilter.SYSTEM_HIGH_PRIORITY
            GlassesInputReceiver.ACTIONS.forEach(::addAction)
        }, ContextCompat.RECEIVER_EXPORTED)
        receiver = input
    }

    private fun systemFlag(key: String): Boolean? = runCatching {
        when (Class.forName("android.os.SystemProperties").getMethod("get", String::class.java, String::class.java)
            .invoke(null, key, "") as String) { "1" -> true; "0" -> false; else -> null }
    }.getOrNull()

    private fun recordingFeedback(): String = when (videoRecorder?.snapshot()?.optString("state")) {
        null -> "第一视角录像服务尚未就绪。"
        "starting" -> "第一视角录像正在启动，请等开录提示。"
        "recording" -> "第一视角正在录像。要保存，请说“停止录像”。"
        "finalizing" -> "正在保存这段第一视角录像。"
        "error" -> "上一段录像失败。要重试，请说“开始录像”。"
        else -> "当前没有录像。要录第一视角，请说“开始录像”。"
    }

    private fun startVideoRecording() {
        if (dataLoggingFailed.get()) { error = DATA_ERROR; return }
        val recorder = videoRecorder ?: run { error = "录像服务尚未就绪"; return }
        if (recorder.wantsFrames) { speak(listOf(recordingFeedback())); return }
        if (recorder.snapshot().optString("state") == "finalizing") { error = "上一段录像正在保存，请稍候"; return }
        if (lastFolded == true) { error = "请先展开镜腿，再开始录像"; return }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            error = "有声录像需要麦克风权限"; return
        }
        if (filesDir.usableSpace < 512L * 1024 * 1024) { error = "存储空间不足，无法开始录像"; return }
        if (sharedMicrophone == null) { error = "麦克风服务尚未就绪"; return }
        error = ""
        try {
            val folder = File(filesDir, "first-person-videos").apply { check(mkdirs() || isDirectory) }
            val file = File(folder, "recording-${System.currentTimeMillis()}.mp4")
            recorder.start(file, withAudio = true)
            if (!recorder.wantsFrames) return
            val sessionPath = file.canonicalPath
            videoSessionPath = sessionPath
            foreground()
            val microphoneRequested = AtomicBoolean()
            videoSinkGeneration = camera.setVideoSink { frame ->
                if (recorder.wantsFrames && videoSessionPath == sessionPath) {
                    recorder.offerVideo(frame)
                    // Do not start audio seconds before a cold camera has produced its first image.
                    if (microphoneRequested.compareAndSet(false, true)) main.post { attachVideoMicrophone(sessionPath, recorder) }
                }
            }
            event("video_requested", JSONObject().put("name", file.name).put("with_audio", true))
            updateCapture()
        } catch (failure: Exception) {
            error = "无法开始录像：${failure.javaClass.simpleName}"
            stopVideoRecording("start_error")
        }
    }

    /** The Activity supplies one fresh, OS-approved token. No token is persisted or logged. */
    fun reportCaptureConsent(stage: String, failureKind: String = "") {
        if (stage !in setOf("requested", "granted", "rejected", "ui_unavailable")) return
        if (stage == "ui_unavailable") error = "系统未提供录屏授权界面，双录制尚未开始"
        if (stage == "rejected") error = "录屏授权已取消，双录制尚未开始"
        if (stage == "requested" || stage == "granted") error = ""
        event("screen_consent", JSONObject().put("stage", stage).put("capture_started", false)
            .put("error_kind", failureKind.take(80)))
    }

    fun startResearchRecording(resultCode: Int, authorization: Intent, width: Int, height: Int, density: Int) {
        if (Looper.myLooper() != Looper.getMainLooper()) { main.post { startResearchRecording(resultCode, authorization, width, height, density) }; return }
        if (closed || dataLoggingFailed.get() || lastFolded == true || screenRecorder?.active == true ||
            videoRecorder?.snapshot()?.optString("state") in setOf("starting", "recording", "finalizing")) {
            error = "请展开镜腿并先结束已有录像，再开始双录制"
            return
        }
        if (filesDir.usableSpace < 512L * 1024 * 1024) { error = "存储不足，无法双录制"; return }
        try {
            val identity = dataJournal().ensureSession("dual_recording")
            val group = "capture-${System.currentTimeMillis()}"
            val folder = File(filesDir, "screen-recordings").apply { check(mkdirs() || isDirectory) }
            val file = File(folder, "screen-${System.currentTimeMillis()}.mp4")
            captureGroupId = group; pairedScreenPath = file.canonicalPath; endingDual = false
            val metadata = JSONObject().put("run_session_id", identity.getString("session_id"))
                .put("boot_id", identity.getString("boot_id")).put("data_run_id", directory!!.name)
                .put("clock_domain", SessionJournal.CLOCK_DOMAIN).put("capture_group_id", group)
            projectionForeground = true
            foreground()
            screenRecorder?.start(file, resultCode, authorization, width, height, density, metadata)
            if (screenRecorder?.active != true) { projectionForeground = false; foreground(); return }
            startVideoRecording()
            pairedVideoPath = videoSessionPath
            if (videoRecorder?.wantsFrames != true) { screenRecorder?.stop("camera_start_failed"); return }
            videoCaptureMetadata[pairedVideoPath!!] = JSONObject(metadata.toString())
            event("dual_recording_requested", metadata.put("screen_file", file.name)
                .put("first_person_file", File(pairedVideoPath!!).name).put("screen_audio", false))
        } catch (_: Exception) {
            error = "双录制未能启动，请检查授权和设备编码能力"
            screenRecorder?.stop("start_failed")
            if (screenRecorder?.active != true) { projectionForeground = false; foreground() }
        }
    }

    fun stopResearchRecording() {
        if (Looper.myLooper() != Looper.getMainLooper()) { main.post { stopResearchRecording() }; return }
        if (endingDual) return
        endingDual = true
        val group = captureGroupId
        markSync()
        event("dual_recording_stop_requested", JSONObject().put("capture_group_id", group))
        // Leave the end SYNC marker on the screen before finalizing both streams.
        main.postDelayed({
            if (!closed && group == captureGroupId) {
                screenRecorder?.stop("user")
                if (videoSessionPath == pairedVideoPath) stopVideoRecording("user")
            }
        }, 1000)
    }

    private fun onScreenEvent(name: String, data: JSONObject) {
        event(name, data)
        if (name == "screen_rejected") { error = data.optString("error"); return }
        if (data.optString("path") != pairedScreenPath) return
        when (name) {
            "screen_started" -> if (!closed) { markSync(); foreground() }
            "screen_saved", "screen_error" -> {
                projectionForeground = false
                if (videoSessionPath == pairedVideoPath && videoRecorder?.wantsFrames == true)
                    stopVideoRecording("screen_${data.optString("reason", name)}")
                if (name == "screen_error") {
                    error = data.optString("error", "录屏失败")
                    if (videoSessionPath == pairedVideoPath) stopVideoRecording("screen_error")
                }
                else if (!closed) speak(listOf("屏幕录制已保存。"))
                if (!closed) foreground()
            }
        }
    }

    private fun attachVideoMicrophone(sessionPath: String, recorder: FirstPersonRecorder) {
        if (closed || videoSessionPath != sessionPath || !recorder.wantsFrames) return
        try {
            videoMicrophone = sharedMicrophone!!.subscribe({ atNs, pcm ->
                if (videoSessionPath == sessionPath) recorder.offerAudio(atNs, pcm)
            }, { name, data ->
                if (name == "ptt_error") main.post { if (videoSessionPath == sessionPath) {
                    error = "录像收音中断，请重新开始录像"
                    event("video_microphone_error", JSONObject(data.toString()))
                    stopVideoRecording("microphone_error")
                    speak(listOf("录像收音中断，已结束本段录像。"))
                } }
            }, raw = true)
        } catch (_: Exception) {
            error = "录像麦克风启动失败"
            stopVideoRecording("microphone_start_error")
        }
    }

    private fun stopVideoRecording(reason: String, update: Boolean = true) {
        val wasRecording = videoRecorder?.wantsFrames == true
        if (wasRecording && videoSessionPath == pairedVideoPath && screenRecorder?.active == true) screenRecorder?.stop("first_person_$reason")
        videoMicrophone?.close(); videoMicrophone = null
        if (::camera.isInitialized) camera.setVideoSink(null)
        videoRecorder?.stop()
        if (wasRecording) {
            event("video_stop_requested", JSONObject().put("reason", reason))
            if (reason == "user") speak(listOf("正在保存录像，请稍候。"))
        }
        if (update && !closed) updateCapture()
    }

    private fun onVideoEvent(name: String, data: JSONObject) {
        event(name, data)
        val currentSession = data.optString("path") == videoSessionPath
        when (name) {
            "video_started" -> if (!closed && currentSession) { foreground(); speak(listOf("已开始第一视角录像。")) }
            "video_saved" -> {
                if (currentSession) {
                    videoMicrophone?.close(); videoMicrophone = null
                    if (::camera.isInitialized) camera.setVideoSink(null)
                    videoSessionPath = null
                }
                val file = File(data.optString("path"))
                val root = File(filesDir, "first-person-videos").canonicalFile
                if (file.isFile && file.canonicalFile.parentFile == root && file.name.matches(Regex("recording-[0-9]+\\.mp4"))) {
                    associateVideoData(file)
                    lastVideoName = file.name
                    lastVideoUri = null
                    getSharedPreferences("video_recordings", MODE_PRIVATE).edit().putString("last_name", file.name).remove("last_uri").apply()
                    publishVideo(file)
                }
                if (!closed && currentSession) { foreground(); speak(listOf("第一视角录像已保存。")); updateCapture() }
            }
            "video_error" -> {
                if (!currentSession) return
                if (videoSessionPath == pairedVideoPath) screenRecorder?.stop("camera_error")
                error = data.optString("error", "录像失败，请查看保存状态")
                videoMicrophone?.close(); videoMicrophone = null
                if (::camera.isInitialized) camera.setVideoSink(null)
                videoSessionPath = null
                if (!closed) { foreground(); speak(listOf("录像遇到问题，已停止。")); updateCapture() }
            }
        }
    }

    /** The private finalized MP4 remains authoritative; publishing failure never removes it. */
    private fun publishVideo(file: File) {
        if (videoPublisher.isShutdown) return
        videoPublisher.execute {
            var created: Uri? = null
            try {
                check(file.usableSpace > file.length() + 64L * 1024 * 1024) { "insufficient space for media copy" }
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Satori")
                    put(MediaStore.Video.Media.IS_PENDING, 1)
                }
                val uri = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("media insert failed")
                created = uri
                contentResolver.openOutputStream(uri)?.use { output -> file.inputStream().use { it.copyTo(output) } }
                    ?: throw IllegalStateException("media output unavailable")
                check(contentResolver.update(uri, ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }, null, null) == 1)
                main.post {
                    val prefs = getSharedPreferences("video_recordings", MODE_PRIVATE)
                    if (prefs.getString("last_name", "") == file.name) {
                        prefs.edit().putString("last_uri", uri.toString()).apply()
                        if (!closed && lastVideoName == file.name) lastVideoUri = uri.toString()
                    }
                }
            } catch (_: Exception) {
                created?.let { runCatching { contentResolver.delete(it, null, null) } }
                main.post { if (!closed) event("video_publish_failed", JSONObject().put("name", file.name)) }
            }
        }
    }

    private fun foreground() {
        if (instance !== this) return // Isolated component tests do not own an Android service token.
        getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel("standalone", "Satori 独立助手", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java).setAction(Intent.ACTION_MAIN)
            .addCategory(Intent.CATEGORY_LAUNCHER), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val recording = videoRecorder?.wantsFrames == true
        val notification = Notification.Builder(this, "standalone").setSmallIcon(android.R.drawable.presence_video_online)
            .setContentTitle(if (screenRecorder?.active == true) "Satori 第一视角＋屏幕录制" else if (recording) "Satori 有声录像" else "Satori 独立助手")
            .setContentText(if (recording) "正在录像，点击返回后停止并保存" else if (listeningEnabled) "语音自动待听；说“停止监听”关闭" else "语音待听已关闭，右键重新开启")
            .setContentIntent(open).setOngoing(true).build()
        var types = ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
        if (projectionForeground) types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        ServiceCompat.startForeground(this, 47803, notification, types)
    }

    override fun onDestroy() {
        closed = true
        if (journaledTask.isNotEmpty() && journaledPhase !in setOf("stopped", "complete"))
            event("trial_interrupted", JSONObject().put("trial_id", journaledTask).put("reason", "service_stop").put("independent_success", JSONObject.NULL))
        stopVideoRecording("service_stop", update = false)
        screenRecorder?.close()
        invalidate()
        runCatching { receiveTone?.release() }; receiveTone = null
        main.removeCallbacksAndMessages(null)
        receiver?.let { unregisterReceiver(it) }
        engine.pause(now()); persist()
        camera.shutdown(); imu.stop()
        speech?.close(); voice?.close(); api?.close(); speechApi?.close(); voiceApi?.close(); dialogueApi?.close()
        sharedMicrophone?.close(); videoRecorder?.close()
        videoPublisher.shutdown()
        worker.shutdownNow()
        // The gate is used only on the camera executor. Camera shutdown stops new work first.
        synchronized(logLock) { gate.close() }
        wakeLock?.let { if (it.isHeld) it.release() }
        instance = null
        super.onDestroy()
    }

    companion object {
        private const val TAG = "SatoriStandalone"
        private const val DATA_ERROR = "实时数据保存失败，已暂停采集，请检查存储后重新打开应用"
        private val QUOTA_ERRORS = setOf("insufficient_user_quota", "insufficient_quota", "token_quota_not_enough")
        private fun now() = SystemClock.elapsedRealtime()
        @Volatile var instance: StandaloneService? = null
            private set
    }
}
