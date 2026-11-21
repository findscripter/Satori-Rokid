package com.satori.glasses.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.Context
import android.media.projection.MediaProjectionManager
import android.graphics.Color
import android.graphics.Canvas
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.TypedValue
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import org.json.JSONArray
import java.io.File

/** Read-only glasses HUD. All interaction uses the official phone remote or spoken commands. */
class StandaloneActivity : Activity() {
    private val main = Handler(Looper.getMainLooper())
    private lateinit var statusView: TextView
    private lateinit var videoView: TextView
    private var interactionView: TextView? = null
    private lateinit var chatRows: LinearLayout
    private lateinit var conversationScroll: ScrollView
    private var renderedChatKey = ""
    private var lastUserMessageId = ""
    private var panelNotice = ""
    private var panelNoticeUntil = 0L
    private var screenConsentPending = false
    private lateinit var syncView: SyncMarkerView
    private var imported = ""
    private val gestures = HashMap<Int, Gesture>()
    private data class Gesture(val action: String, val pending: String?, val downTime: Long = 0)
    private val confirmationGate = VoiceKeyGate { SystemClock.elapsedRealtime() }
    private val recordingKeyGate = RecordingKeyGate()
    private var touchGesture: Gesture? = null
    private var touchX = 0f
    private var touchY = 0f
    private var touchDownAt = 0L
    private val refresh = object : Runnable {
        override fun run() { render(); main.postDelayed(this, 1000) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // No credential is ever displayed or edited in this HUD. Official screen recording is allowed.
        window.clearFlags(WindowManager.LayoutParams.FLAG_SECURE)
        importPrivateProfile()
        val scroll = ScrollView(this).apply { setBackgroundColor(Color.BLACK); tag = "satori_hud_scroll" }
        conversationScroll = scroll
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(16))
            tag = "satori_readonly_hud"
        }
        fun line(size: Float, name: String): TextView = TextView(this).apply {
            textSize = size; setTextColor(Color.GREEN)
            setPadding(0, dp(6), 0, dp(6)); tag = name; visibility = View.GONE
            isClickable = false; isLongClickable = false; isFocusable = false
        }
        chatRows = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; tag = "chat_messages" }
        layout.addView(chatRows, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        statusView = line(16f, "hud_status").also { layout.addView(it) }
        videoView = line(14f, "hud_video").apply { maxLines = 1; setPadding(dp(16), dp(4), dp(16), dp(4)) }
        scroll.addView(layout)
        val screen = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.BLACK) }
        syncView = SyncMarkerView(this).apply {
            tag = "hud_sync_marker"; setTextColor(Color.WHITE); setBackgroundColor(Color.BLACK)
            setTypeface(typeface, Typeface.BOLD); gravity = Gravity.CENTER
            maxLines = 1; setHorizontallyScrolling(false)
            setPadding(dp(8), dp(5), dp(8), dp(5)); minimumHeight = dp(58)
            setAutoSizeTextTypeUniformWithConfiguration(18, 38, 1, TypedValue.COMPLEX_UNIT_SP)
            isClickable = false; isLongClickable = false; isFocusable = false; visibility = View.GONE
        }
        // Recording indicators remain visible while conversation history scrolls.
        screen.addView(syncView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        screen.addView(videoView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        screen.addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(screen)
        if (imported.contains("失败")) showNotice(imported)
        requestAndStart()
    }

    override fun onResume() {
        super.onResume()
        visibleInstance = this
        isPanelVisible = true
        main.post(refresh)
    }
    override fun onPause() {
        recordingKeyGate.reset()
        main.removeCallbacks(refresh); gestures.clear(); touchGesture = null
        clearVisibleInstance()
        super.onPause()
    }
    override fun onDestroy() {
        main.removeCallbacks(refresh); gestures.clear(); touchGesture = null
        clearVisibleInstance()
        super.onDestroy()
    }
    private fun clearVisibleInstance() {
        if (visibleInstance === this) { visibleInstance = null; isPanelVisible = false }
    }

    private fun requestAndStart() {
        val missing = listOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
            .filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), 32)
        else startForegroundService(Intent(this, StandaloneService::class.java))
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != 32) return
        if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            panelNotice = ""
            panelNoticeUntil = 0
            requestAndStart()
        }
        else {
            showNotice("未获得相机或麦克风权限")
            panelNoticeUntil = Long.MAX_VALUE
        }
    }

    private fun render() {
        val state = StandaloneService.instance?.snapshot()
        if (state == null) {
            chatRows.removeAllViews(); interactionView = null; renderedChatKey = ""
            setContent(videoView, "")
            syncView.hide()
            setContent(statusView, if (SystemClock.elapsedRealtime() < panelNoticeUntil) panelNotice else "")
            return
        }
        renderState(state)
    }

    private fun renderState(state: JSONObject) {
        renderSyncMarker(state.optJSONObject("sync_marker"))
        renderChat(state)
        val error = state.optString("device_error").ifBlank {
            if (!state.optBoolean("dialogue_busy") && !state.optBoolean("busy") && !state.isNull("error")) state.optString("error") else ""
        }
        val notice = error.ifBlank {
            if (state.optBoolean("automatic_voice_unavailable")) "语音识别暂不可用"
            else if (SystemClock.elapsedRealtime() < panelNoticeUntil) panelNotice else ""
        }
        val messages = state.optJSONArray("chat_messages")
        val lastMessage = messages?.optJSONObject(messages.length() - 1)
        setContent(statusView, if (lastMessage?.optString("role") == "assistant" && lastMessage.optString("text") == notice) "" else notice)
        val video = state.optJSONObject("video")
        val screen = state.optJSONObject("screen")
        if (video?.optString("state") in setOf("starting", "recording", "finalizing")) {
            val seconds = video!!.optLong("captured_duration_ms", video.optLong("duration_ms")) / 1000
            videoView.text = if (video.optString("state") == "finalizing") "正在保存录像…"
                else "录像中 ${(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}"
            videoView.visibility = View.VISIBLE
        } else videoView.visibility = View.GONE
        if (screen?.optString("state") in setOf("starting", "recording", "finalizing")) {
            val label = when (screen!!.optString("state")) {
                "finalizing" -> "正在保存屏幕录像"
                "starting" -> "正在开始屏幕录制"
                else -> "屏幕录制中"
            }
            videoView.text = "${videoView.text.toString().takeIf { videoView.visibility == View.VISIBLE }.orEmpty()}  $label".trim()
            videoView.visibility = View.VISIBLE
        }
    }

    private fun renderChat(state: JSONObject) {
        val messages = state.optJSONArray("chat_messages") ?: JSONArray()
        val feedback = state.optJSONObject("interaction_feedback")
        val inputBusy = state.optBoolean("recording") || state.optBoolean("voice_busy")
        val progress = if (inputBusy || !feedback?.optString("request_id").isNullOrBlank()) StandaloneInteractionFeedback.text(state) else ""
        val activeTurn = if (inputBusy) "" else feedback?.optString("turn_id").orEmpty()
        val pending = if (messages.length() == 0) StandaloneInteractionFeedback.reply(state) else ""
        val width = (chatRows.width.takeIf { it > 0 } ?: (resources.displayMetrics.widthPixels - dp(32))).coerceAtLeast(dp(120))
        val key = messages.toString() + "|$activeTurn|${progress.isNotBlank()}|$pending|$width"
        if (key != renderedChatKey) {
            val previousScroll = conversationScroll.scrollY
            val atBottom = conversationScroll.scrollY + conversationScroll.height >= (conversationScroll.getChildAt(0)?.height ?: 0) - dp(48)
            val anchor = (0 until chatRows.childCount).map { chatRows.getChildAt(it) }
                .firstOrNull { chatRows.top + it.bottom > previousScroll }
            val anchorTag = anchor?.tag
            val anchorOffset = anchor?.let { chatRows.top + it.top - previousScroll } ?: 0
            var newestUser = ""
            chatRows.removeAllViews()
            interactionView = null
            for (i in 0 until minOf(messages.length(), 40)) {
                val row = messages.optJSONObject(i) ?: continue
                val role = row.optString("role")
                if (role !in setOf("user", "assistant")) continue
                val id = row.optString("id")
                if (role == "user") newestUser = id
                val text = row.optString("text")
                when {
                    text.isNotBlank() -> addChatBubble(role, id, text, width)
                    role == "assistant" && row.optString("state") == "pending" &&
                        row.optString("turn_id") == activeTurn && activeTurn.isNotBlank() && progress.isNotBlank() -> {
                        interactionView = addChatBubble(role, id, progress, width, waiting = true)
                    }
                    role == "assistant" && row.optString("state") == "cancelled" -> addChatBubble(role, id, "已取消", width)
                    role == "assistant" && row.optString("state") == "error" -> addChatBubble(role, id, "这次处理未完成", width)
                }
            }
            if (pending.isNotBlank()) addChatBubble("assistant", "pending-confirmation", pending, width)
            if (progress.isNotBlank() && interactionView == null)
                interactionView = addChatBubble("assistant", "active-request", progress, width, waiting = true)
            renderedChatKey = key
            val userSent = newestUser.isNotBlank() && newestUser != lastUserMessageId
            lastUserMessageId = newestUser
            if (atBottom || userSent) conversationScroll.post { conversationScroll.fullScroll(View.FOCUS_DOWN) }
            else if (anchorTag != null) conversationScroll.post {
                if (conversationScroll.scrollY == previousScroll) chatRows.findViewWithTag<View>(anchorTag)?.let {
                    conversationScroll.scrollTo(0, (chatRows.top + it.top - anchorOffset).coerceAtLeast(0))
                }
            }
        }
        interactionView?.let {
            if (it.text.toString() != progress) it.text = progress
            it.contentDescription = "助手：$progress"
        }
    }

    private fun addChatBubble(role: String, id: String, text: String, availableWidth: Int, waiting: Boolean = false): TextView {
        val user = role == "user"
        return TextView(this).apply {
            this.text = text
            tag = if (waiting) "hud_interaction" else "chat_${role}_$id"
            contentDescription = "${if (user) "你" else "助手"}：$text"
            textSize = if (user) 18f else 19f
            setTextColor(Color.GREEN)
            alpha = if (waiting) 0.8f else 1f
            maxWidth = (availableWidth * if (user) 0.82f else 0.92f).toInt()
            setPadding(dp(12), dp(9), dp(12), dp(9))
            background = GradientDrawable().apply {
                setColor(if (user) Color.rgb(0, 30, 0) else Color.BLACK)
                cornerRadius = dp(12).toFloat()
                setStroke(dp(1), if (user) Color.rgb(0, 130, 0) else Color.rgb(0, 65, 0))
            }
            isClickable = false; isLongClickable = false; isFocusable = false
            chatRows.addView(this, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                gravity = if (user) Gravity.END else Gravity.START
                bottomMargin = dp(10)
            })
        }
    }

    private fun setContent(view: TextView, text: String) {
        view.text = text
        view.visibility = if (text.isBlank()) View.GONE else View.VISIBLE
    }

    private fun showNotice(text: String) {
        panelNotice = text
        panelNoticeUntil = SystemClock.elapsedRealtime() + 8000
        setContent(statusView, text)
    }

    private fun renderSyncMarker(marker: JSONObject?) {
        val id = marker?.optString("id").orEmpty()
        val requested = marker?.optLong("requested_elapsed_ns") ?: 0L
        val until = marker?.optLong("visible_until_elapsed_ns") ?: 0L
        if (id.matches(Regex("S[0-9]{4}")) && requested > 0 && SystemClock.elapsedRealtimeNanos() < until) {
            val session = marker!!.optString("session_tag").take(6).uppercase()
            syncView.show(id, requested, until, if (session.isBlank()) "SYNC $id" else "SYNC $id ·$session")
        } else syncView.hide()
    }

    /** First actual View drawing uses the same elapsed-realtime clock as capture and the journal.
     * It is a draw-submission anchor; it does not assert display scan-out or a phone video frame PTS.
     */
    private inner class SyncMarkerView(context: Context) : TextView(context) {
        private var markerId: String? = null
        private var requestedNs = 0L
        private var untilNs = 0L
        private var reportedIdentity: String? = null

        fun show(id: String, requested: Long, until: Long, caption: String) {
            markerId = id; requestedNs = requested; untilNs = until
            if (text.toString() != caption) text = caption
            visibility = View.VISIBLE
            invalidate()
        }
        fun hide() { markerId = null; visibility = View.GONE }

        override fun onDraw(canvas: Canvas) {
            val id = markerId ?: return
            val requested = requestedNs
            val until = untilNs
            val drawElapsedNs = SystemClock.elapsedRealtimeNanos()
            if (!isShown || !isAttachedToWindow || visibleInstance !== this@StandaloneActivity ||
                drawElapsedNs < requested || drawElapsedNs > until) return
            super.onDraw(canvas)
            val identity = "$id:$requested"
            if (reportedIdentity == identity) return
            reportedIdentity = identity
            // No I/O or view-tree mutation during drawing. A replaced marker cannot claim this draw.
            main.post {
                if (markerId != id || requestedNs != requested || visibleInstance !== this@StandaloneActivity) return@post
                val service = StandaloneService.instance ?: return@post
                val current = service.snapshot().optJSONObject("sync_marker") ?: return@post
                if (current.optString("id") == id && current.optLong("requested_elapsed_ns") == requested)
                    service.onSyncMarkerDrawn(id, drawElapsedNs)
            }
        }
    }

    private fun pendingSignature(state: JSONObject): String? {
        if (state.optString("phase") !in setOf("awaiting_plan_confirmation", "awaiting_step_confirmation")) return null
        val pending = state.optJSONObject("pending") ?: return null
        if (!pending.optString("kind").startsWith("confirm")) return null
        val steps = pending.optJSONArray("step_ids")
        val ids = if (steps == null) emptyList() else (0 until steps.length()).map { steps.optString(it) }.sorted()
        return "${state.optString("phase")}|${pending.optString("kind")}|${ids.joinToString(",")}|${pending.optLong("prompted_ms")}"
    }

    private fun confirmationGesture(state: JSONObject): Gesture {
        val pending = pendingSignature(state)
        return Gesture(when {
            state.optBoolean("recording") -> "stop_voice"
            pending != null -> "confirm"
            else -> "voice_if_unchanged"
        }, pending)
    }

    private fun commitConfirmation(gesture: Gesture, source: String, service: StandaloneService, state: JSONObject) {
        val eligible = when (gesture.action) {
            "confirm", "voice_if_unchanged" -> pendingSignature(state) == gesture.pending
            "stop_voice" -> state.optBoolean("recording")
            else -> false
        }
        if (!eligible || !confirmationGate.accept(source)) return
        Log.i("SatoriHud", "confirm_input source=$source action=${gesture.action}")
        when (gesture.action) {
            "confirm" -> service.command("confirm")
            "stop_voice" -> service.command("voice")
            "voice_if_unchanged" -> service.onVoiceKey("remote:confirm")
        }
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        val wasTracking = touchGesture != null
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                touchGesture = if (event.pointerCount == 1) StandaloneService.instance?.snapshot()?.let(::confirmationGesture) else null
                touchX = event.x; touchY = event.y; touchDownAt = event.eventTime
            }
            MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_CANCEL -> touchGesture = null
            MotionEvent.ACTION_MOVE -> {
                val slop = ViewConfiguration.get(this).scaledTouchSlop
                if (event.pointerCount != 1 || kotlin.math.abs(event.x - touchX) > slop || kotlin.math.abs(event.y - touchY) > slop)
                    touchGesture = null
            }
            MotionEvent.ACTION_UP -> {
                val gesture = touchGesture
                touchGesture = null
                if (gesture != null && event.pointerCount == 1 &&
                    event.eventTime - touchDownAt in 0..ViewConfiguration.getLongPressTimeout().toLong() &&
                    kotlin.math.abs(event.x - touchX) <= ViewConfiguration.get(this).scaledTouchSlop &&
                    kotlin.math.abs(event.y - touchY) <= ViewConfiguration.get(this).scaledTouchSlop) {
                    StandaloneService.instance?.let { commitConfirmation(gesture, "touch:tap", it, it.snapshot()) }
                    render()
                }
            }
        }
        // ScrollView retains the full motion stream; moved/multitouch gestures never confirm.
        return super.dispatchTouchEvent(event) || wasTracking || touchGesture != null
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val key = event.keyCode
        val downKeys = setOf(KeyEvent.KEYCODE_PROG_BLUE, KeyEvent.KEYCODE_F1, KeyEvent.KEYCODE_HEADSETHOOK)
        val confirmKeys = setOf(KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_NUMPAD_ENTER)
        val remoteKeys = confirmKeys + setOf(KeyEvent.KEYCODE_DPAD_RIGHT, KeyEvent.KEYCODE_DPAD_LEFT, KeyEvent.KEYCODE_BACK, KeyEvent.KEYCODE_ESCAPE)
        if (key !in downKeys && key !in remoteKeys) return super.dispatchKeyEvent(event)
        // Only handled control keys are logged; no characters, input text or credentials.
        Log.i("SatoriHud", "control_key code=$key action=${event.action} repeat=${event.repeatCount}")
        if (event.action == KeyEvent.ACTION_DOWN) {
            if (key != KeyEvent.KEYCODE_DPAD_LEFT) recordingKeyGate.reset()
            if (event.repeatCount == 0 && gestures[key]?.downTime != event.downTime) {
                val service = StandaloneService.instance
                val state = service?.snapshot()
                if (key in downKeys) {
                    gestures[key] = Gesture("consumed", null, event.downTime)
                    service?.onVoiceKey("key:$key")
                } else if (state != null) {
                    val gesture = when {
                        key == KeyEvent.KEYCODE_DPAD_LEFT -> Gesture("mark_sync", null)
                        key == KeyEvent.KEYCODE_DPAD_RIGHT -> Gesture("voice", null)
                        key == KeyEvent.KEYCODE_BACK || key == KeyEvent.KEYCODE_ESCAPE ->
                            Gesture(if (state.optString("paused_reason") == "user") "resume" else "pause", null)
                        else -> confirmationGesture(state)
                    }
                    gestures[key] = gesture.copy(downTime = event.downTime)
                }
            }
            return true
        }
        if (event.action == KeyEvent.ACTION_UP) {
            val gesture = gestures.remove(key) ?: return true
            if (event.isCanceled || gesture.action == "consumed") return true
            val service = StandaloneService.instance ?: return true
            val state = service.snapshot()
            if (key in confirmKeys && event.eventTime - gesture.downTime >= 650) {
                toggleResearchRecording(service, state)
                return true
            }
            when (gesture.action) {
                "confirm", "stop_voice", "voice_if_unchanged" -> commitConfirmation(gesture, "key:$key", service, state)
                "voice" -> service.onVoiceKey("remote:voice")
                "mark_sync" -> {
                    if (recordingKeyGate.press(event.eventTime)) {
                        Log.i("SatoriHud", "record_input source=remote:left_double_press")
                        if (state.optJSONObject("screen")?.optString("state") in setOf("starting", "recording", "finalizing")) {
                            service.stopResearchRecording()
                        } else when (state.optJSONObject("video")?.optString("state")) {
                            "starting", "recording" -> service.command("record_stop")
                            "finalizing" -> showNotice("正在保存录像")
                            else -> service.command("record_start")
                        }
                    } else service.command("mark_sync")
                }
                "pause" -> if (state.optString("paused_reason") != "user") service.command("pause")
                "resume" -> if (state.optString("paused_reason") == "user") service.command("resume")
            }
            render()
            return true
        }
        return true
    }

    private fun toggleResearchRecording(service: StandaloneService, state: JSONObject) {
        if (state.optJSONObject("screen")?.optString("state") in setOf("starting", "recording", "finalizing")) {
            service.stopResearchRecording()
            return
        }
        if (screenConsentPending) return
        if (state.optJSONObject("video")?.optString("state") in setOf("starting", "recording", "finalizing")) {
            showNotice("已有第一视角录像，请先停止录像")
            return
        }
        screenConsentPending = true
        service.reportCaptureConsent("requested")
        try {
            val manager = getSystemService(MediaProjectionManager::class.java)
            @Suppress("DEPRECATION")
            startActivityForResult(manager.createScreenCaptureIntent(), 61)
        } catch (failure: Exception) {
            screenConsentPending = false
            service.reportCaptureConsent("ui_unavailable", failure.javaClass.simpleName)
            showNotice("设备未提供录屏授权入口")
        }
    }

    @Deprecated("Activity result API retained for the existing native HUD")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != 61) return
        if (!screenConsentPending) return
        screenConsentPending = false
        if (resultCode != RESULT_OK || data == null) {
            StandaloneService.instance?.reportCaptureConsent("rejected")
            showNotice("录屏授权已取消，未开始双录制"); return
        }
        StandaloneService.instance?.reportCaptureConsent("granted")
        val bounds = windowManager.maximumWindowMetrics.bounds
        StandaloneService.instance?.startResearchRecording(resultCode, data, bounds.width(), bounds.height(), resources.displayMetrics.densityDpi)
    }

    /** Private ADB provisioning only; no exported intent extra or HUD field exposes the credential. */
    private fun importPrivateProfile() {
        val file = File(filesDir, "standalone-import.json")
        if (file.isFile) try {
            require(file.length() in 1..20000)
            val data = JSONObject(file.readText())
            val settings = StandaloneSettings(this)
            settings.save(settings.load().copy(baseUrl = data.getString("base_url"), apiKey = data.getString("api_key"),
                planner = data.getString("planner"), monitor = data.getString("monitor"), asrModel = data.optString("asr_model", "whisper-1"),
                ttsModel = data.optString("tts_model", "gpt-4o-mini-tts"), ttsVoice = data.optString("tts_voice", "alloy"),
                backupApiKey = data.optString("backup_api_key")))
            imported = "接口已配置"
        } catch (_: Exception) { imported = "接口配置导入失败" }
        finally { file.delete() }
        val toolsFile = File(filesDir, "standalone-tools-import.json")
        if (toolsFile.isFile) try {
            require(toolsFile.length() in 1..16000)
            val data = StandaloneApi.decodeObject(toolsFile.readText())
            val settings = StandaloneSettings(this)
            val old = settings.load()
            settings.save(old.copy(toolsBaseUrl = data.optString("tools_base_url"), toolsToken = data.optString("tools_token"),
                manualId = data.getString("manual_id"), assistanceMode = data.optString("assistance_mode", "guided"),
                manualSource = data.optString("manual_source", "remote_api")))
            imported = "手册来源已配置"
        } catch (_: Exception) { imported = "手册服务导入失败，保留原配置" }
        finally { toolsFile.delete() }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    companion object {
        private var visibleInstance: StandaloneActivity? = null // main-thread identity, cleared on pause/destroy
        @Volatile var isPanelVisible = false
            private set

        /** Service commands can refresh the visible marker without waiting for the one-second HUD tick. */
        fun refreshVisiblePanel() {
            Handler(Looper.getMainLooper()).post {
                val panel = visibleInstance ?: return@post
                if (!isPanelVisible || panel.isFinishing || panel.isDestroyed) return@post
                panel.render()
                panel.window.decorView.invalidate()
            }
        }
    }
}
