package com.satori.glasses.capture

import org.json.JSONObject

/** Elapsed time describes waiting already observed, never a completion estimate. */
object StandaloneInteractionFeedback {
    fun text(state: JSONObject): String {
        if (state.optBoolean("recording")) return "正在听你说话…"
        if (state.optBoolean("voice_busy")) return "正在识别语音…"
        val feedback = state.optJSONObject("interaction_feedback")
        if (feedback == null || feedback.optString("request_id").isBlank())
            return if (state.optBoolean("speech_busy")) "正在回答…" else ""
        val seconds = feedback.optLong("elapsed_ms", 0).coerceAtLeast(0) / 1000
        val stage = feedback.optString("stage")
        val label = when (stage) {
            "received" -> "问题已收到"
            "manual_lookup" -> if (seconds >= 15) "仍在查阅资料" else "正在查阅资料"
            "search_planning" -> "正在理解当前请求"
            "current_view" -> "正在查看当前画面"
            "web_search" -> if (seconds >= 15) "仍在搜索网页与说明书" else "正在搜索网页与说明书"
            "model" -> if (seconds >= 15) "仍在等待答复" else "正在等待答复"
            "metadata_answer" -> "正在整理资料"
            "decompose" -> "正在理解任务"
            "plan" -> "正在安排步骤"
            "observe" -> "正在观察当前步骤"
            "remind" -> "正在整理提示"
            else -> "正在处理"
        }
        return "$label · 已等待${seconds}秒"
    }

    fun reply(state: JSONObject): String {
        val actual = if (state.has("chat_messages")) "" else state.optString("last_reply_text")
        if (actual.isNotBlank()) return actual
        if (state.optBoolean("dialogue_busy") || state.optBoolean("busy")) return ""
        return if (state.optString("phase") in setOf("awaiting_plan_confirmation", "awaiting_step_confirmation", "awaiting_clarification"))
            state.optJSONObject("pending")?.optString("text").orEmpty() else ""
    }
}
