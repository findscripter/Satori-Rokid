package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject

/** A clarification grants one bounded reply window, scoped to the task that asked it. */
class ManualConversation {
    private var task = ""
    private var replyUntil = 0L
    private val turns = ArrayDeque<JSONObject>()

    fun reset() { task = ""; replyUntil = 0; turns.clear() }
    fun awaitingReply(taskId: String, nowMs: Long) = taskId == task && nowMs < replyUntil

    fun begin(taskId: String, text: String, nowMs: Long): JSONArray {
        require(taskId.isNotBlank() && text.length in 1..500)
        if (task != taskId) { reset(); task = taskId }
        replyUntil = 0
        append("user", text, nowMs)
        return JSONArray(turns.map { JSONObject(it.toString()) })
    }

    fun finish(taskId: String, answer: JSONObject, nowMs: Long) {
        if (task != taskId) return
        if (answer.optString("kind") == "unrelated") { if (turns.lastOrNull()?.optString("role") == "user") turns.removeLast(); return }
        append("assistant", answer.getString("text"), nowMs)
        replyUntil = if (answer.optString("kind") == "clarify") nowMs + 90_000 else 0
    }

    private fun append(role: String, text: String, at: Long) {
        turns.addLast(JSONObject().put("role", role).put("text", text).put("at_ms", at))
        while (turns.size > 8) turns.removeFirst()
    }
}
