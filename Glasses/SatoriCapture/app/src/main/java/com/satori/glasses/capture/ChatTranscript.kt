package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject

/** Session-only display history, owned by the service's main thread. */
class ChatTranscript {
    private data class Turn(val id: String, val user: String?, var reply: String = "", var state: String = "pending")
    private val turns = ArrayDeque<Turn>()
    private var sequence = 0L

    fun beginUser(text: String): String {
        require(text.isNotBlank() && text.length <= 500)
        val turn = Turn("t${++sequence}", text)
        append(turn)
        return turn.id
    }

    fun answer(turnId: String?, text: String): Boolean {
        require(text.isNotBlank() && text.length <= 4000)
        if (turnId == null) {
            append(Turn("t${++sequence}", null, text, "complete"))
            return true
        }
        val turn = turns.firstOrNull { it.id == turnId } ?: return false
        if (turn.state != "pending") return false
        turn.reply = text
        turn.state = "complete"
        return true
    }

    fun finish(turnId: String, state: String): Boolean {
        require(state in setOf("cancelled", "error", "ignored", "complete"))
        val turn = turns.firstOrNull { it.id == turnId } ?: return false
        if (turn.state != "pending") return false
        turn.state = state
        return true
    }

    fun snapshot(): JSONArray = JSONArray().apply {
        for (turn in turns) {
            if (turn.user != null) put(row(turn, "user", turn.user, "complete"))
            put(row(turn, "assistant", turn.reply, turn.state))
        }
    }

    private fun append(turn: Turn) {
        turns.addLast(turn)
        while (turns.size > 20) turns.removeFirst()
    }

    private fun row(turn: Turn, role: String, text: String, state: String) = JSONObject()
        .put("id", "${turn.id}:$role").put("role", role).put("text", text)
        .put("turn_id", turn.id).put("state", state)
}
