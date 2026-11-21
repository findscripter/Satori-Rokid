package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class StandaloneInteractionFeedbackTest {
    private fun state(stage: String, elapsedMs: Any = 0L, preview: String? = null) = JSONObject()
        .put("interaction_feedback", JSONObject().put("request_id", "d1").put("stage", stage)
            .put("elapsed_ms", elapsedMs).apply { if (preview != null) put("question_preview", preview) })

    private fun compact(text: String) = text.replace(Regex("\\s+"), "")

    @Test fun displaysTheCurrentStageAndElapsedSecondsWithoutInventingProgress() {
        val lookup = StandaloneInteractionFeedback.text(state("manual_lookup", 2_999L))
        assertTrue(lookup.contains("正在查阅资料"))
        assertTrue(compact(lookup).contains("2秒"))
        assertFalse(compact(lookup).contains("3秒"))
        val model = StandaloneInteractionFeedback.text(state("model", 7_321L))
        assertTrue(model.contains("正在等待答复"))
        assertTrue(compact(model).contains("7秒"))
        for (text in listOf(lookup, model)) {
            assertFalse(text.contains("%") || text.contains("％"))
            assertFalse(text.contains("预计") || text.contains("剩余"))
        }
    }

    @Test fun longWaitMessageChangesAtFifteenSeconds() {
        val before = StandaloneInteractionFeedback.text(state("model", 14_999L))
        val boundary = StandaloneInteractionFeedback.text(state("model", 15_000L))
        assertTrue(before.contains("正在等待答复"))
        assertFalse(before.contains("仍在等待答复"))
        assertTrue(boundary.contains("仍在等待答复"))
        assertTrue(compact(boundary).contains("15秒"))
    }

    @Test fun absentRequestsDoNotResurrectOldBusyStateAndInvalidElapsedTimeIsZero() {
        val old = JSONObject().put("dialogue_busy", true)
            .put("last_answer", JSONObject().put("text", "上一次答复"))
        assertEquals("", StandaloneInteractionFeedback.text(old))
        assertEquals("", StandaloneInteractionFeedback.text(old.put("interaction_feedback", JSONObject.NULL)))
        for (id in listOf("", "   ")) {
            val value = state("model", 20_000L)
            value.getJSONObject("interaction_feedback").put("request_id", id)
            assertEquals("", StandaloneInteractionFeedback.text(value))
        }
        val missingId = state("model", 20_000L)
        missingId.getJSONObject("interaction_feedback").remove("request_id")
        assertEquals("", StandaloneInteractionFeedback.text(missingId))
        for (elapsed in listOf(-1L, "invalid", JSONObject.NULL)) {
            val text = StandaloneInteractionFeedback.text(state("model", elapsed))
            assertTrue(compact(text).contains("0秒"))
            assertFalse(text.contains("仍在等待答复"))
        }
    }

    @Test fun progressDoesNotDuplicateTheConversationText() {
        val raw = "甲".repeat(35) + "\r\n" + "乙".repeat(40) + "必须被截断"
        val text = StandaloneInteractionFeedback.text(state("model", 1_000L, raw))
        assertTrue(text.contains("正在等待答复"))
        assertFalse(text.contains("听到：") || text.contains("甲") || text.contains('\n'))
    }
}
