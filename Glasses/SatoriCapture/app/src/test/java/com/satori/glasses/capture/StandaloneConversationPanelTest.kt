package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class StandaloneConversationPanelTest {
    @Test fun savedTasksAndInternalClassificationsDoNotInventAReply() {
        val state = JSONObject().put("phase", "executing")
            .put("goal", JSONObject().put("text", "整理书桌"))
            .put("desires", JSONArray().put(JSONObject().put("title", "预设步骤")
                .put("plan", JSONObject().put("steps", JSONArray().put(JSONObject().put("instruction", "归拢物品"))))))
            .put("manual_id", "tl-sg105-v4").put("assistance_mode", "guided")
            .put("last_answer", JSONObject().put("kind", "unrelated").put("text", "忽略无关语音"))
            .put("last_spoken", "之前的答复")
        assertEquals("", StandaloneInteractionFeedback.reply(state))
        assertEquals("", StandaloneInteractionFeedback.text(state))
        state.put("last_reply_text", "请核对设备铭牌上的硬件版本。")
        assertEquals("请核对设备铭牌上的硬件版本。", StandaloneInteractionFeedback.reply(state))
    }

    @Test fun actualPendingInteractionIsOnlyAFallbackForAnIdleConfirmationOrClarification() {
        val prompt = "这一步需要先确认设备型号，是否继续？"
        for (phase in listOf("awaiting_plan_confirmation", "awaiting_step_confirmation", "awaiting_clarification")) {
            val state = JSONObject().put("phase", phase).put("last_reply_text", "")
                .put("pending", JSONObject().put("text", prompt))
            assertEquals(prompt, StandaloneInteractionFeedback.reply(state))
            for (busy in listOf("dialogue_busy", "busy")) {
                state.put(busy, true)
                assertEquals("pending instruction must not masquerade as an in-flight answer", "", StandaloneInteractionFeedback.reply(state))
                state.put(busy, false)
            }
            state.put("last_reply_text", "刚生成的实际答复").put("dialogue_busy", true)
            assertEquals("刚生成的实际答复", StandaloneInteractionFeedback.reply(state))
        }
        for (phase in listOf("idle", "executing", "complete", "stopped")) {
            val stale = JSONObject().put("phase", phase).put("pending", JSONObject().put("text", prompt))
            assertEquals("stale pending text is not current conversation", "", StandaloneInteractionFeedback.reply(stale))
        }
        assertEquals("", StandaloneInteractionFeedback.reply(JSONObject().put("phase", "awaiting_clarification")))
    }

    @Test fun transientStatusDoesNotDuplicateConversationOrListFeatures() {
        val spoken = "这份手册适用哪个型号"
        val state = JSONObject().put("last_user_text", spoken)
            .put("interaction_feedback", JSONObject().put("request_id", "d1").put("stage", "model")
                .put("elapsed_ms", 3_500L).put("question_preview", spoken))
        val waiting = StandaloneInteractionFeedback.text(state)
        assertTrue(waiting.contains("等待答复"))
        assertTrue(waiting.replace(" ", "").contains("3秒"))
        assertFalse("question is displayed in its own conversation row", waiting.contains(spoken) || waiting.contains("听到："))
        for ((flag, meaning) in listOf("recording" to "听", "voice_busy" to "识别", "speech_busy" to "回答")) {
            val active = StandaloneInteractionFeedback.text(JSONObject().put(flag, true))
            assertTrue("actual $flag state must remain visible", active.contains(meaning))
            for (feature in listOf("右键", "左键", "确认键", "双录制", "设置目标", "我想"))
                assertFalse("status must not become a feature menu", active.contains(feature))
        }
        assertEquals("", StandaloneInteractionFeedback.text(JSONObject().put("listening", true)))
    }
}
