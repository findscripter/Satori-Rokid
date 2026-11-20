package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ManualConversationTest {
    @Test fun modelNumberReplyRetainsTheQuestionButCannotLeakToAnotherTask() {
        val chat = ManualConversation()
        chat.begin("task-a", "网线怎么接", 0)
        chat.finish("task-a", JSONObject().put("kind", "clarify").put("text", "请告诉我铭牌型号"), 10)
        assertTrue(chat.awaitingReply("task-a", 11))
        assertFalse(chat.awaitingReply("task-b", 11))
        val turns = chat.begin("task-a", "TL-SG105第四版", 12)
        assertEquals(3, turns.length())
        assertEquals("请告诉我铭牌型号", turns.getJSONObject(1).getString("text"))
        assertFalse(chat.awaitingReply("task-a", 13))
        assertEquals(1, chat.begin("task-b", "另一台设备", 14).length())
    }

    @Test fun clarificationExpiresAndHistoryIsBounded() {
        val chat = ManualConversation()
        repeat(20) { i ->
            chat.begin("task", "问题$i", i.toLong())
            chat.finish("task", JSONObject().put("kind", "clarify").put("text", "回答$i"), i.toLong())
        }
        assertFalse(chat.awaitingReply("task", 100_000))
        assertEquals(8, chat.begin("task", "最新问题", 100_001).length())
        chat.reset(); assertFalse(chat.awaitingReply("task", 20))
    }
}
