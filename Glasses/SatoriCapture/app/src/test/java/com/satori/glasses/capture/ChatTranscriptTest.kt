package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test

class ChatTranscriptTest {
    @Test fun repeatedUtterancesRemainDistinctTurnsWithEmptyPendingReplies() {
        val chat = ChatTranscript()
        val first = chat.beginUser(" 哪个型号？ ")
        val second = chat.beginUser(" 哪个型号？ ")
        assertNotEquals(first, second)
        val rows = chat.snapshot()
        assertEquals(4, rows.length())
        for ((index, id) in listOf(0 to first, 2 to second)) {
            val user = rows.getJSONObject(index)
            val assistant = rows.getJSONObject(index + 1)
            assertEquals(" 哪个型号？ ", user.getString("text"))
            assertEquals("user", user.getString("role"))
            assertEquals("$id:user", user.getString("id"))
            assertEquals("complete", user.getString("state"))
            assertEquals("$id:assistant", assistant.getString("id"))
            assertEquals(id, assistant.getString("turn_id"))
            assertEquals("assistant", assistant.getString("role"))
            assertEquals("", assistant.getString("text"))
            assertEquals("pending", assistant.getString("state"))
        }
    }

    @Test fun lateAnswerStaysWithItsOriginalQuestionAndCannotReplaceAnotherAnswer() {
        val chat = ChatTranscript()
        val manual = chat.beginUser("手册是什么版本")
        val progress = chat.beginUser("任务完成了吧")
        assertTrue(chat.answer(progress, "任务仍然暂停。"))
        assertTrue(chat.answer(manual, "手册登记硬件V4。"))
        assertFalse(chat.answer(manual, "重复回调不替换原答案。"))
        val rows = chat.snapshot()
        assertEquals(4, rows.length())
        assertEquals(manual, rows.getJSONObject(1).getString("turn_id"))
        assertEquals("手册登记硬件V4。", rows.getJSONObject(1).getString("text"))
        assertEquals(progress, rows.getJSONObject(3).getString("turn_id"))
        assertEquals("任务仍然暂停。", rows.getJSONObject(3).getString("text"))
    }

    @Test fun cancelledFailedAndIgnoredRequestsCannotBeRevivedByLateCallbacks() {
        val chat = ChatTranscript()
        for (state in listOf("cancelled", "error", "ignored")) {
            val id = chat.beginUser("一个未完成的问题")
            assertTrue(chat.finish(id, state))
            assertFalse(chat.finish(id, state))
            assertFalse(chat.answer(id, "迟到的答复"))
            val row = chat.snapshot().getJSONObject(chat.snapshot().length() - 1)
            assertEquals(state, row.getString("state"))
            assertEquals("", row.getString("text"))
        }
        val answered = chat.beginUser("已经回答的问题")
        assertTrue(chat.answer(answered, "有效答复"))
        assertFalse(chat.finish(answered, "cancelled"))
        assertEquals("有效答复", chat.snapshot().getJSONObject(7).getString("text"))
    }

    @Test fun explicitlyBoundUnknownIdsNeverCreateMessages() {
        val chat = ChatTranscript()
        chat.beginUser("保留此问题")
        val before = chat.snapshot().toString()
        for (id in listOf("", "missing", "t999")) {
            assertFalse(chat.answer(id, "无法绑定的答复"))
            assertFalse(chat.finish(id, "error"))
        }
        assertEquals(before, chat.snapshot().toString())
    }

    @Test fun acceptedCommandCanFinishItsEmptyPlaceholderWithoutInventingAReply() {
        val chat = ChatTranscript()
        val id = chat.beginUser("确认")
        assertTrue(chat.finish(id, "complete"))
        val assistant = chat.snapshot().getJSONObject(1)
        assertEquals("complete", assistant.getString("state"))
        assertEquals("", assistant.getString("text"))
        assertFalse(chat.answer(id, "旧请求的迟到回复"))
    }

    @Test fun trimmingKeepsWholeTurnsAndNeverReusesAnEvictedId() {
        val chat = ChatTranscript()
        val ids = (0 until 25).map { chat.beginUser("问题$it") }
        val rows = chat.snapshot()
        assertEquals(40, rows.length())
        assertEquals(ids[5], rows.getJSONObject(0).getString("turn_id"))
        assertEquals(ids.last(), rows.getJSONObject(39).getString("turn_id"))
        assertFalse(chat.answer(ids.first(), "过期轮次的答复"))
        assertFalse(chat.finish(ids.first(), "cancelled"))
        assertEquals(40, chat.snapshot().length())
        val next = chat.beginUser("仍然相同的原话")
        assertFalse(next in ids)
    }

    @Test fun unsolicitedAssistantMessagesAlsoConsumeTheTwentyTurnBudget() {
        val chat = ChatTranscript()
        val ids = (0 until 19).map { chat.beginUser("问题$it") }
        assertTrue(chat.answer(null, "主动提示一"))
        assertTrue(chat.answer(null, "主动提示二"))
        val rows = chat.snapshot()
        assertEquals(38, rows.length())
        assertEquals(ids[1], rows.getJSONObject(0).getString("turn_id"))
        assertFalse(chat.answer(ids[0], "被逐出的轮次不能恢复"))
        for ((index, text) in listOf(36 to "主动提示一", 37 to "主动提示二")) {
            val row = rows.getJSONObject(index)
            assertEquals("assistant", row.getString("role"))
            assertEquals(text, row.getString("text"))
            assertEquals("complete", row.getString("state"))
        }
    }

    @Test fun returnedJsonCannotMutateStoredMessages() {
        val chat = ChatTranscript()
        val id = chat.beginUser("原始输入")
        val copy = chat.snapshot()
        copy.getJSONObject(0).put("text", "被外部修改")
        copy.getJSONObject(1).put("state", "complete").put("text", "伪造答复")
        copy.remove(0)
        assertEquals("原始输入", chat.snapshot().getJSONObject(0).getString("text"))
        assertEquals("pending", chat.snapshot().getJSONObject(1).getString("state"))
        assertTrue(chat.answer(id, "正式答复"))
    }

    @Test fun invalidTextAndTerminalStatesAreRejectedBeforeMutation() {
        val chat = ChatTranscript()
        for (text in listOf("", "   ", "字".repeat(501))) rejects { chat.beginUser(text) }
        assertEquals(0, chat.snapshot().length())
        val id = chat.beginUser("有效问题")
        for (text in listOf("", "   ", "字".repeat(4001))) rejects { chat.answer(id, text) }
        for (state in listOf("", "completed", "pending", "arbitrary")) rejects { chat.finish(id, state) }
        assertEquals("pending", chat.snapshot().getJSONObject(1).getString("state"))
    }

    private fun rejects(action: () -> Unit) {
        try { action(); fail("accepted invalid input") } catch (_: IllegalArgumentException) { }
    }
}
