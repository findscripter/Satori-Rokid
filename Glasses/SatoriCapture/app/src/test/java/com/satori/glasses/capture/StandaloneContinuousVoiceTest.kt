package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test

class StandaloneContinuousVoiceTest {
    @Test fun onlyExplicitRecordingRequestsCanSwitchTheRecorder() {
        for (text in listOf("你现在开始录像", "用眼镜开始录像", "你用眼镜录像", "哈罗开始录像", "嗯开始录像", "Hello，请帮我开始录制。")) {
            assertEquals(text, "record_start", StandaloneVoice.recordingAction(text))
            assertTrue(StandaloneVoice.automaticInputAllowed(text, false))
        }
        for (text in listOf("你现在停止录像", "用眼镜结束录像", "嗯，停止录像", "请结束录制", "结束录像")) {
            assertEquals(text, "record_stop", StandaloneVoice.recordingAction(text))
        }
        for (text in listOf("开始录像？", "怎么开始录像", "停止录像了吗", "可以开始录像吗？",
                "二楼开始录像", "开始录像用水特朗普", "你用眼镜的录像")) {
            assertNull(text, StandaloneVoice.recordingAction(text))
            assertTrue(text, StandaloneVoice.needsRecordingFeedback(text))
            assertTrue(text, StandaloneVoice.automaticInputAllowed(text, false))
        }
        for (text in listOf("不要开始录像", "等我说完再开始录像", "帮我查录像机说明书", "开始录屏", "开始绿瓶", "他说开始录像")) {
            assertNull(text, StandaloneVoice.recordingAction(text))
            assertFalse(text, StandaloneVoice.needsRecordingFeedback(text))
            assertFalse(text, StandaloneVoice.automaticInputAllowed(text, false))
        }
    }

    @Test fun backgroundSpeechCannotBecomeAnAutomaticGoal() {
        assertFalse(StandaloneVoice.automaticInputAllowed("今天天气不错", false))
        assertFalse(StandaloneVoice.automaticInputAllowed("整理书桌", false))
        assertTrue(StandaloneVoice.automaticInputAllowed("我想整理书桌", false))
        assertTrue(StandaloneVoice.automaticInputAllowed("暂停", false))
        assertTrue(StandaloneVoice.automaticInputAllowed("停止监听", false))
    }

    @Test fun aProgressQuestionIsNeverACompletionConfirmation() {
        assertTrue(StandaloneVoice.isProgressQuery("任务完成了吧？"))
        assertTrue(StandaloneVoice.automaticInputAllowed("任务完成了吧", false))
        assertFalse(StandaloneVoice.isProgressQuery("确认"))
        assertFalse(StandaloneVoice.automaticInputAllowed("不要停止", false))
        assertFalse(StandaloneVoice.automaticInputAllowed("还没完成", false))
    }

    @Test fun freeAnswersOnlyEnterAnExplicitClarificationTurn() {
        assertTrue(StandaloneVoice.automaticInputAllowed("先整理左边", true))
        assertFalse(StandaloneVoice.automaticInputAllowed("先整理左边", false))
        assertFalse(StandaloneVoice.automaticInputAllowed("", true))
        assertFalse(StandaloneVoice.automaticInputAllowed("答".repeat(301), true))
    }
}
