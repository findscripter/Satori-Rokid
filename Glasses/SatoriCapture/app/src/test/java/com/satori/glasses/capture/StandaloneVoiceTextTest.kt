package com.satori.glasses.capture

import org.junit.Assert.assertEquals
import org.junit.Test

class StandaloneVoiceTextTest {
    @Test fun chineseModelTokenSpacesBecomeUsableCommands() {
        assertEquals("重复", StandaloneVoice.normalizeOfflineText(" 重 复 "))
        assertEquals("我要整理书桌", StandaloneVoice.normalizeOfflineText("我 要 整理 书桌"))
    }

    @Test fun latinProductNamesKeepTheirWordBoundaries() {
        assertEquals("我要整理 Apple Pencil", StandaloneVoice.normalizeOfflineText("我 要 整理 Apple Pencil"))
        assertEquals("", StandaloneVoice.normalizeOfflineText(" \n "))
    }
}
