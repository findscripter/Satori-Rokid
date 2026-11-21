package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class StandaloneOfflineCandidateTest {
    private fun candidates(vararg values: Pair<String, Double>): JSONObject = JSONObject().put("alternatives",
        JSONArray(values.map { JSONObject().put("text", it.first).put("confidence", it.second) }))
    private fun choose(value: JSONObject) = StandaloneVoice.selectOfflineCandidate(value)

    @Test fun actualSupportedGoalPrefixMaySelectAGenuineCloseScoringCandidate() {
        val choice = choose(candidates("火药 整理 书桌" to 212.533844, "活跃 整理 书桌" to 211.9,
            "我 要 整理 书桌" to 211.466019))
        assertEquals("火药整理书桌", choice.getString("raw_top1"))
        assertEquals("我要整理书桌", choice.getString("selected"))
        assertEquals("goal_prefix_candidate_within_score_gap", choice.getString("reason"))
        assertEquals(3, choice.getInt("candidate_count"))
        assertEquals(211.466019, choice.getDouble("selected_score"), .000001)
    }

    @Test fun bestAlreadySupportedGoalIsNeverReplaced() {
        val choice = choose(candidates("我想整理书桌" to 100.0, "我要泡茶" to 99.9))
        assertEquals("我想整理书桌", choice.getString("selected"))
        assertEquals("best_is_supported_command", choice.getString("reason"))
    }

    @Test fun supportedControlWordWinsOverGoalPrefixAlternative() {
        listOf("确认", "完成", "下一步", "确定", "暂停", "停一下", "继续", "恢复", "停止", "结束任务",
            "取消任务", "重复", "再说一遍", "重做", "重新做", "跳过", "重新规划").forEach { word ->
            assertEquals(word, choose(candidates(word to 100.0, "我要整理书桌" to 99.9)).getString("selected"))
        }
    }

    @Test fun recordingControlTopOneCannotBeOverwrittenByAGoalCandidate() {
        listOf("开始录像", "开始录制", "停止录像", "停止录制", "结束录像").forEach { word ->
            val choice = choose(candidates(word to 100.0, "我要整理书桌" to 99.9))
            assertEquals(word, choice.getString("selected"))
            assertEquals("best_is_supported_command", choice.getString("reason"))
        }
    }

    @Test fun scoreGapBoundIsInclusiveAndNeverTurnsIntoProbability() {
        assertEquals("我要整理书桌", choose(candidates("火药整理书桌" to 100.0, "我要整理书桌" to 98.0)).getString("selected"))
        assertEquals("火药整理书桌", choose(candidates("火药整理书桌" to 100.0, "我要整理书桌" to 97.99)).getString("selected"))
    }

    @Test fun noGoalPrefixKeepsKnownBareGoalFailureAndDoesNotGuessControls() {
        assertEquals("整理收着", choose(candidates("整理收着" to 100.0, "整理书桌" to 99.9)).getString("selected"))
        assertEquals("赞庭", choose(candidates("赞庭" to 100.0, "暂停" to 99.9)).getString("selected"))
    }

    @Test fun missingOrNonNumericScoresCannotTriggerContextReranking() {
        val result = candidates("火药整理书桌" to 100.0, "我要整理书桌" to 99.0)
        result.getJSONArray("alternatives").getJSONObject(0).remove("confidence")
        assertEquals("火药整理书桌", choose(result).getString("selected"))
        result.getJSONArray("alternatives").getJSONObject(0).put("confidence", 100.0)
        result.getJSONArray("alternatives").getJSONObject(1).put("confidence", "99.0")
        assertEquals("火药整理书桌", choose(result).getString("selected"))
    }

    @Test fun onlyExistingServiceGoalPrefixesAreEligibleAndNeedAnArgument() {
        listOf("目标改为", "新的目标是", "设置目标", "我的目标是", "我要", "我想").forEach { prefix ->
            assertEquals(prefix + "整理书桌", choose(candidates("模糊句子" to 100.0, prefix + "整理书桌" to 99.0)).getString("selected"))
        }
        assertEquals("模糊句子", choose(candidates("模糊句子" to 100.0, "我要" to 99.0)).getString("selected"))
        assertEquals("模糊句子", choose(candidates("模糊句子" to 100.0, "请帮我整理书桌" to 99.0)).getString("selected"))
    }

    @Test fun highestOriginalScoringEligibleCandidateIsChosenWithoutWordReplacement() {
        val choice = choose(candidates("模糊句子" to 100.0, "我要泡茶" to 98.5, "我想整理书桌" to 99.3))
        assertEquals("我想整理书桌", choice.getString("selected"))
        assertEquals(100.0, choice.getDouble("raw_top1_score"), .0)
    }

    @Test fun legacySingleResultAndEmptyFinalSegmentStayUnchanged() {
        assertEquals("确认", choose(JSONObject().put("text", "确认")).getString("selected"))
        val empty = choose(JSONObject().put("alternatives", JSONArray().put(JSONObject().put("text", "").put("confidence", 0.0))))
        assertEquals("", empty.getString("selected"))
        assertEquals(0, empty.getInt("candidate_count"))
    }
}
