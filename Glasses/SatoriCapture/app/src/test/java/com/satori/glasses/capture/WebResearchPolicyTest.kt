package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class WebResearchPolicyTest {
    private fun decision(search: Boolean, query: String) = JSONObject().put("needs_search", search).put("query", query)

    private fun rejects(raw: JSONObject) {
        try { WebResearchPolicy.plan(raw); fail("Invalid search decision accepted") }
        catch (_: IllegalArgumentException) { }
    }

    @Test fun searchCanBeSkippedWithoutSupplyingAnObjectOrDomain() {
        val plan = WebResearchPolicy.plan(decision(false, ""))
        assertFalse(plan.getBoolean("needs_search"))
        assertEquals("", plan.getString("query"))
        assertEquals(setOf("needs_search", "query"), plan.keys().asSequence().toSet())
    }

    @Test fun searchPreservesTheRuntimeQuery() {
        val raw = decision(true, "当地展览开放时间")
        val plan = WebResearchPolicy.plan(raw)
        assertEquals("当地展览开放时间", plan.getString("query"))
        raw.put("query", "changed")
        assertEquals("当地展览开放时间", plan.getString("query"))
        assertEquals("x".repeat(200), WebResearchPolicy.plan(decision(true, "x".repeat(200))).getString("query"))
    }

    @Test fun schemaAndQueryAreValidatedBeforeSearching() {
        for (bad in listOf("true", 1, JSONObject.NULL)) rejects(decision(false, "").put("needs_search", bad))
        rejects(JSONObject().put("needs_search", false))
        rejects(JSONObject().put("query", ""))
        rejects(decision(false, "").put("query", 42))
        rejects(decision(false, "").put("requires_model", true))
        rejects(decision(false, "query"))
        for (bad in listOf("", " ", " x", "x ", "a\nb", "x".repeat(201))) rejects(decision(true, bad))
    }

    @Test fun generalPromptsDoNotIntroduceAnObjectOrRequireAnInvocationPhrase() {
        val prompts = WebResearchPolicy.PLAN_PROMPT + WebResearchPolicy.WEB_RULES + WebResearchPolicy.DIALOGUE_PROMPT
        for (preset in listOf("小米", "米家", "风扇", "清洁", "TP-Link", "整理书桌", "铭牌", "看看眼前", "requires_model", "model_source"))
            assertFalse("prompt preselected $preset", prompts.contains(preset, ignoreCase = true))
    }
}
