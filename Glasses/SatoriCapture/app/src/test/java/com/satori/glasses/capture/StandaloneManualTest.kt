package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class StandaloneManualTest {
    @Test fun exhaustedPlanningAttemptsAreNotReportedAsManualRetrievalFailure() {
        for (diagnostic in listOf("timeout_connect_and_write", "timeout_response_headers")) {
            val failure = ApiFailure(0, 1234, "模型响应超时", diagnostic)
            val reported = StandaloneManuals.lookupFailure("search_planning", failure)
            assertEquals("请求理解超时，请稍后重试", reported.message)
            assertEquals(failure.status, reported.status)
            assertEquals(failure.retryAfterMs, reported.retryAfterMs)
            assertEquals(diagnostic, reported.diagnostic)
        }
    }

    @Test fun retrievalTimeoutsNameTheFailingStageAndKeepTransportDetails() {
        for ((stage, expected) in mapOf("search_planning" to "请求理解", "web_search" to "网页检索",
            "manual_lookup" to "手册检索或原件校验")) {
            for (status in listOf(408, 504)) {
                val reported = StandaloneManuals.lookupFailure(stage, ApiFailure(status, 2500, "请求超时", "http_timeout"))
                assertEquals("${expected}超时，请稍后重试", reported.message)
                assertEquals(status, reported.status)
                assertEquals(2500L, reported.retryAfterMs)
                assertEquals("http_timeout", reported.diagnostic)
            }
        }
    }

    @Test fun knownLookupFailuresKeepTheirSpecificMessagesAndUnknownErrorsStayPrivate() {
        val failures = listOf(ApiFailure(message = "请求已取消", diagnostic = "cancelled"),
            ApiFailure(message = "请求已取消"), ApiFailure(401, message = "接口认证失败"),
            ApiFailure(403, message = "网页检索服务拒绝当前匿名访问"),
            ApiFailure(429, 60000, "网页检索服务已限流", "rate_limit"),
            ApiFailure(message = "厂商原手册版本已变化，需核对后更新本机摘录"))
        for (stage in listOf("search_planning", "web_search", "manual_lookup")) {
            for (failure in failures) assertSame(failure, StandaloneManuals.lookupFailure(stage, failure))
            val reported = StandaloneManuals.lookupFailure(stage, IllegalArgumentException("private upstream detail"))
            assertFalse(reported.message.orEmpty().contains("private upstream detail"))
            assertEquals(0, reported.status)
            assertEquals(0L, reported.retryAfterMs)
            assertEquals("", reported.diagnostic)
        }
    }

    @Test fun unknownWebModelWaitsForIdentityWithoutInventingSourcesOrPlan() {
        val engine = StandaloneEngine()
        engine.setGoal("清洁小米风扇", true, 0)
        val request = engine.nextRequest(1, listOf("f1"))!!
        assertEquals(listOf("请核对铭牌型号"), engine.clarifyManualIdentity(request, "请核对铭牌型号", 2))
        assertEquals("awaiting_clarification", engine.snapshot().getString("phase"))
        assertEquals(0, engine.snapshot().getJSONArray("desires").length())
        assertTrue(engine.clarifyManualIdentity(request, "旧请求", 3).isEmpty())
        engine.answerClarification("型号是BPLDS02DM", 4)
        assertEquals("decomposing", engine.snapshot().getString("phase"))
    }
    @Test fun manualModeDoesNotKeepTheLegacyAssumptionForHiddenPhysicalProperties() {
        val legacy = "按JSON回复。画面看不出的属性（例如水温、成分）按常规情况理解，不要因此判为未完成或偏离，也不要因此让用户停下或更换正在用的东西。"
        val grounded = StandaloneManuals.stagePrompt(legacy)
        assertFalse(grounded.contains("按常规情况理解"))
        assertTrue(grounded.contains("未知保持UNKNOWN"))
        assertTrue(grounded.contains("source_sections数组"))
        assertTrue(grounded.startsWith("按JSON回复。"))
    }

    private fun passage() = JSONObject().put("manual_id", "switch-v4").put("section_id", "cabling")
        .put("title", "安装指南").put("model", "TL-SG105 V4").put("version", "4.0")
        .put("section_title", "连接").put("source_url", "https://example.org/manual.pdf").put("page", 1)
        .put("sha256", "a".repeat(64)).put("synthetic", false).put("text", "仅按匹配型号手册连接LAN端口。")

    @Test fun toolEndpointOnlyAllowsHttpsOrExplicitLoopbackHttp() {
        assertEquals("https://example.org/tools", StandaloneManuals.endpoint("https://example.org/tools/"))
        assertEquals("http://127.0.0.1:8766", StandaloneManuals.endpoint("http://127.0.0.1:8766"))
        listOf("http://example.org", "http://192.168.1.2", "https://u:p@example.org", "https://example.org/?token=x",
            "http://127.0.0.1.evil.org", "https://example.org/#x", "file:///data").forEach {
            try { StandaloneManuals.endpoint(it); fail("accepted unsafe endpoint") } catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun retrievalRejectsWrongManualAndUntrustedSourceFields() {
        val data = JSONObject().put("matches", JSONArray().put(passage()))
        val selected = StandaloneManuals.context("switch-v4", "网线", data)
        assertEquals("switch-v4:cabling", selected.getJSONArray("passages").getJSONObject(0).getString("citation_id"))
        for (bad in listOf(passage().put("manual_id", "wrong"), passage().put("sha256", "broken"),
            passage().put("source_url", "javascript:alert(1)"), passage().put("synthetic", "false"))) {
            try { StandaloneManuals.context("switch-v4", "网线", JSONObject().put("matches", JSONArray().put(bad))); fail("accepted corrupt evidence") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun dialogueCannotInventCitationsOrTreatReportsAsVerifiedCompletion() {
        val evidence = StandaloneManuals.context("switch-v4", "怎么接", JSONObject().put("matches", JSONArray().put(passage())))
        val answer = JSONObject().put("kind", "answer").put("text", "请核对LAN标识。")
            .put("source_sections", JSONArray().put("switch-v4:cabling")).put("suggest_replan", false)
        assertEquals("answer", StandaloneManuals.answer(answer, evidence).getString("kind"))
        answer.put("source_sections", JSONArray().put("invented:9"))
        try { StandaloneManuals.answer(answer, evidence); fail("invented citation") } catch (_: IllegalArgumentException) { }
        answer.put("source_sections", JSONArray()).put("kind", "complete")
        try { StandaloneManuals.answer(answer, evidence); fail("dialogue cannot finish a physical step") } catch (_: IllegalArgumentException) { }
    }

    @Test fun feedbackRetainsTaskScopeAndInvalidatesOldObservationWithoutCompletingTask() {
        val engine = StandaloneEngine()
        engine.setGoal("连接教室网络", true, 0)
        val old = engine.nextRequest(1, listOf("f1"))!!
        val task = engine.snapshot().getString("task_instance_id")
        engine.addUserFeedback("这台是交换机，不是路由器", 2)
        val snapshot = engine.snapshot()
        assertEquals("decomposing", snapshot.getString("phase"))
        assertEquals(task, snapshot.getString("task_instance_id"))
        assertEquals("user_report_unverified", snapshot.getJSONArray("user_feedback").getJSONObject(0).getString("source"))
        assertTrue(engine.accept(old, JSONObject(), 3).isEmpty())
        assertEquals(1, engine.nextRequest(4, listOf("f2"))!!.payload.getJSONArray("user_feedback").length())
        engine.setGoal("另一个教室", true, 5)
        assertEquals(0, engine.snapshot().getJSONArray("user_feedback").length())
    }

    @Test fun passiveAmbientSpeechIsNotANewGoalAndTaskQuestionsAreAdmitted() {
        assertTrue(StandaloneManuals.isTaskUtterance("网线应该插哪个接口"))
        assertTrue(StandaloneManuals.isTaskUtterance("这个灯还没亮"))
        assertTrue(StandaloneManuals.isTaskUtterance("这分手车型与那个型号")) // Captured ASR output; do not rewrite it into a guessed transcript.
        assertTrue(StandaloneManuals.isTaskUtterance("铭牌上是V4"))
        assertTrue(StandaloneManuals.isTaskUtterance("硬件版本V4"))
        assertFalse(StandaloneManuals.isTaskUtterance("可能"))
        assertFalse(StandaloneManuals.isTaskUtterance("明天吃饭吧"))
        assertFalse(StandaloneManuals.isTaskUtterance("怎么".repeat(300)))
        assertFalse(StandaloneVoice.isExplicitGoal("我想问一下网线怎么接"))
        assertFalse(StandaloneVoice.isExplicitGoal("我想确认这个接口是不是LAN"))
        assertTrue(StandaloneVoice.isExplicitGoal("我想配置教室网络"))
    }

    @Test fun proactivePlanRejectsInventedManualReferences() {
        val engine = StandaloneEngine()
        engine.setGoal("配置网络", true, 0)
        val request = engine.nextRequest(1, listOf("f1"))!!
        request.payload.put("manual_context", StandaloneManuals.context("switch-v4", "网络", JSONObject().put("matches", JSONArray().put(passage()))))
        val response = JSONObject().put("goal_summary", "配置网络").put("horizon", "short")
            .put("desires", JSONArray().put(JSONObject().put("desire_id", "d1").put("title", "连接交换机").put("success_criteria", "资源可访问")))
            .put("clarifying_question", "请核对设备铭牌").put("confidence", .8).put("evidence_frame_ids", JSONArray().put("f1"))
            .put("source_sections", JSONArray().put("wrong:fake"))
        engine.accept(request, response, 2)
        assertEquals("decomposing", engine.snapshot().getString("phase"))
        assertEquals(1, engine.snapshot().getInt("failures"))
        val next = engine.nextRequest(6000, listOf("f1"))!!
        next.payload.put("manual_context", request.payload.getJSONObject("manual_context"))
        engine.accept(next, response.put("source_sections", JSONArray().put("switch-v4:cabling")), 6001)
        assertEquals("awaiting_clarification", engine.snapshot().getString("phase"))
    }
}
