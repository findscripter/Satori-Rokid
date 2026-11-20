package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ManualMetadataAnswerTest {
    private val assets = File("src/main/assets/manuals")
    private val store by lazy {
        LocalManualStore(listOf("tl-sg105-v4.json", "classroom-demo.json").map { File(assets, it).readBytes() })
    }
    private val question = "这份手册适用哪个型号"

    private fun context(): JSONObject {
        val query = "型号 铭牌 版本"
        val evidence = StandaloneManuals.context("tl-sg105-v4", query, store.search("tl-sg105-v4", query))
        val sources = JSONObject(File(assets, "source-index.json").readText(Charsets.UTF_8)).getJSONArray("sources")
        val original = (0 until sources.length()).map { sources.getJSONObject(it) }
            .single { it.getString("manual_id") == "tl-sg105-v4" }
        // Fixture for an already-verified download; this test never performs or claims a live PDF check.
        return evidence.put("transport", "device_asset_search").put("source_check", JSONObject()
            .put("status", "original_pdf_sha256_verified").put("source_url", original.getString("source_url"))
            .put("original_pdf_sha256", original.getString("sha256")).put("bytes", original.getInt("document_bytes"))
            .put("research_section_ids", original.getJSONArray("research_section_ids"))
            .put("verified_unix_ms", 1L).put("cache_hit", true))
    }

    private fun passages(context: JSONObject): List<JSONObject> = context.getJSONArray("passages").let { rows ->
        (0 until rows.length()).map { rows.getJSONObject(it) }
    }

    @Test fun wholeMetadataQuestionsAndTheCapturedAsrPhraseAreEligible() {
        for (text in listOf(question, "当前手册对应什么型号", "这份手册是什么版本",
            "请问这份手册适用哪个型号？", "问这份手册适用于那的型号的设备", " 这份手册适用哪个型号？ ")) {
            val evidence = context()
            val answer = StandaloneManuals.metadataAnswer(text, evidence)
            assertNotNull("metadata question must use verified local metadata: $text", answer)
            assertEquals("answer", StandaloneManuals.answer(answer!!, evidence).getString("kind"))
        }
    }

    @Test fun physicalIdentificationCompatibilityCommandsAndCompoundQuestionsStayWithTheModel() {
        for (text in listOf(
            "我这台是什么型号", "这份手册适合我的设备吗", "V3也适用吗", "那是什么型号",
            "手册里的网线怎么接", "这份手册版本怎么升级", "这份手册适用哪个接口",
            "这份手册适用哪个型号，线怎么接", "这份手册适用哪个型号？我这台V3也行吗",
            "他说这份手册适用哪个型号", "不要回答这份手册适用哪个型号", "我不是问这份手册适用哪个型号",
            "这份手册适用型号不是V4吗", "把手册型号改成V3", "TL-SG105第四版",
            "这分手车型与那个型号", "嗯这份手册适用于拉个性化", "型号", "版本", "", " ",
        )) assertNull("not an unambiguous current-manual metadata question: $text", StandaloneManuals.metadataAnswer(text, context()))
    }

    @Test fun answerKeepsTheFourFieldContractAndCitesTheRetrievedResearchSection() {
        val evidence = context()
        val before = evidence.toString()
        val answer = StandaloneManuals.metadataAnswer(question, evidence)!!
        assertEquals(setOf("kind", "text", "source_sections", "suggest_replan"), answer.keys().asSequence().toSet())
        assertEquals("answer", answer.getString("kind"))
        assertFalse(answer.getBoolean("suggest_replan"))
        assertEquals(listOf("tl-sg105-v4:applicability"), answer.getJSONArray("source_sections").let { ids ->
            (0 until ids.length()).map { ids.getString(it) }
        })
        val metadata = store.lookup("tl-sg105-v4")!!
        val text = answer.getString("text")
        assertTrue(text.contains("当前本机摘录登记的型号"))
        assertTrue(text.contains(metadata.getString("model")))
        assertTrue(text.contains("版本") && text.contains(metadata.getString("version")))
        assertTrue(text.contains("现场") && text.contains("核对铭牌"))
        assertEquals(answer.toString(), StandaloneManuals.answer(answer, evidence).toString())
        assertEquals("metadata answering must not rewrite source provenance", before, evidence.toString())
        val applicability = passages(evidence).single { it.getString("section_id") == "applicability" }
        assertTrue(applicability.has("page") && applicability.isNull("page"))
        assertTrue(applicability.getString("section_title").contains("研究整理"))
        assertNotEquals(applicability.getString("sha256"), evidence.getJSONObject("source_check").getString("original_pdf_sha256"))
    }

    @Test fun missingOrUnverifiedEvidenceReturnsNullWithoutThrowing() {
        val invalid: List<(JSONObject) -> Unit> = listOf(
            { it.remove("matched") },
            { it.put("matched", false) },
            { it.remove("transport") },
            { it.put("transport", "http_json_api") },
            { it.remove("passages") },
            { it.put("passages", JSONArray()) },
            { it.remove("source_check") },
            { it.put("source_check", JSONObject.NULL) },
            { it.getJSONObject("source_check").remove("status") },
            { it.getJSONObject("source_check").put("status", "unknown_manual") },
            { it.getJSONObject("source_check").put("status", "synthetic_no_remote_source") },
            { it.getJSONObject("source_check").remove("source_url") },
            { it.getJSONObject("source_check").put("source_url", "https://other.invalid/manual.pdf") },
            { it.getJSONObject("source_check").remove("original_pdf_sha256") },
            { it.getJSONObject("source_check").put("original_pdf_sha256", "g".repeat(64)) },
            { it.getJSONObject("source_check").put("original_pdf_sha256", "a".repeat(63)) },
        )
        assertNull(StandaloneManuals.metadataAnswer(question, JSONObject()))
        for ((index, change) in invalid.withIndex()) {
            val evidence = context().also(change)
            assertNull("invalid evidence case $index must not answer", StandaloneManuals.metadataAnswer(question, evidence))
        }
    }

    @Test fun syntheticMissingOrConflictingPassageMetadataCannotProduceAnAnswer() {
        val conflicts = mapOf(
            "manual_id" to "other-manual", "model" to "TL-SG105(UN) V3.0", "version" to "different-version",
            "source_url" to "https://other.invalid/manual.pdf", "sha256" to "f".repeat(64),
        )
        for ((field, value) in conflicts) {
            val evidence = context()
            assertTrue("fixture must exercise multiple retrieved passages", passages(evidence).size >= 2)
            passages(evidence).last().put(field, value)
            assertNull("conflicting $field must not be silently ignored", StandaloneManuals.metadataAnswer(question, evidence))
        }
        for (field in listOf("manual_id", "model", "version", "source_url", "sha256", "synthetic")) {
            val evidence = context()
            passages(evidence).first().remove(field)
            assertNull("missing $field must not throw or answer", StandaloneManuals.metadataAnswer(question, evidence))
        }
        val synthetic = context()
        passages(synthetic).forEach { it.put("synthetic", true) }
        assertNull("synthetic material is not manufacturer evidence", StandaloneManuals.metadataAnswer(question, synthetic))
        val wrongManual = context().put("manual_id", "other-manual")
        assertNull("context and passage manual ids must agree", StandaloneManuals.metadataAnswer(question, wrongManual))
    }

    @Test fun absentApplicabilityOrForgedCitationsCannotBeUsedAsGrounding() {
        val noApplicability = context()
        noApplicability.put("passages", JSONArray(passages(noApplicability).filter { it.getString("section_id") != "applicability" }))
        assertNull("must actually retrieve the cited applicability section", StandaloneManuals.metadataAnswer(question, noApplicability))
        for (citation in listOf("other-manual:applicability", "tl-sg105-v4:invented-section", "", "invented-citation")) {
            val evidence = context()
            passages(evidence).single { it.getString("section_id") == "applicability" }.put("citation_id", citation)
            assertNull("citation must match the retrieved manual and section", StandaloneManuals.metadataAnswer(question, evidence))
        }
        val missingCitation = context()
        passages(missingCitation).single { it.getString("section_id") == "applicability" }.remove("citation_id")
        assertNull(StandaloneManuals.metadataAnswer(question, missingCitation))
    }

    @Test fun onlyTheCurrentUtteranceChoosesThePathNotTheHistorySearchQuery() {
        val oldMetadataQuestion = context().put("query", "$question 网线怎么接")
        assertNull(StandaloneManuals.metadataAnswer("网线怎么接", oldMetadataQuestion))
        val oldWiringQuestion = context().put("query", "好了好了好了 网线怎么接")
        assertNotNull(StandaloneManuals.metadataAnswer(question, oldWiringQuestion))
    }

    @Test fun metadataAnswerIsStoredAsAnAnswerWithoutOpeningAClarificationWindow() {
        val conversation = ManualConversation()
        conversation.begin("task-a", "网线怎么接", 0)
        conversation.finish("task-a", JSONObject().put("kind", "clarify").put("text", "请告诉我铭牌型号"), 1)
        assertTrue(conversation.awaitingReply("task-a", 2))
        conversation.begin("task-a", question, 3)
        val answer = StandaloneManuals.metadataAnswer(question, context())!!
        conversation.finish("task-a", answer, 4)
        assertFalse(conversation.awaitingReply("task-a", 5))
        assertFalse(conversation.awaitingReply("task-b", 5))
        val history = conversation.begin("task-a", "网线怎么接", 6)
        assertEquals("assistant", history.getJSONObject(history.length() - 2).getString("role"))
        assertEquals(answer.getString("text"), history.getJSONObject(history.length() - 2).getString("text"))
        assertEquals(1, conversation.begin("task-b", "另一台设备", 7).length())
    }
}
