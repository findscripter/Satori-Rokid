package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.security.MessageDigest

class WebEvidenceTest {
    private val primaryUrl = "https://docs.example.org/page-a"
    private val otherUrl = "https://support.example.net/page-b"
    private fun plan() = JSONObject().put("needs_search", true).put("query", "reference documentation")
    private fun row(url: String, text: String, title: String = "Source $url") = JSONObject()
        .put("url", url).put("title", title).put("text", text)
    private fun answer(ids: List<String>) = JSONObject().put("source_sections", JSONArray(ids))
    private fun hash(text: String) = MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it) }

    @Test fun fetchedBodyAndSearchExcerptRemainDifferentEvidenceKinds() {
        val expectedBody = "页面正文：按对应说明书核对型号。"
        val hits = JSONArray().put(row(primaryUrl, "搜索摘要", "Search title"))
            .put(row(otherUrl, "另一个搜索摘要"))
        val evidence = WebEvidence.context(plan(), hits, JSONArray().put(row(primaryUrl, expectedBody, "Fetched title")), 1234)
        val pages = evidence.getJSONArray("passages")
        assertEquals(2, pages.length())
        assertEquals(expectedBody, pages.getJSONObject(0).getString("text"))
        assertEquals("Fetched title", pages.getJSONObject(0).getString("title"))
        assertEquals("fetched_page", pages.getJSONObject(0).getString("retrieval_kind"))
        assertEquals("search_excerpt", pages.getJSONObject(1).getString("retrieval_kind"))
        assertEquals("mcp_streamable_http", evidence.getString("transport"))
        assertEquals(1234L, evidence.getLong("retrieved_unix_ms"))
        assertTrue(evidence.getBoolean("matched"))
    }

    @Test fun anEmptyFetchCannotPromoteASearchExcerptToFetchedEvidence() {
        val evidence = WebEvidence.context(plan(), JSONArray().put(row(primaryUrl, "只有搜索摘要")),
            JSONArray().put(row(primaryUrl, "   ", "Empty fetch")), 1)
        val passage = evidence.getJSONArray("passages").getJSONObject(0)
        assertEquals("只有搜索摘要", passage.getString("text"))
        assertEquals("search_excerpt", passage.getString("retrieval_kind"))
    }

    @Test fun hashesDescribeRetrievedTextAndAHumanReadableDomainDoesNotVerifyAuthorityOrIdentity() {
        val source = row(primaryUrl, "仅取得文本；其他信息可能在图片中。").put("truncated", true)
        val evidence = WebEvidence.context(plan(), JSONArray().put(source), JSONArray().put(source), 1)
        val passage = evidence.getJSONArray("passages").getJSONObject(0)
        assertEquals("docs.example.org", passage.getString("source_host"))
        assertEquals("unverified", passage.getString("authority"))
        assertFalse(passage.has("official_xiaomi"))
        assertFalse(passage.has("model_match"))
        assertFalse(passage.getBoolean("images_reviewed"))
        assertTrue(passage.getBoolean("truncated"))
        assertEquals("retrieved_text_utf8", passage.getString("hash_scope"))
        assertEquals(hash(source.getString("text")), passage.getString("sha256"))
        assertTrue(passage.isNull("page"))
        assertFalse(passage.has("model"))
        assertFalse(evidence.getJSONObject("source_check").has("original_pdf_sha256"))
        assertEquals("retrieved_unverified", evidence.getJSONObject("source_check").getString("status"))
        val other = WebEvidence.context(plan(), JSONArray().put(row(otherUrl, "候选")), JSONArray(), 1)
        assertEquals("support.example.net", other.getJSONArray("passages").getJSONObject(0).getString("source_host"))
        assertEquals("unverified", other.getJSONArray("passages").getJSONObject(0).getString("authority"))
    }

    @Test fun legacySearchPlanDoesNotCarryADeviceTaskIntoNewEvidence() {
        val previous = plan().put("requires_model", true).put("model", "DEVICE-42")
        val result = WebEvidence.context(previous, JSONArray(), JSONArray(), 1)
        assertEquals(setOf("needs_search", "query"), result.getJSONObject("search_plan").keys().asSequence().toSet())
        assertTrue(result.getJSONObject("search_plan").getBoolean("needs_search"))
        assertEquals(previous.getString("query"), result.getString("query"))
    }

    @Test fun skippedSearchHasNoPassagesOrProvider() {
        val skipped = JSONObject().put("needs_search", false).put("query", "")
        val evidence = WebEvidence.context(skipped, JSONArray(), JSONArray(), 1)
        assertFalse(evidence.getBoolean("matched"))
        assertEquals("not_requested", evidence.getString("transport"))
        assertTrue(evidence.isNull("provider"))
        assertEquals(0, evidence.getJSONArray("passages").length())
    }

    @Test fun sourceNoteListsOnlyActualCitationsOrClearlyLabeledCandidates() {
        val thirdUrl = "https://example.edu/page-c"
        val evidence = WebEvidence.context(plan(), JSONArray().put(row(primaryUrl, "A", "First candidate"))
            .put(row(otherUrl, "B", "Other candidate")).put(row(thirdUrl, "C", "Third candidate")), JSONArray(), 1)
        val pages = evidence.getJSONArray("passages")
        val cited = WebEvidence.sourceNote(evidence, answer(listOf(pages.getJSONObject(1).getString("citation_id"))))
        assertTrue(cited.contains("资料来源") && cited.contains(otherUrl))
        assertFalse(cited.contains(primaryUrl) || cited.contains(thirdUrl))
        assertEquals("", WebEvidence.sourceNote(evidence, answer(listOf("web-discovery:invented"))))
        val candidates = WebEvidence.sourceNote(evidence, answer(emptyList()))
        assertTrue(candidates.contains("检索候选"))
        assertTrue(candidates.contains(primaryUrl) && candidates.contains(otherUrl))
        assertFalse(candidates.contains(thirdUrl))
        assertEquals("", WebEvidence.sourceNote(WebEvidence.context(plan(), JSONArray(), JSONArray(), 1), answer(emptyList())))
    }

    @Test fun contextsKeepOnlyRetrievedHitUrlsAndCopyThePlan() {
        val original = plan()
        val hits = JSONArray().put(row(primaryUrl, "A")).put(row(primaryUrl, "duplicate"))
        val evidence = WebEvidence.context(original, hits, JSONArray().put(row(otherUrl, "unrequested fetch")), 1)
        val pages = evidence.getJSONArray("passages")
        assertEquals(1, pages.length())
        assertEquals(primaryUrl, pages.getJSONObject(0).getString("source_url"))
        assertEquals("search_excerpt", pages.getJSONObject(0).getString("retrieval_kind"))
        assertEquals("web-discovery:" + pages.getJSONObject(0).getString("section_id"), pages.getJSONObject(0).getString("citation_id"))
        evidence.getJSONObject("search_plan").put("query", "changed")
        assertEquals("reference documentation", original.getString("query"))
    }
}
