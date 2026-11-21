package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.net.URI

/** Search hits are candidates. Only fetch results provide page text. */
object WebEvidence {
    private fun hash(value: String) = MessageDigest.getInstance("SHA-256").digest(value.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it) }

    fun context(plan: JSONObject, hits: JSONArray, fetched: JSONArray, atMs: Long): JSONObject {
        val byUrl = (0 until fetched.length()).map { fetched.getJSONObject(it) }.associateBy { it.getString("url") }
        val passages = JSONArray()
        val seen = HashSet<String>()
        for (i in 0 until hits.length()) {
            if (passages.length() == 4) break
            val hit = hits.getJSONObject(i)
            val url = hit.getString("url")
            if (!seen.add(url)) continue
            val page = byUrl[url]?.takeIf { it.optString("text").isNotBlank() }
            val title = (page?.optString("title")?.takeIf { it.isNotBlank() } ?: hit.getString("title")).take(240)
            val text = (page?.optString("text")?.takeIf { it.isNotBlank() } ?: hit.optString("text")).take(16000)
            if (text.isBlank()) continue
            val section = "source-${hash(url).take(16)}"
            passages.put(JSONObject().put("manual_id", "web-discovery").put("section_id", section)
                .put("citation_id", "web-discovery:$section").put("title", title).put("section_title", title)
                .put("source_url", url).put("sha256", hash(text)).put("hash_scope", "retrieved_text_utf8")
                .put("synthetic", false).put("page", JSONObject.NULL).put("text", text)
                .put("retrieval_kind", if (page == null) "search_excerpt" else "fetched_page")
                .put("truncated", (page ?: hit).optBoolean("truncated", false))
                .put("source_host", URI(url).host).put("authority", "unverified")
                .put("images_reviewed", false))
        }
        return JSONObject().put("schema", "satori.manual-context/1").put("manual_id", "web-discovery")
            .put("query", plan.getString("query")).put("passages", passages).put("matched", passages.length() > 0)
            .put("transport", if (plan.optBoolean("needs_search", true)) "mcp_streamable_http" else "not_requested")
            .put("provider", if (plan.optBoolean("needs_search", true)) "exa_keyless" else JSONObject.NULL)
            .put("retrieved_unix_ms", atMs).put("search_plan", JSONObject()
                .put("needs_search", plan.getBoolean("needs_search")).put("query", plan.getString("query")))
            .put("source_check", JSONObject().put("status", if (plan.getBoolean("needs_search")) "retrieved_unverified" else "not_requested"))
    }

    fun sourceNote(context: JSONObject, answer: JSONObject): String {
        val ids = answer.optJSONArray("source_sections") ?: JSONArray()
        val cited = (0 until ids.length()).map { ids.optString(it) }.toSet()
        val passages = context.optJSONArray("passages") ?: return ""
        val rows = (0 until passages.length()).map { passages.getJSONObject(it) }
        val selected = (if (cited.isEmpty()) rows.take(2) else rows.filter { it.optString("citation_id") in cited }).take(3)
        if (selected.isEmpty()) return ""
        return "\n\n${if (cited.isEmpty()) "检索候选" else "资料来源"}：\n" + selected.joinToString("\n") {
            it.getString("title").take(80) + "\n" + it.getString("source_url")
        }
    }
}
