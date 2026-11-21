package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ExaMcpTest {
    private fun rejects(block: () -> Unit) {
        try { block(); fail("Accepted invalid MCP data") } catch (_: Exception) { }
    }
    private fun result(text: String) = JSONObject().put("content", JSONArray().put(JSONObject().put("type", "text").put("text", text)))
    private fun envelope(id: Any, result: JSONObject) = JSONObject().put("jsonrpc", "2.0").put("id", id).put("result", result).toString()
    private fun searchText(title: String = "Manufacturer guide", url: String = "https://example.org/manual") =
        "Title: $title\nURL: $url\nPublished: N/A\nAuthor: N/A\nHighlights:\nRead the model label before following a manual."

    @Test fun requestsUseTheObservedPublicToolSchemaWithoutCredentials() {
        val init = ExaMcp.initialize(1)
        assertEquals("initialize", init.getString("method"))
        assertEquals(ExaMcp.PROTOCOL, init.getJSONObject("params").getString("protocolVersion"))
        assertFalse(ExaMcp.initialized().has("id"))
        assertEquals("tools/list", ExaMcp.listTools(2).getString("method"))
        val search = ExaMcp.search(3, "runtime query for original documentation")
        val args = search.getJSONObject("params").getJSONObject("arguments")
        assertEquals(setOf("query", "objective", "numResults"), args.keys().asSequence().toSet())
        assertEquals(4, args.getInt("numResults"))
        assertTrue(args.getString("objective").contains("supplied query"))
        val fetch = ExaMcp.fetch(4, listOf("https://example.org/manual"))
        assertEquals("web_fetch_exa", fetch.getJSONObject("params").getString("name"))
        assertEquals(16000, fetch.getJSONObject("params").getJSONObject("arguments").getInt("maxCharacters"))
        rejects { ExaMcp.search(1, "\nmalformed") }
        rejects { ExaMcp.search(1, "x".repeat(501)) }
        rejects { ExaMcp.fetch(1, listOf("https://example.org/a", "https://example.org/a")) }
        rejects { ExaMcp.fetch(1, emptyList()) }
    }

    @Test fun jsonResponseBindsIdentityAndRejectsErrorsMalformedUtf8AndTrailingData() {
        val expected = JSONObject().put("tools", JSONArray())
        assertEquals(expected.toString(), ExaMcp.response(envelope(1, expected).toByteArray(), 1).toString())
        rejects { ExaMcp.response(envelope(2, expected).toByteArray(), 1) }
        rejects { ExaMcp.response(envelope("1", expected).toByteArray(), 1) }
        rejects { ExaMcp.response(envelope(1, JSONObject().put("isError", true)).toByteArray(), 1) }
        rejects { ExaMcp.response(envelope(1, JSONObject().put("isError", "false")).toByteArray(), 1) }
        rejects { ExaMcp.response("""{"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"private error"}}""".toByteArray(), 1) }
        rejects { ExaMcp.response((envelope(1, expected) + " {} ").toByteArray(), 1) }
        rejects { ExaMcp.response(byteArrayOf(0xc3.toByte(), 0x28), 1) }
        rejects { ExaMcp.response(ByteArray(ExaMcp.MAX_RESPONSE_BYTES + 1), 1) }
    }

    @Test fun sseAllowsNotificationsButOnlyOneMatchingResponse() {
        val response = envelope(7, JSONObject().put("ok", true))
        val notification = """{"jsonrpc":"2.0","method":"notifications/progress","params":{}}"""
        val stream = ": keepalive\r\n\r\nevent: message\r\ndata: $notification\r\n\r\nevent: message\r\ndata: $response\r\n\r\n"
        assertTrue(ExaMcp.response(stream.toByteArray(), 7).getBoolean("ok"))
        rejects { ExaMcp.response((stream + "event: message\ndata: $response\n\n").toByteArray(), 7) }
        rejects { ExaMcp.response("event: message\ndata: $notification\n\n".toByteArray(), 7) }
        rejects { ExaMcp.response("event: message\ndata: ${envelope(8, JSONObject())}\n\n".toByteArray(), 7) }
    }

    @Test fun observedSearchHighlightsStaySeparateFromFetchedPageClaims() {
        val second = searchText("Second source", "https://example.org/second")
        val rows = ExaMcp.searchResults(result(searchText() + "\n\n---\n\n" + second))
        assertEquals(2, rows.length())
        assertEquals("Manufacturer guide", rows.getJSONObject(0).getString("title"))
        assertEquals("https://example.org/manual", rows.getJSONObject(0).getString("url"))
        assertTrue(rows.getJSONObject(0).getString("text").startsWith("Read the model"))
        assertEquals("search_highlights", rows.getJSONObject(0).getString("source_kind"))
        assertFalse(rows.getJSONObject(0).has("manual_id"))
        assertFalse(rows.getJSONObject(0).has("model"))
        assertFalse(rows.getJSONObject(0).getBoolean("truncated"))
        rejects { ExaMcp.searchResults(result("Answer without any source URL")) }
        rejects { ExaMcp.searchResults(result(searchText().replace("Highlights:", "Unknown format:"))) }
    }

    @Test fun fetchedContentMustEchoTheRequestedPublicHttpsSource() {
        val raw = "# Manufacturer guide\nURL: https://example.org/manual\n\n# Cleaning\nUnplug before maintenance."
        val rows = ExaMcp.fetchResults(result(raw), listOf("https://example.org/manual"))
        assertEquals("fetched_page", rows.getJSONObject(0).getString("source_kind"))
        assertEquals("# Cleaning\nUnplug before maintenance.", rows.getJSONObject(0).getString("text"))
        rejects { ExaMcp.fetchResults(result(raw), listOf("https://example.org/other")) }
        for (url in listOf("http://example.org/a", "https://user:pass@example.org/a", "https://localhost/a",
            "https://127.0.0.1/a", "https://192.168.1.2/a", "https://[::1]/a", "https://printer.local/a",
            "https://example.org:8443/a", "https://example.org/a#fragment", "https://example.org/a b")) {
            rejects { ExaMcp.fetch(1, listOf(url)) }
            val rejected = mutableListOf<String>()
            assertEquals(0, ExaMcp.searchResults(result(searchText(url = url))) { rejected += it }.length())
            assertEquals(listOf("unsupported_source_url"), rejected)
            rejects { ExaMcp.fetchResults(result("# Guide\nURL: $url\n\nFetched body"), listOf("https://example.org/manual")) }
        }
        rejects { ExaMcp.fetch(1, listOf("https://example.org/a\n")) }
    }

    @Test fun mixedSearchCandidatesKeepHttpsSourcesWithoutInventingHttpsReplacements() {
        val safeOne = "https://example.org/first"
        val safeTwo = "https://example.org/second"
        val text = listOf(searchText("First", safeOne), searchText("Legacy", "http://legacy.example.org/manual"),
            searchText("Second", safeTwo)).joinToString("\n\n---\n\n")
        val rejected = mutableListOf<String>()
        val rows = ExaMcp.searchResults(result(text)) { rejected += it }
        assertEquals(listOf(safeOne, safeTwo), (0 until rows.length()).map { rows.getJSONObject(it).getString("url") })
        assertEquals(listOf("unsupported_source_url"), rejected)
        assertFalse(rows.toString().contains("legacy.example.org"))
    }

    @Test fun entirelyUnsupportedStructuredSearchResultsAreEmptyAndReportOnlyFixedReasons() {
        val entries = JSONArray().put(JSONObject().put("title", "Legacy").put("url", "http://legacy.example.org/a").put("text", "A"))
            .put(JSONObject().put("title", "Credentials").put("url", "https://user:private@example.org/a").put("text", "B"))
        val rejected = mutableListOf<String>()
        val rows = ExaMcp.searchResults(JSONObject().put("structuredContent", JSONObject().put("results", entries))) { rejected += it }
        assertEquals(0, rows.length())
        assertEquals(listOf("unsupported_source_url", "unsupported_source_url"), rejected)
        assertFalse(rejected.toString().contains("private"))
        entries.getJSONObject(0).put("text", "")
        rejects { ExaMcp.searchResults(JSONObject().put("structuredContent", JSONObject().put("results", entries))) }
    }

    @Test fun structuredResultsAreBoundedExplicitlyTruncatedAndDeduplicated() {
        val data = JSONArray()
        for (i in 0..5) data.put(JSONObject().put("title", "Guide $i").put("url", "https://example.org/$i").put("text", "a".repeat(17000)))
        val structured = JSONObject().put("structuredContent", JSONObject().put("results", data))
        val rows = ExaMcp.searchResults(structured)
        assertEquals(4, rows.length())
        assertEquals(16000, rows.getJSONObject(0).getString("text").length)
        assertTrue(rows.getJSONObject(0).getBoolean("truncated"))
        val duplicate = result(searchText() + "\n\n---\n\n" + searchText())
        assertEquals(1, ExaMcp.searchResults(duplicate).length())
        assertEquals(0, ExaMcp.searchResults(JSONObject().put("structuredContent", JSONObject().put("results", JSONArray()))).length())
        rejects { ExaMcp.searchResults(structured.put("isError", true)) }
    }

    @Test fun handshakeRejectsMissingOrChangedRequiredToolArguments() {
        ExaMcp.validateInitialize(JSONObject().put("protocolVersion", ExaMcp.PROTOCOL).put("capabilities", JSONObject().put("tools", JSONObject())))
        rejects { ExaMcp.validateInitialize(JSONObject().put("protocolVersion", "2099-01-01").put("capabilities", JSONObject())) }
        fun tool(name: String, required: List<String>) = JSONObject().put("name", name).put("inputSchema", JSONObject()
            .put("type", "object").put("required", JSONArray(required)).put("properties", JSONObject().apply {
                required.forEach { put(it, JSONObject().put("type", "string")) }
            }))
        val search = tool("web_search_exa", listOf("query", "objective"))
        val fetch = tool("web_fetch_exa", listOf("urls"))
        val catalog = JSONObject().put("tools", JSONArray().put(search).put(fetch))
        ExaMcp.validateTools(catalog)
        rejects { ExaMcp.validateTools(JSONObject().put("tools", JSONArray().put(search))) }
        search.getJSONObject("inputSchema").getJSONArray("required").put("credential")
        rejects { ExaMcp.validateTools(catalog) }
    }
}
