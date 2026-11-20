package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.net.URL
import java.security.cert.Certificate
import javax.net.ssl.HttpsURLConnection

class WebResearchApiTest {
    private val firstSource = "https://example.org/page-a"
    private val secondSource = "https://docs.example.net/page-b"
    private val thirdSource = "https://support.example.com/page-c"
    private val fourthSource = "https://example.edu/page-d"
    private val modelSecret = "private-model-key-never-send-to-exa"
    private val session = "fixture-public-mcp-session"
    private fun profile() = StandaloneProfile("https://model.invalid/v1", modelSecret, "planner", "monitor",
        manualId = "web-discovery", manualSource = "web", backupApiKey = "private-backup-key-never-send-to-exa")
    private fun plan() = JSONObject().put("needs_search", true).put("query", "document revision procedure reference")

    private fun envelope(id: Int, result: JSONObject) = "event: message\ndata: " +
        JSONObject().put("jsonrpc", "2.0").put("id", id).put("result", result).toString() + "\n\n"
    private fun textResult(text: String) = JSONObject().put("content", JSONArray()
        .put(JSONObject().put("type", "text").put("text", text)))
    private fun initialize() = FakeConnection(200, envelope(1, JSONObject().put("protocolVersion", ExaMcp.PROTOCOL)
        .put("capabilities", JSONObject().put("tools", JSONObject()))), session)
    private fun catalog(): JSONObject {
        fun tool(name: String, required: List<String>) = JSONObject().put("name", name).put("inputSchema", JSONObject()
            .put("type", "object").put("required", JSONArray(required)).put("properties", JSONObject().apply {
                required.forEach { put(it, JSONObject().put("type", if (it == "urls") "array" else "string")) }
            }))
        return JSONObject().put("tools", JSONArray().put(tool("web_search_exa", listOf("query", "objective")))
            .put(tool("web_fetch_exa", listOf("urls"))))
    }
    private fun handshake() = mutableListOf(initialize(), FakeConnection(202, ""), FakeConnection(200, envelope(2, catalog())))
    private fun searchText() = listOf(firstSource, secondSource, thirdSource, fourthSource).mapIndexed { index, url ->
        "Title: Candidate ${index + 1}\nURL: $url\nPublished: N/A\nAuthor: N/A\nHighlights:\nSearch excerpt ${index + 1}."
    }.joinToString("\n\n---\n\n")
    private fun fetchedText(url: String = firstSource) = "# Retrieved document\nURL: $url\n\nFetched page text: reference contents for the current question."
    private fun fetchedPages() = listOf(firstSource, secondSource, thirdSource).joinToString("\n\n---\n\n") { fetchedText(it) }

    @Test fun anonymousMcpLifecyclePreservesSessionsAndSeparatesExcerptsFromFetchedText() {
        val connections = handshake().apply {
            add(FakeConnection(200, envelope(3, textResult(searchText()))))
            add(FakeConnection(200, envelope(4, textResult(fetchedPages()))))
        }
        val addresses = mutableListOf<String>()
        var opened = 0
        StandaloneApi(profile(), connectionFactory = { url -> addresses += url; connections[opened++] }).use { api ->
            val context = api.researchWeb(plan())
            assertTrue(context.getBoolean("matched"))
            assertEquals("mcp_streamable_http", context.getString("transport"))
            assertEquals("web-discovery", context.getString("manual_id"))
            assertEquals("retrieved_unverified", context.getJSONObject("source_check").getString("status"))
            val passages = context.getJSONArray("passages")
            assertEquals(4, passages.length())
            val page = passages.getJSONObject(0)
            assertEquals(firstSource, page.getString("source_url"))
            assertEquals("fetched_page", page.getString("retrieval_kind"))
            assertTrue(page.getString("text").startsWith("Fetched page text"))
            assertEquals("retrieved_text_utf8", page.getString("hash_scope"))
            assertTrue(page.getString("sha256").matches(Regex("[a-f0-9]{64}")))
            assertEquals("example.org", page.getString("source_host"))
            assertEquals("unverified", page.getString("authority"))
            assertFalse(page.has("official_xiaomi"))
            assertFalse(page.has("model"))
            assertFalse(page.getBoolean("images_reviewed"))
            assertTrue(page.isNull("page"))
            assertEquals(listOf(firstSource, secondSource, thirdSource, fourthSource),
                (0 until passages.length()).map { passages.getJSONObject(it).getString("source_url") })
            assertEquals("fetched_page", passages.getJSONObject(1).getString("retrieval_kind"))
            assertEquals("fetched_page", passages.getJSONObject(2).getString("retrieval_kind"))
            assertEquals("search_excerpt", passages.getJSONObject(3).getString("retrieval_kind"))
            val answer = JSONObject().put("kind", "answer").put("text", "根据本次取得的资料回答。")
                .put("source_sections", JSONArray().put(page.getString("citation_id"))).put("suggest_replan", false)
            assertEquals("answer", StandaloneManuals.answer(answer, context).getString("kind"))
        }
        assertEquals(5, opened)
        assertTrue(addresses.all { it == ExaMcp.ENDPOINT })
        val requests = connections.map { JSONObject(it.sent.toString("UTF-8")) }
        assertEquals(listOf("initialize", "notifications/initialized", "tools/list", "tools/call", "tools/call"), requests.map { it.getString("method") })
        assertFalse(requests[1].has("id"))
        val args = requests[3].getJSONObject("params").getJSONObject("arguments")
        assertEquals(plan().getString("query"), args.getString("query"))
        assertTrue(args.getString("objective").isNotBlank())
        assertEquals(4, args.getInt("numResults"))
        val urls = requests[4].getJSONObject("params").getJSONObject("arguments").getJSONArray("urls")
        assertEquals(3, urls.length())
        assertEquals(listOf(firstSource, secondSource, thirdSource), (0 until urls.length()).map { urls.getString(it) })
        for ((i, conn) in connections.withIndex()) {
            assertEquals("POST", conn.requestMethod)
            assertEquals("application/json, text/event-stream", conn.getRequestProperty("Accept"))
            assertEquals("application/json", conn.getRequestProperty("Content-Type"))
            assertEquals(ExaMcp.PROTOCOL, conn.getRequestProperty("MCP-Protocol-Version"))
            assertEquals(if (i == 0) null else session, conn.getRequestProperty("Mcp-Session-Id"))
            assertNull(conn.getRequestProperty("Authorization"))
            assertNull(conn.getRequestProperty("x-api-key"))
            assertFalse(conn.sent.toString("UTF-8").contains(modelSecret))
            assertFalse(conn.instanceFollowRedirects)
            assertTrue(conn.disconnected)
        }
    }

    @Test fun aPlanWithoutExternalInformationNeedsDoesNotOpenAnyConnectionOrEmitMcpEvents() {
        var opened = 0
        val events = mutableListOf<String>()
        StandaloneApi(profile(), connectionFactory = { opened++; throw AssertionError("no-search plan opened a connection") }).use { api ->
            val context = api.researchWeb(plan().put("needs_search", false).put("query", "")) { name, _ -> events += name }
            assertEquals("not_requested", context.getString("transport"))
            assertFalse(context.getBoolean("matched"))
            assertEquals(0, context.getJSONArray("passages").length())
        }
        assertEquals(0, opened)
        assertTrue(events.isEmpty())
    }

    @Test fun validatedMcpSessionIsReusedForTheNextSearchWithoutReinitializing() {
        val connections = handshake().apply {
            add(FakeConnection(200, envelope(3, textResult(searchText()))))
            add(FakeConnection(200, envelope(4, textResult(fetchedText()))))
            add(FakeConnection(200, envelope(5, textResult(searchText()))))
            add(FakeConnection(200, envelope(6, textResult(fetchedText()))))
        }
        var opened = 0
        StandaloneApi(profile(), connectionFactory = { connections[opened++] }).use { api ->
            assertTrue(api.researchWeb(plan()).getBoolean("matched"))
            assertTrue(api.researchWeb(plan().put("query", "A different document reference question")).getBoolean("matched"))
        }
        assertEquals(7, opened)
        val next = JSONObject(connections[5].sent.toString("UTF-8"))
        assertEquals("web_search_exa", next.getJSONObject("params").getString("name"))
        assertEquals(5, next.getInt("id"))
        assertEquals(session, connections[5].getRequestProperty("Mcp-Session-Id"))
    }

    @Test fun emptySearchDoesNotPretendToReadAPageOrCallFetch() {
        val connections = handshake().apply { add(FakeConnection(200, envelope(3, JSONObject().put("content", JSONArray())))) }
        var opened = 0
        StandaloneApi(profile(), connectionFactory = { connections[opened++] }).use { api ->
            val context = api.researchWeb(plan())
            assertFalse(context.getBoolean("matched"))
            assertEquals(0, context.getJSONArray("passages").length())
        }
        assertEquals(4, opened)
    }

    @Test fun redirectsRateLimitsAndMcpToolErrorsDoNotBecomeSuccessfulEvidence() {
        for (status in listOf(302, 429)) {
            val connection = FakeConnection(status, "")
            var opened = 0
            StandaloneApi(profile(), connectionFactory = { opened++; connection }).use { api ->
                val failure = failure { api.researchWeb(plan()) }
                assertEquals(status, failure.status)
                if (status == 429) assertTrue(failure.retryAfterMs >= 60_000)
                assertFalse(failure.message.orEmpty().contains(modelSecret))
            }
            assertEquals(1, opened)
            assertFalse(connection.instanceFollowRedirects)
            assertNull(connection.getRequestProperty("Authorization"))
        }
        val connections = handshake().apply {
            add(FakeConnection(200, envelope(3, JSONObject().put("isError", true)
                .put("content", JSONArray().put(JSONObject().put("type", "text").put("text", "upstream private error"))))))
        }
        var opened = 0
        StandaloneApi(profile(), connectionFactory = { connections[opened++] }).use { api ->
            val failure = failure { api.researchWeb(plan()) }
            assertFalse(failure.message.orEmpty().contains("upstream private error"))
        }
        assertEquals(4, opened)
    }

    @Test fun mismatchedFetchSourceOrSessionInjectionFailsClosed() {
        val connections = handshake().apply {
            add(FakeConnection(200, envelope(3, textResult(searchText()))))
            add(FakeConnection(200, envelope(4, textResult(fetchedText("https://attacker.invalid/forged")))))
        }
        var opened = 0
        StandaloneApi(profile(), connectionFactory = { connections[opened++] }).use { api -> failure { api.researchWeb(plan()) } }
        assertEquals(5, opened)
        var injectedCalls = 0
        val badSession = FakeConnection(200, envelope(1, JSONObject()), "bad\r\nAuthorization: secret")
        StandaloneApi(profile(), connectionFactory = { injectedCalls++; badSession }).use { api -> failure { api.researchWeb(plan()) } }
        assertEquals(1, injectedCalls)
    }

    @Test fun webProfileCannotCarryLegacyManualCredentialsOrASelectedFixedManual() {
        profile().validate()
        for (invalid in listOf(profile().copy(toolsToken = "t".repeat(32)), profile().copy(toolsBaseUrl = "https://manual.invalid"),
            profile().copy(manualId = "tl-sg105-v4"))) {
            try { invalid.validate(); fail("web profile accepted legacy credentials or fixed manual") }
            catch (_: IllegalArgumentException) { }
        }
        profile().copy(manualSource = "remote_api", manualId = "tl-sg105-v4", toolsBaseUrl = "https://manual.invalid",
            toolsToken = "t".repeat(32)).validate()
    }

    private fun failure(block: () -> Unit): ApiFailure {
        try { block(); fail("Expected failed web research") } catch (failure: ApiFailure) { return failure }
        throw AssertionError("unreachable")
    }

    private class FakeConnection(private val status: Int, text: String, private val responseSession: String? = null) :
        HttpsURLConnection(URL(ExaMcp.ENDPOINT)) {
        val sent = ByteArrayOutputStream()
        private val response = text.toByteArray(Charsets.UTF_8)
        var disconnected = false
        override fun getOutputStream() = sent
        override fun getInputStream() = ByteArrayInputStream(response)
        override fun getErrorStream() = ByteArrayInputStream(response)
        override fun getResponseCode() = status
        override fun getContentLengthLong() = response.size.toLong()
        override fun getHeaderField(name: String?): String? = if (name.equals("Mcp-Session-Id", true)) responseSession else null
        override fun connect() = Unit
        override fun disconnect() { disconnected = true }
        override fun usingProxy() = false
        override fun getCipherSuite() = "fixture-no-network"
        override fun getLocalCertificates(): Array<Certificate>? = null
        override fun getServerCertificates(): Array<Certificate> = emptyArray()
    }
}
