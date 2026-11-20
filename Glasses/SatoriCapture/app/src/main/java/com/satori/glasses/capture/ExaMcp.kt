package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction

/** Exa's public Search MCP protocol only; transport, cancellation and credentials stay with the caller. */
object ExaMcp {
    const val ENDPOINT = "https://mcp.exa.ai/mcp"
    const val PROTOCOL = "2025-11-25"
    const val MAX_RESPONSE_BYTES = 1024 * 1024
    const val MAX_SOURCES = 4
    const val MAX_TEXT = 16000

    fun initialize(id: Int): JSONObject = request(id, "initialize", JSONObject()
        .put("protocolVersion", PROTOCOL).put("capabilities", JSONObject())
        .put("clientInfo", JSONObject().put("name", "satori-glasses").put("version", "1")))

    fun initialized(): JSONObject = JSONObject().put("jsonrpc", "2.0").put("method", "notifications/initialized")
    fun listTools(id: Int): JSONObject = request(id, "tools/list", JSONObject())

    fun search(id: Int, query: String): JSONObject {
        require(query.isNotBlank() && query.length <= 500 && query.none { it.isISOControl() })
        return request(id, "tools/call", JSONObject().put("name", "web_search_exa").put("arguments", JSONObject()
            .put("query", query).put("numResults", MAX_SOURCES)
            .put("objective", "Return relevant evidence and source URLs for the supplied query. Prefer reliable primary sources.")))
    }

    fun fetch(id: Int, urls: List<String>): JSONObject {
        require(urls.size in 1..MAX_SOURCES && urls.distinct().size == urls.size)
        urls.forEach(::httpsUrl)
        return request(id, "tools/call", JSONObject().put("name", "web_fetch_exa").put("arguments", JSONObject()
            .put("urls", JSONArray(urls)).put("maxCharacters", MAX_TEXT)))
    }

    fun validateInitialize(result: JSONObject) {
        require(result.getString("protocolVersion") == PROTOCOL) { "Unsupported MCP protocol" }
        require(result.getJSONObject("capabilities").optJSONObject("tools") != null) { "MCP tools unavailable" }
    }

    fun validateTools(result: JSONObject) {
        val tools = result.getJSONArray("tools")
        require(tools.length() in 1..64 && !result.has("nextCursor")) { "Incomplete MCP tool catalog" }
        for ((name, required) in mapOf("web_search_exa" to setOf("query", "objective"), "web_fetch_exa" to setOf("urls"))) {
            val tool = (0 until tools.length()).map { tools.getJSONObject(it) }.single { it.getString("name") == name }
            val schema = tool.getJSONObject("inputSchema")
            require(schema.getString("type") == "object")
            val properties = schema.getJSONObject("properties")
            val actual = schema.optJSONArray("required") ?: JSONArray()
            require((0 until actual.length()).map { actual.getString(it) }.toSet() == required) { "MCP input contract changed" }
            require(required.all { properties.has(it) })
        }
    }

    /** Accept JSON or bounded SSE messages, ignore only JSON-RPC notifications, and bind the response id. */
    fun response(raw: ByteArray, expectedId: Int): JSONObject {
        require(expectedId >= 0 && raw.size in 1..MAX_RESPONSE_BYTES)
        val text = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(raw)).toString().trim()
        val envelopes = if (text.startsWith("{")) listOf(objectFrom(text)) else {
            val events = text.replace("\r\n", "\n").replace('\r', '\n').split(Regex("\n\n+"))
            require(events.size <= 64)
            events.mapNotNull { event ->
                val data = event.lines().filter { it.startsWith("data:") }.map { it.removePrefix("data:").removePrefix(" ") }
                if (data.isEmpty()) null else objectFrom(data.joinToString("\n"))
            }
        }
        var answer: JSONObject? = null
        for (envelope in envelopes) {
            require(envelope.getString("jsonrpc") == "2.0")
            if (!envelope.has("id")) {
                require(envelope.optString("method").isNotBlank() && !envelope.has("result") && !envelope.has("error"))
                continue
            }
            val id = envelope.get("id")
            require(id is Number && id.toDouble() == expectedId.toDouble()) { "MCP response id mismatch" }
            require(answer == null && !envelope.has("error")) { "MCP request failed or duplicated" }
            answer = envelope.getJSONObject("result")
            require(!answer.has("isError") || answer.get("isError") == false) { "MCP tool failed" }
        }
        return answer ?: throw IllegalArgumentException("MCP response missing")
    }

    fun searchResults(result: JSONObject, onRejected: (String) -> Unit = {}): JSONArray =
        results(result, fetch = false, requestedUrls = emptyList(), onRejected = onRejected)
    fun fetchResults(result: JSONObject, requestedUrls: List<String>): JSONArray {
        require(requestedUrls.size in 1..MAX_SOURCES)
        requestedUrls.forEach(::httpsUrl)
        return results(result, fetch = true, requestedUrls = requestedUrls)
    }

    private fun results(result: JSONObject, fetch: Boolean, requestedUrls: List<String>, onRejected: (String) -> Unit = {}): JSONArray {
        require(!result.has("isError") || result.get("isError") == false)
        val structured = result.optJSONObject("structuredContent")
        val rows = structured?.optJSONArray("results")
        val entries = if (rows != null) {
            require(rows.length() <= 32)
            (0 until rows.length()).map { row -> rows.getJSONObject(row).let {
                Triple(it.getString("title"), it.getString("url"), it.getString("text"))
            } }
        } else {
            val content = result.getJSONArray("content")
            require(content.length() <= 8)
            (0 until content.length()).flatMap { i ->
                val item = content.getJSONObject(i)
                require(item.getString("type") == "text")
                val text = item.getString("text").replace("\r\n", "\n").trim()
                require(text.length <= MAX_RESPONSE_BYTES)
                if (text.isBlank()) emptyList() else parseText(text, fetch)
            }
        }
        require(entries.size <= 32)
        val seen = HashSet<String>()
        return JSONArray().apply {
            for ((title, url, body) in entries) {
                require(title.isNotBlank() && title.length <= 1000 && title.none { it.isISOControl() })
                require(body.isNotBlank())
                if (fetch) httpsUrl(url) else {
                    try { httpsUrl(url) } catch (_: Exception) {
                        onRejected("unsupported_source_url")
                        continue
                    }
                }
                if (fetch) require(url in requestedUrls) { "Fetched source URL changed" }
                if (!seen.add(url) || length() >= MAX_SOURCES) continue
                put(JSONObject().put("title", title.take(240)).put("url", url).put("text", body.take(MAX_TEXT))
                    .put("truncated", body.length > MAX_TEXT).put("source_kind", if (fetch) "fetched_page" else "search_highlights"))
            }
        }
    }

    private fun parseText(text: String, fetch: Boolean): List<Triple<String, String, String>> {
        val prefix = if (fetch) "# " else "Title: "
        val chunks = text.split(Regex("\n\n---\n\n(?=${Regex.escape(prefix)})"))
        require(chunks.size <= 32)
        return chunks.map { block ->
            val lines = block.lines()
            require(lines.size >= 3 && lines[0].startsWith(prefix) && lines[1].startsWith("URL: ")) { "MCP source format changed" }
            val title = lines[0].removePrefix(prefix).trim()
            val url = lines[1].removePrefix("URL: ").trim()
            val body = if (fetch) lines.drop(2).joinToString("\n").trim() else {
                val contentLine = lines.indexOfFirst { it == "Highlights:" || it == "Text:" || it == "Content:" }
                require(contentLine >= 2) { "Search text missing" }
                lines.drop(contentLine + 1).joinToString("\n").trim()
            }
            Triple(title, url, body)
        }
    }

    fun httpsUrl(value: String): String {
        require(value.length in 1..2048 && value.none { it.isWhitespace() || it.isISOControl() || it == '\\' })
        val uri = URI(value)
        val host = uri.host.orEmpty().lowercase()
        require(uri.scheme == "https" && host.contains('.') && !host.endsWith(".local") && !host.endsWith(".localhost") &&
            !host.matches(Regex("[0-9.]+")) && !host.contains(':') && uri.rawUserInfo == null && uri.rawFragment == null &&
            (uri.port == -1 || uri.port == 443)) { "Only public HTTPS source URLs are accepted" }
        return value
    }

    private fun request(id: Int, method: String, params: JSONObject): JSONObject {
        require(id >= 0)
        return JSONObject().put("jsonrpc", "2.0").put("id", id).put("method", method).put("params", params)
    }

    private fun objectFrom(text: String): JSONObject {
        val parser = JSONTokener(text)
        val result = parser.nextValue() as? JSONObject ?: throw IllegalArgumentException("MCP object required")
        require(parser.nextClean() == '\u0000') { "Trailing MCP data" }
        return result
    }
}
