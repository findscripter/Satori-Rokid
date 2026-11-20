package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.net.URI
import java.net.HttpURLConnection
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Base64
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Callable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutionException
import java.util.concurrent.Future
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicLong
import javax.net.ssl.HttpsURLConnection

class ApiFailure(val status: Int = 0, val retryAfterMs: Long = 0, message: String, val diagnostic: String = "") : Exception(message)

/** Bounded HTTPS transport. The caller gets an absolute deadline, including serialization, DNS and slow responses.
 * DNS itself cannot always be interrupted by Android; at most two bounded workers may remain in OS code.
 * No raw provider response or request headers enter exceptions or logs. */
class StandaloneApi(
    private val profile: StandaloneProfile,
    private val manualAsset: ((String) -> ByteArray)? = null,
    private val onTransportEvent: (String, JSONObject) -> Unit = { _, _ -> },
    private val connectionFactory: (String) -> HttpURLConnection = { URI(it).toURL().openConnection() as HttpURLConnection },
) : Closeable {
    private val base = normalizeBaseUrl(profile.baseUrl)
    init { profile.validate() }
    private val closed = AtomicBoolean()
    private val cancellationGeneration = AtomicLong()
    private val workers = ThreadPoolExecutor(2, 2, 30, TimeUnit.SECONDS, ArrayBlockingQueue<Runnable>(2),
        { task -> Thread(task, "standalone-https").apply { isDaemon = true } }, ThreadPoolExecutor.AbortPolicy())
    private val timer = ScheduledThreadPoolExecutor(1) { task -> Thread(task, "standalone-deadline").apply { isDaemon = true } }
        .apply { removeOnCancelPolicy = true }
    private val closers = ThreadPoolExecutor(2, 2, 30, TimeUnit.SECONDS, ArrayBlockingQueue<Runnable>(4),
        { task -> Thread(task, "standalone-disconnect").apply { isDaemon = true } }, ThreadPoolExecutor.DiscardPolicy())
    private val active = ConcurrentHashMap.newKeySet<Call>()
    private val localManuals by lazy {
        val read = manualAsset ?: throw ApiFailure(message = "本机手册资源未加载")
        LocalManualStore(listOf("tl-sg105-v4.json", "classroom-demo.json").map(read))
    }
    private val verifiedManualSources = ConcurrentHashMap<String, String>()
    private var mcpSession: String? = null
    private var mcpReady = false
    private var mcpSequence = 0

    private class Call(val deadlineNs: Long, val permitted: () -> Boolean, val disconnect: (HttpURLConnection) -> Unit) {
        val cancelled = AtomicBoolean()
        @Volatile var connection: HttpURLConnection? = null
        @Volatile var future: Future<*>? = null
        @Volatile var phase = "queued"
        private val deadlinePhase = AtomicReference<String?>(null)
        fun timeoutFailure(): ApiFailure {
            deadlinePhase.compareAndSet(null, phase)
            return ApiFailure(message = "模型响应超时", diagnostic = "timeout_${deadlinePhase.get()}")
        }
        fun remaining(): Int {
            if (!permitted()) throw ApiFailure(message = "请求已取消", diagnostic = "cancelled")
            if (cancelled.get()) {
                if (System.nanoTime() >= deadlineNs) throw timeoutFailure()
                throw ApiFailure(message = "请求已取消")
            }
            val left = (deadlineNs - System.nanoTime()) / 1_000_000L
            if (left <= 0) throw timeoutFailure()
            return left.coerceAtMost(Int.MAX_VALUE.toLong()).toInt().coerceAtLeast(1)
        }
        fun cancel() {
            if (System.nanoTime() >= deadlineNs) deadlinePhase.compareAndSet(null, phase)
            cancelled.set(true)
            future?.cancel(true)
            connection?.let(disconnect)
        }
    }

    fun complete(model: String, prompt: String, payload: JSONObject, frames: List<ByteArray>, timeoutMs: Long = 75000): JSONObject {
        require(timeoutMs in 1000..180000)
        val started = System.nanoTime()
        val deadline = started + timeoutMs * 1_000_000L
        val ticket = cancellationGeneration.get()
        val callId = java.util.UUID.randomUUID().toString()
        fun permitted() = !closed.get() && ticket == cancellationGeneration.get() && !Thread.currentThread().isInterrupted
        fun attempt(slot: String, credential: String, until: Long): JSONObject {
            if (!permitted()) throw ApiFailure(message = "请求已取消", diagnostic = "cancelled")
            onTransportEvent("model_attempt", JSONObject().put("call_id", callId).put("credential_slot", slot)
                .put("remaining_budget_ms", ((until - System.nanoTime()) / 1_000_000L).coerceAtLeast(0)))
            return completeAttempt(model, prompt, payload, frames, until, ticket, credential)
        }
        val primaryDeadline = if (profile.backupApiKey.isEmpty()) deadline else started + timeoutMs * 600_000L
        return try { attempt("primary", profile.apiKey, primaryDeadline) }
        catch (failure: ApiFailure) {
            val timedOut = failure.diagnostic.startsWith("timeout_") || failure.status in setOf(408, 504)
            if (!timedOut || profile.backupApiKey.isEmpty() || !permitted() || System.nanoTime() >= deadline) throw failure
            onTransportEvent("model_failover", JSONObject().put("call_id", callId).put("from_slot", "primary").put("to_slot", "backup")
                .put("reason", if (failure.status in setOf(408, 504)) "http_timeout" else "deadline_timeout").put("http_status", failure.status)
                .put("diagnostic", failure.diagnostic))
            attempt("backup", profile.backupApiKey, deadline)
        }
    }

    private fun completeAttempt(model: String, prompt: String, payload: JSONObject, frames: List<ByteArray>,
                                deadlineNs: Long, ticket: Long, credential: String): JSONObject =
        boundedUntil(deadlineNs, ticket) { call ->
            require(model.isNotBlank() && model.length <= 200)
            require(prompt.length <= 64000 && payload.toString().length <= 256000)
            require(frames.size <= 12 && frames.all { it.isNotEmpty() && it.size <= 1024 * 1024 } &&
                frames.sumOf { it.size.toLong() } <= 8L * 1024 * 1024) { "画面数据超出限制" }
            val metadata = payload.optJSONArray("frames_note") ?: JSONArray()
            val envelope = JSONObject().put("payload", payload).put("frames", metadata)
            val content = JSONArray().put(JSONObject().put("type", "text").put("text", envelope.toString()))
            frames.forEachIndexed { index, jpeg ->
                call.remaining()
                val info = metadata.optJSONObject(index)
                content.put(JSONObject().put("type", "text").put("text", "Frame ID: ${info?.optString("frame_id") ?: "f$index"}; timestamp_ms: ${info?.optLong("timestamp_ms") ?: 0}"))
                content.put(JSONObject().put("type", "image_url").put("image_url", JSONObject()
                    .put("url", "data:image/jpeg;base64," + Base64.getEncoder().encodeToString(jpeg))))
            }
            val body = JSONObject().put("model", model).put("stream", false).put("max_tokens", 4096)
                .put("messages", JSONArray().put(JSONObject().put("role", "system").put("content", prompt))
                    .put(JSONObject().put("role", "user").put("content", content)))
            val bytes = exchange(call, "/chat/completions", "application/json", body.toString().toByteArray(Charsets.UTF_8), credential = credential)
            call.phase = "parsing_response"
            val response = JSONObject(String(bytes, Charsets.UTF_8))
            val choice = response.getJSONArray("choices").getJSONObject(0)
            val reason = choice.optString("finish_reason")
            if (reason == "length") throw ApiFailure(message = "模型回复被截断，请重试")
            val message = choice.getJSONObject("message")
            if (!message.isNull("refusal") && message.optString("refusal").isNotBlank()) throw ApiFailure(400, message = "模型拒绝处理这组画面")
            val raw = message.opt("content")
            val text = when (raw) {
                is String -> raw
                is JSONArray -> (0 until raw.length()).joinToString("") { raw.getJSONObject(it).optString("text") }
                else -> throw ApiFailure(message = "模型没有返回可读内容")
            }
            decodeObject(text)
        }

    fun transcribe(wav: ByteArray): String = bounded(45000) { call ->
        validateWav(wav)
        require(wav.size <= MAX_PCM_BYTES + 256) { "录音超过时长限制" }
        val boundary = "Satori" + java.util.UUID.randomUUID().toString().replace("-", "")
        val out = ByteArrayOutputStream(wav.size + 1024)
        fun part(text: String) = out.write(text.toByteArray(Charsets.UTF_8))
        part("--$boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n${profile.asrModel}\r\n")
        part("--$boundary\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nzh\r\n")
        part("--$boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
        out.write(wav)
        part("\r\n--$boundary--\r\n")
        val result = JSONObject(String(exchange(call, "/audio/transcriptions", "multipart/form-data; boundary=$boundary", out.toByteArray()), Charsets.UTF_8))
        val text = result.optString("text").trim()
        if (text.isEmpty() || text.length > 2000) throw ApiFailure(message = "没有识别到有效语音，请重说")
        text
    }

    /** Device mode uses bundled excerpts and verifies the original PDF once per API instance. */
    fun searchManual(query: String): JSONObject = bounded(if (profile.manualSource == "device") 45000 else 15000) { call ->
        require(query.isNotBlank() && query.length <= 500 && profile.manualId.isNotBlank())
        if (profile.manualSource == "device") {
            val result = StandaloneManuals.context(profile.manualId, query, localManuals.search(profile.manualId, query))
                .put("transport", "device_asset_search")
            val metadata = localManuals.lookup(profile.manualId)
            if (metadata == null) return@bounded result.put("source_check", JSONObject().put("status", "unknown_manual"))
            val cached = verifiedManualSources[profile.manualId]
            val source = if (cached != null) JSONObject(cached).put("cache_hit", true) else {
                val index = decodeObject(String(manualAsset!!.invoke("source-index.json"), Charsets.UTF_8))
                require(index.getInt("schema_version") == 1)
                val entries = index.getJSONArray("sources")
                val entry = (0 until entries.length()).map { entries.getJSONObject(it) }
                    .single { it.getString("manual_id") == profile.manualId }
                require(entry.getBoolean("synthetic") == metadata.getBoolean("synthetic"))
                val check = if (entry.getBoolean("synthetic")) JSONObject().put("status", "synthetic_no_remote_source") else {
                    val url = normalizeBaseUrl(entry.getString("source_url"))
                    require(url == metadata.getString("source_url"))
                    val expectedHash = entry.getString("sha256")
                    require(expectedHash.matches(Regex("[a-f0-9]{64}")))
                    val maxBytes = entry.getInt("max_bytes")
                    require(maxBytes in 1..MAX_RESPONSE_BYTES)
                    val bytes = exchange(call, "", "application/pdf", byteArrayOf(), url, null, "GET")
                    require(bytes.size <= maxBytes && bytes.size == entry.getInt("document_bytes"))
                    val actualHash = java.security.MessageDigest.getInstance("SHA-256").digest(bytes)
                        .joinToString("") { "%02x".format(it) }
                    if (actualHash != expectedHash) throw ApiFailure(message = "厂商原手册版本已变化，需核对后更新本机摘录")
                    JSONObject().put("status", "original_pdf_sha256_verified").put("source_url", url)
                        .put("original_pdf_sha256", actualHash).put("bytes", bytes.size)
                        .put("verified_unix_ms", System.currentTimeMillis())
                        .put("research_section_ids", entry.getJSONArray("research_section_ids"))
                }
                call.remaining()
                verifiedManualSources[profile.manualId] = check.toString()
                check.put("cache_hit", false)
            }
            return@bounded result.put("source_check", source)
        }
        val endpoint = StandaloneManuals.endpoint(profile.toolsBaseUrl)
        val body = JSONObject().put("name", "search_manual").put("arguments", JSONObject()
            .put("manual_id", profile.manualId).put("query", query).put("limit", 4))
        val bytes = exchange(call, "/tools/call", "application/json", body.toString().toByteArray(Charsets.UTF_8), endpoint, profile.toolsToken)
        require(bytes.size <= 1024 * 1024)
        val envelope = decodeObject(String(bytes, Charsets.UTF_8))
        if (envelope.opt("ok") != true) throw ApiFailure(message = "手册服务没有返回可用结果")
        StandaloneManuals.context(profile.manualId, query, envelope.getJSONObject("result")).put("transport", "http_json_api")
    }

    fun synthesize(text: String): ByteArray = bounded(45000) { call ->
        require(text.isNotBlank() && text.length <= 2000)
        val body = JSONObject().put("model", profile.ttsModel).put("voice", profile.ttsVoice)
            .put("input", text).put("response_format", "wav")
        exchange(call, "/audio/speech", "application/json", body.toString().toByteArray(Charsets.UTF_8)).also { validateWav(it) }
    }

    fun researchWeb(plan: JSONObject, onToolEvent: (String, JSONObject) -> Unit = { _, _ -> }): JSONObject = bounded(60000) { call ->
        val decision = WebResearchPolicy.plan(JSONObject().put("needs_search", plan.opt("needs_search")).put("query", plan.opt("query")))
        if (!decision.getBoolean("needs_search")) {
            return@bounded WebEvidence.context(decision, JSONArray(), JSONArray(), System.currentTimeMillis())
        }
        fun send(body: JSONObject, id: Int?): JSONObject {
            call.remaining()
            val started = System.nanoTime()
            onToolEvent("mcp_request", JSONObject().put("method", body.getString("method")).put("id", id ?: JSONObject.NULL)
                .put("tool", body.optJSONObject("params")?.optString("name").orEmpty())
                .put("arguments", body.optJSONObject("params")?.optJSONObject("arguments") ?: JSONObject()))
            val headers = mutableMapOf("MCP-Protocol-Version" to ExaMcp.PROTOCOL)
            mcpSession?.let { headers["Mcp-Session-Id"] = it }
            val raw = try { exchange(call, "", "application/json", body.toString().toByteArray(Charsets.UTF_8),
                ExaMcp.ENDPOINT, null, "POST", "application/json, text/event-stream", headers) { connection ->
                connection.getHeaderField("Mcp-Session-Id")?.let { session ->
                    require(session.length in 1..256 && session.all { it.code in 33..126 })
                    mcpSession = session
                }
            } } catch (failure: ApiFailure) {
                if (failure.status == 404) { mcpReady = false; mcpSession = null }
                if (failure.status == 429) throw ApiFailure(429, maxOf(60000, failure.retryAfterMs), "网页检索服务已限流，请稍后重试或配置正式检索账号", failure.diagnostic)
                if (failure.status in setOf(401, 403)) throw ApiFailure(failure.status, message = "网页检索服务拒绝当前匿名访问，请检查网络或配置正式检索账号", diagnostic = failure.diagnostic)
                throw failure
            }
            val result = if (id == null) JSONObject() else ExaMcp.response(raw, id)
            onToolEvent("mcp_result", JSONObject().put("method", body.getString("method")).put("id", id ?: JSONObject.NULL)
                .put("bytes", raw.size).put("elapsed_ms", (System.nanoTime() - started) / 1e6))
            return result
        }
        if (!mcpReady) {
            mcpSession = null
            val id = ++mcpSequence
            ExaMcp.validateInitialize(send(ExaMcp.initialize(id), id))
            send(ExaMcp.initialized(), null)
            val toolsId = ++mcpSequence
            ExaMcp.validateTools(send(ExaMcp.listTools(toolsId), toolsId))
            mcpReady = true
        }
        val searchId = ++mcpSequence
        val hits = ExaMcp.searchResults(send(ExaMcp.search(searchId, decision.getString("query")), searchId)) { reason ->
            onToolEvent("mcp_source_skipped", JSONObject().put("id", searchId).put("reason", reason))
        }
        val selected = (0 until hits.length()).map { hits.getJSONObject(it) }.take(3)
        val urls = selected.map { it.getString("url") }
        val fetched = if (urls.isEmpty()) JSONArray() else {
            val fetchId = ++mcpSequence
            ExaMcp.fetchResults(send(ExaMcp.fetch(fetchId, urls), fetchId), urls)
        }
        WebEvidence.context(decision, hits, fetched, System.currentTimeMillis())
    }

    /** Cancels current requests; the transport remains usable for the next task. */
    fun cancel() { cancellationGeneration.incrementAndGet(); active.toList().forEach { it.cancel() }; workers.purge() }
    override fun close() {
        if (closed.compareAndSet(false, true)) { cancel(); workers.shutdownNow(); timer.shutdownNow(); closers.shutdown() }
    }

    private fun <T> bounded(timeoutMs: Long, work: (Call) -> T): T {
        require(timeoutMs in 1000..180000)
        return boundedUntil(System.nanoTime() + timeoutMs * 1_000_000L, cancellationGeneration.get(), work)
    }

    private fun <T> boundedUntil(deadlineNs: Long, ticket: Long, work: (Call) -> T): T {
        if (closed.get()) throw ApiFailure(message = "网络服务已关闭")
        val call = Call(deadlineNs, { !closed.get() && ticket == cancellationGeneration.get() }) { conn -> closers.execute { runCatching { conn.disconnect() } } }
        active.add(call)
        val watchdog = timer.schedule({ call.cancel() }, (deadlineNs - System.nanoTime()).coerceAtLeast(0), TimeUnit.NANOSECONDS)
        try {
            call.remaining()
            val future = workers.submit(Callable { call.phase = "preparation"; work(call) })
            call.future = future
            if (call.cancelled.get()) future.cancel(true)
            return future.get(call.remaining().toLong(), TimeUnit.MILLISECONDS)
        } catch (_: TimeoutException) {
            val failure = call.timeoutFailure()
            call.cancel()
            throw failure
        } catch (_: java.util.concurrent.CancellationException) {
            if (ticket != cancellationGeneration.get() || closed.get()) throw ApiFailure(message = "请求已取消", diagnostic = "cancelled")
            if (System.nanoTime() >= call.deadlineNs) throw call.timeoutFailure()
            throw ApiFailure(message = "请求已取消")
        } catch (_: InterruptedException) {
            call.cancel(); Thread.currentThread().interrupt()
            throw ApiFailure(message = "请求已取消")
        } catch (e: ExecutionException) {
            val cause = e.cause
            if (cause is ApiFailure) throw cause
            if (cause is IllegalArgumentException) throw ApiFailure(message = "请求配置或数据无效")
            throw ApiFailure(message = if (call.cancelled.get()) "请求已取消或超时" else "网络请求失败，请检查网络、接口和模型设置")
        } catch (_: RejectedExecutionException) {
            throw ApiFailure(message = "网络请求仍在结束，请稍后重试")
        } finally {
            watchdog.cancel(false)
            active.remove(call)
        }
    }

    private fun exchange(call: Call, path: String, type: String, body: ByteArray,
                         origin: String = base, credential: String? = profile.apiKey, method: String = "POST",
                         accept: String? = null, headers: Map<String, String> = emptyMap(),
                         onResponse: (HttpURLConnection) -> Unit = {}): ByteArray {
        call.remaining()
        if (body.size > 12 * 1024 * 1024) throw ApiFailure(message = "请求数据过大")
        call.phase = "opening_connection"
        val conn = connectionFactory(origin + path)
        call.connection = conn
        try {
            call.remaining()
            conn.instanceFollowRedirects = false
            conn.requestMethod = method
            conn.connectTimeout = call.remaining()
            conn.readTimeout = call.remaining()
            conn.useCaches = false
            conn.doOutput = method == "POST"
            if (credential != null) conn.setRequestProperty("Authorization", "Bearer $credential")
            conn.setRequestProperty("Content-Type", type)
            conn.setRequestProperty("User-Agent", "SatoriAR-Android/${BuildConfig.VERSION_NAME}")
            conn.setRequestProperty("Accept", accept ?: if (method == "GET") type else if (path == "/audio/speech") "audio/wav" else "application/json")
            headers.forEach { (key, value) -> conn.setRequestProperty(key, value) }
            if (method == "POST") conn.setFixedLengthStreamingMode(body.size)
            call.phase = "connecting"
            conn.connect()
            call.remaining()
            if (method == "POST") {
              call.phase = "request_stream"
              conn.outputStream.use { output ->
                call.phase = "request_body"
                var offset = 0
                while (offset < body.size) {
                    call.remaining()
                    val length = minOf(16384, body.size - offset)
                    output.write(body, offset, length)
                    offset += length
                }
                call.phase = "request_body_finish"
              }
            }
            call.remaining()
            conn.readTimeout = call.remaining()
            call.phase = "response_headers"
            val status = conn.responseCode
            if (status !in 200..299) {
                val cooldown = if (status == 429) retryAfterMs(conn.getHeaderField("Retry-After")) else 0
                // A known error code, never the raw error message, distinguishes quota from credential failures.
                val diagnostic = errorDiagnostic(conn, call)
                val message = if (diagnostic in setOf("insufficient_user_quota", "insufficient_quota", "token_quota_not_enough"))
                    "接口额度不足，请充值或更换接口" else when (status) {
                    401 -> "接口认证失败（HTTP 401），请检查 API 密钥"
                    403 -> "接口拒绝访问（HTTP 403），请检查接口权限和账户状态"
                    404 -> "接口或模型不存在，请检查模型设置"
                    429 -> "接口限流，稍后自动重试"
                    in 300..399 -> "接口发生重定向，请配置最终 HTTPS 地址"
                    else -> "接口请求失败（HTTP $status）"
                }
                throw ApiFailure(status, cooldown, message, diagnostic)
            }
            onResponse(conn)
            val expected = conn.contentLengthLong
            if (expected > MAX_RESPONSE_BYTES) throw ApiFailure(message = "接口回复过大")
            val out = ByteArrayOutputStream(if (expected in 1..MAX_RESPONSE_BYTES.toLong()) expected.toInt() else 8192)
            call.phase = "response_body"
            conn.inputStream.use { input ->
                val buffer = ByteArray(16384)
                while (true) {
                    conn.readTimeout = call.remaining()
                    val count = input.read(buffer)
                    if (count < 0) break
                    if (out.size() + count > MAX_RESPONSE_BYTES) throw ApiFailure(message = "接口回复过大")
                    out.write(buffer, 0, count)
                }
            }
            call.remaining()
            if (expected >= 0 && expected != out.size().toLong()) throw ApiFailure(message = "接口回复不完整")
            return out.toByteArray()
        } catch (_: java.net.SocketTimeoutException) {
            call.remaining()
            throw call.timeoutFailure()
        } finally {
            call.phase = "cleanup"
            conn.disconnect()
            call.connection = null
        }
    }

    /** Inspect at most one 4 KB read within the same deadline. Only fixed, recognized labels escape this method. */
    private fun errorDiagnostic(conn: HttpURLConnection, call: Call): String {
        val headerIsHtml = conn.contentType.orEmpty().contains("html", ignoreCase = true)
        if (headerIsHtml) return "html_error"
        return try {
            conn.readTimeout = minOf(1000, call.remaining())
            val bytes = ByteArray(4096)
            val count = conn.errorStream?.use { it.read(bytes) } ?: return "empty_error"
            if (count <= 0) return "empty_error"
            val text = String(bytes, 0, count, Charsets.UTF_8).trimStart()
            if (text.startsWith("<html", true) || text.startsWith("<!doctype", true)) return "html_error"
            val data = JSONObject(text)
            val code = data.optJSONObject("error")?.optString("code") ?: data.optString("code")
            if (code in setOf("insufficient_user_quota", "insufficient_quota", "token_quota_not_enough",
                    "invalid_api_key", "invalid_token", "token_expired", "access_denied", "model_not_found",
                    "rate_limit_exceeded", "unsupported_country_region_territory")) code else "json_error"
        } catch (_: Exception) { "unreadable_error" }
    }

    companion object {
        const val MAX_RESPONSE_BYTES = 4 * 1024 * 1024
        const val MAX_PCM_BYTES = 16000 * 2 * 20

        fun normalizeBaseUrl(value: String): String {
            val text = value.trim().trimEnd('/')
            val uri = try { URI(text) } catch (_: Exception) { throw IllegalArgumentException("请输入有效 HTTPS 接口地址") }
            require(uri.scheme.equals("https", true) && !uri.host.isNullOrBlank() && uri.rawUserInfo == null &&
                uri.rawQuery == null && uri.rawFragment == null && (uri.port == -1 || uri.port in 1..65535)) { "接口必须是无凭据、无查询参数的 HTTPS 地址" }
            require(uri.rawPath.orEmpty().none { it == '\\' } && text.length <= 2048) { "接口地址无效" }
            return text
        }

        fun decodeObject(raw: String): JSONObject {
            var text = raw.trim()
            if (text.startsWith("```")) {
                val match = Regex("^```(?:json)?\\s*\\n?([\\s\\S]*?)\\n?```$", RegexOption.IGNORE_CASE).matchEntire(text)
                    ?: throw ApiFailure(message = "模型没有返回完整 JSON")
                text = match.groupValues[1].trim()
            }
            try {
                if (!text.startsWith("{") || !text.endsWith("}") || text.length > MAX_RESPONSE_BYTES) throw IllegalArgumentException()
                StrictJson(text).validate()
                val tokener = JSONTokener(text)
                val result = tokener.nextValue()
                if (result !is JSONObject || tokener.nextClean() != 0.toChar()) throw IllegalArgumentException()
                return result
            } catch (_: Exception) { throw ApiFailure(message = "模型返回的 JSON 无效，请重试") }
        }

        fun retryAfterMs(value: String?, nowMs: Long = System.currentTimeMillis()): Long {
            val numeric = value?.trim()?.toLongOrNull()
            if (numeric != null) return numeric.coerceIn(1, 300) * 1000
            val date = try { ZonedDateTime.parse(value, DateTimeFormatter.RFC_1123_DATE_TIME).toInstant().toEpochMilli() }
                catch (_: Exception) { return 60000 }
            return (date - nowMs).coerceIn(1000, 300000)
        }

        fun wavFromPcm(pcm: ByteArray): ByteArray {
            require(pcm.isNotEmpty() && pcm.size % 2 == 0 && pcm.size <= MAX_PCM_BYTES) { "录音数据无效或超时" }
            return ByteBuffer.allocate(pcm.size + 44).order(ByteOrder.LITTLE_ENDIAN).apply {
                put("RIFF".toByteArray(Charsets.US_ASCII)); putInt(pcm.size + 36); put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
                putInt(16); putShort(1); putShort(1); putInt(16000); putInt(32000); putShort(2); putShort(16)
                put("data".toByteArray(Charsets.US_ASCII)); putInt(pcm.size); put(pcm)
            }.array()
        }

        fun validateWav(bytes: ByteArray) {
            try {
                require(bytes.size in 44..MAX_RESPONSE_BYTES)
                val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" && String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE")
                val riffSize = buf.getInt(4).toLong() and 0xffffffffL
                require(riffSize + 8 == bytes.size.toLong() && riffSize + 8 >= 44)
                var pos = 12
                var validFormat = false
                var frameBytes = 0
                while (pos + 8 <= bytes.size) {
                    val id = String(bytes, pos, 4, Charsets.US_ASCII)
                    val size = buf.getInt(pos + 4).toLong() and 0xffffffffL
                    val body = pos + 8
                    require(size <= bytes.size - body)
                    if (id == "fmt ") {
                        require(size >= 16)
                        val format = buf.getShort(body).toInt() and 65535
                        val channels = buf.getShort(body + 2).toInt() and 65535
                        val rate = buf.getInt(body + 4)
                        val bits = buf.getShort(body + 14).toInt() and 65535
                        require(format == 1 && channels in 1..2 && rate in 8000..48000 && bits == 16)
                        frameBytes = channels * 2
                        require((buf.getShort(body + 12).toInt() and 65535) == frameBytes)
                        validFormat = true
                    }
                    if (id == "data") { require(validFormat && size > 0 && size % frameBytes == 0L); return }
                    pos = (body + size + (size and 1)).toInt()
                }
                throw IllegalArgumentException()
            } catch (_: Exception) { throw ApiFailure(message = "语音接口未返回可播放的完整 PCM16 WAV") }
        }
    }

    /** org.json intentionally accepts JavaScript extensions. Provider output must use RFC JSON syntax. */
    private class StrictJson(private val text: String) {
        private var pos = 0
        fun validate() { value(0); spaces(); require(pos == text.length) }
        private fun spaces() { while (pos < text.length && text[pos] in " \r\n\t") pos++ }
        private fun take(c: Char): Boolean { spaces(); return if (pos < text.length && text[pos] == c) { pos++; true } else false }
        private fun expect(c: Char) { require(take(c)) }
        private fun value(depth: Int) {
            require(depth <= 48)
            spaces(); require(pos < text.length)
            when (text[pos]) {
                '{' -> {
                    pos++; val keys = HashSet<String>()
                    if (take('}')) return
                    do {
                        spaces(); require(pos < text.length && text[pos] == '"')
                        val key = string(); require(keys.add(key)); expect(':'); value(depth + 1)
                    } while (take(','))
                    expect('}')
                }
                '[' -> {
                    pos++
                    if (take(']')) return
                    do { value(depth + 1) } while (take(','))
                    expect(']')
                }
                '"' -> string()
                't' -> literal("true")
                'f' -> literal("false")
                'n' -> literal("null")
                else -> number()
            }
        }
        private fun literal(value: String) { require(text.startsWith(value, pos)); pos += value.length }
        private fun string(): String {
            require(text[pos++] == '"')
            val result = StringBuilder()
            while (pos < text.length) {
                val c = text[pos++]
                if (c == '"') return result.toString()
                require(c >= ' ')
                if (c != '\\') result.append(c) else {
                    require(pos < text.length)
                    when (val escaped = text[pos++]) {
                        '"', '\\', '/' -> result.append(escaped)
                        'b' -> result.append('\b')
                        'f' -> result.append(12.toChar())
                        'n' -> result.append('\n')
                        'r' -> result.append('\r')
                        't' -> result.append('\t')
                        'u' -> {
                            require(pos + 4 <= text.length)
                            val digits = text.substring(pos, pos + 4)
                            require(digits.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' })
                            val code = digits.toInt(16)
                            result.append(code.toChar()); pos += 4
                        }
                        else -> throw IllegalArgumentException()
                    }
                }
            }
            throw IllegalArgumentException()
        }
        private fun number() {
            val start = pos
            if (pos < text.length && text[pos] == '-') pos++
            require(pos < text.length)
            if (text[pos] == '0') pos++ else {
                require(text[pos] in '1'..'9'); pos++
                while (pos < text.length && text[pos] in '0'..'9') pos++
            }
            if (pos < text.length && text[pos] == '.') {
                pos++; val before = pos
                while (pos < text.length && text[pos] in '0'..'9') pos++
                require(pos > before)
            }
            if (pos < text.length && text[pos] in "eE") {
                pos++; if (pos < text.length && text[pos] in "+-") pos++
                val before = pos
                while (pos < text.length && text[pos] in '0'..'9') pos++
                require(pos > before)
            }
            require(pos > start)
        }
    }
}
