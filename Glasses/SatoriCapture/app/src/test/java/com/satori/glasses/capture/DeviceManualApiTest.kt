package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.URL
import java.security.MessageDigest
import java.security.cert.Certificate
import javax.net.ssl.HttpsURLConnection

class DeviceManualApiTest {
    private val assets = File("src/main/assets/manuals")
    private val pdf = "%PDF-1.4\nControlled JVM transport fixture; not a manufacturer document.\n%%EOF\n".toByteArray()
    private fun asset(name: String) = File(assets, name).readBytes()
    private fun originalUrl() = JSONObject(String(asset("tl-sg105-v4.json"), Charsets.UTF_8)).getString("source_url")
    private fun profile(manualId: String = "tl-sg105-v4") = StandaloneProfile(
        baseUrl = "https://provider.invalid/v1", apiKey = "model-secret-for-test", planner = "planner", monitor = "monitor",
        manualId = manualId, manualSource = "device")

    private fun indexedAssets(change: (JSONObject) -> Unit = {}): (String) -> ByteArray {
        val index = JSONObject(String(asset("source-index.json"), Charsets.UTF_8))
        val entries = index.getJSONArray("sources")
        val original = (0 until entries.length()).map { entries.getJSONObject(it) }
            .single { it.getString("manual_id") == "tl-sg105-v4" }
        original.put("sha256", sha256(pdf)).put("document_bytes", pdf.size)
        change(original)
        val bytes = index.toString().toByteArray(Charsets.UTF_8)
        return { name -> if (name == "source-index.json") bytes else asset(name) }
    }

    @Test fun deviceSearchVerifiesOriginalWithoutCredentialsThenReusesVerification() {
        val connection = FakeConnection(200, pdf)
        val requested = mutableListOf<String>()
        StandaloneApi(profile(), manualAsset = indexedAssets(), connectionFactory = { url -> requested += url; connection }).use { api ->
            val first = api.searchManual("网线 LAN 接线")
            assertEquals("device_asset_search", first.getString("transport"))
            assertTrue(first.getBoolean("matched"))
            val source = first.getJSONObject("source_check")
            assertEquals("original_pdf_sha256_verified", source.getString("status"))
            assertFalse(source.getBoolean("cache_hit"))
            assertEquals(sha256(pdf), source.getString("original_pdf_sha256"))
            assertEquals(originalUrl(), source.getString("source_url"))
            assertEquals(pdf.size, source.getInt("bytes"))
            assertTrue(source.getLong("verified_unix_ms") > 0)
            assertEquals(setOf("applicability", "verification-boundary"), source.getJSONArray("research_section_ids").let {
                (0 until it.length()).map(it::getString).toSet()
            })
            val passage = first.getJSONArray("passages").getJSONObject(0)
            assertEquals("tl-sg105-v4", passage.getString("manual_id"))
            assertEquals(sha256(asset("tl-sg105-v4.json")), passage.getString("sha256"))
            assertFalse(passage.getBoolean("synthetic"))

            val second = api.searchManual("互联网 教学资源 验证")
            assertEquals("互联网 教学资源 验证", second.getString("query"))
            val cached = second.getJSONObject("source_check")
            assertTrue(cached.getBoolean("cache_hit"))
            assertEquals(source.getLong("verified_unix_ms"), cached.getLong("verified_unix_ms"))
            assertEquals(source.getString("original_pdf_sha256"), cached.getString("original_pdf_sha256"))
            val rows = second.getJSONArray("passages")
            val boundary = (0 until rows.length()).map { rows.getJSONObject(it) }
                .single { it.getString("section_id") == "verification-boundary" }
            assertTrue("research provenance must keep an explicit null page", boundary.has("page") && boundary.isNull("page"))
        }
        assertEquals(listOf(originalUrl()), requested)
        assertEquals("https", URL(requested.single()).protocol)
        assertReadOnlyDownload(connection)
        assertEquals("application/pdf", connection.getRequestProperty("Accept"))
        assertEquals(1, connection.inputRequests)
    }

    @Test fun wrongHashIsRejectedAndDoesNotPoisonTheSuccessCache() {
        val changed = pdf.copyOf().also { it[10] = (it[10].toInt() xor 1).toByte() }
        val wrong = FakeConnection(200, changed)
        val correct = FakeConnection(200, pdf)
        var opened = 0
        StandaloneApi(profile(), manualAsset = indexedAssets(), connectionFactory = { if (opened++ == 0) wrong else correct }).use { api ->
            val failure = expectFailure { api.searchManual("LAN") }
            assertTrue(failure.message.orEmpty().contains("原手册版本已变化"))
            assertFalse(api.searchManual("LAN").getJSONObject("source_check").getBoolean("cache_hit"))
            assertTrue(api.searchManual("LAN").getJSONObject("source_check").getBoolean("cache_hit"))
        }
        assertEquals(2, opened)
        assertReadOnlyDownload(wrong)
        assertReadOnlyDownload(correct)
    }

    @Test fun redirectsAreRejectedWithoutFollowingOrSendingCredentials() {
        for (code in listOf(301, 302, 307, 308)) {
            val redirect = FakeConnection(code, byteArrayOf())
            var opened = 0
            StandaloneApi(profile(), manualAsset = indexedAssets(), connectionFactory = { opened++; redirect }).use { api ->
                assertEquals(code, expectFailure { api.searchManual("LAN") }.status)
            }
            assertEquals("redirect must not create a second connection", 1, opened)
            assertEquals(0, redirect.inputRequests)
            assertReadOnlyDownload(redirect)
        }
    }

    @Test fun declaredOrStreamedOversizeCannotBeVerifiedOrCached() {
        val cases = listOf(
            FakeConnection(200, pdf, StandaloneApi.MAX_RESPONSE_BYTES + 1L),
            FakeConnection(200, ByteArray(StandaloneApi.MAX_RESPONSE_BYTES + 1), -1),
            FakeConnection(200, pdf, pdf.size + 1L),
        )
        for (bad in cases) {
            val correct = FakeConnection(200, pdf)
            var opened = 0
            StandaloneApi(profile(), manualAsset = indexedAssets(), connectionFactory = { if (opened++ == 0) bad else correct }).use { api ->
                expectFailure { api.searchManual("LAN") }
                assertFalse(api.searchManual("LAN").getJSONObject("source_check").getBoolean("cache_hit"))
            }
            assertEquals("failed download must be retried", 2, opened)
            assertReadOnlyDownload(bad)
        }
        assertEquals("oversized Content-Length must be rejected before reading", 0, cases.first().inputRequests)
    }

    @Test fun indexDocumentLimitAndExactLengthAreEnforcedEvenWithTheCorrectHash() {
        val invalidIndexes: List<(JSONObject) -> Unit> = listOf(
            { it.put("max_bytes", pdf.size - 1) },
            { it.put("document_bytes", pdf.size + 1) },
        )
        for (change in invalidIndexes) {
            val connection = FakeConnection(200, pdf)
            var opened = 0
            StandaloneApi(profile(), manualAsset = indexedAssets(change), connectionFactory = { opened++; connection }).use { api ->
                repeat(2) { expectFailure { api.searchManual("LAN") } }
            }
            assertEquals("invalid metadata must never populate the cache", 2, opened)
            assertReadOnlyDownload(connection)
        }
    }

    @Test fun syntheticAndUnknownManualsNeverOpenAnHttpConnection() {
        for ((id, status) in listOf("classroom-demo" to "synthetic_no_remote_source", "unknown-device-v9" to "unknown_manual")) {
            var opened = 0
            StandaloneApi(profile(id), manualAsset = indexedAssets(), connectionFactory = {
                opened++
                throw AssertionError("synthetic or unknown manual must not use HTTP")
            }).use { api ->
                repeat(2) { index ->
                    val result = api.searchManual("网线 LAN 教室")
                    assertEquals("device_asset_search", result.getString("transport"))
                    assertEquals(status, result.getJSONObject("source_check").getString("status"))
                    val rows = result.getJSONArray("passages")
                    if (id == "classroom-demo") {
                        assertTrue(result.getBoolean("matched"))
                        assertEquals(index > 0, result.getJSONObject("source_check").getBoolean("cache_hit"))
                        assertTrue((0 until rows.length()).all { rows.getJSONObject(it).getBoolean("synthetic") })
                    } else {
                        assertFalse(result.getBoolean("matched"))
                        assertEquals(0, rows.length())
                    }
                }
            }
            assertEquals(0, opened)
        }
    }

    @Test fun sourceIndexCannotRedirectDownloadToAnUnapprovedUrl() {
        for (url in listOf(originalUrl().replace("https://", "http://"), "https://other.invalid/manual.pdf", originalUrl() + "?key=secret")) {
            var opened = 0
            StandaloneApi(profile(), manualAsset = indexedAssets { it.put("source_url", url) }, connectionFactory = {
                opened++
                throw AssertionError("invalid source must be rejected before HTTP")
            }).use { api -> expectFailure { api.searchManual("LAN") } }
            assertEquals(0, opened)
        }
    }

    @Test fun deviceConfigurationRejectsEveryExternalToolCredentialCombination() {
        profile().validate()
        for (bad in listOf(profile().copy(toolsBaseUrl = "http://127.0.0.1:8766"),
            profile().copy(toolsToken = "t".repeat(32)),
            profile().copy(toolsBaseUrl = "http://127.0.0.1:8766", toolsToken = "t".repeat(32)))) {
            try { bad.validate(); fail("device mode accepted external tool credentials") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun legacyRemoteConfigurationStillUsesItsSeparateToolTokenAndPostBody() {
        val config = StandaloneProfile("https://provider.invalid/v1", "model-secret-for-test", "planner", "monitor",
            toolsBaseUrl = "http://127.0.0.1:8766", toolsToken = "t".repeat(32), manualId = "tl-sg105-v4")
        assertEquals("remote_api", config.manualSource)
        config.validate()
        val response = """{"ok":true,"result":{"matches":[]}}""".toByteArray()
        val connection = FakeConnection(200, response)
        var requested = ""
        StandaloneApi(config, manualAsset = { throw AssertionError("remote mode must not read device assets") },
            connectionFactory = { requested = it; connection }).use { api ->
            val result = api.searchManual("LAN")
            assertEquals("http_json_api", result.getString("transport"))
            assertFalse(result.getBoolean("matched"))
        }
        assertEquals("http://127.0.0.1:8766/tools/call", requested)
        assertEquals("POST", connection.requestMethod)
        assertTrue(connection.doOutput)
        assertEquals("Bearer ${config.toolsToken}", connection.getRequestProperty("Authorization"))
        assertEquals(1, connection.outputRequests)
        val sent = JSONObject(connection.sent.toString("UTF-8"))
        assertEquals("search_manual", sent.getString("name"))
        assertEquals("tl-sg105-v4", sent.getJSONObject("arguments").getString("manual_id"))
        assertEquals("LAN", sent.getJSONObject("arguments").getString("query"))
        assertFalse(connection.sent.toString("UTF-8").contains(config.apiKey))
        assertFalse(connection.instanceFollowRedirects)
        assertTrue(connection.disconnected)
    }

    private fun expectFailure(action: () -> Unit): ApiFailure {
        try { action() } catch (failure: ApiFailure) { return failure }
        throw AssertionError("unverified original manual was accepted")
    }

    private fun assertReadOnlyDownload(connection: FakeConnection) {
        assertEquals("GET", connection.requestMethod)
        assertNull(connection.getRequestProperty("Authorization"))
        assertFalse(connection.doOutput)
        assertEquals("GET must never open the request body stream", 0, connection.outputRequests)
        assertEquals(0, connection.sent.size())
        assertFalse(connection.instanceFollowRedirects)
        assertTrue(connection.disconnected)
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private class FakeConnection(private val code: Int, private val response: ByteArray,
                                 private val declaredLength: Long = response.size.toLong()) : HttpsURLConnection(URL("https://fixture.invalid/manual.pdf")) {
        val sent = ByteArrayOutputStream()
        var disconnected = false
        var inputRequests = 0
        var outputRequests = 0
        override fun getOutputStream(): java.io.OutputStream { outputRequests++; return sent }
        override fun getInputStream(): java.io.InputStream { inputRequests++; return ByteArrayInputStream(response) }
        override fun getErrorStream(): java.io.InputStream = ByteArrayInputStream(byteArrayOf())
        override fun getResponseCode() = code
        override fun getContentLengthLong() = declaredLength
        override fun getHeaderField(name: String?): String? = if (name.equals("Location", true)) "https://other.invalid/manual.pdf" else null
        override fun connect() = Unit
        override fun disconnect() { disconnected = true }
        override fun usingProxy() = false
        override fun getCipherSuite() = "TLS_AES_128_GCM_SHA256"
        override fun getLocalCertificates(): Array<Certificate>? = null
        override fun getServerCertificates(): Array<Certificate> = emptyArray()
    }
}
