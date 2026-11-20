package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.net.URL
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.cert.Certificate
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import javax.net.ssl.HttpsURLConnection

class StandaloneApiTest {
    @Test fun endpointOnlyAllowsHttpsWithoutCredentialsOrQuery() {
        assertEquals("https://example.org/v1", StandaloneApi.normalizeBaseUrl(" https://example.org/v1/ "))
        listOf("http://example.org/v1", "https://user:pass@example.org", "https://example.org/?key=secret",
            "https://example.org/#key", "https://", "file:///secret").forEach { value ->
            try { StandaloneApi.normalizeBaseUrl(value); fail("accepted unsafe endpoint") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun jsonMustBeOneCompleteObject() {
        assertEquals(1, StandaloneApi.decodeObject("```json\n{\"a\":1}\n```").getInt("a"))
        assertEquals("中文", StandaloneApi.decodeObject("{\"a\":\"中文\"}").getString("a"))
        listOf("{\"a\":1} trailing", "{}{}", "[]", "prefix {}", "```json\n{}\n``` extra", "{'a':1}",
            "{a:1}", "{\"a\":NaN}", "{\"a\":01}", "{\"a\":1,}", "{\"a\":1,\"a\":2}").forEach {
            try { StandaloneApi.decodeObject(it); fail("accepted mixed response") }
            catch (_: ApiFailure) { }
        }
    }

    @Test fun rateLimitHasMinimumAndBoundedCooldown() {
        assertEquals(60_000L, StandaloneApi.retryAfterMs(null))
        assertEquals(2_000L, StandaloneApi.retryAfterMs("2"))
        assertEquals(300_000L, StandaloneApi.retryAfterMs("9999999999"))
        assertEquals(1_000L, StandaloneApi.retryAfterMs("0"))
        assertEquals(60_000L, StandaloneApi.retryAfterMs("nonsense"))
    }

    @Test fun unicodeEscapeRequiresExactlyFourHexDigitsWithoutSign() {
        assertEquals("1", StandaloneApi.decodeObject("{\"a\":\"\\u0031\"}").getString("a"))
        listOf("{\"a\":\"\\u+031\"}", "{\"a\":\"\\u-001\"}", "{\"a\":\"\\u00g1\"}").forEach {
            try { StandaloneApi.decodeObject(it); fail("accepted signed/nonhex Unicode escape") }
            catch (_: ApiFailure) { }
        }
    }

    @Test fun wavWrapPreservesPcmAndRejectsInvalidSizes() {
        val pcm = byteArrayOf(1, 2, 3, 4)
        val wav = StandaloneApi.wavFromPcm(pcm)
        assertEquals("RIFF", String(wav, 0, 4, Charsets.US_ASCII))
        assertEquals(16000, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(24))
        assertArrayEquals(pcm, wav.copyOfRange(44, wav.size))
        StandaloneApi.validateWav(wav)
        listOf(ByteArray(0), byteArrayOf(1), ByteArray(StandaloneApi.MAX_PCM_BYTES + 2)).forEach {
            try { StandaloneApi.wavFromPcm(it); fail("accepted invalid PCM") }
            catch (_: IllegalArgumentException) { }
        }
        try { StandaloneApi.validateWav(ByteArray(44)); fail("accepted non-WAV") }
        catch (_: ApiFailure) { }
    }

    @Test fun truncatedWavIsNotPlayed() {
        val wav = StandaloneApi.wavFromPcm(byteArrayOf(1, 2, 3, 4))
        try { StandaloneApi.validateWav(wav.copyOf(wav.size - 1)); fail("accepted truncated audio") }
        catch (_: ApiFailure) { }
    }

    @Test fun redirectDoesNotForwardCredentials() {
        val conn = FakeConnection(302, "")
        StandaloneApi(profile()) { conn }.use { api ->
            try { api.complete("model", "prompt", org.json.JSONObject(), emptyList()); fail("accepted redirect") }
            catch (e: ApiFailure) { assertEquals(302, e.status); assertFalse(e.message!!.contains("secret")) }
        }
        assertFalse(conn.instanceFollowRedirects)
        assertTrue(conn.disconnected)
    }

    @Test fun providerErrorCannotExposeResponseBody() {
        val conn = FakeConnection(429, "provider secret-test-key response")
        StandaloneApi(profile()) { conn }.use { api ->
            try { api.complete("model", "prompt", org.json.JSONObject(), emptyList()); fail("accepted limit") }
            catch (e: ApiFailure) {
                assertEquals(429, e.status); assertEquals(60000L, e.retryAfterMs)
                assertFalse(e.message!!.contains("secret-test-key"))
            }
        }
    }

    @Test fun quotaFailureIsDistinguishedWithoutExposingProviderMessage() {
        val conn = object : FakeConnection(403, "") {
            override fun getErrorStream(): java.io.InputStream = ByteArrayInputStream(
                """{"error":{"code":"insufficient_user_quota","message":"secret-test-key private diagnostic"}}""".toByteArray())
        }
        StandaloneApi(profile()) { conn }.use { api ->
            try { api.synthesize("你好"); fail("accepted exhausted quota") }
            catch (e: ApiFailure) {
                assertEquals(403, e.status)
                assertEquals("insufficient_user_quota", e.diagnostic)
                assertEquals("接口额度不足，请充值或更换接口", e.message)
                assertFalse(e.toString().contains("secret-test-key"))
                assertFalse(e.toString().contains("private diagnostic"))
            }
        }
    }

    @Test fun completionIncludesImagesAndExtractsStructuredReply() {
        val conn = FakeConnection(200, """{"choices":[{"finish_reason":"stop","message":{"content":"{\"ok\":true}"}}]}""")
        StandaloneApi(profile()) { conn }.use { api ->
            assertTrue(api.complete("monitor", "prompt", org.json.JSONObject().put("goal", "整理"), listOf(byteArrayOf(1, 2))).getBoolean("ok"))
        }
        val request = org.json.JSONObject(conn.sent.toString("UTF-8"))
        assertEquals("monitor", request.getString("model"))
        val content = request.getJSONArray("messages").getJSONObject(1).getJSONArray("content")
        assertEquals("整理", org.json.JSONObject(content.getJSONObject(0).getString("text")).getJSONObject("payload").getString("goal"))
        assertTrue(content.getJSONObject(1).getString("text").startsWith("Frame ID:"))
        assertTrue(content.getJSONObject(2).getJSONObject("image_url").getString("url").startsWith("data:image/jpeg;base64,"))
    }

    @Test fun absoluteDeadlineReturnsEvenWhenDisconnectIsSlow() {
        val block = CountDownLatch(1)
        val conn = object : FakeConnection(200, "") {
            override fun getOutputStream(): java.io.OutputStream { block.await(5, TimeUnit.SECONDS); return sent }
            override fun disconnect() { block.await(3, TimeUnit.SECONDS); super.disconnect() }
        }
        val start = System.nanoTime()
        StandaloneApi(profile()) { conn }.use { api ->
            try { api.complete("model", "prompt", org.json.JSONObject(), emptyList(), 1000); fail("accepted late request") }
            catch (_: ApiFailure) { }
            finally { block.countDown() }
        }
        assertTrue("deadline was not bounded", (System.nanoTime() - start) / 1_000_000 < 2200)
    }

    @Test fun timeoutDistinguishesConnectionStreamUploadAndBodyFinish() {
        for (stage in listOf("connecting", "request_stream", "request_body", "request_body_finish")) {
            fun stall(at: String) { if (stage == at) throw java.net.SocketTimeoutException("private provider details") }
            val conn = object : FakeConnection(200, "") {
                override fun connect() { stall("connecting") }
                override fun getOutputStream(): java.io.OutputStream {
                    stall("request_stream")
                    return object : java.io.OutputStream() {
                        override fun write(value: Int) { stall("request_body") }
                        override fun close() { stall("request_body_finish") }
                    }
                }
            }
            StandaloneApi(profile()) { conn }.use { api ->
                try { api.complete("model", "prompt", org.json.JSONObject(), emptyList(), 1000); fail("timeout accepted") }
                catch (failure: ApiFailure) {
                    assertEquals("timeout_$stage", failure.diagnostic)
                    assertFalse(failure.toString().contains("private provider details"))
                }
            }
        }
    }

    @Test fun manualTransportNeverReusesProviderCredentialAndDoesNotFollowRedirects() {
        val config = profile().copy(toolsBaseUrl = "http://127.0.0.1:8766", toolsToken = "t".repeat(32), manualId = "classroom-demo")
        val conn = FakeConnection(200, """{"ok":true,"result":{"matches":[]}}""")
        var requested = ""
        StandaloneApi(config) { url -> requested = url; conn }.use { api ->
            assertFalse(api.searchManual("LAN").getBoolean("matched"))
        }
        assertEquals("http://127.0.0.1:8766/tools/call", requested)
        assertEquals("Bearer ${config.toolsToken}", conn.getRequestProperty("Authorization"))
        assertFalse(conn.instanceFollowRedirects)
        assertFalse(conn.sent.toString("UTF-8").contains(config.apiKey))
        val request = org.json.JSONObject(conn.sent.toString("UTF-8"))
        assertEquals("classroom-demo", request.getJSONObject("arguments").getString("manual_id"))
        val redirect = FakeConnection(302, "")
        StandaloneApi(config) { redirect }.use { api ->
            try { api.searchManual("LAN"); fail("followed tool redirect") } catch (failure: ApiFailure) { assertEquals(302, failure.status) }
        }
    }

    @Test fun timeoutRecordsHeadersPhaseWithoutExposingCredentials() {
        val release = CountDownLatch(1)
        val conn = object : FakeConnection(200, "") {
            override fun getResponseCode(): Int { release.await(3, TimeUnit.SECONDS); return super.getResponseCode() }
        }
        StandaloneApi(profile()) { conn }.use { api ->
            try { api.complete("model", "prompt", org.json.JSONObject(), emptyList(), 1000); fail("late headers accepted") }
            catch (failure: ApiFailure) { assertEquals("timeout_response_headers", failure.diagnostic) }
            finally { release.countDown() }
        }
    }

    @Test fun timeoutCanIdentifyCleanupAfterTheCompleteBodyWasRead() {
        val release = CountDownLatch(1)
        val conn = object : FakeConnection(200, """{"choices":[{"message":{"content":"{\"ok\":true}"}}]}""") {
            override fun disconnect() { release.await(3, TimeUnit.SECONDS); super.disconnect() }
        }
        StandaloneApi(profile()) { conn }.use { api ->
            try { api.complete("model", "prompt", org.json.JSONObject(), emptyList(), 1000); fail("late cleanup accepted") }
            catch (failure: ApiFailure) { assertEquals("timeout_cleanup", failure.diagnostic) }
            finally { release.countDown() }
        }
    }

    private fun profile() = StandaloneProfile("https://example.org/v1", "secret-test-key", "planner", "monitor")

    private open class FakeConnection(private val code: Int, text: String) : HttpsURLConnection(URL("https://example.org/v1")) {
        val sent = ByteArrayOutputStream()
        private val response = text.toByteArray(Charsets.UTF_8)
        var disconnected = false
        override fun getOutputStream(): java.io.OutputStream = sent
        override fun getInputStream(): java.io.InputStream = ByteArrayInputStream(response)
        override fun getResponseCode() = code
        override fun getContentLengthLong() = response.size.toLong()
        override fun getHeaderField(name: String?): String? = null
        override fun connect() = Unit
        override fun disconnect() { disconnected = true }
        override fun usingProxy() = false
        override fun getCipherSuite() = "TLS_AES_128_GCM_SHA256"
        override fun getLocalCertificates(): Array<Certificate>? = null
        override fun getServerCertificates(): Array<Certificate> = emptyArray()
    }
}
