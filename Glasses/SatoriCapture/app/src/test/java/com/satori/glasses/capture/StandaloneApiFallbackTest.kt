package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.net.SocketTimeoutException
import java.net.URL
import java.security.cert.Certificate
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import javax.net.ssl.HttpsURLConnection

class StandaloneApiFallbackTest {
    private val primaryKey = "fixture-primary-credential"
    private val backupKey = "fixture-backup-credential"
    private val endpoint = "https://provider.example.invalid/v1/chat/completions"
    private val success = """{"choices":[{"finish_reason":"stop","message":{"content":"{\"ok\":true}"}}]}"""
    private fun profile() = StandaloneProfile("https://provider.example.invalid/v1", primaryKey, "planner", "monitor", backupApiKey = backupKey)
    private fun invoke(api: StandaloneApi, timeoutMs: Long = 2000) = api.complete("planner", "fixture prompt", JSONObject().put("question", "fixture"), emptyList(), timeoutMs)

    @Test fun primarySuccessDoesNotOpenABackupConnection() {
        val connections = mutableListOf<FakeConnection>()
        val events = CopyOnWriteArrayList<JSONObject>()
        StandaloneApi(profile(), onTransportEvent = { name, data -> events += JSONObject(data.toString()).put("event", name) }) { url ->
            assertEquals(endpoint, url)
            FakeConnection(200, success).also { connections += it }
        }.use { api -> assertTrue(invoke(api).getBoolean("ok")) }
        assertEquals(1, connections.size)
        assertEquals("Bearer $primaryKey", connections.single().getRequestProperty("Authorization"))
        assertFalse(events.any { it.getString("event") == "model_failover" })
        assertNoSecrets(events.toString())
    }

    @Test fun httpTimeoutsSwitchOnceAndReuseTheSameRequestBody() {
        for (status in listOf(408, 504)) {
            val first = FakeConnection(status, "untrusted $primaryKey $backupKey")
            val second = FakeConnection(200, success)
            val opened = AtomicInteger()
            val events = CopyOnWriteArrayList<JSONObject>()
            StandaloneApi(profile(), onTransportEvent = { name, data -> events += JSONObject(data.toString()).put("event", name) }) { url ->
                assertEquals(endpoint, url)
                when (opened.incrementAndGet()) { 1 -> first; 2 -> second; else -> throw AssertionError("more than one fallback") }
            }.use { api -> assertTrue(invoke(api).getBoolean("ok")) }
            assertEquals(2, opened.get())
            assertEquals("Bearer $primaryKey", first.getRequestProperty("Authorization"))
            assertEquals("Bearer $backupKey", second.getRequestProperty("Authorization"))
            assertArrayEquals(first.sent.toByteArray(), second.sent.toByteArray())
            assertEquals(1, events.count { it.getString("event") == "model_failover" })
            assertFalse(first.instanceFollowRedirects || second.instanceFollowRedirects)
            assertNoSecrets(events.toString())
        }
    }

    @Test fun enforcedPrimaryDeadlineCanFallbackEvenIfOldIoIgnoresInterrupts() {
        val release = CountDownLatch(1)
        val primary = object : FakeConnection(200, success) {
            override fun getResponseCode(): Int { awaitIgnoringInterrupts(release); return super.getResponseCode() }
        }
        val backup = FakeConnection(200, success)
        val opened = AtomicInteger()
        val started = System.nanoTime()
        try {
            StandaloneApi(profile()) { when (opened.incrementAndGet()) { 1 -> primary; 2 -> backup; else -> throw AssertionError("third attempt") } }.use { api ->
                assertTrue(invoke(api, 1200).getBoolean("ok"))
            }
            assertEquals(2, opened.get())
            assertEquals("Bearer $backupKey", backup.getRequestProperty("Authorization"))
            val elapsed = (System.nanoTime() - started) / 1_000_000
            assertTrue("primary must actually consume its deadline slice", elapsed >= 500)
            assertTrue("fallback must stay inside the original budget plus scheduling tolerance", elapsed < 1800)
        } finally { release.countDown() }
    }

    @Test fun socketTimeoutAlsoUsesBackupButOtherHttpErrorsNeverDo() {
        val opened = AtomicInteger()
        val failovers = mutableListOf<JSONObject>()
        StandaloneApi(profile(), onTransportEvent = { name, data -> if (name == "model_failover") failovers += JSONObject(data.toString()) }) {
            when (opened.incrementAndGet()) {
                1 -> object : FakeConnection(200, success) { override fun getResponseCode(): Int = throw SocketTimeoutException("sensitive $primaryKey") }
                2 -> FakeConnection(200, success)
                else -> throw AssertionError("third attempt")
            }
        }.use { assertTrue(invoke(it).getBoolean("ok")) }
        assertEquals(2, opened.get())
        assertEquals("timeout_response_headers", failovers.single().getString("diagnostic"))
        assertNoSecrets(failovers.toString())
        for (status in listOf(401, 403, 429, 500)) {
            var count = 0
            val events = mutableListOf<JSONObject>()
            StandaloneApi(profile(), onTransportEvent = { name, data -> events += JSONObject(data.toString()).put("event", name) }) {
                count++
                FakeConnection(status, "untrusted $primaryKey $backupKey")
            }.use { api ->
                val failure = fails { invoke(api) }
                assertEquals(status, failure.status)
                assertNoSecrets(failure.toString())
            }
            assertEquals("HTTP $status must not rotate credentials", 1, count)
            assertFalse(events.any { it.getString("event") == "model_failover" })
            assertNoSecrets(events.toString())
        }
    }

    @Test fun cancellationInsideFailoverCallbackClosesTheAttemptGap() {
        var count = 0
        var failovers = 0
        lateinit var api: StandaloneApi
        api = StandaloneApi(profile(), onTransportEvent = { name, _ ->
            if (name == "model_failover") { failovers++; api.cancel() }
        }) { count++; FakeConnection(408, "timeout") }
        api.use { fails { invoke(api) } }
        assertEquals(1, failovers)
        assertEquals("cancel between attempts must prevent the backup HTTP request", 1, count)
    }

    @Test fun bothTimeoutsRemainWithinOneTotalBudgetAndAtMostTwoAttempts() {
        val release = CountDownLatch(1)
        val opened = AtomicInteger()
        val started = System.nanoTime()
        try {
            StandaloneApi(profile()) {
                check(opened.incrementAndGet() <= 2)
                object : FakeConnection(200, success) {
                    override fun getResponseCode(): Int { awaitIgnoringInterrupts(release); return super.getResponseCode() }
                }
            }.use { api -> assertNoSecrets(fails { invoke(api, 1200) }.toString()) }
            assertEquals(2, opened.get())
            val elapsed = (System.nanoTime() - started) / 1_000_000
            assertTrue("two attempts must not each receive a new total timeout", elapsed < 1800)
            assertTrue("a real total timeout was exercised", elapsed >= 1000)
        } finally { release.countDown() }
    }

    @Test fun invalidBackupAndRequestParametersNeverOpenConnectionsOrExposeSecrets() {
        for (bad in listOf(primaryKey, " ", "line\nbreak", "非ASCII", "x".repeat(4097))) {
            try { profile().copy(backupApiKey = bad).validate(); fail("invalid backup accepted") }
            catch (failure: IllegalArgumentException) { assertNoSecrets(failure.toString()) }
        }
        assertNoSecrets(profile().toString())
        var count = 0
        StandaloneApi(profile()) { count++; FakeConnection(200, success) }.use { api ->
            try { api.complete("", "prompt", JSONObject(), emptyList(), 2000); fail("invalid request accepted") }
            catch (failure: ApiFailure) { assertNoSecrets(failure.toString()) }
            catch (failure: IllegalArgumentException) { assertNoSecrets(failure.toString()) }
        }
        assertEquals(0, count)
    }

    private fun fails(block: () -> Any): ApiFailure {
        try { block() } catch (failure: ApiFailure) { return failure }
        throw AssertionError("request unexpectedly succeeded")
    }
    private fun assertNoSecrets(value: String) {
        assertFalse(value.contains(primaryKey))
        assertFalse(value.contains(backupKey))
    }
    private fun awaitIgnoringInterrupts(release: CountDownLatch) {
        val until = System.nanoTime() + TimeUnit.SECONDS.toNanos(4)
        while (release.count != 0L && System.nanoTime() < until) {
            try { release.await(20, TimeUnit.MILLISECONDS) } catch (_: InterruptedException) { }
        }
    }
    private open class FakeConnection(private val code: Int, text: String) : HttpsURLConnection(URL("https://provider.example.invalid/v1")) {
        val sent = ByteArrayOutputStream()
        private val response = text.toByteArray(Charsets.UTF_8)
        override fun getOutputStream() = sent
        override fun getInputStream() = ByteArrayInputStream(response)
        override fun getErrorStream() = ByteArrayInputStream(response)
        override fun getResponseCode() = code
        override fun getContentLengthLong() = response.size.toLong()
        override fun getHeaderField(name: String?): String? = null
        override fun connect() = Unit
        override fun disconnect() = Unit
        override fun usingProxy() = false
        override fun getCipherSuite() = "fixture-no-network"
        override fun getLocalCertificates(): Array<Certificate>? = null
        override fun getServerCertificates(): Array<Certificate> = emptyArray()
    }
}
