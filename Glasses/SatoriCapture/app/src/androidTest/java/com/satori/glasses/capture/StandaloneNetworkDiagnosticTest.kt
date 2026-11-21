package com.satori.glasses.capture

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.net.SocketTimeoutException
import java.net.URI
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import javax.net.ssl.HttpsURLConnection

/**
 * Explicit opt-in only. Makes exactly three authenticated GET /models attempts, in this order:
 * default Android user agent, Python-urllib/3.12, SatoriAR-Android/<app version>.
 * Each attempt has a ten-second caller deadline and reads at most 4096 response bytes.
 * Only the six-field summaries are printed; provider text, headers and credentials never leave probe().
 * TLS validation and hostname verification use unmodified platform defaults. Redirects are disabled.
 */
@RunWith(AndroidJUnit4::class)
class StandaloneNetworkDiagnosticTest {
    @Test(timeout = 40_000) fun compareThreeUserAgentsWithoutLoggingResponseBodies() {
        val selected = InstrumentationRegistry.getArguments().getString("class").orEmpty()
            .split(',').map { it.trim().substringBefore('#') }
        assumeTrue("network diagnostics require explicit class/method selection", javaClass.name in selected)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val profile = StandaloneSettings(context).load()
        profile.validate()
        val executor = Executors.newFixedThreadPool(3) { task -> Thread(task, "network-diagnostic").apply { isDaemon = true } }
        val closers = Executors.newFixedThreadPool(3) { task -> Thread(task, "network-diagnostic-close").apply { isDaemon = true } }
        try {
            listOf<String?>(null, "Python-urllib/3.12", "SatoriAR-Android/${BuildConfig.VERSION_NAME}").forEach { userAgent ->
                val active = AtomicReference<HttpsURLConnection?>(null)
                val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
                val future = executor.submit(Callable {
                    try { probe(profile, userAgent, active, deadline) }
                    catch (_: Exception) { summary() }
                    finally {
                        active.getAndSet(null)?.let { connection ->
                            closers.execute { runCatching { connection.disconnect() } }
                        }
                    }
                })
                val result = try { future.get(10, TimeUnit.SECONDS) }
                catch (_: Exception) {
                    future.cancel(true)
                    active.getAndSet(null)?.let { connection ->
                        closers.execute { runCatching { connection.disconnect() } }
                    }
                    summary()
                }
                // Fixed ordering identifies the UA variant without adding a provider/header/log payload.
                println(result.toString())
            }
        } finally {
            executor.shutdownNow()
            closers.shutdown()
        }
    }

    private fun probe(profile: StandaloneProfile, userAgent: String?, active: AtomicReference<HttpsURLConnection?>,
                      deadline: Long): JSONObject {
        fun remaining(): Int {
            val milliseconds = TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime())
            if (milliseconds <= 0 || Thread.currentThread().isInterrupted) throw SocketTimeoutException()
            return milliseconds.coerceAtMost(10_000).toInt().coerceAtLeast(1)
        }
        val url = StandaloneApi.normalizeBaseUrl(profile.baseUrl) + "/models"
        val connection = URI(url).toURL().openConnection() as HttpsURLConnection
        active.set(connection)
        connection.instanceFollowRedirects = false
        connection.requestMethod = "GET"
        connection.connectTimeout = remaining()
        connection.readTimeout = remaining()
        connection.useCaches = false
        connection.setRequestProperty("Authorization", "Bearer ${profile.apiKey}")
        connection.setRequestProperty("Accept", "application/json")
        if (userAgent != null) connection.setRequestProperty("User-Agent", userAgent)
        val status = connection.responseCode
        val contentType = connection.contentType.orEmpty().substringBefore(';').trim().lowercase(Locale.ROOT)
            .let { if (it.matches(Regex("[a-z0-9.+-]{1,64}/[a-z0-9.+-]{1,64}"))) it else "" }
        val bytes = ByteArrayOutputStream(4_096)
        val input = if (status in 200..399) connection.inputStream else connection.errorStream
        input?.use { stream ->
            val buffer = ByteArray(1_024)
            while (bytes.size() < 4_096) {
                connection.readTimeout = remaining()
                val count = stream.read(buffer, 0, minOf(buffer.size, 4_096 - bytes.size()))
                if (count < 0) break
                if (count == 0) continue
                bytes.write(buffer, 0, count)
            }
        }
        val text = bytes.toString(Charsets.UTF_8.name()).lowercase(Locale.ROOT)
        val compact = text.replace(Regex("\\s+"), " ")
        return summary(status, contentType,
            html = contentType == "text/html" || text.contains("<html") || text.contains("<!doctype html"),
            invalidKey = listOf("invalid api key", "invalid_api_key", "incorrect api key", "api key is invalid",
                "invalid token", "无效的密钥", "密钥无效", "令牌无效", "无效的令牌", "api密钥不正确").any(compact::contains),
            ipDenied = listOf("ip denied", "ip not allowed", "ip address not allowed", "ip blocked", "blocked ip",
                "ip白名单", "ip 白名单", "白名单限制", "ip限制", "ip 限制", "ip不在", "ip 不在").any(compact::contains) ||
                Regex("\\bip\\b.{0,64}(not allowed|denied|blocked|forbidden|restricted)").containsMatchIn(compact),
            cloudflare = listOf("cloudflare", "cf-ray", "cf_chl", "attention required!").any(compact::contains))
    }

    private fun summary(status: Int = 0, contentType: String = "", html: Boolean = false,
                        invalidKey: Boolean = false, ipDenied: Boolean = false, cloudflare: Boolean = false): JSONObject =
        JSONObject().put("status", status).put("content_type", contentType).put("html", html)
            .put("invalid_key_phrase", invalidKey).put("ip_denied_phrase", ipDenied).put("cloudflare_phrase", cloudflare)
}
