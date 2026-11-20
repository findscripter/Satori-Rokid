package com.satori.glasses.capture

import android.content.Intent
import android.os.Bundle
import org.json.JSONObject

/** Runtime settings. Defaults suit wired debugging through `adb reverse tcp:47800 tcp:47800`. */
data class CaptureConfig(
    val host: String = "127.0.0.1",
    val port: Int = 47800,
    val fps: Double = 2.0,
    val width: Int = 640,
    val height: Int = 480,
    val jpegQuality: Int = 80,
    val imuHz: Int = 100,
    val streaming: Boolean = true,
    val pauseWhenRemoved: Boolean = true,
    val interceptAiKey: Boolean = true,
    /** STREAM_MUSIC level 0..15 applied on change; -1 leaves the system volume alone. */
    val volume: Int = -1,
) {
    fun sanitized(): CaptureConfig = copy(
        host = safeHost(host),
        port = port.coerceIn(1, 65535),
        fps = if (fps.isFinite()) fps.coerceIn(0.1, 15.0) else 2.0,
        width = width.coerceIn(160, 4032),
        height = height.coerceIn(120, 3024),
        jpegQuality = jpegQuality.coerceIn(30, 100),
        imuHz = imuHz.coerceIn(10, 400),
        volume = volume.coerceIn(-1, 15),
    )

    /**
     * Applies the keys present in a PC `config` message; absent keys keep their value.
     * Throws IllegalArgumentException for a wrongly typed or non-finite value, so the caller can reject the message.
     */
    fun merged(json: JSONObject): CaptureConfig {
        fun number(key: String): Double? {
            if (!json.has(key)) return null
            val value = json.opt(key)
            if (value !is Number) throw IllegalArgumentException("$key must be a number") // org.json booleans are not Numbers
            return value.toDouble().takeIf { it.isFinite() } ?: throw IllegalArgumentException("$key must be finite")
        }
        fun flag(key: String): Boolean? {
            if (!json.has(key)) return null
            return json.opt(key) as? Boolean ?: throw IllegalArgumentException("$key must be true or false")
        }
        return copy(
            fps = number("fps") ?: fps,
            width = number("width")?.toInt() ?: width,
            height = number("height")?.toInt() ?: height,
            jpegQuality = number("jpeg_quality")?.toInt() ?: jpegQuality,
            imuHz = number("imu_hz")?.toInt() ?: imuHz,
            volume = number("volume")?.toInt() ?: volume,
            streaming = flag("streaming") ?: streaming,
            pauseWhenRemoved = flag("pause_when_removed") ?: pauseWhenRemoved,
            interceptAiKey = flag("intercept_ai_key") ?: interceptAiKey,
        ).sanitized()
    }

    fun toJson(): JSONObject = JSONObject()
        .put("host", host)
        .put("port", port)
        .put("fps", fps)
        .put("width", width)
        .put("height", height)
        .put("jpeg_quality", jpegQuality)
        .put("imu_hz", imuHz)
        .put("streaming", streaming)
        .put("pause_when_removed", pauseWhenRemoved)
        .put("intercept_ai_key", interceptAiKey)
        .put("volume", volume)

    companion object {
        private val IPV4 = Regex("^(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})$")

        /**
         * The activity is exported (launcher), so any app could pass a host extra; frames may only go to
         * loopback (adb reverse) or a private LAN address given as a numeric IPv4 literal — never to a DNS name,
         * which could resolve anywhere (e.g. "10.evil.example").
         */
        fun safeHost(value: String): String {
            val host = value.trim()
            if (host == "localhost") return host
            val octets = IPV4.matchEntire(host)?.groupValues?.drop(1)?.map { it.toInt() } ?: return "127.0.0.1"
            if (octets.any { it > 255 }) return "127.0.0.1"
            val (a, b) = octets
            val allowed = a == 127 || a == 10 || (a == 192 && b == 168) || (a == 172 && b in 16..31)
            return if (allowed) host else "127.0.0.1"
        }

        fun fromJson(text: String?): CaptureConfig {
            if (text.isNullOrBlank()) return CaptureConfig()
            val json = runCatching { JSONObject(text) }.getOrNull() ?: return CaptureConfig()
            val base = CaptureConfig()
            return runCatching {
                base.copy(host = json.optString("host", base.host), port = json.optInt("port", base.port)).merged(json)
            }.getOrDefault(base)
        }

        /** Overrides from `am start ... --es host 192.0.2.10 --ei port 47800 --ef fps 2 --ez streaming true`. */
        fun fromIntent(base: CaptureConfig, intent: Intent?): CaptureConfig {
            val extras = intent?.extras ?: return base
            return base.copy(
                host = extras.text("host") ?: base.host,
                port = extras.number("port")?.toInt() ?: base.port,
                fps = extras.number("fps") ?: base.fps,
                width = extras.number("width")?.toInt() ?: base.width,
                height = extras.number("height")?.toInt() ?: base.height,
                jpegQuality = extras.number("jpeg_quality")?.toInt() ?: base.jpegQuality,
                imuHz = extras.number("imu_hz")?.toInt() ?: base.imuHz,
                volume = extras.number("volume")?.toInt() ?: base.volume,
                streaming = extras.flag("streaming") ?: base.streaming,
                pauseWhenRemoved = extras.flag("pause_when_removed") ?: base.pauseWhenRemoved,
                interceptAiKey = extras.flag("intercept_ai_key") ?: base.interceptAiKey,
            ).sanitized()
        }

        @Suppress("DEPRECATION")
        private fun Bundle.text(key: String): String? = get(key)?.toString()

        @Suppress("DEPRECATION")
        private fun Bundle.number(key: String): Double? = when (val value = get(key)) {
            is Number -> value.toDouble().takeIf { it.isFinite() }
            is String -> value.toDoubleOrNull()?.takeIf { it.isFinite() }
            else -> null
        }

        @Suppress("DEPRECATION")
        private fun Bundle.flag(key: String): Boolean? = when (val value = get(key)) {
            is Boolean -> value
            is String -> value.toBooleanStrictOrNull()
            else -> null
        }
    }
}
