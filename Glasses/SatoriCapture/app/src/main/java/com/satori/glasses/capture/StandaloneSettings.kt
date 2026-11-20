package com.satori.glasses.capture

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import org.json.JSONObject
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Runtime configuration. Credentials never belong in resources, logs or the APK. */
data class StandaloneProfile(
    val baseUrl: String = "",
    val apiKey: String = "",
    val planner: String = "",
    val monitor: String = "",
    val asrModel: String = "whisper-1",
    val ttsModel: String = "gpt-4o-mini-tts",
    val ttsVoice: String = "alloy",
    val toolsBaseUrl: String = "",
    val toolsToken: String = "",
    val manualId: String = "",
    val assistanceMode: String = "guided",
    val manualSource: String = "remote_api",
    val backupApiKey: String = "",
) {
    fun validate() {
        StandaloneApi.normalizeBaseUrl(baseUrl)
        require(apiKey.isNotBlank() && apiKey.length <= 4096 && apiKey.none { it.isISOControl() }) { "请配置 API 密钥" }
        require(backupApiKey.isEmpty() || backupApiKey.length <= 4096 && backupApiKey.all { it.code in 33..126 } && backupApiKey != apiKey) {
            "备用密钥必须是与主密钥不同的有效ASCII密钥"
        }
        listOf(planner, monitor, asrModel, ttsModel, ttsVoice).forEach {
            require(it.isNotBlank() && it.length <= 200 && it.none { c -> c.isISOControl() }) { "请配置有效的模型和语音名称" }
        }
        require(assistanceMode in setOf("guided", "reactive")) { "任务模式无效" }
        require(manualSource in setOf("device", "remote_api", "web")) { "手册来源无效" }
        if (manualSource == "web") {
            require(manualId == "web-discovery" && toolsBaseUrl.isEmpty() && toolsToken.isEmpty()) { "联网检索配置无效" }
        } else if (manualSource == "device") {
            require(manualId.matches(Regex("[A-Za-z0-9][A-Za-z0-9_.-]{0,79}"))) { "手册编号无效" }
            require(toolsBaseUrl.isEmpty() && toolsToken.isEmpty()) { "本机手册模式不使用外部服务令牌" }
        } else if (toolsBaseUrl.isNotEmpty() || manualId.isNotEmpty() || toolsToken.isNotEmpty()) {
            StandaloneManuals.endpoint(toolsBaseUrl)
            require(manualId.matches(Regex("[A-Za-z0-9][A-Za-z0-9_.-]{0,79}"))) { "手册编号无效" }
            require(toolsToken.length in 32..512 && toolsToken.all { it.code in 33..126 }) { "请配置32到512位ASCII手册服务令牌" }
        }
    }

    // Prevent accidental interpolation from exposing the credential.
    override fun toString() = "StandaloneProfile(configured=${baseUrl.isNotBlank()}, apiKey=<redacted>)"
}

/** AES-GCM envelope; the non-exportable key stays in AndroidKeyStore, ciphertext in app-private preferences. */
class StandaloneSettings(context: Context) {
    private val assets = context.applicationContext.assets
    private val prefs = context.applicationContext.getSharedPreferences("standalone_secure", Context.MODE_PRIVATE)
    private val alias = "satori.standalone.config.v1"

    @Synchronized fun load(): StandaloneProfile {
        val encoded = prefs.getString("envelope", null) ?: return StandaloneProfile()
        try {
            val envelope = JSONObject(encoded)
            val iv = Base64.decode(envelope.getString("iv"), Base64.NO_WRAP)
            val data = Base64.decode(envelope.getString("data"), Base64.NO_WRAP)
            require(iv.size == 12 && data.size in 16..20000)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
            cipher.updateAAD(alias.toByteArray(Charsets.UTF_8))
            val value = JSONObject(String(cipher.doFinal(data), Charsets.UTF_8))
            return StandaloneProfile(value.optString("base_url"), value.optString("api_key"),
                value.optString("planner"), value.optString("monitor"), value.optString("asr_model", "whisper-1"),
                value.optString("tts_model", "gpt-4o-mini-tts"), value.optString("tts_voice", "alloy"),
                value.optString("tools_base_url"), value.optString("tools_token"), value.optString("manual_id"),
                value.optString("assistance_mode", "guided"), value.optString("manual_source", "remote_api"), value.optString("backup_api_key"))
        } catch (_: Exception) {
            throw IllegalStateException("无法解密本机配置，请重新保存 API 设置")
        }
    }

    @Synchronized fun save(profile: StandaloneProfile) {
        profile.validate()
        if (profile.manualSource == "device") {
            val bundled = assets.open("manuals/${profile.manualId}.json").use { it.readBytes() }
            require(LocalManualStore(listOf(bundled)).lookup(profile.manualId) != null) { "安装包中没有所选手册" }
        }
        val value = JSONObject().put("base_url", StandaloneApi.normalizeBaseUrl(profile.baseUrl))
            .put("api_key", profile.apiKey).put("planner", profile.planner).put("monitor", profile.monitor)
            .put("backup_api_key", profile.backupApiKey)
            .put("asr_model", profile.asrModel).put("tts_model", profile.ttsModel).put("tts_voice", profile.ttsVoice)
            .put("tools_base_url", profile.toolsBaseUrl).put("tools_token", profile.toolsToken)
            .put("manual_id", profile.manualId).put("assistance_mode", profile.assistanceMode)
            .put("manual_source", profile.manualSource)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        val plaintext = value.toString().toByteArray(Charsets.UTF_8)
        require(plaintext.size <= 19984) { "配置内容过长，无法安全保存" }
        cipher.init(Cipher.ENCRYPT_MODE, key())
        cipher.updateAAD(alias.toByteArray(Charsets.UTF_8))
        val envelope = JSONObject().put("iv", Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .put("data", Base64.encodeToString(cipher.doFinal(plaintext), Base64.NO_WRAP))
        check(prefs.edit().putString("envelope", envelope.toString()).commit()) { "保存配置失败" }
    }

    @Synchronized fun clear() {
        check(prefs.edit().clear().commit()) { "清除配置失败" }
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (store.containsAlias(alias)) store.deleteEntry(alias)
    }

    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(alias, null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256).setRandomizedEncryptionRequired(true).build())
        }.generateKey()
    }
}
