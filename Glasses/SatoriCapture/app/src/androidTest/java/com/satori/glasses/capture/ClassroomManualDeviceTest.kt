package com.satori.glasses.capture

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Explicit opt-in: one real manual lookup and at most one paid model call, with no sensor input.
 * Never creates/replaces a Service, changes a saved profile, or writes the user's task state.
 */
@RunWith(AndroidJUnit4::class)
class ClassroomManualDeviceTest {
    @Test fun unknownBundledManualCannotReplaceSavedConfiguration() {
        assumeTrue("use -e satoriManualTools true for a configured-device check",
            InstrumentationRegistry.getArguments().getString("satoriManualTools") == "true")
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val settings = StandaloneSettings(target)
        val original = settings.load().also { it.validate() }
        val prefs = target.getSharedPreferences("standalone_secure", android.content.Context.MODE_PRIVATE)
        val before = prefs.getString("envelope", null)
        var saved = false
        try {
            try {
                settings.save(original.copy(manualSource = "device", manualId = "absent-acceptance-fixture", toolsBaseUrl = "", toolsToken = ""))
                saved = true
            } catch (_: java.io.FileNotFoundException) { }
            assertFalse("unknown bundled manual must not be saved", saved)
            assertTrue("encrypted configuration must remain byte-for-byte unchanged", before == prefs.getString("envelope", null))
        } finally {
            if (saved) settings.save(original)
        }
    }

    @Test(timeout = 55_000) fun realToolLookupWithoutCloud() {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue("use -e satoriManualTools true for this manual/source lookup",
            arguments.getString("satoriManualTools") == "true")
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val report = JSONObject().put("schema", "satori.classroom-manual-tool-check/1")
            .put("model_calls_attempted", 0).put("camera_frames_sent", 0).put("microphone_used", false)
            .put("passed", false)
        val started = SystemClock.elapsedRealtime()
        var failureKind: String? = null
        try {
            val profile = StandaloneSettings(target).load().also { it.validate() }
            val expectedCredentialHash = arguments.getString("expected_key_sha256")
            if (expectedCredentialHash != null) {
                val actual = java.security.MessageDigest.getInstance("SHA-256").digest(profile.apiKey.toByteArray(Charsets.UTF_8))
                    .joinToString("") { "%02x".format(it.toInt() and 255) }
                assertTrue("configured model credential differs from the preserved approved source", actual == expectedCredentialHash)
                report.put("model_credential_preserved", true)
            }
            assertTrue("configure the official TL-SG105 V4 manual first", profile.manualId == "tl-sg105-v4")
            val expectedHash = arguments.getString("expectedManualSha256")
            if (expectedHash != null) assertTrue("expected source hash must be SHA256", expectedHash.matches(Regex("[a-f0-9]{64}")))
            StandaloneApi(profile, manualAsset = { name -> target.assets.open("manuals/$name").use { it.readBytes() } }).use { api ->
                val evidence = api.searchManual("TL-SG105 LAN 网线 Link Act 互联网")
                report.put("manual_source", profile.manualSource).put("transport", evidence.getString("transport"))
                if (profile.manualSource == "device") {
                    val check = evidence.getJSONObject("source_check")
                    assertEquals("original_pdf_sha256_verified", check.getString("status"))
                    assertFalse(check.getBoolean("cache_hit"))
                    report.put("source_check", check)
                }
                assertTrue("real HTTP lookup must return matching evidence", evidence.getBoolean("matched"))
                val passages = evidence.getJSONArray("passages")
                assertTrue("expected one to four bounded passages", passages.length() in 1..4)
                val metadata = JSONArray()
                for (i in 0 until passages.length()) {
                    val row = passages.getJSONObject(i)
                    assertTrue("wrong manual returned", row.getString("manual_id") == "tl-sg105-v4")
                    assertTrue("wrong hardware revision returned", row.getString("model") == "TL-SG105(UN) V4.0")
                    assertFalse("official fixture must not be synthetic", row.getBoolean("synthetic"))
                    val hash = row.getString("sha256")
                    assertTrue("source JSON SHA256 is required", hash.matches(Regex("[a-f0-9]{64}")))
                    if (expectedHash != null) assertTrue("source JSON differs from expected corpus", hash == expectedHash)
                    assertTrue("citation must identify its own manual and section",
                        row.getString("citation_id") == "tl-sg105-v4:" + row.getString("section_id"))
                    metadata.put(JSONObject().put("manual_id", "tl-sg105-v4").put("model", "TL-SG105(UN) V4.0")
                        .put("sha256", hash).put("citation_id", row.getString("citation_id")).put("page", row.get("page")))
                }
                report.put("retrieved_metadata", metadata).put("expected_source_hash_checked", expectedHash != null).put("passed", true)
            }
        } catch (failure: Throwable) {
            failureKind = failure.javaClass.simpleName
            report.put("failure_kind", failureKind)
        } finally {
            report.put("tool_latency_ms", SystemClock.elapsedRealtime() - started)
            val folder = File(target.filesDir, "diagnostics").apply { check(mkdirs() || isDirectory) }
            val destination = File(folder, "classroom-manual-tool-only-${System.currentTimeMillis()}-${SystemClock.elapsedRealtimeNanos()}.json")
            check(destination.createNewFile())
            destination.writeText(report.toString(2))
            println(JSONObject().put("classroom_manual_tool_passed", report.optBoolean("passed"))
                .put("model_calls_attempted", 0).put("diagnostic_file", destination.name))
        }
        if (failureKind != null) throw AssertionError("Tool-only device check failed ($failureKind); inspect private diagnostics")
    }

    @Test(timeout = 135_000) fun realManualLookupGroundsOneTypedQuestionWithoutObservingTheUser() {
        assumeTrue("use -e satoriManualLive true to authorize this one real model request",
            InstrumentationRegistry.getArguments().getString("satoriManualLive") == "true")
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val report = JSONObject().put("schema", "satori.classroom-manual-live-check/1")
            .put("started_unix_ms", System.currentTimeMillis()).put("input_kind", "synthetic_typed_question")
            .put("camera_frames_sent", 0).put("microphone_used", false).put("model_calls_attempted", 0)
            .put("passed", false)
        var profile: StandaloneProfile? = null
        var failureKind: String? = null
        val started = SystemClock.elapsedRealtime()
        try {
            val loaded = StandaloneSettings(target).load().also { it.validate() }
            profile = loaded
            assertTrue("configure the specific public TL-SG105 V4 manual before the live check", loaded.manualId == "tl-sg105-v4")
            assertTrue("a device manual or separately configured endpoint is required",
                loaded.manualSource == "device" || loaded.toolsBaseUrl.isNotBlank())
            report.put("model", loaded.planner).put("manual_id", loaded.manualId).put("manual_source", loaded.manualSource)
            StandaloneApi(loaded, manualAsset = { name -> target.assets.open("manuals/$name").use { it.readBytes() } }).use { api ->
                val lookupAt = SystemClock.elapsedRealtime()
                val evidence = api.searchManual("TL-SG105 的 Link Act 指示灯亮，能否确认互联网可用，LAN 网线")
                report.put("transport", evidence.getString("transport")).put("source_check", evidence.optJSONObject("source_check"))
                report.put("tool_latency_ms", SystemClock.elapsedRealtime() - lookupAt)
                assertTrue("real service must return matching manual evidence", evidence.getBoolean("matched"))
                val passages = evidence.getJSONArray("passages")
                val metadata = JSONArray()
                var hasNetworkBoundary = false
                for (i in 0 until passages.length()) {
                    val passage = passages.getJSONObject(i)
                    assertTrue("wrong manual returned", passage.getString("manual_id") == "tl-sg105-v4")
                    assertTrue("manual hardware revision must match the fixture", passage.getString("model") == "TL-SG105(UN) V4.0")
                    assertFalse("the official manual fixture must not be replaced by a synthetic manual", passage.getBoolean("synthetic"))
                    assertTrue("imported source JSON hash is required", passage.getString("sha256").matches(Regex("[a-f0-9]{64}")))
                    if (passage.getString("section_id") == "verification-boundary") hasNetworkBoundary = true
                    val entry = JSONObject()
                    for (key in listOf("citation_id", "manual_id", "model", "version", "sha256", "section_id", "page", "synthetic"))
                        entry.put(key, passage.get(key))
                    metadata.put(entry)
                }
                report.put("retrieved_metadata", metadata)
                assertTrue("lookup must include the explicit link-versus-Internet evidence", hasNetworkBoundary)
                val question = "这是手册阅读测试，不是在报告现场状态。只根据给定的 TL-SG105(UN) V4.0 手册说明：Link/Act 指示灯亮能证明互联网已经可用吗？请引用本次提供的段落，简短说明还需要核验什么。"
                val payload = JSONObject().put("user_utterance", question).put("manual_context", evidence)
                    .put("task", JSONObject().put("goal", JSONObject().put("text", "理解指定手册的联网验证条件")))
                    .put("frames_note", JSONArray())
                val modelAt = SystemClock.elapsedRealtime()
                report.put("model_calls_attempted", 1)
                val raw = api.complete(loaded.planner, StandaloneManuals.DIALOGUE_PROMPT + StandaloneManuals.GROUNDED_RULES,
                    payload, emptyList(), 75_000)
                report.put("model_latency_ms", SystemClock.elapsedRealtime() - modelAt)
                val answer = StandaloneManuals.answer(raw, evidence)
                report.put("answer", answer)
                assertTrue("manual explanation must be a grounded answer", answer.getString("kind") == "answer")
                assertTrue("the actual response must cite retrieved evidence", answer.getJSONArray("source_sections").length() > 0)
                assertFalse("a document question must not request task replanning", answer.getBoolean("suggest_replan"))
                val text = answer.getString("text")
                assertTrue("the actual reply must distinguish link state from Internet availability",
                    Regex("不代表|不等于|不能|无法|不足以").containsMatchIn(text) &&
                        Regex("互联网|上网|联网|Internet|internet").containsMatchIn(text))
                report.put("passed", true)
            }
        } catch (failure: Throwable) {
            failureKind = failure.javaClass.simpleName
            report.put("failure_kind", failureKind)
        } finally {
            report.put("total_latency_ms", SystemClock.elapsedRealtime() - started)
            val secrets = profile?.let { listOf(it.apiKey, it.backupApiKey, it.toolsToken, it.baseUrl, it.toolsBaseUrl).filter(String::isNotEmpty) } ?: emptyList()
            fun redacted(value: Any): Any = when (value) {
                is JSONObject -> JSONObject().apply { value.keys().asSequence().forEach { key -> put(key, redacted(value.get(key))) } }
                is JSONArray -> JSONArray().apply { for (i in 0 until value.length()) put(redacted(value.get(i))) }
                is String -> secrets.fold(value) { text, secret -> text.replace(secret, "<redacted>") }
                else -> value
            }
            val encoded = (redacted(report) as JSONObject).toString(2)
            val folder = File(target.filesDir, "diagnostics").apply { check(mkdirs() || isDirectory) }
            val destination = File(folder, "classroom-manual-live-${System.currentTimeMillis()}-${SystemClock.elapsedRealtimeNanos()}.json")
            check(destination.createNewFile())
            destination.writeText(encoded)
            println(JSONObject().put("classroom_manual_live_passed", report.optBoolean("passed"))
                .put("model_calls_attempted", report.optInt("model_calls_attempted"))
                .put("camera_frames_sent", 0).put("diagnostic_file", destination.name))
        }
        if (failureKind != null) throw AssertionError("Controlled manual live check failed ($failureKind); inspect sanitized private diagnostics")
    }
}
