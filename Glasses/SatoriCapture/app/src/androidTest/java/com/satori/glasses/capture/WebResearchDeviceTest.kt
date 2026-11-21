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

/** Public-text probe only: no Service creation, task change, camera or microphone input. */
@RunWith(AndroidJUnit4::class)
class WebResearchDeviceTest {
    @Test(timeout = 110_000) fun runtimeInputChoosesWhetherToSearchWithoutAPresetTask() {
        assumeTrue("explicit live opt-in required", InstrumentationRegistry.getArguments().getString("satoriWebLive") == "true")
        val arguments = InstrumentationRegistry.getArguments()
        val input = arguments.getString("satoriWebQuery") ?: throw AssertionError("Supply the public test query at runtime")
        require(input.length in 1..500)
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val report = JSONObject().put("schema", "satori.web-research-device/1").put("passed", false)
            .put("started_unix_ms", System.currentTimeMillis()).put("camera_frames_sent", 0).put("microphone_used", false)
            .put("model_calls_attempted", 0).put("mcp_events", JSONArray())
        var stage = "configuration"
        var failureKind: String? = null
        val started = SystemClock.elapsedRealtime()
        try {
            val profile = StandaloneSettings(target).load().also { it.validate() }
            assertEquals("web", profile.manualSource)
            assertEquals("web-discovery", profile.manualId)
            assertTrue(profile.toolsBaseUrl.isEmpty() && profile.toolsToken.isEmpty())
            val expected = InstrumentationRegistry.getArguments().getString("expected_key_sha256")
            if (expected != null) {
                val actual = java.security.MessageDigest.getInstance("SHA-256").digest(profile.apiKey.toByteArray(Charsets.UTF_8))
                    .joinToString("") { "%02x".format(it) }
                assertTrue("model credential changed", actual == expected)
                report.put("model_credential_preserved", true)
            }
            arguments.getString("expected_backup_sha256")?.let { expectedBackup ->
                assertTrue(profile.backupApiKey.isNotEmpty())
                val actual = java.security.MessageDigest.getInstance("SHA-256").digest(profile.backupApiKey.toByteArray(Charsets.UTF_8))
                    .joinToString("") { "%02x".format(it) }
                assertTrue("backup credential differs from the authorized value", actual == expectedBackup)
                report.put("backup_credential_verified", true)
            }
            report.put("model_attempt_events", JSONArray())
            StandaloneApi(profile, onTransportEvent = { name, data ->
                report.getJSONArray("model_attempt_events").put(JSONObject(data.toString()).put("event", name))
            }).use { api ->
                report.put("runtime_query", input)
                stage = "search_planning"
                val directTransportCheck = arguments.getString("satoriWebDirect") == "true"
                report.put("model_calls_attempted", if (directTransportCheck) 0 else 1)
                    .put("decision_mode", if (directTransportCheck) "explicit_query_transport_check" else "model_runtime_decision")
                val planAt = SystemClock.elapsedRealtime()
                val rawPlan = if (directTransportCheck) JSONObject().put("needs_search", true).put("query", input)
                    else api.complete(profile.planner, WebResearchPolicy.PLAN_PROMPT,
                        JSONObject().put("userText", input).put("frames_note", JSONArray()), emptyList(), 30000)
                val plan = WebResearchPolicy.plan(rawPlan)
                report.put("plan_latency_ms", SystemClock.elapsedRealtime() - planAt).put("search_plan", plan)
                arguments.getString("expectedSearch")?.let { expected -> assertEquals(expected == "true", plan.getBoolean("needs_search")) }
                stage = "mcp_search_fetch"
                val searchAt = SystemClock.elapsedRealtime()
                val evidence = api.researchWeb(plan) { name, data -> report.getJSONArray("mcp_events").put(JSONObject(data.toString()).put("event", name)) }
                report.put("search_latency_ms", SystemClock.elapsedRealtime() - searchAt).put("evidence", evidence)
                val pages = evidence.getJSONArray("passages")
                if (plan.getBoolean("needs_search")) {
                    assertEquals("mcp_streamable_http", evidence.getString("transport"))
                    assertTrue(evidence.getBoolean("matched"))
                    assertTrue((0 until pages.length()).any { pages.getJSONObject(it).getString("retrieval_kind") == "fetched_page" })
                } else {
                    assertEquals("not_requested", evidence.getString("transport"))
                    assertEquals(0, report.getJSONArray("mcp_events").length())
                    assertEquals(0, pages.length())
                }
                for (i in 0 until pages.length()) {
                    assertEquals("unverified", pages.getJSONObject(i).getString("authority"))
                    assertFalse(pages.getJSONObject(i).getBoolean("images_reviewed"))
                }
                report.put("passed", true)
            }
        } catch (failure: Throwable) {
            failureKind = failure.javaClass.simpleName
            report.put("failure_kind", failureKind).put("stage", stage)
            if (failure is ApiFailure) report.put("http_status", failure.status).put("diagnostic", failure.diagnostic)
        } finally {
            report.put("total_latency_ms", SystemClock.elapsedRealtime() - started)
            val dir = File(target.filesDir, "diagnostics").apply { check(mkdirs() || isDirectory) }
            val file = File(dir, "web-research-${System.currentTimeMillis()}-${SystemClock.elapsedRealtimeNanos()}.json")
            check(file.createNewFile()); file.writeText(report.toString(2))
            println(JSONObject().put("web_research_passed", report.getBoolean("passed")).put("diagnostic_file", file.name))
        }
        if (failureKind != null) throw AssertionError("Web discovery failed at $stage ($failureKind); inspect private report")
    }
}
