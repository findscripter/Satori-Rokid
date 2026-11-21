package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

/** Opt-in production transport/engine contract; never a wearing or physical completion test. */
class ClassroomGuidanceLiveTest {
    @Test fun actualManualRetrievalAndModelResponsesPassTheProductionEngine() {
        assumeTrue("Explicit SATORI_GUIDANCE_LIVE=1 is required", System.getenv("SATORI_GUIDANCE_LIVE") == "1")
        val report = JSONObject().put("schema", "satori.classroom-guidance-contract/1")
            .put("started_unix_ms", System.currentTimeMillis()).put("passed", false)
            .put("input_kind", "synthetic_instruction_fixture").put("real_camera_frames", 0)
            .put("microphone_used", false).put("device_accessed", false)
            .put("physical_completion_verified", false).put("internet_verified", false)
            .put("model_calls_attempted", 0).put("timings_are", "API latency, not participant task duration")
        val calls = JSONArray()
        report.put("calls", calls)
        val engine = StandaloneEngine()
        var profile: StandaloneProfile? = null
        var failed = false
        val started = System.nanoTime()
        try {
            val configured = StandaloneProfile(baseUrl = env("SATORI_GUIDANCE_BASE_URL"), apiKey = env("SATORI_GUIDANCE_API_KEY"),
                planner = env("SATORI_GUIDANCE_PLANNER"), monitor = env("SATORI_GUIDANCE_MONITOR"),
                toolsBaseUrl = env("SATORI_GUIDANCE_TOOLS_URL"), toolsToken = env("SATORI_GUIDANCE_TOOLS_TOKEN"),
                manualId = "tl-sg105-v4")
            profile = configured
            configured.validate()
            val sourceBytes = Files.readAllBytes(Path.of(env("SATORI_GUIDANCE_MANUAL")))
            assertEquals("approved manual source changed", MANUAL_SHA, sha256(sourceBytes))
            val source = JSONObject(String(sourceBytes, Charsets.UTF_8))
            val fixture = Files.readAllBytes(Path.of(env("SATORI_GUIDANCE_FIXTURE")))
            assertEquals("only the approved synthetic scene may be sent", FIXTURE_SHA, sha256(fixture))
            val promptBytes = Files.readAllBytes(Path.of(env("SATORI_GUIDANCE_PROMPTS")))
            val prompts = JSONObject(String(promptBytes, Charsets.UTF_8))
            report.put("model", configured.planner).put("manual_source_sha256", MANUAL_SHA)
                .put("fixture_sha256", FIXTURE_SHA).put("fixture_archive_member", "e2e-1793891288763/scene-0.jpg")
                .put("prompt_source_sha256", sha256(promptBytes))
            engine.setGoal(GOAL, true, 0)
            StandaloneApi(configured).use { api ->
                repeat(2) { index ->
                    val at = (index + 1) * 10_000L
                    val request = engine.nextRequest(at, listOf("synthetic-instruction-$index"))
                        ?: throw AssertionError("production engine did not issue the next request")
                    assertTrue("only decomposition/planning is authorized", request.stage in setOf("decompose", "plan"))
                    val call = JSONObject().put("stage", request.stage).put("request_id", request.requestId)
                    calls.put(call)
                    val lookupStarted = System.nanoTime()
                    val query = (request.payload.optString("goal") + " " +
                        request.payload.optJSONObject("active_step")?.optString("instruction").orEmpty()).take(500)
                    val manual = api.searchManual(query)
                    call.put("manual_latency_ms", elapsedMs(lookupStarted)).put("manual_context", manual)
                    verifyManual(manual, source)
                    request.payload.put("manual_context", manual)
                        .put("input_provenance", "合成几何图片仅为图像传输夹具，不是教室或设备照片。目标是按指定型号手册安排尚未执行的接线演练；现场状态、接点和联网结果均未知。")
                        .put("frames_note", JSONArray().put(JSONObject().put("frame_id", request.frameIds.single())
                            .put("timestamp_ms", at).put("synthetic", true)))
                    call.put("request_payload", JSONObject(request.payload.toString()))
                    val prompt = StandaloneManuals.stagePrompt(prompts.getString(request.stage))
                    call.put("effective_prompt_sha256", sha256(prompt.toByteArray(Charsets.UTF_8)))
                    report.put("model_calls_attempted", index + 1)
                    val modelStarted = System.nanoTime()
                    val response = try {
                        api.complete(configured.planner, prompt, request.payload, listOf(fixture), 120_000)
                    } finally { call.put("model_latency_ms", elapsedMs(modelStarted)) }
                    call.put("model_response", response)
                    // Keep the provider object intact; production accept validates a separate copy.
                    val original = response.toString()
                    val speech = engine.accept(request, response, at + 1)
                    val accepted = engine.snapshot(at + 1)
                    call.put("engine_state", accepted).put("spoken_hints", JSONArray(speech))
                        .put("accepted", accepted.getInt("failures") == 0)
                    assertEquals("Engine must not rewrite the provider response", original, response.toString())
                    assertEquals("production engine rejected the actual response", 0, accepted.getInt("failures"))
                    assertTrue("production error must remain empty", accepted.isNull("error"))
                    assertTrue("actual response must retain manual references", response.getJSONArray("source_sections").length() > 0)
                    when (accepted.getString("phase")) {
                        "awaiting_clarification" -> {
                            assertEquals("two calls exhausted without an executable plan", 0, index)
                            call.put("scripted_clarification", CLARIFICATION)
                            engine.answerClarification(CLARIFICATION, at + 2)
                            report.put("route", "clarification_then_nested_first_plan")
                        }
                        "awaiting_plan_confirmation" -> {
                            call.put("scripted_command", "confirm_decomposition")
                            engine.confirm(at + 2)
                            if (index == 0) {
                                // Exercise the real standalone planning path even when first_plan was reusable.
                                engine.replan(at + 3)
                                report.put("route", "decompose_then_independent_plan")
                            }
                        }
                        "executing" -> assertEquals("unexpected initial execution before confirmation", 1, index)
                        else -> throw AssertionError("unexpected production phase after acceptance")
                    }
                }
            }
            val finalState = engine.snapshot(30_000)
            report.put("final_state", finalState)
            assertEquals("two requests must produce a plan", "executing", finalState.getString("phase"))
            val desires = finalState.getJSONArray("desires")
            val active = (0 until desires.length()).map { desires.getJSONObject(it) }
                .single { it.getString("desire_id") == finalState.getString("active_desire_id") }
            assertTrue("the accepted plan must concern the classroom task", Regex("网线|LAN|交换机|教学|路由|型号|铭牌|端口")
                .containsMatchIn(active.getJSONObject("plan").getJSONArray("steps").toString()))
            assertTrue("planning alone cannot complete any desire", (0 until desires.length()).none {
                desires.getJSONObject(it).getString("status") == "done"
            })
            report.put("passed", true)
        } catch (failure: Throwable) {
            failed = true
            report.put("failure_kind", failure.javaClass.simpleName).put("failure_message", failure.message ?: "")
                .put("failure_state", engine.snapshot())
            if (failure is ApiFailure) report.put("http_status", failure.status).put("api_diagnostic", failure.diagnostic)
        } finally {
            report.put("total_latency_ms", elapsedMs(started))
            val secrets = profile?.let { listOf(it.baseUrl, it.apiKey, it.backupApiKey, it.toolsBaseUrl, it.toolsToken) } ?: emptyList()
            val destination = Path.of(env("SATORI_GUIDANCE_EVIDENCE"), "guidance-report.json")
            Files.write(destination, ((redact(report, secrets) as JSONObject).toString(2) + "\n").toByteArray(Charsets.UTF_8),
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)
        }
        if (failed) throw AssertionError("Live guidance contract failed; inspect the sanitized guidance report")
    }

    private fun verifyManual(context: JSONObject, source: JSONObject) {
        assertTrue("real HTTP service must return evidence", context.getBoolean("matched"))
        val rows = context.getJSONArray("passages")
        val sourceRows = source.getJSONArray("sections")
        val sections = (0 until sourceRows.length()).associate { sourceRows.getJSONObject(it).getString("section_id") to sourceRows.getJSONObject(it) }
        val retrieved = (0 until rows.length()).map { rows.getJSONObject(it) }
        assertTrue("retrieval must include the network verification boundary", retrieved.any { it.getString("section_id") == "verification-boundary" })
        for (row in retrieved) {
            assertEquals("source hash is locked", MANUAL_SHA, row.getString("sha256"))
            for (field in listOf("manual_id", "model", "version", "source_url", "synthetic"))
                assertEquals("manual metadata changed: $field", source.get(field), row.get(field))
            assertFalse("official manual must remain non-synthetic", row.getBoolean("synthetic"))
            val section = sections[row.getString("section_id")] ?: throw AssertionError("unknown official section")
            assertTrue("page must exist even when null", row.has("page"))
            assertEquals("page/null provenance changed", section.get("page"), row.get("page"))
            assertEquals("retrieved text differs from locked source", section.getString("text"), row.getString("text"))
        }
    }

    private fun env(name: String): String = System.getenv(name)?.takeIf { it.isNotBlank() }
        ?: throw IllegalArgumentException("missing required live test environment: $name")

    private fun elapsedMs(start: Long) = (System.nanoTime() - start) / 1_000_000
    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    private fun redact(value: Any, secrets: List<String>): Any = when (value) {
        is JSONObject -> JSONObject().apply { value.keys().asSequence().forEach { put(redact(it, secrets) as String, redact(value.get(it), secrets)) } }
        is JSONArray -> JSONArray().apply { for (i in 0 until value.length()) put(redact(value.get(i), secrets)) }
        is String -> secrets.filter(String::isNotEmpty).fold(value) { text, secret -> text.replace(secret, "<redacted>") }
        else -> value
    }

    companion object {
        private const val MANUAL_SHA = "0a1152f2bc22c2f9295eb5ba0b11ef305cc2ebbde1a63c1f0060a17ac4831258"
        private const val FIXTURE_SHA = "78c47f8aeb09639b59c7fd41ab3e38edc378837eead2674f9c0c254f376fac7a"
        private const val GOAL = "按 TL-SG105(UN) V4.0 手册演练教室接线：路由器 LAN 接交换机，再接教学电脑；查看 Link/Act 指示灯并在该电脑核验教学资源能否访问。我还没开始，型号已作为演练条件给定，不把演练当作现场完成。"
        private const val CLARIFICATION = "演练型号固定为 TL-SG105(UN) V4.0，使用该版本手册。画面是合成几何夹具，没有设备实景；请安排尚未执行的接线与教学资源验证步骤，现场状态仍待核实。"
    }
}
