package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class EngineRequest(
    val stage: String,
    val payload: JSONObject,
    val revision: Long,
    val frameIds: List<String>,
    val requestId: Long = 0
)

/**
 * Device-local BDI state machine. No Android, network, file, thread or clock dependencies.
 * The app loads the stage contracts from the bundled standalone-prompts.json.
 * The caller owns image capture, deadlines, speech cancellation and atomic persistence.
 *
 * User commands invalidate in-flight requests immediately. A request can be committed once, and
 * only to the exact state for which it was issued. Completion requires two reliable observations;
 * this port deliberately does not enable the Python implementation's single-view fast shortcut.
 */
class StandaloneEngine {
    private var state = freshState()
    private var inFlight: EngineRequest? = null
    private var requestSequence = 0L

    @Synchronized
    fun setGoal(goal: String, confirmMode: Boolean = true, nowMs: Long = 0): List<String> {
        val text = userText(goal, 300)
        val previous = state
        state = freshState().put("revision", previous.optLong("revision"))
            .put("mode", if (confirmMode) "confirm" else "auto")
            .put("presence", previous.optString("presence", "unknown"))
            .put("folded", previous.optBoolean("folded"))
            .put("paused_reason", previous.opt("paused_reason") ?: JSONObject.NULL)
            .put("rate_limit_until_ms", previous.optLong("rate_limit_until_ms"))
            .put("memory", clone(previous.getJSONObject("memory")))
            .put("goal", JSONObject().put("text", text).put("summary", JSONObject.NULL).put("horizon", JSONObject.NULL))
            .put("phase", "decomposing")
        touch(nowMs)
        return listOf("好的，我先看一下，马上给你安排。")
    }

    @Synchronized
    fun pause(nowMs: Long = 0): List<String> {
        withdrawStepConfirmation()
        breakSequence()
        state.put("paused_reason", "user")
        touch(nowMs)
        return listOf("已暂停。")
    }

    @Synchronized
    fun resume(nowMs: Long = 0): List<String> {
        if (state.optString("phase") in listOf("idle", "stopped", "complete")) return emptyList()
        state.put("paused_reason", if (state.optBoolean("folded")) "folded" else JSONObject.NULL)
        succeeded()
        state.put("retry_at_ms", 0)
        breakSequence()
        touch(nowMs)
        if (!state.isNull("paused_reason")) return emptyList()
        resetConfirmationTimer(nowMs)
        return if (state.optJSONObject("pending") != null) pendingSpeech() else repeat()
    }

    @Synchronized
    fun confirm(nowMs: Long = 0): List<String> {
        if (!mayInteract()) return emptyList()
        val pending = state.optJSONObject("pending") ?: return emptyList()
        if (pending.optString("kind") == "clarify") return listOf(pending.getString("text"))
        val speech = mutableListOf<String>()
        if (pending.optString("kind") == "confirm_decomposition") {
            state.put("pending", JSONObject.NULL)
            activateNext(nowMs, speech)
            applyFirstPlan(nowMs, speech)
        } else {
            val streaks = clone(state.getJSONObject("step_streaks"))
            val desireId = state.optString("active_desire_id")
            finishSteps(pending.getJSONArray("step_ids").strings().toSet(),
                pending.getJSONArray("inferred_ids").strings().toSet(), nowMs, speech, "user",
                nullableString(pending, "prefer"))
            carryStreaks(streaks, desireId)
        }
        touch(nowMs)
        return speech
    }

    @Synchronized
    fun redo(nowMs: Long = 0): List<String> {
        if (!mayInteract()) return emptyList()
        val step = activeStep() ?: return emptyList()
        step.put("needs_fresh_incomplete", true).put("done_by", JSONObject.NULL).put("done_at_ms", JSONObject.NULL)
        state.put("pending", JSONObject.NULL)
        state.getJSONObject("desire_streaks").put(state.getString("active_desire_id"), 0)
        breakSequence()
        val speech = mutableListOf<String>()
        startStep(step, nowMs, speech)
        touch(nowMs)
        return speech
    }

    @Synchronized
    fun skip(nowMs: Long = 0): List<String> = finishCurrent(nowMs, "skipped")

    @Synchronized
    fun done(nowMs: Long = 0): List<String> = finishCurrent(nowMs, "done")

    @Synchronized
    fun replan(nowMs: Long = 0): List<String> {
        if (!mayInteract() || activeDesire() == null) return emptyList()
        activeStep()?.put("status", "pending")
        state.put("phase", "planning").put("active_step_id", JSONObject.NULL)
            .put("pending", JSONObject.NULL).put("remind_due", false).put("issue", "用户要求重新规划")
        breakSequence()
        touch(nowMs)
        return listOf("好的，重新安排接下来的步骤。")
    }

    @Synchronized
    fun answerClarification(text: String, nowMs: Long = 0): List<String> {
        if (state.optJSONObject("pending")?.optString("kind") != "clarify") return emptyList()
        val answer = userText(text, 300)
        val goal = state.getJSONObject("goal").getString("text")
        require(answer != goal) { "请回答刚才的问题" }
        // Bound a series of clarifications; never silently discard the user's original goal.
        val combined = "$goal（补充：$answer）"
        require(combined.length <= 900) { "补充内容过长，请重新设定目标" }
        state.getJSONObject("goal").put("text", combined)
        state.put("phase", "decomposing").put("pending", JSONObject.NULL).put("desires", JSONArray())
            .put("active_desire_id", JSONObject.NULL).put("active_step_id", JSONObject.NULL)
        breakSequence()
        succeeded()
        touch(nowMs)
        return listOf("好的，我按你的补充重新安排。")
    }

    @Synchronized
    fun repeat(): List<String> {
        if (!state.isNull("paused_reason")) return emptyList()
        if (state.optJSONObject("pending") != null) return pendingSpeech()
        return activeStep()?.optString("spoken_hint")?.takeIf { it.isNotBlank() }?.let { listOf(it) } ?: emptyList()
    }

    /** User reports affect later reasoning, but never count as camera evidence or finish a step. */
    @Synchronized
    fun addUserFeedback(text: String, nowMs: Long = 0) {
        require(!state.isNull("goal") && state.optString("phase") !in setOf("idle", "stopped", "complete")) { "请先设置或继续任务" }
        val value = userText(text, 500)
        val previous = state.optJSONArray("user_feedback") ?: JSONArray()
        state.put("user_feedback", JSONArray(previous.objects().takeLast(11)).put(JSONObject()
            .put("text", value).put("source", "user_report_unverified").put("at_ms", nowMs)
            .put("task_instance_id", state.getString("task_instance_id"))))
        withdrawStepConfirmation()
        breakSequence()
        touch(nowMs)
    }

    @Synchronized
    fun stop(nowMs: Long = 0): List<String> {
        state.put("phase", "stopped").put("pending", JSONObject.NULL).put("remind_due", false)
        invalidateCandidatePlan()
        touch(nowMs)
        return listOf("任务已停止。")
    }

    @Synchronized
    fun presence(worn: Boolean?, nowMs: Long = 0, folded: Boolean = false): List<String> {
        val before = state.optString("presence")
        val wasFolded = state.optBoolean("folded")
        val presence = when (worn) { true -> "ok"; false -> "removed"; null -> if (before == "removed") "removed" else "unknown" }
        if (before == presence && wasFolded == folded) return heartbeat(nowMs)
        state.put("presence", presence).put("folded", folded).put("updated_at_ms", nowMs)
        // Wearing is telemetry, not a task revision: removal must not discard an in-flight observation.
        if (wasFolded == folded) return emptyList()
        invalidateCandidatePlan()
        val userPaused = state.optString("paused_reason") in listOf("user", "idle")
        if (folded) {
            withdrawStepConfirmation()
            breakSequence()
            if (!userPaused) state.put("paused_reason", "folded")
        } else if (!userPaused) {
            state.put("paused_reason", JSONObject.NULL)
            state.put("last_progress_ms", nowMs)
            if (!state.isNull("failing_since_ms")) state.put("failing_since_ms", nowMs)
            resetConfirmationTimer(nowMs)
        }
        touch(nowMs)
        return if (wasFolded && !folded && !userPaused) repeat() else emptyList()
    }

    /** Camera/session/configuration gaps invalidate every prior judgement and outstanding model request. */
    @Synchronized
    fun gap(nowMs: Long = 0) {
        withdrawStepConfirmation()
        breakSequence()
        touch(nowMs, progress = false)
    }

    /** Cancel transport work for a user gesture without erasing the evidence behind an open confirmation. */
    @Synchronized
    fun cancelRequest(nowMs: Long = 0) {
        touch(nowMs, progress = false)
    }

    /** Called by the service about once a second; step-confirmation observation retains the idle limit. */
    @Synchronized
    fun heartbeat(nowMs: Long): List<String> {
        val pending = state.optJSONObject("pending")
        if (pending != null && pending.optString("kind") != "clarify") {
            if (!state.isNull("paused_reason")) {
                if (pending.isNull("held_since_ms")) pending.put("held_since_ms", nowMs)
                return emptyList()
            }
            if (!pending.isNull("held_since_ms")) {
                pending.put("held_ms", pending.optLong("held_ms") + (nowMs - pending.optLong("held_since_ms")).coerceAtLeast(0))
                pending.put("held_since_ms", JSONObject.NULL)
            }
            if (!pending.optBoolean("reminded") && nowMs - pending.optLong("prompted_ms") - pending.optLong("held_ms") >= 60_000) {
                pending.put("reminded", true)
                touch(nowMs, progress = false)
                return listOf("还在等你确认。${pendingSpeech().firstOrNull().orEmpty()}")
            }
        }
        if (state.optString("phase") in listOf("executing", "awaiting_step_confirmation") && state.isNull("paused_reason") &&
            nowMs - state.optLong("last_progress_ms") >= 600_000) {
            state.put("paused_reason", "idle").put("issue", "很久没有进展，已暂停观察")
            withdrawStepConfirmation()
            breakSequence()
            touch(nowMs)
            return listOf("很久没有进展，我先暂停观察，需要时说继续。")
        }
        return emptyList()
    }

    @Synchronized
    fun needsVisualInput(): Boolean = state.optString("phase") in listOf("decomposing", "planning", "executing", "awaiting_step_confirmation") &&
        state.isNull("paused_reason") && state.isNull("error") && !state.optBoolean("folded")

    @Synchronized
    fun nextRequest(nowMs: Long, frameIds: List<String> = emptyList()): EngineRequest? {
        if (inFlight != null || !needsVisualInput() || nowMs < maxOf(state.optLong("retry_at_ms"), state.optLong("rate_limit_until_ms"))) return null
        if (frameIds.isEmpty()) return null
        require(frameIds.size <= 12 && frameIds.distinct().size == frameIds.size && frameIds.all { it.matches(Regex("[A-Za-z0-9_.-]{1,64}")) }) {
            "每次请求需要1到12个不同的画面ID"
        }
        val stage = when (state.optString("phase")) {
            "decomposing" -> "decompose"
            "planning" -> "plan"
            "executing" -> if (state.optBoolean("remind_due")) "remind" else "observe"
            "awaiting_step_confirmation" -> "observe"
            else -> return null
        }
        val payload = payload(stage, nowMs)
        val req = EngineRequest(stage, payload, state.getLong("revision"), frameIds.toList(), ++requestSequence)
        inFlight = req
        return req
    }

    @Synchronized
    fun accept(request: EngineRequest, response: JSONObject, nowMs: Long): List<String> {
        if (!current(request)) return emptyList()
        val answer = try { validate(request, response) } catch (_: Exception) {
            return fail(request, 422, "模型返回格式不合格", nowMs)
        }
        inFlight = null
        val speech = mutableListOf<String>()
        when (request.stage) {
            "decompose" -> applyDecomposition(answer, nowMs, speech)
            "plan" -> installPlan(answer, nowMs, speech)
            "observe" -> applyObservation(answer, nowMs, speech)
            "remind" -> {
                state.put("remind_due", false)
                speech += answer.getString("text")
                val desire = activeDesire()
                if (answer.getBoolean("suggest_replan") && desire != null && desire.optInt("replans") < 3) {
                    desire.put("replans", desire.optInt("replans") + 1)
                    activeStep()?.put("status", "pending")
                    state.put("phase", "planning").put("active_step_id", JSONObject.NULL)
                        .put("issue", "原计划不适用，重新规划剩余步骤")
                }
            }
        }
        succeeded()
        touch(nowMs, progress = false)
        return speech
    }

    @Synchronized
    fun fail(request: EngineRequest, status: Int?, message: String, nowMs: Long): List<String> {
        if (!current(request)) return emptyList()
        inFlight = null
        val transient = status == null || status == 0 || status == 408 || status == 429 || status >= 500
        if (state.isNull("failing_since_ms")) state.put("failing_since_ms", nowMs)
        val countKey = if (transient) "transient_failures" else "failures"
        val count = state.optInt(countKey) + 1
        state.put(countKey, count)
        // Do not persist arbitrary network error text: it can contain authorization headers or provider responses.
        val safeMessage = when (status) {
            401, 403 -> "模型接口凭据无效或无权限，请检查设置"
            429 -> "模型接口限流，60秒后重试"
            422 -> "模型返回格式不合格"
            null, 0, 408 -> "网络连接失败或模型响应超时"
            else -> if (status >= 500) "模型服务暂时不可用" else "模型请求失败（HTTP $status）"
        }
        state.put("last_failure", JSONObject().put("stage", request.stage).put("status", status ?: JSONObject.NULL)
            .put("message", safeMessage).put("at_ms", nowMs))
        val delay = when {
            status == 429 -> 60_000L
            status == 401 || status == 403 -> 60_000L
            else -> (5_000L * (1L shl (count - 1).coerceIn(0, 4))).coerceAtMost(60_000)
        }
        state.put("retry_at_ms", nowMs + delay)
        if (status == 429) state.put("rate_limit_until_ms", maxOf(state.optLong("rate_limit_until_ms"), nowMs + 60_000))
        if ((!transient && count >= 5) || (transient && nowMs - state.optLong("failing_since_ms") >= 600_000))
            state.put("error", safeMessage)
        touch(nowMs, progress = false)
        return if (count == 1 || !state.isNull("error")) listOf(safeMessage) else emptyList()
    }

    @Synchronized
    fun snapshot(nowMs: Long = 0): JSONObject = clone(state).put("saved_at_ms", nowMs)

    /** Restoring never silently restarts camera/model requests. The user must explicitly resume. */
    @Synchronized
    fun restore(snapshot: JSONObject, nowMs: Long = 0) {
        require(snapshot.optString("schema") == SCHEMA) { "任务记录版本不受支持" }
        val restored = clone(snapshot)
        if (!restored.has("user_feedback")) restored.put("user_feedback", JSONArray())
        val feedback = restored.getJSONArray("user_feedback")
        require(feedback.length() <= 12) { "对话记录过长" }
        feedback.objects().forEach {
            text(it, "text", 500)
            require(it.getString("source") == "user_report_unverified" && it.get("at_ms") is Number)
            require(it.getString("task_instance_id") == restored.optString("task_instance_id"))
        }
        if (!restored.has("folded")) restored.put("folded", false)
        require(restored.get("folded") is Boolean) { "镜腿状态记录损坏" }
        if (!restored.has("task_instance_id")) restored.put("task_instance_id", UUID.randomUUID().toString())
        text(restored, "task_instance_id", 80)
        require(restored.optString("phase") in PHASES) { "任务记录损坏" }
        require(restored.optJSONArray("desires") != null && restored.optJSONObject("memory") != null) { "任务记录损坏" }
        val memory = restored.getJSONObject("memory")
        require(memory.optJSONObject("objects") != null && memory.optJSONArray("activities") != null) { "记忆记录损坏" }
        require(memory.getJSONObject("objects").length() <= 200 && memory.getJSONArray("activities").length() <= 200) { "记忆记录过大" }
        memory.getJSONObject("objects").let { objects ->
            objects.keys().asSequence().forEach { name ->
                val item = objects.getJSONObject(name)
                require(item.get("state") is String && item.get("name") is String && item.get("last_seen_ms") is Number)
                if (item.has("task_instance_id")) text(item, "task_instance_id", 80)
                optionalText(item, "container_id", 40)
                if (item.has("scope_uncertain")) require(item.get("scope_uncertain") is Boolean)
                require(item.getJSONArray("places").length() <= 8)
                item.getJSONArray("places").objects().forEach { require(it.get("state") is String) }
            }
        }
        memory.getJSONArray("activities").objects().forEach {
            require(it.get("action") is String && it.get("until_ms") is Number && it.get("from_ms") is Number)
            if (it.has("task_instance_id")) text(it, "task_instance_id", 80)
        }
        memory.optJSONObject("hands")?.let {
            require(it.get("text") is String && it.get("at_ms") is Number)
            if (it.has("task_instance_id")) text(it, "task_instance_id", 80)
        }
        listOf("revision", "last_progress_ms", "retry_at_ms", "failures", "transient_failures").forEach {
            require(restored.get(it) is Number) { "任务计数记录损坏" }
        }
        require(restored.optJSONObject("step_streaks") != null && restored.optJSONObject("desire_streaks") != null &&
            restored.optJSONObject("switch_votes") != null && restored.optJSONArray("belief_history") != null) { "任务计数记录损坏" }
        require(restored.optString("mode") in listOf("auto", "confirm")) { "任务模式损坏" }
        require(restored.getJSONArray("belief_history").length() <= 5) { "任务历史过长" }
        restored.getJSONArray("belief_history").objects().forEach { belief ->
            require(belief.get("at_ms") is Number)
            listOf("scene", "hands", "action", "changes").forEach { require(belief.get(it) is String) }
        }
        restored.optJSONObject("pending")?.let {
            require(it.optString("kind") in listOf("clarify", "confirm_decomposition", "confirm_step_done")) { "确认记录损坏" }
            require(it.get("text") is String) { "确认文本损坏" }
        }
        if (restored.optString("phase") !in listOf("idle", "stopped", "complete"))
            require(restored.optJSONObject("goal")?.opt("text") is String) { "目标记录损坏" }
        val desires = restored.getJSONArray("desires").objects()
        require(desires.size <= 24 && desires.map { it.getString("desire_id") }.distinct().size == desires.size)
        for (desire in desires) {
            require(desire.getString("status") in listOf("pending", "active", "done"))
            require(desire.get("title") is String && desire.get("success_criteria") is String)
            desire.optJSONObject("plan")?.let { plan ->
                val steps = plan.getJSONArray("steps").objects()
                require(steps.size in 1..256 && steps.map { it.getString("step_id") }.distinct().size == steps.size)
                steps.forEach {
                    require(it.getString("status") in listOf("pending", "active", "done", "skipped"))
                    require(it.get("instruction") is String && it.get("spoken_hint") is String)
                    stringList(it.getJSONArray("required_objects"), 10, 40)
                    val checks = it.getJSONArray("checkpoints").objects()
                    require(checks.size in 1..3 && checks.map { it.getString("checkpoint_id") }.distinct().size == checks.size)
                    checks.forEach { check -> require(check.get("question") is String) }
                }
            }
        }
        val activeDid = nullableString(restored, "active_desire_id")
        if (activeDid != null) require(desires.any { it.getString("desire_id") == activeDid }) { "任务记录子目标损坏" }
        val active = desires.firstOrNull { it.optString("desire_id") == activeDid }
        val activeSid = nullableString(restored, "active_step_id")
        if (activeSid != null) require(active?.optJSONObject("plan")?.optJSONArray("steps")?.objects()?.any { it.optString("step_id") == activeSid } == true) { "任务记录步骤损坏" }
        if (restored.optString("phase") in listOf("executing", "awaiting_step_confirmation")) {
            require(activeSid != null && active?.optString("status") == "active") { "任务记录缺少当前步骤" }
            require(active!!.getJSONObject("plan").getJSONArray("steps").objects()
                .any { it.optString("step_id") == activeSid && it.optString("status") == "active" }) { "当前步骤状态损坏" }
        }
        if (restored.optString("phase") == "planning") require(activeDid != null && active?.optString("status") == "active") { "任务记录缺少当前子目标" }
        if (restored.optString("phase") in listOf("awaiting_clarification", "awaiting_plan_confirmation", "awaiting_step_confirmation"))
            require(restored.optJSONObject("pending")?.opt("kind") is String) { "确认记录损坏" }
        state = restored
        if (state.optString("phase") == "awaiting_step_confirmation")
            state.put("phase", "executing").put("pending", JSONObject.NULL)
        if (state.optString("phase") !in listOf("idle", "stopped", "complete")) state.put("paused_reason", "user")
        state.put("presence", "unknown").put("retry_at_ms", 0).put("last_progress_ms", nowMs)
            .put("issue", "已恢复上次任务，请明确继续")
        invalidateCandidatePlan()
        // A monotonic clock may have restarted with the device. Retain at most one fresh cooldown on restore.
        state.put("rate_limit_until_ms", if (state.optLong("rate_limit_until_ms") > snapshot.optLong("saved_at_ms")) nowMs + 60_000 else 0)
        state.getJSONObject("memory").getJSONObject("objects").keys().asSequence().toList().forEach {
            state.getJSONObject("memory").getJSONObject("objects").getJSONObject(it).put("restored", true)
        }
        succeeded()
        breakSequence()
        resetConfirmationTimer(nowMs)
        touch(nowMs, progress = false)
    }

    private fun applyDecomposition(answer: JSONObject, now: Long, speech: MutableList<String>) {
        state.getJSONObject("goal").put("summary", answer.getString("goal_summary")).put("horizon", answer.getString("horizon"))
        state.put("desires", JSONArray().apply {
            answer.getJSONArray("desires").objects().forEach { put(clone(it).put("status", "pending").put("plan", JSONObject.NULL).put("replans", 0)) }
        }).put("active_desire_id", JSONObject.NULL).put("active_step_id", JSONObject.NULL)
        state.put("candidate_first_plan", answer.opt("first_plan") ?: JSONObject.NULL)
            .put("candidate_first_plan_at_ms", if (answer.optJSONObject("first_plan") != null) now else JSONObject.NULL)
        val question = nullableString(answer, "clarifying_question")
        if (question != null) {
            state.put("phase", "awaiting_clarification").put("pending", JSONObject().put("kind", "clarify").put("text", question))
            speech += question
        } else if (state.optString("mode") == "confirm") {
            state.put("phase", "awaiting_plan_confirmation")
            val titles = desires().joinToString("，") { it.getString("title") }
            newPending("confirm_decomposition", now).put("text", "目标分为：$titles。请确认。")
            speech += pendingSpeech()
        } else {
            activateNext(now, speech)
            applyFirstPlan(now, speech)
        }
    }

    private fun applyFirstPlan(now: Long, speech: MutableList<String>) {
        val first = state.optJSONObject("candidate_first_plan")
        val createdAt = state.optLong("candidate_first_plan_at_ms", -1)
        // A bounded reuse window avoids replaying a plan after the wearer waited or changed the scene.
        // This is a conservative time guard, not a guarantee that the visual context is still unchanged.
        val fresh = createdAt >= 0 && now >= createdAt && now - createdAt < FIRST_PLAN_MAX_AGE_MS
        invalidateCandidatePlan()
        if (first != null && fresh && state.optString("phase") == "planning" && activeDesire()?.optString("desire_id") == first.optString("desire_id"))
            installPlan(first, now, speech)
    }

    private fun installPlan(answer: JSONObject, now: Long, speech: MutableList<String>) {
        val desire = activeDesire() ?: return
        val previous = desire.optJSONObject("plan")
        val kept = previous?.optJSONArray("steps")?.objects()?.filter { it.optString("status") in listOf("done", "skipped") } ?: emptyList()
        val steps = JSONArray()
        kept.forEach { steps.put(clone(it)) }
        answer.getJSONArray("steps").objects().forEach {
            steps.put(clone(it).put("status", "pending").put("done_by", JSONObject.NULL)
                .put("done_at_ms", JSONObject.NULL).put("needs_fresh_incomplete", false))
        }
        steps.objects().forEachIndexed { index, step -> step.put("step_id", "s${index + 1}") }
        desire.put("plan", JSONObject().put("steps", steps).put("assumptions", answer.getJSONArray("assumptions"))
            .put("revision", (previous?.optInt("revision") ?: 0) + 1).put("planned_at_ms", now))
        state.put("pending", JSONObject.NULL)
        startStep(steps.getJSONObject(kept.size), now, speech)
    }

    private fun applyObservation(answer: JSONObject, now: Long, speech: MutableList<String>) {
        val waiting = state.optString("phase") == "awaiting_step_confirmation"
        val belief = clone(answer.getJSONObject("belief")).put("at_ms", now).put("confidence", answer.getDouble("confidence"))
            .put("uncertain", answer.getBoolean("uncertain"))
        state.put("belief", belief)
        val history = state.getJSONArray("belief_history").objects().toMutableList()
        history += JSONObject().apply { listOf("scene", "hands", "action", "changes", "at_ms").forEach { put(it, belief.get(it)) } }
        state.put("belief_history", JSONArray(history.takeLast(5)))
        if (answer.getBoolean("uncertain") || answer.getDouble("confidence") < 0.6) {
            if (waiting) return // An occluded view cannot refute an earlier observation or renew its approval.
            val count = state.optInt("unclear_count") + 1
            state.put("unclear_count", count)
            if (count >= 3 && !state.optBoolean("camera_hint_given")) {
                state.put("camera_hint_given", true)
                speech += "我看不太清，请把手和要用的东西放在眼前。"
            }
            return
        }
        remember(belief, now, answer.getJSONArray("evidence_frame_ids").getString(answer.getJSONArray("evidence_frame_ids").length() - 1))
        state.put("unclear_count", 0)
        if (waiting) {
            val check = answer.getJSONObject("step_check")
            val contradicted = check.getString("overall_status") in listOf("NOT_STARTED", "PROCESSING") ||
                check.getJSONArray("checkpoints").objects().any { it.getString("status") in listOf("NOT_STARTED", "IN_PROGRESS") } ||
                !check.isNull("deviation")
            if (contradicted) {
                withdrawStepConfirmation()
                breakSequence()
                state.put("issue", "新观察与完成候选不一致")
                speech += "看到状态变化，原完成提示已撤回，请继续操作。"
            }
            return // Refresh memory without changing the steps the user was asked to approve.
        }
        val step = activeStep() ?: return
        val check = answer.optJSONObject("step_check") ?: return
        val seen = check.getJSONArray("checkpoints").objects().associate { it.getString("checkpoint_id") to it.getString("status") }
        val overall = check.getString("overall_status")
        val unfinished = overall in listOf("NOT_STARTED", "PROCESSING") || seen.values.any { it in listOf("NOT_STARTED", "IN_PROGRESS") }
        val reported = answer.getJSONObject("progress").getJSONArray("completed_step_ids").strings().toMutableSet()
        val sid = step.getString("step_id")
        if (unfinished) reported.remove(sid)
        else if (step.getJSONArray("checkpoints").objects().all { seen[it.getString("checkpoint_id")] == "COMPLETED" }) reported += sid
        val doneDesires = answer.getJSONObject("progress").getJSONArray("completed_desire_ids").strings().toSet()
        val working = nullableString(answer.getJSONObject("progress"), "working_on_desire_id")
        if (overall == "UNKNOWN" && reported.isEmpty() && doneDesires.isEmpty() && working == null) return
        if (reported.isNotEmpty() || doneDesires.isNotEmpty() || working != null || overall == "PROCESSING") state.put("last_progress_ms", now)
        val newly = mutableSetOf<String>()
        val streaks = state.getJSONObject("step_streaks")
        for (candidate in steps().filter { isOpen(it) }) {
            val id = candidate.getString("step_id")
            if (id in reported && !candidate.optBoolean("needs_fresh_incomplete")) {
                val count = streaks.optInt(id) + 1
                streaks.put(id, count)
                if (count >= 2) newly += id
            } else if (id !in reported) {
                streaks.put(id, 0)
                if (id == sid && unfinished) candidate.put("needs_fresh_incomplete", false)
            }
        }
        val desireStreaks = state.getJSONObject("desire_streaks")
        for (desire in desires().filter { isOpen(it) }) {
            val id = desire.getString("desire_id")
            val latched = desire.optJSONObject("plan")?.optJSONArray("steps")?.objects()?.any { isOpen(it) && it.optBoolean("needs_fresh_incomplete") } ?: false
            desireStreaks.put(id, if (id in doneDesires && !latched) desireStreaks.optInt(id) + 1 else 0)
        }
        val currentDid = state.getString("active_desire_id")
        val elsewhere = working?.takeIf { it != currentDid && it !in doneDesires }
        val votes = state.getJSONObject("switch_votes")
        if (working == currentDid) state.put("switch_votes", JSONObject().put("desire_id", JSONObject.NULL).put("count", 0))
        else if (elsewhere != null) state.put("switch_votes", JSONObject().put("desire_id", elsewhere)
            .put("count", if (votes.optString("desire_id") == elsewhere) votes.optInt("count") + 1 else 1))
        if (state.optString("mode") == "auto") {
            for (other in desires().filter { it.optString("status") == "pending" && desireStreaks.optInt(it.getString("desire_id")) >= 2 }) {
                other.put("status", "done").put("completed_at_ms", now)
                other.optJSONObject("plan")?.optJSONArray("steps")?.objects()?.filter { isOpen(it) }?.forEach { mark(it, "done", "inferred", now) }
                speech += "${other.getString("title")}看起来已经完成了。"
            }
        }
        val closing = desireStreaks.optInt(currentDid) >= 2
        if (closing || newly.isNotEmpty()) {
            val open = steps().filter { isOpen(it) }
            val furthest = steps().indexOfLast { it.getString("step_id") in newly }
            val inferred = if (closing) open.map { it.getString("step_id") }.toSet() - newly else
                steps().take(furthest.coerceAtLeast(0)).filter { isOpen(it) && !it.optBoolean("needs_fresh_incomplete") &&
                    !(it.getString("step_id") == sid && unfinished) }.map { it.getString("step_id") }.toSet() - newly
            if (state.optString("mode") == "confirm") {
                val ids = if (closing) open.map { it.getString("step_id") }.toSet() else newly + inferred
                state.put("phase", "awaiting_step_confirmation")
                newPending("confirm_step_done", now).put("step_ids", JSONArray(ids.toList())).put("inferred_ids", JSONArray(inferred.toList()))
                    .put("prefer", elsewhere ?: JSONObject.NULL).put("close_desire", closing)
                    .put("text", if (closing) "这个子目标好像做完了，请确认。" else "这一步好像做完了，请确认。")
                speech += pendingSpeech()
            } else {
                val savedStreaks = clone(streaks)
                finishSteps(newly + inferred, inferred, now, speech, "model", elsewhere)
                carryStreaks(savedStreaks, currentDid)
            }
            return
        }
        val switch = state.getJSONObject("switch_votes")
        val target = desires().firstOrNull { it.optString("desire_id") == switch.optString("desire_id") && it.optString("status") == "pending" }
        if (switch.optInt("count") >= 2 && target != null && target.getString("desire_id") !in doneDesires) {
            activeStep()?.put("status", "pending")
            activeDesire()?.put("status", "pending")
            target.put("status", "active")
            state.put("active_desire_id", target.getString("desire_id")).put("active_step_id", JSONObject.NULL)
                .put("phase", "planning").put("issue", "用户正在做另一个子目标，跟随当前进展")
            resetTracking()
            return
        }
        val deviation = nullableString(check, "deviation")
        val deviationCount = if (deviation != null) state.optInt("deviation_count") + 1 else 0
        val stalled = reported.isEmpty() && doneDesires.isEmpty() && working == null && overall in listOf("NOT_STARTED", "UNRELATED")
        val stallCount = if (stalled) state.optInt("stall_count") + 1 else 0
        state.put("deviation_count", deviationCount).put("stall_count", stallCount)
        val reminders = state.optInt("remind_count")
        if (reminders < 3 && (deviationCount >= 2 || stallCount >= 4 * (1 shl reminders))) {
            state.put("remind_due", true).put("remind_count", reminders + 1)
                .put("issue", deviation ?: "很久没有进展").put("deviation_count", 0).put("stall_count", 0)
        }
    }

    private fun finishCurrent(now: Long, status: String): List<String> {
        if (!mayInteract()) return emptyList()
        val step = activeStep() ?: return emptyList()
        mark(step, status, "user", now)
        val speech = mutableListOf<String>()
        advance(now, speech)
        touch(now)
        return speech
    }

    private fun finishSteps(ids: Set<String>, inferred: Set<String>, now: Long, speech: MutableList<String>, by: String, prefer: String?) {
        steps().filter { isOpen(it) && it.getString("step_id") in ids }.forEach {
            mark(it, "done", if (it.getString("step_id") in inferred) "inferred" else by, now)
        }
        state.put("pending", JSONObject.NULL)
        if (activeStep()?.let { isOpen(it) } != true) advance(now, speech, prefer)
        else state.put("phase", "executing")
    }

    private fun advance(now: Long, speech: MutableList<String>, prefer: String? = null) {
        state.put("pending", JSONObject.NULL).put("active_step_id", JSONObject.NULL)
        val next = steps().firstOrNull { isOpen(it) }
        if (next != null) startStep(next, now, speech)
        else {
            activeDesire()?.let { it.put("status", "done").put("completed_at_ms", now); speech += "${it.getString("title")}完成了。" }
            activateNext(now, speech, prefer)
        }
    }

    private fun activateNext(now: Long, speech: MutableList<String>, prefer: String? = null) {
        val waiting = desires().filter { it.optString("status") == "pending" }
        val next = waiting.firstOrNull { it.optString("desire_id") == prefer } ?: waiting.firstOrNull()
        state.put("active_step_id", JSONObject.NULL).put("pending", JSONObject.NULL)
            .put("switch_votes", JSONObject().put("desire_id", JSONObject.NULL).put("count", 0))
        resetTracking()
        if (next == null) {
            state.put("phase", "complete").put("active_desire_id", JSONObject.NULL)
            speech += "目标已经全部完成。"
        } else {
            next.put("status", "active")
            state.put("active_desire_id", next.getString("desire_id")).put("phase", "planning")
                .put("last_progress_ms", now)
        }
    }

    private fun startStep(step: JSONObject, now: Long, speech: MutableList<String>) {
        step.put("status", "active").put("started_at_ms", now)
        state.put("active_step_id", step.getString("step_id")).put("phase", "executing").put("last_progress_ms", now)
        resetTracking()
        speech += step.getString("spoken_hint")
    }

    private fun payload(stage: String, now: Long): JSONObject {
        val history = JSONArray(state.getJSONArray("belief_history").objects().takeLast(3).map { clone(it) })
        val goal = state.getJSONObject("goal")
        val p = JSONObject().put("goal", nullableString(goal, "summary") ?: goal.getString("text"))
            .put("task_instance_id", state.getString("task_instance_id"))
            .put("user_feedback", clone(state).optJSONArray("user_feedback") ?: JSONArray())
            .put("belief_history", history)
        val desire = activeDesire()
        when (stage) {
            "decompose" -> p.put("goal", goal.getString("text")).put("completed_desires", JSONArray())
            "plan" -> {
                p.put("desire", briefDesire(desire!!)).put("completed_desires", JSONArray(desires().filter { it.optString("status") == "done" }.map { JSONObject().put("title", it.getString("title")) }))
                    .put("previous_plan", desire.opt("plan") ?: JSONObject.NULL).put("issue", state.opt("issue") ?: JSONObject.NULL)
            }
            "observe" -> p.put("desire", desire?.let { briefDesire(it) } ?: JSONObject.NULL)
                .put("active_step", activeStep()?.let { briefStep(it) } ?: JSONObject.NULL)
                .put("steps", JSONArray(steps().map { JSONObject().put("step_id", it.getString("step_id")).put("instruction", it.getString("instruction")).put("status", it.getString("status")) }))
                .put("desires", JSONArray(desires().filter { isOpen(it) }.map { briefDesire(it).put("status", it.getString("status")) }))
            "remind" -> p.put("active_step", activeStep()?.let { briefStep(it) } ?: JSONObject.NULL)
                .put("issue", nullableString(state, "issue") ?: "很久没有进展")
        }
        if (stage != "remind") p.put("memory", memoryContext(now))
        return p
    }

    @Synchronized
    fun clarifyManualIdentity(request: EngineRequest, question: String, nowMs: Long): List<String> {
        require(question.isNotBlank() && question.length <= 500)
        if (!current(request) || request.stage != "decompose") return emptyList()
        inFlight = null
        state.put("phase", "awaiting_clarification").put("pending", JSONObject().put("kind", "clarify").put("text", question).put("manual_identity", true))
        succeeded()
        touch(nowMs)
        return listOf(question)
    }

    private fun validate(request: EngineRequest, raw: JSONObject): JSONObject {
        val a = clone(raw)
        request.payload.optJSONObject("manual_context")?.let { context ->
            val references = a.getJSONArray("source_sections").strings()
            val known = context.getJSONArray("passages").objects().map { it.getString("citation_id") }.toSet()
            require(references.size <= 4 && references.distinct().size == references.size && references.all { it in known })
            if (request.stage != "observe" && context.optString("transport") !in setOf("mcp_streamable_http", "not_requested")) require(references.isNotEmpty())
            a.remove("source_sections")
        }
        fun evidence() {
            val ids = a.getJSONArray("evidence_frame_ids").strings()
            require(ids.isNotEmpty() && ids.distinct().size == ids.size && ids.all { it in request.frameIds })
        }
        if (request.stage != "remind") {
            val value = a.get("confidence")
            require(value is Number && value.toDouble().isFinite() && value.toDouble() in 0.0..1.0)
        }
        evidence()
        when (request.stage) {
            "decompose" -> {
                keys(a, setOf("goal_summary", "horizon", "desires", "clarifying_question", "confidence", "evidence_frame_ids"), setOf("first_plan"))
                text(a, "goal_summary", 160)
                require(a.getString("horizon") in listOf("short", "long"))
                optionalText(a, "clarifying_question", 160)
                val ds = a.getJSONArray("desires").objects()
                require(ds.size in 1..6)
                sequenceIds(ds, "desire_id", "d")
                ds.forEach { keys(it, setOf("desire_id", "title", "success_criteria")); text(it, "title", 80); text(it, "success_criteria", 240) }
                val first = a.optJSONObject("first_plan")
                a.put("first_plan", JSONObject.NULL)
                if (first != null && a.isNull("clarifying_question")) {
                    try {
                        keys(first, setOf("steps"), setOf("assumptions"))
                        val plan = clone(first).put("desire_id", ds.first().getString("desire_id"))
                            .put("assumptions", first.optJSONArray("assumptions") ?: JSONArray())
                            .put("confidence", a.getDouble("confidence")).put("evidence_frame_ids", a.getJSONArray("evidence_frame_ids"))
                        validatePlan(plan, ds.first().getString("desire_id"))
                        a.put("first_plan", plan)
                    } catch (_: Exception) { /* A malformed optional first plan falls back to the normal planning stage. */ }
                }
            }
            "plan" -> validatePlan(a, request.payload.getJSONObject("desire").getString("desire_id"))
            "observe" -> {
                keys(a, setOf("belief", "uncertain", "confidence", "step_check", "progress", "evidence_frame_ids"))
                require(a.get("uncertain") is Boolean)
                val belief = a.getJSONObject("belief")
                keys(belief, setOf("scene", "objects", "hands", "action", "changes"))
                listOf("scene", "hands", "action", "changes").forEach { text(belief, it, 160, true) }
                val os = belief.getJSONArray("objects").objects()
                require(os.size <= 16)
                os.forEach {
                    keys(it, setOf("name", "state"), setOf("container_id"))
                    text(it, "name", 40); text(it, "state", 60, true); optionalText(it, "container_id", 40)
                }
                require(os.map { it.getString("name") to nullableString(it, "container_id") }.distinct().size == os.size)
                val active = request.payload.optJSONObject("active_step")
                val check = a.optJSONObject("step_check")
                if (active == null) require(a.isNull("step_check")) else {
                    require(check != null)
                    keys(check, setOf("step_id", "checkpoints", "overall_status", "deviation"))
                    require(check.getString("step_id") == active.getString("step_id"))
                    val expected = active.getJSONArray("checkpoints").objects().map { it.getString("checkpoint_id") }
                    val seen = check.getJSONArray("checkpoints").objects()
                    require(seen.size == expected.size && seen.map { it.getString("checkpoint_id") }.toSet() == expected.toSet())
                    seen.forEach { keys(it, setOf("checkpoint_id", "status")); require(it.getString("status") in CHECKPOINT_STATUSES) }
                    require(check.getString("overall_status") in OVERALL_STATUSES)
                    if (check.getString("overall_status") == "COMPLETED") require(seen.all { it.getString("status") == "COMPLETED" })
                    optionalText(check, "deviation", 160)
                }
                val progress = a.getJSONObject("progress")
                keys(progress, setOf("completed_step_ids", "completed_desire_ids", "working_on_desire_id"))
                val knownSteps = request.payload.getJSONArray("steps").objects().map { it.getString("step_id") }.toSet()
                val knownDesires = request.payload.getJSONArray("desires").objects().map { it.getString("desire_id") }.toSet()
                knownIds(progress.getJSONArray("completed_step_ids"), knownSteps)
                knownIds(progress.getJSONArray("completed_desire_ids"), knownDesires)
                nullableString(progress, "working_on_desire_id")?.let { require(it in knownDesires) }
            }
            "remind" -> {
                keys(a, setOf("text", "suggest_replan", "evidence_frame_ids"))
                text(a, "text", 80)
                require(a.get("suggest_replan") is Boolean)
            }
        }
        return a
    }

    private fun validatePlan(a: JSONObject, did: String) {
        keys(a, setOf("desire_id", "steps", "assumptions", "confidence", "evidence_frame_ids"))
        require(a.getString("desire_id") == did)
        val plan = a.getJSONArray("steps").objects()
        require(plan.size in 1..8)
        sequenceIds(plan, "step_id", "s")
        plan.forEach {
            keys(it, setOf("step_id", "instruction", "spoken_hint", "required_objects", "checkpoints"))
            text(it, "instruction", 200); text(it, "spoken_hint", 60)
            stringList(it.getJSONArray("required_objects"), 10, 40)
            val cs = it.getJSONArray("checkpoints").objects()
            require(cs.size in 1..3)
            sequenceIds(cs, "checkpoint_id", "c")
            cs.forEach { c -> keys(c, setOf("checkpoint_id", "question")); text(c, "question", 160) }
        }
        stringList(a.getJSONArray("assumptions"), 6, 160)
    }

    private fun remember(belief: JSONObject, now: Long, frameId: String) {
        val memory = state.getJSONObject("memory")
        val objects = memory.getJSONObject("objects")
        val task = state.getString("task_instance_id")
        val listed = mutableSetOf<String>()
        val seenNames = mutableSetOf<String>()
        for (o in belief.getJSONArray("objects").objects()) {
            val name = o.getString("name").trim()
            val place = o.getString("state").trim()
            val container = nullableString(o, "container_id")
            val key = JSONArray().put(task).put(container ?: JSONObject.NULL).put(name).toString()
            seenNames += name
            if (place.isNotEmpty()) listed += key
            var entry = objects.optJSONObject(key)
            if (entry == null) {
                entry = JSONObject().put("name", name).put("state", place).put("first_seen_ms", now).put("places", JSONArray())
                    .put("task_instance_id", task).put("container_id", container ?: JSONObject.NULL)
                objects.put(key, entry)
            } else if (place.isNotEmpty() && place != entry.optString("state")) {
                rememberPlace(entry, now)
                entry.put("state", place)
            }
            requireNotNull(entry).put("last_seen_ms", now).put("last_frame_id", frameId).put("restored", false)
                .put("scope_uncertain", container == null || place.isBlank())
        }
        // shortcut: no persistent object tracking; an absent same-name scope becomes uncertain until observed again.
        for (key in objects.keys().asSequence().toList()) {
            val entry = objects.getJSONObject(key)
            if (entry.optString("task_instance_id") == task && key !in listed && entry.getString("name") in seenNames)
                entry.put("scope_uncertain", true)
        }
        val hands = belief.getString("hands").trim()
        if (hands.isNotEmpty()) {
            for (key in objects.keys().asSequence().toList()) {
                val entry = objects.getJSONObject(key)
                if (entry.optString("task_instance_id") == task && key !in listed && !namedInHands(entry.getString("name"), hands) && held(entry.optString("state"))) {
                    rememberPlace(entry, now)
                    entry.put("state", UNKNOWN_HELD)
                }
            }
            memory.put("hands", JSONObject().put("text", hands).put("at_ms", now).put("task_instance_id", task))
        }
        val action = belief.getString("action").trim()
        val actions = memory.getJSONArray("activities").objects().toMutableList()
        if (action.isNotEmpty()) {
            val last = actions.lastOrNull()
            if (last != null && last.optString("task_instance_id") == task && last.optString("action") == action) last.put("until_ms", now)
            else actions += JSONObject().put("action", action).put("from_ms", now).put("until_ms", now).put("task_instance_id", task)
            memory.put("activities", JSONArray(actions.takeLast(200)))
        }
        val sorted = objects.keys().asSequence().toList().sortedBy { objects.getJSONObject(it).optLong("last_seen_ms") }
        sorted.take((sorted.size - 200).coerceAtLeast(0)).forEach { objects.remove(it) }
    }

    private fun memoryContext(now: Long): JSONObject {
        val memory = state.getJSONObject("memory")
        val task = state.getString("task_instance_id")
        val objects = memory.getJSONObject("objects").let { all -> all.keys().asSequence().map { all.getJSONObject(it) }
            .filter { it.optString("task_instance_id") == task }.toList() }
        val unknown = objects.filter { it.optString("state") == UNKNOWN_HELD }
        fun ago(at: Long) = ((now - at).coerceAtLeast(0) / 1000)
        val context = objects.sortedByDescending { it.optLong("last_seen_ms") }.take(12).map { e ->
            val stale = e.optBoolean("restored") || e.optBoolean("scope_uncertain") || ago(e.optLong("last_seen_ms")) > 120 || e.optString("state") == UNKNOWN_HELD ||
                unknown.any { e.optString("state").contains(it.optString("name")) && e.optLong("last_seen_ms") <= it.optLong("last_seen_ms") }
            JSONObject().put("name", e.getString("name")).put("state", e.getString("state")).put("seen_s_ago", ago(e.optLong("last_seen_ms")))
                .put("container_id", e.opt("container_id") ?: JSONObject.NULL).put("scope_uncertain", e.optBoolean("scope_uncertain"))
                .apply {
                    val places = e.getJSONArray("places")
                    if (places.length() > 0) put("was", places.getJSONObject(places.length() - 1).getString("state"))
                    if (stale) put("stale", true)
                }
        }
        val hands = memory.optJSONObject("hands")?.takeIf { it.optString("task_instance_id") == task }
        return JSONObject().put("task_instance_id", task).put("objects", JSONArray(context))
            .put("hands", hands?.let { JSONObject().put("text", it.getString("text")).put("seen_s_ago", ago(it.optLong("at_ms"))) } ?: JSONObject.NULL)
            .put("recent_actions", JSONArray(memory.getJSONArray("activities").objects().filter { it.optString("task_instance_id") == task }.takeLast(6).reversed().map {
                JSONObject().put("action", it.getString("action")).put("s_ago", ago(it.optLong("until_ms")))
            }))
    }

    private fun rememberPlace(entry: JSONObject, now: Long) {
        val place = entry.optString("state")
        if (place.isBlank() || place == UNKNOWN_HELD) return
        entry.put("places", JSONArray((entry.getJSONArray("places").objects() + JSONObject().put("state", place).put("until_ms", now)).takeLast(8)))
    }

    private fun namedInHands(name: String, hands: String): Boolean {
        if (HANDS_EMPTY.containsMatchIn(hands)) return false
        var start = 0
        while (start <= hands.length) {
            val at = hands.indexOf(name, start)
            if (at < 0) break
            val prefix = hands.substring(0, at).split(Regex("[，,。；;、]")).last()
            if (!NEGATED_BEFORE.containsMatchIn(prefix)) return true
            start = at + name.length.coerceAtLeast(1)
        }
        return false
    }

    private fun held(text: String) = HELD.containsMatchIn(text) && !RELEASED.containsMatchIn(text)
    private fun current(req: EngineRequest) = inFlight?.requestId == req.requestId && req.revision == state.optLong("revision")
    private fun mayInteract() = state.optString("phase") !in listOf("idle", "stopped", "complete")
    private fun desires() = state.getJSONArray("desires").objects()
    private fun activeDesire() = desires().firstOrNull { it.optString("desire_id") == state.optString("active_desire_id") }
    private fun steps() = activeDesire()?.optJSONObject("plan")?.optJSONArray("steps")?.objects() ?: emptyList()
    private fun activeStep() = steps().firstOrNull { it.optString("step_id") == state.optString("active_step_id") }
    private fun isOpen(o: JSONObject) = o.optString("status") in listOf("active", "pending")
    private fun mark(step: JSONObject, status: String, by: String, now: Long) { step.put("status", status).put("done_by", by).put("done_at_ms", now).put("needs_fresh_incomplete", false) }
    private fun briefDesire(d: JSONObject) = JSONObject().apply { listOf("desire_id", "title", "success_criteria").forEach { put(it, d.get(it)) } }
    private fun briefStep(s: JSONObject) = JSONObject().apply { listOf("step_id", "instruction", "required_objects", "checkpoints").forEach { put(it, s.get(it)) } }
    private fun succeeded() { state.put("failures", 0).put("transient_failures", 0).put("failing_since_ms", JSONObject.NULL).put("error", JSONObject.NULL).put("retry_at_ms", 0) }
    private fun touch(now: Long, progress: Boolean = true) {
        inFlight = null
        state.put("revision", state.optLong("revision") + 1).put("updated_at_ms", now)
        if (progress) state.put("last_progress_ms", now)
    }
    private fun resetTracking() {
        state.put("step_streaks", JSONObject()).put("stall_count", 0).put("deviation_count", 0).put("unclear_count", 0)
            .put("camera_hint_given", false).put("remind_count", 0).put("remind_due", false)
    }
    private fun withdrawStepConfirmation() {
        if (state.optString("phase") == "awaiting_step_confirmation")
            state.put("phase", "executing").put("pending", JSONObject.NULL)
    }
    private fun breakSequence() {
        invalidateCandidatePlan()
        state.put("step_streaks", JSONObject()).put("desire_streaks", JSONObject())
            .put("switch_votes", JSONObject().put("desire_id", JSONObject.NULL).put("count", 0))
            .put("stall_count", 0).put("deviation_count", 0)
        inFlight = null
    }
    private fun invalidateCandidatePlan() {
        state.put("candidate_first_plan", JSONObject.NULL).put("candidate_first_plan_at_ms", JSONObject.NULL)
    }
    private fun carryStreaks(old: JSONObject, did: String) {
        if (state.optString("active_desire_id") != did || state.optString("phase") != "executing") return
        state.put("step_streaks", JSONObject().apply { steps().filter { isOpen(it) }.forEach { val id = it.getString("step_id"); if (old.optInt(id) > 0) put(id, old.optInt(id)) } })
    }
    private fun newPending(kind: String, now: Long): JSONObject {
        val pending = JSONObject().put("kind", kind).put("prompted_ms", now).put("held_ms", 0)
            .put("held_since_ms", JSONObject.NULL).put("reminded", false)
        state.put("pending", pending)
        return pending
    }
    private fun pendingSpeech(): List<String> = state.optJSONObject("pending")?.let {
        listOf(it.optString("text").ifBlank { "目标拆好了，请确认。" })
    } ?: emptyList()
    private fun resetConfirmationTimer(now: Long) { state.optJSONObject("pending")?.put("prompted_ms", now)?.put("held_ms", 0)?.put("held_since_ms", JSONObject.NULL)?.put("reminded", false) }

    companion object {
        private const val SCHEMA = "satori.android-bdi/1"
        private const val FIRST_PLAN_MAX_AGE_MS = 10_000L
        private const val UNKNOWN_HELD = "位置未知（上次在手里）"
        private val PHASES = setOf("idle", "decomposing", "awaiting_clarification", "awaiting_plan_confirmation", "planning", "executing", "awaiting_step_confirmation", "complete", "stopped")
        private val CHECKPOINT_STATUSES = setOf("COMPLETED", "IN_PROGRESS", "NOT_STARTED", "UNKNOWN")
        private val OVERALL_STATUSES = setOf("COMPLETED", "PROCESSING", "NOT_STARTED", "UNRELATED", "UNKNOWN")
        private val HELD = Regex("手里|手中(?!间)|手持(?!式)|手上(?![方面边侧])|在[左右双]手(?![边旁侧上下前后附臂])|拿着|拿在|拿起|握着|握持|握住|握在|举着|举起|捏着|抓着|端着|提着|托着|扶着|按着|手握|指间|手心|单手拿")
        private val RELEASED = Regex("未知|不在手|没在手|没有在手|未在手|放下|放开|松开|脱手|离手|从手|后放|没拿|没有拿|未拿|手里没有|手中无")
        private val NEGATED_BEFORE = Regex("(?:没有|没|未)[拿握持举抓捏端提着起住在了手里中]{0,4}$")
        private val HANDS_EMPTY = Regex("(?:双|两只?|左右)手(?:都|均|全|皆)?[空闲]|空着[双两]?手|(?:(?:双|两只?)手[里上中]?|(?<![左右只])手[里上中])都?(?:什么(?:东西)?(?:都|也)没|(?:没有?|无)拿?着?(?:任何)?(?:东西|物品))")
        private fun freshState() = JSONObject().put("schema", SCHEMA).put("revision", 0).put("phase", "idle").put("mode", "confirm")
            .put("task_instance_id", UUID.randomUUID().toString())
            .put("user_feedback", JSONArray())
            .put("goal", JSONObject.NULL).put("desires", JSONArray()).put("active_desire_id", JSONObject.NULL).put("active_step_id", JSONObject.NULL)
            .put("belief", JSONObject.NULL).put("belief_history", JSONArray()).put("pending", JSONObject.NULL)
            .put("candidate_first_plan", JSONObject.NULL).put("candidate_first_plan_at_ms", JSONObject.NULL)
            .put("step_streaks", JSONObject()).put("desire_streaks", JSONObject()).put("switch_votes", JSONObject().put("desire_id", JSONObject.NULL).put("count", 0))
            .put("paused_reason", JSONObject.NULL).put("presence", "unknown").put("folded", false).put("error", JSONObject.NULL).put("issue", JSONObject.NULL)
            .put("failures", 0).put("transient_failures", 0).put("failing_since_ms", JSONObject.NULL).put("retry_at_ms", 0)
            .put("rate_limit_until_ms", 0)
            .put("stall_count", 0).put("deviation_count", 0).put("unclear_count", 0).put("remind_count", 0).put("remind_due", false)
            .put("camera_hint_given", false).put("last_progress_ms", 0).put("updated_at_ms", 0)
            .put("memory", JSONObject().put("objects", JSONObject()).put("hands", JSONObject.NULL).put("activities", JSONArray()))
        private fun clone(o: JSONObject) = JSONObject(o.toString())
        private fun JSONArray.objects() = (0 until length()).map { getJSONObject(it) }
        private fun JSONArray.strings() = (0 until length()).map { get(it).also { v -> require(v is String) } as String }
        private fun nullableString(o: JSONObject, name: String): String? = if (o.isNull(name)) null else o.get(name).also { require(it is String) } as String
        private fun keys(o: JSONObject, required: Set<String>, optional: Set<String> = emptySet()) {
            val found = o.keys().asSequence().toSet()
            require(found.containsAll(required) && found.all { it in required || it in optional })
        }
        private fun text(o: JSONObject, name: String, max: Int, empty: Boolean = false) {
            val value = o.get(name)
            require(value is String && value.length <= max && (empty || value.isNotBlank()))
            o.put(name, value.trim())
        }
        private fun optionalText(o: JSONObject, name: String, max: Int) { if (!o.isNull(name)) text(o, name, max) }
        private fun userText(value: String, limit: Int): String { require(value.isNotBlank() && value.length <= limit) { "内容需要1到${limit}个字" }; return value.trim() }
        private fun sequenceIds(objects: List<JSONObject>, key: String, prefix: String) {
            val ids = objects.map { it.get(key).also { v -> require(v is String && v.isNotBlank()) } as String }
            require(ids.distinct().size == ids.size)
            objects.forEachIndexed { i, o -> o.put(key, "$prefix${i + 1}") }
        }
        private fun knownIds(ids: JSONArray, known: Set<String>) { val list = ids.strings(); require(list.distinct().size == list.size && list.all { it in known }) }
        private fun stringList(a: JSONArray, count: Int, max: Int) { require(a.length() <= count); a.strings().forEach { require(it.isNotBlank() && it.length <= max) } }
    }
}
