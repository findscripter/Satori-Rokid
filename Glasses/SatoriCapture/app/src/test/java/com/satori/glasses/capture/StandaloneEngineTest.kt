package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Behavioral fixtures exercise the real JSON stage contract, not the implementation's helper methods. */
class StandaloneEngineTest {
    private fun step(id: String = "s1") = JSONObject("""{"step_id":"$id","instruction":"把书放好","spoken_hint":"请把书放好","required_objects":["书"],"checkpoints":[{"checkpoint_id":"c1","question":"书已放好吗"}]}""")
    private fun decomposition(twoSteps: Boolean = false) = JSONObject("""{"goal_summary":"整理桌面","horizon":"short","desires":[{"desire_id":"d1","title":"收好书","success_criteria":"书已放好"}],"clarifying_question":null,"confidence":0.9,"evidence_frame_ids":["f1"]}""")
        .put("first_plan", JSONObject().put("steps", JSONArray().put(step()).apply { if (twoSteps) put(step("s2")) }))
    private fun view(done: Boolean, confidence: Double = 0.8, uncertain: Boolean = false) = JSONObject("""{"belief":{"scene":"桌面","objects":[{"name":"书","state":"在桌上"}],"hands":"双手空闲","action":"整理书","changes":""},"uncertain":$uncertain,"confidence":$confidence,"step_check":{"step_id":"s1","checkpoints":[{"checkpoint_id":"c1","status":"${if (done) "COMPLETED" else "NOT_STARTED"}"}],"overall_status":"${if (done) "COMPLETED" else "NOT_STARTED"}","deviation":null},"progress":{"completed_step_ids":[],"completed_desire_ids":[],"working_on_desire_id":null},"evidence_frame_ids":["f1"]}""")
        .apply { if (done) getJSONObject("progress").put("completed_step_ids", JSONArray().put("s1")) }
    private fun request(e: StandaloneEngine, now: Long = 1L) = e.nextRequest(now, listOf("f1"))!!
    private fun plan(desireId: String = "d1", action: String = "把书放好") = JSONObject().put("desire_id", desireId)
        .put("steps", JSONArray().put(step().put("instruction", action).put("spoken_hint", "请$action")))
        .put("assumptions", JSONArray()).put("confidence", 0.8).put("evidence_frame_ids", JSONArray().put("f1"))
    private fun twoDesires() = decomposition().apply {
        getJSONArray("desires").put(JSONObject().put("desire_id", "d2").put("title", "擦净桌面").put("success_criteria", "桌面没有污渍"))
    }
    private fun stepView(id: String, done: Boolean = true) = view(done).apply {
        getJSONObject("step_check").put("step_id", id)
        getJSONObject("progress").put("completed_step_ids", JSONArray().apply { if (done) put(id) })
    }
    private fun ready(confirm: Boolean = false, twoSteps: Boolean = false): StandaloneEngine {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", confirm, 0)
        e.accept(request(e), decomposition(twoSteps), 2)
        if (confirm) e.confirm(3)
        return e
    }

    @Test fun autoTaskNeedsTwoReliableViewsBeforeCompletion() {
        val e = ready()
        assertEquals("executing", e.snapshot().getString("phase"))
        e.accept(request(e, 3), view(true), 4)
        assertEquals("executing", e.snapshot().getString("phase"))
        e.accept(request(e, 5), view(true), 6)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun confirmationUsesIncludedFirstPlanAndWaitsBeforeMoving() {
        val e = ready(confirm = true)
        assertEquals("executing", e.snapshot().getString("phase"))
        e.accept(request(e, 4), view(true), 5)
        val speech = e.accept(request(e, 6), view(true), 7)
        assertEquals("awaiting_step_confirmation", e.snapshot().getString("phase"))
        assertTrue(speech.any { it.contains("确认") })
        assertEquals("observe", e.nextRequest(8, listOf("f1"))!!.stage)
        e.confirm(9)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun pausedAndReplacedGoalsRejectInflightResponses() {
        val e = ready()
        val old = request(e, 3)
        e.pause(4)
        e.accept(old, view(true), 5)
        assertEquals(0, e.snapshot().getJSONObject("step_streaks").optInt("s1"))
        e.resume(6)
        val oldGoal = request(e, 7)
        e.setGoal("泡茶", true, 8)
        e.accept(oldGoal, view(true), 9)
        assertEquals("泡茶", e.snapshot().getJSONObject("goal").getString("text"))
        assertEquals("decomposing", e.snapshot().getString("phase"))
    }

    @Test fun foldingBreaksCompletionStreakAndUserPauseSurvivesUnfolding() {
        val e = ready()
        e.accept(request(e, 3), view(true), 4)
        e.presence(false, 5, folded = true)
        assertNull(e.nextRequest(6, listOf("f1")))
        e.presence(true, 7)
        e.accept(request(e, 8), view(true), 9)
        assertEquals("executing", e.snapshot().getString("phase"))
        e.pause(10)
        e.presence(false, 11, folded = true)
        e.presence(true, 12)
        assertEquals("user", e.snapshot().getString("paused_reason"))
        assertNull(e.nextRequest(13, listOf("f1")))
    }

    @Test fun redoRequiresSeeingUnfinishedBeforeCompletingAgain() {
        val e = ready()
        e.redo(3)
        e.accept(request(e, 4), view(true), 5)
        e.accept(request(e, 6), view(true), 7)
        assertEquals("executing", e.snapshot().getString("phase"))
        e.accept(request(e, 8), view(false), 9)
        e.accept(request(e, 10), view(true), 11)
        e.accept(request(e, 12), view(true), 13)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun checkpointContradictionOverridesClaimedCompletion() {
        val e = ready()
        val contradiction = view(false).apply { getJSONObject("progress").put("completed_step_ids", JSONArray().put("s1")) }
        repeat(3) { e.accept(request(e, (it * 2 + 3).toLong()), contradiction, (it * 2 + 4).toLong()) }
        assertEquals("executing", e.snapshot().getString("phase"))
    }

    @Test fun invalidEvidenceOrForeignStepCannotAdvanceState() {
        val e = ready()
        val invalid = view(true).put("evidence_frame_ids", JSONArray().put("not-sent"))
        e.accept(request(e, 3), invalid, 4)
        assertEquals("executing", e.snapshot().getString("phase"))
        assertEquals(1, e.snapshot().getInt("failures"))
    }

    @Test fun restoredTaskIsPausedAndOldCompletionProposalIsDiscarded() {
        val e = ready(confirm = true)
        e.accept(request(e, 4), view(true), 5)
        e.accept(request(e, 6), view(true), 7)
        val restored = StandaloneEngine()
        restored.restore(e.snapshot(7), 100)
        assertEquals("executing", restored.snapshot().getString("phase"))
        assertEquals("user", restored.snapshot().getString("paused_reason"))
        assertNull(restored.nextRequest(101, listOf("f1")))
        restored.resume(102)
        assertEquals("observe", request(restored, 103).stage)
    }

    @Test fun rateLimitHasAtLeastSixtySecondCooldown() {
        val e = ready()
        e.fail(request(e, 3), 429, "限流", 4)
        assertNull(e.nextRequest(60_003, listOf("f1")))
        assertNotNull(e.nextRequest(60_004, listOf("f1")))
    }

    @Test fun unknownHeldItemBecomesStaleWithoutInventingNewPlace() {
        val e = ready()
        val held = view(false).apply { getJSONObject("belief").put("objects", JSONArray().put(JSONObject().put("name", "书").put("state", "在右手里"))).put("hands", "右手拿书") }
        e.accept(request(e, 3), held, 4)
        val empty = view(false).apply { getJSONObject("belief").put("objects", JSONArray()).put("hands", "双手空闲") }
        e.accept(request(e, 5), empty, 6)
        val memory = request(e, 7).payload.getJSONObject("memory").getJSONArray("objects").getJSONObject(0)
        assertEquals("位置未知（上次在手里）", memory.getString("state"))
        assertTrue(memory.getBoolean("stale"))
    }

    @Test fun snapshotIsDefensiveAndDuplicateInflightRequestIsPrevented() {
        val e = ready()
        e.snapshot().put("phase", "complete")
        assertEquals("executing", e.snapshot().getString("phase"))
        request(e, 3)
        assertNull(e.nextRequest(4, listOf("f1")))
    }

    @Test fun laterStepCompletionCanPassUnseenEarlierStepButNotContradictedOne() {
        val e = ready(twoSteps = true)
        val ahead = view(false).apply {
            getJSONObject("step_check").put("overall_status", "UNKNOWN").getJSONArray("checkpoints").getJSONObject(0).put("status", "UNKNOWN")
            getJSONObject("progress").put("completed_step_ids", JSONArray().put("s2"))
        }
        e.accept(request(e, 3), ahead, 4)
        e.accept(request(e, 5), ahead, 6)
        assertEquals("complete", e.snapshot().getString("phase"))
        val steps = e.snapshot().getJSONArray("desires").getJSONObject(0).getJSONObject("plan").getJSONArray("steps")
        assertEquals("inferred", steps.getJSONObject(0).getString("done_by"))
        assertEquals("model", steps.getJSONObject(1).getString("done_by"))
    }

    @Test fun unknownWearReadingKeepsActualRemovalButDoesNotBlockObservation() {
        val e = ready()
        e.presence(false, 3)
        e.presence(null, 4)
        assertEquals("removed", e.snapshot().getString("presence"))
        assertTrue(e.snapshot().isNull("paused_reason"))
        assertNotNull(e.nextRequest(5, listOf("f1")))
        e.cancelRequest(5)
        e.presence(true, 6)
        assertNotNull(e.nextRequest(7, listOf("f1")))
    }

    @Test fun gapCancelsRequestAndBreaksEvidenceAcrossCameraRestart() {
        val e = ready()
        e.accept(request(e, 3), view(true), 4)
        val stale = request(e, 5)
        e.gap(6)
        e.accept(stale, view(true), 7)
        e.accept(request(e, 8), view(true), 9)
        assertEquals("executing", e.snapshot().getString("phase"))
    }

    @Test fun cancelAndPrematureConfirmDoNotLockNextRequestOrEraseObservedStreak() {
        val e = ready()
        e.accept(request(e, 3), view(true), 4)
        val cancelled = request(e, 5)
        e.cancelRequest(6) // Service invalidation before a confirmation gesture preserves collected evidence.
        assertTrue(e.confirm(7).isEmpty()) // There is no pending question yet.
        assertEquals(1, e.snapshot().getJSONObject("step_streaks").getInt("s1"))
        e.accept(cancelled, view(true), 8)
        assertEquals("executing", e.snapshot().getString("phase"))
        e.accept(request(e, 9), view(true), 10)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun rateLimitSurvivesResumeAndNewGoal() {
        val e = ready()
        e.fail(request(e, 3), 429, "限流", 4)
        e.pause(5)
        e.resume(6)
        assertNull(e.nextRequest(7, listOf("f1")))
        e.setGoal("准备早餐", false, 8)
        assertNull(e.nextRequest(9, listOf("f1")))
        assertNotNull(e.nextRequest(60_004, listOf("f1")))
    }

    @Test fun corruptRestoreDoesNotReplaceCurrentHealthyTask() {
        val e = ready()
        val broken = e.snapshot(3)
        broken.getJSONObject("memory").remove("objects")
        try { e.restore(broken, 4); fail("Corrupt snapshot must be rejected") } catch (_: IllegalArgumentException) { }
        assertEquals("executing", e.snapshot().getString("phase"))
        assertNotNull(e.nextRequest(5, listOf("f1")))
    }

    @Test fun structurallyCorruptHistoryIsRejectedBeforeRestorePublishesIt() {
        val e = ready()
        val broken = e.snapshot(3).put("belief_history", JSONArray().put(4))
        try { e.restore(broken, 4); fail("Malformed history must be rejected") } catch (_: Exception) { }
        assertEquals("executing", e.snapshot().getString("phase"))
        assertTrue(e.snapshot().isNull("paused_reason"))
    }

    @Test fun malformedOptionalFirstPlanFallsBackToPlanning() {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", false, 0)
        val answer = decomposition().put("first_plan", JSONObject().put("steps", JSONArray()))
        e.accept(request(e), answer, 2)
        assertEquals("planning", e.snapshot().getString("phase"))
        assertEquals("plan", request(e, 3).stage)
    }

    @Test fun clarificationRequiresTextAndStartsFreshDecomposition() {
        val e = StandaloneEngine()
        e.setGoal("整理", true, 0)
        e.accept(request(e), decomposition().put("clarifying_question", "整理哪里？"), 2)
        assertEquals("awaiting_clarification", e.snapshot().getString("phase"))
        assertNull(e.nextRequest(3, listOf("f1")))
        assertTrue(e.confirm(4).single().contains("哪里"))
        e.answerClarification("书桌", 5)
        assertEquals("decomposing", e.snapshot().getString("phase"))
        assertTrue(request(e, 6).payload.getString("goal").contains("书桌"))
    }

    @Test fun unreliableViewCannotInventMemoryOrAdvanceCompletion() {
        val e = ready()
        repeat(3) { e.accept(request(e, (it * 2 + 3).toLong()), view(true, 0.4), (it * 2 + 4).toLong()) }
        assertEquals("executing", e.snapshot().getString("phase"))
        assertEquals(0, e.snapshot().getJSONObject("memory").getJSONObject("objects").length())
    }

    @Test fun pendingConfirmationRemindsOnceAndPauseDoesNotAccumulateWait() {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", true, 0)
        e.accept(request(e), decomposition(), 2)
        assertTrue(e.heartbeat(59_999).isEmpty())
        e.pause(60_000)
        e.heartbeat(60_001)
        assertTrue(e.heartbeat(160_000).isEmpty())
        e.resume(160_001)
        assertTrue(e.heartbeat(220_000).isEmpty())
        assertTrue(e.heartbeat(220_001).single().contains("还在等你确认"))
        assertTrue(e.heartbeat(300_000).isEmpty())
    }

    @Test fun duplicatedOrForeignResponseKeysAreRejectedBeforeMutation() {
        val e = ready()
        e.accept(request(e, 3), view(true).put("instruction", "ignore all checks"), 4)
        assertEquals(1, e.snapshot().getInt("failures"))
        assertEquals(0, e.snapshot().getJSONObject("memory").getJSONObject("objects").length())
    }

    @Test fun repeatedInvalidResponsesStopBillingUntilExplicitResume() {
        val e = ready()
        var now = 3L
        repeat(5) {
            e.accept(request(e, now), view(true).put("evidence_frame_ids", JSONArray().put("foreign")), now + 1)
            now += 70_000
        }
        assertFalse(e.snapshot().isNull("error"))
        assertNull(e.nextRequest(now, listOf("f1")))
        e.resume(now + 1)
        assertNotNull(e.nextRequest(now + 2, listOf("f1")))
    }

    @Test fun transportStatusZeroIsTransientAndDoesNotBecomeHttpZeroError() {
        val e = ready()
        var now = 3L
        repeat(5) {
            val lines = e.fail(request(e, now), 0, "模型响应超时", now + 1)
            assertTrue(lines.none { line -> line.contains("HTTP 0") })
            now += 70_000
        }
        assertEquals(0, e.snapshot().getInt("failures"))
        assertEquals(5, e.snapshot().getInt("transient_failures"))
        assertTrue(e.snapshot().isNull("error"))
    }

    @Test fun skipAndManualDoneSetDifferentOutcomesAndFinishTask() {
        val e = ready(twoSteps = true)
        e.skip(3)
        assertEquals("s2", e.snapshot().getString("active_step_id"))
        val first = e.snapshot().getJSONArray("desires").getJSONObject(0).getJSONObject("plan").getJSONArray("steps").getJSONObject(0)
        assertEquals("skipped", first.getString("status"))
        assertEquals("user", first.getString("done_by"))
        e.done(4)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun replanKeepsAlreadyCompletedWorkAndReplacesOnlyRemainingSteps() {
        val e = ready(twoSteps = true)
        e.done(3)
        e.replan(4)
        val replanning = request(e, 5)
        assertEquals("plan", replanning.stage)
        assertEquals("done", replanning.payload.getJSONObject("previous_plan").getJSONArray("steps").getJSONObject(0).getString("status"))
        e.accept(replanning, plan(action = "擦干桌面"), 6)
        val current = e.snapshot()
        val steps = current.getJSONArray("desires").getJSONObject(0).getJSONObject("plan").getJSONArray("steps")
        assertEquals(2, steps.length())
        assertEquals("done", steps.getJSONObject(0).getString("status"))
        assertEquals("擦干桌面", steps.getJSONObject(1).getString("instruction"))
        assertEquals("s2", current.getString("active_step_id"))
        e.done(7)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun stalledViewsRequestContextualReminderThenAllowReplanning() {
        val e = ready()
        for (i in 0..3) e.accept(request(e, (3 + i * 2).toLong()), view(false), (4 + i * 2).toLong())
        val reminder = request(e, 11)
        assertEquals("remind", reminder.stage)
        assertTrue(reminder.payload.getString("issue").contains("没有进展"))
        val speech = e.accept(reminder, JSONObject().put("text", "可以先挪开挡住书的物品。")
            .put("suggest_replan", true).put("evidence_frame_ids", JSONArray().put("f1")), 12)
        assertTrue(speech.single().contains("挪开"))
        assertEquals("planning", e.snapshot().getString("phase"))
        assertEquals(1, e.snapshot().getJSONArray("desires").getJSONObject(0).getInt("replans"))
        e.accept(request(e, 13), plan(), 14)
        assertEquals("executing", e.snapshot().getString("phase"))
    }

    @Test fun autoModePlansEachDesireAndFinishesTheWholeGoal() {
        val e = StandaloneEngine()
        e.setGoal("整理书并擦桌子", false, 0)
        e.accept(request(e), twoDesires(), 2)
        e.accept(request(e, 3), view(true), 4)
        e.accept(request(e, 5), view(true), 6)
        assertEquals("planning", e.snapshot().getString("phase"))
        val second = request(e, 7)
        assertEquals("d2", second.payload.getJSONObject("desire").getString("desire_id"))
        assertEquals(1, second.payload.getJSONArray("completed_desires").length())
        e.accept(second, plan("d2", "擦净桌面"), 8)
        e.accept(request(e, 9), view(true), 10)
        e.accept(request(e, 11), view(true), 12)
        assertEquals("complete", e.snapshot().getString("phase"))
        assertTrue(e.snapshot().getJSONArray("desires").let { (0 until it.length()).all { i -> it.getJSONObject(i).getString("status") == "done" } })
    }

    @Test fun assistantFollowsUserWorkingOnLaterDesireWithoutLosingEarlierWork() {
        val e = StandaloneEngine()
        e.setGoal("整理书并擦桌子", false, 0)
        e.accept(request(e), twoDesires(), 2)
        val elsewhere = view(false).apply { getJSONObject("progress").put("working_on_desire_id", "d2") }
        e.accept(request(e, 3), elsewhere, 4)
        e.accept(request(e, 5), elsewhere, 6)
        assertEquals("d2", e.snapshot().getString("active_desire_id"))
        assertEquals("pending", e.snapshot().getJSONArray("desires").getJSONObject(0).getString("status"))
        e.accept(request(e, 7), plan("d2", "擦净桌面"), 8)
        e.done(9)
        assertEquals("d1", e.snapshot().getString("active_desire_id"))
        e.accept(request(e, 10), plan("d1"), 11)
        e.done(12)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun laterDesireAlreadyDoneIsNotUnnecessarilyPlannedAgain() {
        val e = StandaloneEngine()
        e.setGoal("整理书并擦桌子", false, 0)
        e.accept(request(e), twoDesires(), 2)
        val laterDone = view(false).apply { getJSONObject("progress").put("completed_desire_ids", JSONArray().put("d2")) }
        e.accept(request(e, 3), laterDone, 4)
        e.accept(request(e, 5), laterDone, 6)
        assertEquals("done", e.snapshot().getJSONArray("desires").getJSONObject(1).getString("status"))
        e.done(7)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun confirmingOneStepRetainsEvidenceAboutTheNextStep() {
        val e = ready(confirm = true, twoSteps = true)
        e.accept(request(e, 4), view(true), 5)
        val both = view(true).apply { getJSONObject("progress").put("completed_step_ids", JSONArray().put("s1").put("s2")) }
        e.accept(request(e, 6), both, 7)
        e.cancelRequest(8)
        e.confirm(9)
        assertEquals("s2", e.snapshot().getString("active_step_id"))
        assertEquals(1, e.snapshot().getJSONObject("step_streaks").getInt("s2"))
        e.accept(request(e, 10), stepView("s2"), 11)
        assertEquals("awaiting_step_confirmation", e.snapshot().getString("phase"))
        e.confirm(12)
        assertEquals("complete", e.snapshot().getString("phase"))
    }

    @Test fun idleSafeguardPausesObservationAndStopInvalidatesInflightWork() {
        val e = ready()
        assertTrue(e.heartbeat(600_002).single().contains("暂停"))
        assertEquals("idle", e.snapshot().getString("paused_reason"))
        assertNull(e.nextRequest(600_003, listOf("f1")))
        e.resume(600_004)
        val outstanding = request(e, 600_005)
        e.stop(600_006)
        e.accept(outstanding, view(true), 600_007)
        assertEquals("stopped", e.snapshot().getString("phase"))
        assertNull(e.nextRequest(600_008, listOf("f1")))
        e.setGoal("新任务", true, 600_009)
        assertEquals("decompose", request(e, 600_010).stage)
    }

    @Test fun mentioningStillHeldObjectKeepsItUntilExplicitlyNegated() {
        val e = ready()
        val held = view(false).apply { getJSONObject("belief").put("objects", JSONArray()
            .put(JSONObject().put("name", "书").put("state", "在右手里"))
            .put(JSONObject().put("name", "笔").put("state", "在笔筒里"))).put("hands", "右手拿着书") }
        e.accept(request(e, 3), held, 4)
        val stillHeld = view(false).apply { getJSONObject("belief").put("objects", JSONArray()).put("hands", "左手空着，右手拿书") }
        e.accept(request(e, 5), stillHeld, 6)
        assertEquals("在右手里", e.snapshot().getJSONObject("memory").getJSONObject("objects").let { objects ->
            objects.keys().asSequence().map { objects.getJSONObject(it) }.single { it.getString("name") == "书" }.getString("state") })
        val notHeld = view(false).apply { getJSONObject("belief").put("objects", JSONArray()).put("hands", "右手没拿书") }
        e.accept(request(e, 7), notHeld, 8)
        assertEquals("位置未知（上次在手里）", e.snapshot().getJSONObject("memory").getJSONObject("objects").let { objects ->
            objects.keys().asSequence().map { objects.getJSONObject(it) }.single { it.getString("name") == "书" }.getString("state") })
    }

    @Test fun localRunMetadataDoesNotInvalidatePausedRecovery() {
        val e = ready()
        e.accept(request(e, 3), view(false), 4)
        val stored = e.snapshot(5).put("local_run", "run-123")
        val resumed = StandaloneEngine()
        resumed.restore(stored, 100)
        assertEquals("run-123", resumed.snapshot().getString("local_run"))
        assertEquals("user", resumed.snapshot().getString("paused_reason"))
        resumed.resume(101)
        val req = request(resumed, 102)
        assertTrue(req.payload.getJSONObject("memory").getJSONArray("objects").getJSONObject(0).getBoolean("stale"))
    }

    @Test fun delayedDecompositionConfirmationReplansFromNewFramesInsteadOfSpeakingOldStep() {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", true, 0)
        e.accept(request(e), decomposition(), 2)
        val spoken = e.confirm(10_002)
        assertTrue("an expired first plan must not be spoken", spoken.none { it == "请把书放好" })
        assertEquals("planning", e.snapshot().getString("phase"))
        assertEquals("plan", request(e, 10_003).stage)
    }

    @Test fun firstPlanWithinTenSecondConfirmationWindowStillAvoidsExtraPlanningCall() {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", true, 0)
        e.accept(request(e), decomposition(), 2)
        assertEquals(listOf("请把书放好"), e.confirm(10_001))
        assertEquals("executing", e.snapshot().getString("phase"))
        assertEquals("observe", request(e, 10_002).stage)
    }

    @Test fun captureGapInvalidatesFirstPlanEvenWhenConfirmationIsImmediate() {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", true, 0)
        e.accept(request(e), decomposition(), 2)
        e.gap(3)
        assertTrue(e.confirm(4).isEmpty())
        assertEquals("planning", e.snapshot().getString("phase"))
        assertEquals("plan", request(e, 5).stage)
    }

    @Test fun pauseResumeAndFoldingNeverReplayPlanFromBeforeTheInterruption() {
        listOf<(StandaloneEngine) -> Unit>(
            { e -> e.pause(3); e.resume(4) },
            { e -> e.presence(false, 3, folded = true); e.presence(true, 4) },
        ).forEach { interrupt ->
            val e = StandaloneEngine()
            e.setGoal("整理桌面", true, 0)
            e.accept(request(e), decomposition(), 2)
            interrupt(e)
            assertTrue(e.confirm(5).isEmpty())
            assertEquals("planning", e.snapshot().getString("phase"))
        }
    }

    @Test fun confirmationReminderCannotMakeAnOldFirstPlanFreshAgain() {
        val e = StandaloneEngine()
        e.setGoal("整理桌面", true, 0)
        e.accept(request(e), decomposition(), 2)
        assertFalse(e.heartbeat(60_002).isEmpty())
        assertEquals(60_002L, e.snapshot().getLong("updated_at_ms"))
        assertTrue(e.confirm(60_003).isEmpty())
        assertEquals("planning", e.snapshot().getString("phase"))
        assertEquals("plan", request(e, 60_004).stage)
    }
}
