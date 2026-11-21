package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class StandaloneResearchTest {
    private fun ready(): StandaloneEngine {
        val engine = StandaloneEngine()
        engine.setGoal("把笔装入A盒和B盒", true, 0)
        val response = JSONObject("""{"goal_summary":"装盒","horizon":"short","desires":[{"desire_id":"d1","title":"装盒","success_criteria":"每盒装好"}],"clarifying_question":null,"confidence":0.9,"evidence_frame_ids":["f1"],"first_plan":{"steps":[{"step_id":"s1","instruction":"把笔放入A盒","spoken_hint":"请装A盒","required_objects":["笔"],"checkpoints":[{"checkpoint_id":"c1","question":"A盒装好了吗"}]}]}}""")
        engine.accept(engine.nextRequest(1, listOf("f1"))!!, response, 2)
        engine.confirm(3)
        return engine
    }

    private fun view(done: Boolean, frame: String, objects: JSONArray = JSONArray()): JSONObject = JSONObject("""{"belief":{"scene":"桌面","objects":[],"hands":"双手空闲","action":"装盒","changes":""},"uncertain":false,"confidence":0.9,"step_check":{"step_id":"s1","checkpoints":[{"checkpoint_id":"c1","status":"${if(done) "COMPLETED" else "IN_PROGRESS"}"}],"overall_status":"${if(done) "COMPLETED" else "PROCESSING"}","deviation":null},"progress":{"completed_step_ids":[],"completed_desire_ids":[],"working_on_desire_id":null},"evidence_frame_ids":["$frame"]}""")
        .apply { getJSONObject("belief").put("objects", objects) }

    private fun objectIn(container: String?, state: String) = JSONObject().put("name", "笔").put("state", state)
        .put("container_id", container ?: JSONObject.NULL)

    private fun observe(engine: StandaloneEngine, now: Long, response: JSONObject): List<String> {
        val frame = response.getJSONArray("evidence_frame_ids").getString(0)
        val request = engine.nextRequest(now, listOf(frame))
        assertNotNull("new frames must be observed while a step awaits approval", request)
        return engine.accept(request!!, response, now + 1)
    }

    private fun pending(): StandaloneEngine = ready().apply {
        observe(this, 4, view(true, "f2"))
        observe(this, 6, view(true, "f3"))
        assertEquals("awaiting_step_confirmation", snapshot().getString("phase"))
    }

    @Test fun waitingUpdatesEvidenceWithoutRepromptingOrAdvancing() {
        val engine = pending()
        val oldPrompt = engine.snapshot().getJSONObject("pending").toString()
        assertTrue(observe(engine, 8, view(true, "f4", JSONArray().put(objectIn("A盒", "1支")))).isEmpty())
        assertEquals("awaiting_step_confirmation", engine.snapshot().getString("phase"))
        assertEquals(oldPrompt, engine.snapshot().getJSONObject("pending").toString())
        assertEquals(1, engine.snapshot().getJSONObject("memory").getJSONObject("objects").length())
        engine.confirm(10)
        assertEquals("complete", engine.snapshot().getString("phase"))
    }

    @Test fun removingUnfoldedGlassesKeepsObservationAndApprovalWithoutInvalidatingTheRequest() {
        val engine = pending()
        val approval = engine.snapshot().getJSONObject("pending").toString()
        val request = engine.nextRequest(8, listOf("f4"))!!
        engine.presence(false, 9)
        assertEquals("removed", engine.snapshot().getString("presence"))
        assertTrue(engine.snapshot().isNull("paused_reason"))
        assertTrue(engine.needsVisualInput())
        assertEquals(request.revision, engine.snapshot().getLong("revision"))
        engine.accept(request, view(true, "f4", JSONArray().put(objectIn("A盒", "1支"))), 10)
        assertEquals(1, engine.snapshot().getJSONObject("memory").getJSONObject("objects").length())
        assertEquals(approval, engine.snapshot().getJSONObject("pending").toString())
    }

    @Test fun offHeadResumeWorksButWearEventsCannotUndoManualPause() {
        val engine = ready()
        engine.presence(false, 4)
        engine.pause(5)
        engine.presence(true, 6)
        engine.presence(false, 7)
        assertEquals("user", engine.snapshot().getString("paused_reason"))
        engine.resume(8)
        assertTrue(engine.snapshot().isNull("paused_reason"))
        assertEquals("removed", engine.snapshot().getString("presence"))
        assertNotNull(engine.nextRequest(9, listOf("f2")))
    }

    @Test fun contradictionWithdrawsOldApprovalAndLateResponseCannotRestoreIt() {
        val engine = pending()
        val spoken = observe(engine, 8, view(false, "f4"))
        assertEquals("executing", engine.snapshot().getString("phase"))
        assertTrue(engine.snapshot().isNull("pending"))
        assertTrue(spoken.single().contains("撤回"))
        assertTrue(engine.confirm(10).isEmpty())
        observe(engine, 11, view(true, "f5"))
        assertEquals("executing", engine.snapshot().getString("phase"))
        observe(engine, 13, view(true, "f6"))
        val old = engine.nextRequest(15, listOf("f7"))!!
        engine.confirm(16)
        engine.accept(old, view(false, "f7"), 17)
        assertEquals("complete", engine.snapshot().getString("phase"))
    }

    @Test fun unknownViewKeepsExplicitApprovalButDoesNotRenewOrExpandIt() {
        val engine = pending()
        val old = engine.snapshot().getJSONObject("pending").toString()
        val unknown = view(true, "f4").put("uncertain", true)
        repeat(3) { observe(engine, 8L + it * 2, unknown) }
        assertEquals(old, engine.snapshot().getJSONObject("pending").toString())
        assertEquals("awaiting_step_confirmation", engine.snapshot().getString("phase"))
    }

    @Test fun actualGapsAndPausesWithdrawStepApproval() {
        for (change in listOf<(StandaloneEngine) -> Unit>({ it.gap(8) }, { it.pause(8) }, { it.presence(false, 8, folded = true) })) {
            val engine = pending()
            change(engine)
            assertTrue(engine.snapshot().isNull("pending"))
            assertEquals("executing", engine.snapshot().getString("phase"))
            assertTrue(engine.confirm(9).isEmpty())
        }
    }

    @Test fun captureAndRequestSchedulingUseTheSameActivePhases() {
        val engine = pending()
        assertTrue(engine.needsVisualInput())
        engine.pause(8)
        assertFalse(engine.needsVisualInput())
        engine.resume(9)
        assertTrue(engine.needsVisualInput())
        engine.presence(false, 10, folded = true)
        assertFalse(engine.needsVisualInput())
        assertFalse(StandaloneEngine().needsVisualInput())
    }

    @Test fun foldingStillBlocksNewGoalsAndManualPauseSurvivesUnfolding() {
        val engine = ready()
        engine.presence(false, 4, folded = true)
        assertEquals("folded", engine.snapshot().getString("paused_reason"))
        engine.setGoal("新目标", true, 5)
        engine.resume(6)
        assertFalse(engine.needsVisualInput())
        engine.presence(false, 7, folded = false)
        assertTrue(engine.needsVisualInput())
        engine.pause(8)
        engine.presence(false, 9, folded = true)
        engine.presence(false, 10, folded = false)
        assertEquals("user", engine.snapshot().getString("paused_reason"))
    }

    @Test fun transportCancellationForAConfirmGesturePreservesApproval() {
        val engine = pending()
        val old = engine.nextRequest(8, listOf("f4"))!!
        engine.cancelRequest(9)
        engine.confirm(10)
        engine.accept(old, view(false, "f4"), 11)
        assertEquals("complete", engine.snapshot().getString("phase"))
    }

    @Test fun waitingObservationIsBoundedByIdlePause() {
        val engine = pending()
        engine.heartbeat(60_008)
        engine.heartbeat(600_008)
        assertEquals("idle", engine.snapshot().getString("paused_reason"))
        assertNull(engine.nextRequest(600_009, listOf("f4")))
    }

    @Test fun newGoalKeepsHistoryButCannotReuseAnotherTasksCurrentFacts() {
        val engine = ready()
        observe(engine, 4, view(false, "f2", JSONArray().put(JSONObject().put("name", "笔").put("state", "已装入"))))
        val oldTask = engine.snapshot().optString("task_instance_id")
        assertTrue(oldTask.isNotBlank())
        engine.setGoal("把笔装入A盒和B盒", true, 6)
        assertNotEquals(oldTask, engine.snapshot().getString("task_instance_id"))
        assertEquals(1, engine.snapshot().getJSONObject("memory").getJSONObject("objects").length())
        val context = engine.nextRequest(7, listOf("f3"))!!.payload.getJSONObject("memory")
        assertEquals(0, context.getJSONArray("objects").length())
        assertEquals(0, context.getJSONArray("recent_actions").length())
        assertTrue(context.isNull("hands"))
    }

    @Test fun sameNameInTwoContainersAndUnknownContainerNeverOverwriteEachOther() {
        val engine = ready()
        observe(engine, 4, view(false, "f2", JSONArray().put(objectIn("A盒", "1支")).put(objectIn("B盒", "2支"))))
        observe(engine, 6, view(false, "f3", JSONArray().put(objectIn(null, "手里"))))
        val objects = engine.nextRequest(8, listOf("f4"))!!.payload.getJSONObject("memory").getJSONArray("objects")
        assertEquals(3, objects.length())
        val byContainer = (0 until objects.length()).map { objects.getJSONObject(it) }.associateBy { it.optString("container_id") }
        assertEquals("1支", byContainer.getValue("A盒").getString("state"))
        assertEquals("2支", byContainer.getValue("B盒").getString("state"))
        assertTrue(byContainer.getValue("A盒").getBoolean("scope_uncertain"))
        assertTrue(byContainer.getValue("B盒").getBoolean("scope_uncertain"))
    }

    @Test fun restoreRetainsTaskScopeAndLegacyUnscopedFactsStayHistorical() {
        val engine = ready()
        observe(engine, 4, view(false, "f2", JSONArray().put(objectIn("A盒", "1支"))))
        val saved = engine.snapshot(6)
        val resumed = StandaloneEngine()
        resumed.restore(saved, 100)
        assertEquals(saved.getString("task_instance_id"), resumed.snapshot().getString("task_instance_id"))
        resumed.resume(101)
        assertEquals("A盒", resumed.nextRequest(102, listOf("f3"))!!.payload.getJSONObject("memory").getJSONArray("objects").getJSONObject(0).getString("container_id"))
        saved.remove("task_instance_id")
        val entries = saved.getJSONObject("memory").getJSONObject("objects")
        entries.keys().asSequence().forEach { entries.getJSONObject(it).remove("task_instance_id") }
        resumed.restore(saved, 200)
        resumed.resume(201)
        assertEquals(1, resumed.snapshot().getJSONObject("memory").getJSONObject("objects").length())
        assertEquals(0, resumed.nextRequest(202, listOf("f4"))!!.payload.getJSONObject("memory").getJSONArray("objects").length())
    }

    @Test fun invalidOrDuplicateContainerFactsAreRejectedAtomically() {
        for (objects in listOf(JSONArray().put(objectIn("A盒", "1支")).put(objectIn("A盒", "2支")),
                JSONArray().put(objectIn("A盒", "1支").put("container_id", 7)),
                JSONArray().put(objectIn("A盒", "1支").put("task_instance_id", "other-task")))) {
            val engine = ready()
            observe(engine, 4, view(false, "f2", objects))
            assertEquals(1, engine.snapshot().getInt("failures"))
            assertEquals(0, engine.snapshot().getJSONObject("memory").getJSONObject("objects").length())
        }
    }
}
