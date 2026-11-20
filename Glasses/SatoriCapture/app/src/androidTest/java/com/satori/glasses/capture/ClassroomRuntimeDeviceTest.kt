package com.satori.glasses.capture

import android.content.Context
import android.content.ContextWrapper
import android.os.Handler
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.json.JSONArray
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.net.URL
import java.security.MessageDigest
import java.security.cert.Certificate
import javax.net.ssl.HttpsURLConnection
import java.lang.reflect.InvocationTargetException
import java.util.UUID
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.FutureTask

/** Real service routing/journaling on a cache-only context, without Service.onCreate.
 * No APIs, sensors or speech objects are installed. A missing dialogue client stops at the
 * production askManual boundary: these tests prove routing and task preservation, not an answer.
 */
@RunWith(AndroidJUnit4::class)
class ClassroomRuntimeDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test fun restoredCompletedTrialDoesNotGetANewEndTime() = terminalSnapshot("complete")

    @Test fun restoredStoppedTrialDoesNotGetANewEndTime() = terminalSnapshot("stopped")

    private fun terminalSnapshot(phase: String) = isolated { service, dir ->
        val engine = field(service, "engine") as StandaloneEngine
        engine.setGoal("只用于隔离日志测试的任务", true, SystemClock.elapsedRealtime())
        // Boundary input fixture, not a claim that an actual physical task was completed.
        val snapshot = engine.snapshot().put("phase", phase)
        val task = snapshot.getString("task_instance_id")
        set(service, "restoredTaskId", task)
        repeat(2) { call(service, "recordTrialBoundary", JSONObject::class.java, snapshot) }
        val rows = events(dir)
        assertEquals(listOf("trial_terminal_snapshot"), rows.map { it.getString("name") })
        val event = rows.single()
        assertEquals(task, event.getJSONObject("data").getString("trial_id"))
        assertEquals(phase, event.getJSONObject("data").getString("phase"))
        assertFalse(rows.any { it.getString("name") in setOf("trial_started", "trial_resumed", "trial_ended") })
        assertEquals(phase, field(service, "journaledPhase"))
        assertFalse("boundary journaling must not persist a replacement production task",
            File(service.filesDir, "standalone-state.json").exists())
    }

    @Test fun progressQuestionDoesNotConfirmOrReplaceTheTask() = isolated { service, dir ->
        val engine = setupTask(service)
        val before = engine.snapshot()
        set(service, "inFlight", true)
        say(service, "任务完成了吧？")
        assertTaskUnchanged(before, engine.snapshot())
        assertTrue("status questions must not cancel an in-flight task request", field(service, "inFlight") as Boolean)
        assertTrue((field(service, "lastSpoken") as String).contains("还未确认"))
        assertEquals(listOf("voice_text"), events(dir).map { it.getString("name") })
        assertNull(field(service, "pendingRecognitionTrace"))
    }

    @Test fun recordingTranscriptsRouteLocallyForManualAndAutomaticInput() = isolated { service, dir ->
        val engine = setupTask(service)
        engine.pause(SystemClock.elapsedRealtime())
        val before = engine.snapshot()
        val profile = StandaloneProfile("https://fixture.invalid/v1", "fixture-only-key", "planner", "monitor",
            manualSource = "web", manualId = "web-discovery")
        set(service, "profile", profile)
        fun transcribe(text: String, automatic: Boolean): List<JSONObject> {
            val start = events(dir).size
            set(service, "pendingRecognitionTrace", JSONObject().put("automatic", automatic))
            call(service, "onText", String::class.java, text)
            return events(dir).drop(start)
        }

        for (automatic in listOf(false, true)) {
            for (text in listOf("不要开始录像", "等我说完再开始录像")) {
                val rows = transcribe(text, automatic)
                val message = "automatic=$automatic text=$text"
                assertTrue(message, rows.any { it.getString("name") in setOf("voice_text", "voice_ignored") })
                assertFalse(message, rows.any { it.getString("name") == "command" })
            }
        }

        // Make a wrong dialogue route observable in the journal while preventing any network access.
        set(service, "dialogueApi", StandaloneApi(profile, connectionFactory = { error("unexpected network call") }))
        for (automatic in listOf(false, true)) {
            for (text in listOf("二楼开始录像", "开始录像用水特朗普", "你用眼镜的录像", "可以开始录像吗？")) {
                val rows = transcribe(text, automatic)
                val message = "automatic=$automatic text=$text"
                assertTrue(message, rows.any { it.getString("name") == "voice_text" })
                assertFalse(message, rows.any { it.getString("name") in setOf("command", "dialogue_request") })
                val reply = field(service, "lastReplyText") as String
                assertTrue("$message reply=$reply", reply.contains("录像") || reply.contains("录制"))
                assertTaskUnchanged(before, engine.snapshot())
            }
            for ((text, action) in listOf("嗯开始录像" to "record_start", "哈罗开始录像" to "record_start",
                    "你现在开始录像" to "record_start", "用眼镜开始录像" to "record_start", "你用眼镜录像" to "record_start",
                    "结束录像" to "record_stop", "用眼镜结束录像" to "record_stop", "你现在停止录像" to "record_stop")) {
                val rows = transcribe(text, automatic)
                val message = "automatic=$automatic text=$text"
                val commanded = rows.filter { it.getString("name") == "command" }.map { it.getJSONObject("data").getString("action") }
                assertEquals(message, listOf(action), commanded)
                assertEquals(message, text, rows.single { it.getString("name") == "voice_text" }.getJSONObject("data").getString("text"))
                assertFalse(message, rows.any { it.getString("name") == "dialogue_request" })
            }
        }
        assertTaskUnchanged(before, engine.snapshot())
    }

    @Test fun questionPrefixesRouteWithoutCreatingANewGoal() = isolated { service, dir ->
        val engine = setupTask(service)
        val before = engine.snapshot()
        for (question in listOf("我想问一下网线怎么接", "我要问一下哪个端口")) {
            val count = events(dir).size
            set(service, "lastReplyText", "之前的回答")
            say(service, question)
            assertTaskUnchanged(before, engine.snapshot())
            val newEvents = events(dir).drop(count)
            assertEquals(listOf("voice_text"), newEvents.map { it.getString("name") })
            assertEquals(question, newEvents.single().getJSONObject("data").getString("text"))
            assertEquals(question, field(service, "lastUserText"))
            assertEquals("", field(service, "lastReplyText"))
            assertEquals("a manual question must not fall through to generic command help", "", field(service, "error"))
        }
        assertNull("dialogue is deliberately unavailable in this routing fixture", field(service, "dialogueApi"))
    }

    @Test fun shortModelReplyPassesAutomaticInputGateForTheAskingTask() = isolated { service, dir ->
        val engine = setupTask(service)
        val before = engine.snapshot()
        val task = before.getString("task_instance_id")
        val conversation = field(service, "manualConversation") as ManualConversation
        val at = SystemClock.elapsedRealtime()
        conversation.begin(task, "请根据手册指导我接网线", at)
        conversation.finish(task, JSONObject().put("kind", "clarify").put("text", "请确认设备型号和版本"), at)
        assertTrue(conversation.awaitingReply(task, at))
        say(service, "TL-SG105第四版")
        assertTaskUnchanged(before, engine.snapshot())
        assertEquals(listOf("voice_text"), events(dir).map { it.getString("name") })
        assertEquals("TL-SG105第四版", events(dir).single().getJSONObject("data").getString("text"))
        assertEquals("", field(service, "error"))
        assertNull(field(service, "pendingRecognitionTrace"))
    }

    @Test fun openDialogueIgnoresEmptyAudioAndAcceptsOneCharacterClarifications() = isolated { service, dir ->
        val engine = field(service, "engine") as StandaloneEngine
        set(service, "profile", StandaloneProfile("https://fixture.invalid/v1", "fixture-only-key", "planner", "monitor",
            manualSource = "web", manualId = "web-discovery"))
        for (automatic in listOf(true, false)) {
            for (noise in listOf("嗯啊", "？！", "哎，呃", "字".repeat(501))) {
                val start = events(dir).size
                set(service, "pendingRecognitionTrace", JSONObject().put("automatic", automatic))
                call(service, "onText", String::class.java, noise)
                assertTrue(events(dir).drop(start).none { it.getString("name") in setOf("voice_text", "command", "dialogue_request") })
            }
        }
        say(service, "蓝")
        assertEquals("voice_ignored", events(dir).last().getString("name"))
        val history = field(service, "manualConversation") as ManualConversation
        val at = SystemClock.elapsedRealtime()
        history.begin("dialogue-run-fixture", "选一个", at)
        history.finish("dialogue-run-fixture", JSONObject().put("kind", "clarify").put("text", "哪一种颜色？"), at)
        say(service, "蓝")
        assertEquals("voice_text", events(dir).last().getString("name"))
        assertTrue(engine.snapshot().isNull("goal"))
    }

    @Test fun capturedModelQuestionPassesOnlyWithinAManualTask() = isolated { service, dir ->
        val engine = setupTask(service)
        engine.pause(SystemClock.elapsedRealtime())
        val before = engine.snapshot()
        val captured = "这分手车型与那个型号"
        say(service, captured)
        assertTaskUnchanged(before, engine.snapshot())
        assertEquals(listOf("voice_text"), events(dir).map { it.getString("name") })
        assertEquals(captured, events(dir).single().getJSONObject("data").getString("text"))
        assertEquals("", field(service, "error"))
        say(service, "可能")
        assertEquals("voice_ignored", events(dir).last().getString("name"))
        val profile = field(service, "profile") as StandaloneProfile
        set(service, "profile", profile.copy(manualId = "", toolsBaseUrl = "", toolsToken = ""))
        say(service, captured)
        assertEquals("voice_ignored", events(dir).last().getString("name"))
        assertTaskUnchanged(before, engine.snapshot())
    }

    @Test fun expiredOrOtherTaskClarificationDoesNotAdmitBackgroundReply() = isolated { service, dir ->
        val engine = setupTask(service)
        val before = engine.snapshot()
        val task = before.getString("task_instance_id")
        val conversation = field(service, "manualConversation") as ManualConversation
        val at = SystemClock.elapsedRealtime()
        for ((owner, time) in listOf("other-task" to at, task to (at - 100_000))) {
            conversation.begin(owner, "请确认型号", time)
            conversation.finish(owner, JSONObject().put("kind", "clarify").put("text", "是什么型号？"), time)
            assertFalse(conversation.awaitingReply(task, at))
            val count = events(dir).size
            say(service, "TL-SG105第四版")
            assertTaskUnchanged(before, engine.snapshot())
            assertEquals(listOf("voice_ignored"), events(dir).drop(count).map { it.getString("name") })
        }
    }

    @Test fun manualIdentityReplyPreemptsAnOlderDialogueReplyWindow() = isolated { service, _ ->
        val engine = setupTask(service)
        val task = engine.snapshot().getString("task_instance_id")
        val request = engine.nextRequest(SystemClock.elapsedRealtime(), listOf("fixture-frame"))!!
        engine.clarifyManualIdentity(request, "请读出完整铭牌型号", SystemClock.elapsedRealtime())
        val conversation = field(service, "manualConversation") as ManualConversation
        conversation.begin(task, "手册问题", SystemClock.elapsedRealtime())
        conversation.finish(task, JSONObject().put("kind", "clarify").put("text", "还有什么问题"), SystemClock.elapsedRealtime())
        say(service, "型号BPLDS02DM")
        assertEquals("decomposing", engine.snapshot().getString("phase"))
        assertTrue(engine.snapshot().getJSONObject("goal").getString("text").contains("BPLDS02DM"))
        assertFalse(conversation.awaitingReply(task, SystemClock.elapsedRealtime()))
    }

    @Test fun ordinaryWebQuestionDoesNotSearchOrReuseAPausedTask() = ordinaryQuestionWithoutSearch(pausedTask = true)

    @Test fun ordinaryWebQuestionWorksWithoutCreatingATask() = ordinaryQuestionWithoutSearch(pausedTask = false)

    @Test fun everydayStatementsReachDialogueWithoutTaskKeywords() {
        for (utterance in listOf("明早赶早班车，行李有点乱", "两张便条上的日期好像不一样")) {
            ordinaryQuestionWithoutSearch(pausedTask = false, utterance = utterance)
        }
    }

    @Test fun ordinaryFollowUpSendsItsCurrentImageToPlannerAndAnswer() = ordinaryQuestionWithoutSearch(pausedTask = false, followUp = true)

    @Test fun ordinaryFollowUpReportsMissingViewAndDoesNotSendStaleImage() = ordinaryQuestionWithoutSearch(pausedTask = false, fresh = false, followUp = true)

    private fun ordinaryQuestionWithoutSearch(pausedTask: Boolean, fresh: Boolean = true, followUp: Boolean = false, utterance: String? = null) {
        val sent = mutableListOf<ByteArrayOutputStream>()
        var before = JSONObject()
        val question = utterance ?: if (followUp) "嗯告诉方法就行" else "二加二等于几"
        val expectedReply = when {
            utterance != null -> "收到。"
            followUp -> "请按步骤操作。"
            else -> "二加二等于4。"
        }
        val endpoint = "https://fixture.invalid/v1/chat/completions"
        isolated(afterMain = { service, dir ->
            val deadline = SystemClock.elapsedRealtime() + 5000
            while (onMain { field(service, "dialogueBusy") as Boolean } && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(25)
            onMain {
                assertFalse("ordinary question must finish", field(service, "dialogueBusy") as Boolean)
                val rows = events(dir)
                assertFalse(rows.any { it.getString("name") in setOf("dialogue_error", "mcp_request", "mcp_result") })
                val response = rows.single { it.getString("name") == "dialogue_response" }.getJSONObject("data")
                assertEquals("answer", response.getJSONObject("answer").getString("kind"))
                val reply = response.getJSONObject("answer").getString("text")
                assertTrue(reply.contains(expectedReply))
                assertEquals(!fresh, reply.contains("没取到眼镜画面"))
                assertTrue(response.getBoolean("model_called"))
                val evidence = response.getJSONObject("evidence")
                assertEquals("not_requested", evidence.getString("transport"))
                assertEquals(0, evidence.getJSONArray("passages").length())
                assertEquals("the planner and answer are the only model calls", 2, sent.size)
                val oldGoal = before.optJSONObject("goal")?.optString("text").orEmpty()
                for (body in sent) {
                    val encoded = body.toString("UTF-8")
                    assertTrue(encoded.contains(question))
                    assertFalse(encoded.contains("old-scene"))
                    val messages = JSONObject(encoded).getJSONArray("messages")
                    val system = messages.getJSONObject(0).getString("content")
                    assertFalse("ordinary turns must not require device identity", system.contains("requires_model"))
                    assertFalse("view is attached without a prescribed phrase", system.contains("看看眼前"))
                    val content = messages.getJSONObject(1).getJSONArray("content")
                    val images = (0 until content.length()).map { content.getJSONObject(it) }
                        .filter { it.getString("type") == "image_url" }.map { it.getJSONObject("image_url").getString("url") }
                    assertEquals(if (fresh) listOf("data:image/jpeg;base64,AQID") else emptyList<String>(), images)
                    if (fresh) assertTrue(encoded.contains("current-scene"))
                    if (!fresh && body === sent.last()) assertTrue(encoded.contains("本轮没有眼镜图像"))
                    if (oldGoal.isNotBlank()) assertFalse("paused task must not become a preset for this question", encoded.contains(oldGoal))
                }
                val after = (field(service, "engine") as StandaloneEngine).snapshot()
                assertEquals(before.opt("task_instance_id"), after.opt("task_instance_id"))
                assertEquals(before.opt("goal").toString(), after.opt("goal").toString())
                assertEquals(before.opt("paused_reason"), after.opt("paused_reason"))
                if (pausedTask) assertEquals("user", after.getString("paused_reason"))
                else { assertEquals("idle", after.getString("phase")); assertTrue(after.isNull("goal")) }
            }
        }) { service, _ ->
            val engine = if (pausedTask) setupTask(service).apply { pause(SystemClock.elapsedRealtime()) }
                else field(service, "engine") as StandaloneEngine
            before = engine.snapshot()
            val config = StandaloneProfile("https://fixture.invalid/v1", "fixture-only-key", "planner", "monitor",
                manualSource = "web", manualId = "web-discovery")
            val plan = JSONObject().put("needs_search", false).put("query", "")
            val answer = JSONObject().put("kind", "answer").put("text", expectedReply)
                .put("source_sections", JSONArray()).put("suggest_replan", false)
            val api = StandaloneApi(config, connectionFactory = { url ->
                assertEquals("no Exa or other endpoint is needed", endpoint, url)
                val output = ByteArrayOutputStream()
                sent.add(output)
                val content = when (sent.size) {
                    1 -> plan
                    2 -> answer
                    else -> throw AssertionError("unexpected additional model call")
                }
                val bytes = JSONObject().put("choices", JSONArray().put(JSONObject().put("finish_reason", "stop")
                    .put("message", JSONObject().put("content", content.toString())))).toString().toByteArray(Charsets.UTF_8)
                object : HttpsURLConnection(URL(url)) {
                    override fun getOutputStream() = output
                    override fun getResponseCode() = 200
                    override fun getInputStream() = ByteArrayInputStream(bytes)
                    override fun getContentLengthLong() = bytes.size.toLong()
                    override fun connect() = Unit
                    override fun disconnect() = Unit
                    override fun usingProxy() = false
                    override fun getCipherSuite() = "fixture-no-network"
                    override fun getLocalCertificates(): Array<Certificate>? = null
                    override fun getServerCertificates(): Array<Certificate> = emptyArray()
                }
            })
            set(service, "profile", config)
            set(service, "dialogueApi", api)
            val at = SystemClock.elapsedRealtime()
            if (followUp) {
                val scope = "dialogue-run-fixture"
                val history = field(service, "manualConversation") as ManualConversation
                history.begin(scope, "我想清洁风扇", at)
                history.finish(scope, JSONObject().put("kind", "clarify").put("text", "请告诉我设备型号。"), at)
            }
            val frames = field(service, "frames") as StandaloneFrameBuffer
            frames.add(StandaloneFrame("old-scene", at - 8000, byteArrayOf(4)))
            if (fresh) frames.add(StandaloneFrame("current-scene", at, byteArrayOf(1, 2, 3)))
            say(service, question)
        }
    }

    @Test fun explicitVisualQuestionWithoutACurrentFrameExplainsFailureLocally() {
        val config = StandaloneProfile("https://fixture.invalid/v1", "fixture-only-key", "planner", "monitor",
            manualSource = "web", manualId = "web-discovery")
        isolated(afterMain = { service, dir ->
            val deadline = SystemClock.elapsedRealtime() + 5000
            while (onMain { field(service, "dialogueBusy") as Boolean } && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(25)
            val rows = events(dir)
            assertTrue(rows.none { it.getString("name") in setOf("tool_request", "dialogue_route") })
            val failed = rows.single { it.getString("name") == "dialogue_error" }.getJSONObject("data")
            assertEquals("current_view", failed.getString("stage"))
            assertTrue((field(service, "lastReplyText") as String).contains("没有取得眼镜画面"))
        }) { service, _ ->
            set(service, "profile", config)
            set(service, "dialogueApi", StandaloneApi(config, connectionFactory = { error("unexpected request") }))
            say(service, "请问眼前是什么")
        }
    }

    @Test fun automaticMetadataQuestionAnswersThroughServiceWithoutCallingAModel() {
        var httpCalls = 0
        var modelCalls = 0
        var task = ""
        isolated(afterMain = { service, dir ->
            val deadline = SystemClock.elapsedRealtime() + 5000
            while (onMain { field(service, "dialogueBusy") as Boolean } && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(25)
            onMain {
                assertFalse(field(service, "dialogueBusy") as Boolean)
                val rows = events(dir)
                assertFalse(rows.any { it.getString("name") == "dialogue_error" })
                val response = rows.single { it.getString("name") == "dialogue_response" }.getJSONObject("data")
                val feedback = rows.filter { it.getString("name") == "interaction_feedback" }.map { it.getJSONObject("data") }
                assertEquals("received", feedback.first().getString("stage"))
                assertTrue(feedback.any { it.optString("stage") == "manual_lookup" })
                assertTrue(feedback.any { it.optString("stage") == "metadata_answer" })
                assertEquals("finished", feedback.last().getString("transition"))
                assertEquals("completed", feedback.last().getString("reason"))
                assertEquals(0, (field(service, "interactionFeedback") as JSONObject).length())
                assertEquals("verified_manual_metadata", response.getString("answer_source"))
                assertFalse(response.getBoolean("model_called"))
                val answer = response.getJSONObject("answer")
                assertTrue(answer.getString("text").contains("TL-SG105(UN) V4.0"))
                assertFalse(answer.getBoolean("suggest_replan"))
                assertEquals(answer.getString("text"), field(service, "lastSpoken"))
                assertEquals(answer.getString("text"), field(service, "lastReplyText"))
                assertEquals("问这份手册适用于那的型号的设备", field(service, "lastUserText"))
                val history = (field(service, "chatTranscript") as ChatTranscript).snapshot()
                assertEquals(4, history.length())
                assertEquals("问这份手册适用于那的型号的设备", history.getJSONObject(0).getString("text"))
                assertEquals(answer.getString("text"), history.getJSONObject(1).getString("text"))
                assertEquals("任务完成了吧", history.getJSONObject(2).getString("text"))
                assertTrue(history.getJSONObject(3).getString("text").contains("暂停"))
                assertEquals(task, (field(service, "engine") as StandaloneEngine).snapshot().getString("task_instance_id"))
                assertEquals(1, httpCalls)
                assertEquals(0, modelCalls)
            }
        }) { service, _ ->
            val engine = setupTask(service)
            engine.pause(SystemClock.elapsedRealtime())
            task = engine.snapshot().getString("task_instance_id")
            val config = (field(service, "profile") as StandaloneProfile).copy(manualSource = "device", toolsBaseUrl = "", toolsToken = "")
            val pdf = "%PDF-1.4 isolated service fixture".toByteArray(Charsets.UTF_8)
            val index = JSONObject(target.assets.open("manuals/source-index.json").bufferedReader().use { it.readText() })
            index.getJSONArray("sources").getJSONObject(0)
                .put("sha256", MessageDigest.getInstance("SHA-256").digest(pdf).joinToString("") { "%02x".format(it) })
                .put("document_bytes", pdf.size)
            val api = StandaloneApi(config, manualAsset = { name ->
                if (name == "source-index.json") index.toString().toByteArray(Charsets.UTF_8)
                else target.assets.open("manuals/$name").use { it.readBytes() }
            }, connectionFactory = { url ->
                httpCalls++
                if (!url.endsWith(".pdf")) { modelCalls++; throw AssertionError("metadata question attempted a model call") }
                object : HttpsURLConnection(URL(url)) {
                    override fun getResponseCode() = 200
                    override fun getInputStream() = ByteArrayInputStream(pdf)
                    override fun getContentLengthLong() = pdf.size.toLong()
                    override fun connect() = Unit
                    override fun disconnect() = Unit
                    override fun usingProxy() = false
                    override fun getCipherSuite() = "fixture-no-network"
                    override fun getLocalCertificates(): Array<Certificate>? = null
                    override fun getServerCertificates(): Array<Certificate> = emptyArray()
                }
            })
            set(service, "profile", config)
            set(service, "dialogueApi", api)
            set(service, "error", "上一次模型响应超时")
            say(service, "问这份手册适用于那的型号的设备")
            assertEquals("", field(service, "error"))
            assertEquals("received", (field(service, "interactionFeedback") as JSONObject).getString("stage"))
            say(service, "任务完成了吧？") // Progress queries deliberately do not cancel the pending manual answer.
            assertEquals("任务完成了吧", field(service, "lastUserText"))
            assertTrue((field(service, "lastReplyText") as String).contains("暂停"))
        }
    }

    @Test fun cancelledFeedbackCannotOverwriteOrFinishTheNextRequest() {
        fun begin(service: StandaloneService, id: String) {
            service.javaClass.getDeclaredMethod("beginFeedback", String::class.java, String::class.java, String::class.java)
                .apply { isAccessible = true }.invoke(service, id, "received", "哪个型号")
        }
        fun advance(service: StandaloneService, id: String, ticket: Long) {
            service.javaClass.getDeclaredMethod("advanceFeedback", String::class.java, Long::class.javaPrimitiveType, String::class.java)
                .apply { isAccessible = true }.invoke(service, id, ticket, "model")
        }
        fun finish(service: StandaloneService, id: String) {
            service.javaClass.getDeclaredMethod("finishFeedback", String::class.java, String::class.java)
                .apply { isAccessible = true }.invoke(service, id, "completed")
        }
        isolated(afterMain = { service, _ ->
            onMain {
                val feedback = field(service, "interactionFeedback") as JSONObject
                assertEquals("d-new", feedback.getString("request_id"))
                assertEquals("received", feedback.getString("stage"))
                val epoch = (field(service, "epoch") as java.util.concurrent.atomic.AtomicLong).get()
                advance(service, "d-new", epoch)
            }
            onMain {
                assertEquals("model", (field(service, "interactionFeedback") as JSONObject).getString("stage"))
                finish(service, "d-old")
                assertEquals("d-new", (field(service, "interactionFeedback") as JSONObject).getString("request_id"))
                finish(service, "d-new")
                assertEquals(0, (field(service, "interactionFeedback") as JSONObject).length())
            }
        }) { service, _ ->
            val epoch = field(service, "epoch") as java.util.concurrent.atomic.AtomicLong
            val chat = field(service, "chatTranscript") as ChatTranscript
            val oldTurn = chat.beginUser("旧问题")
            chat.beginUser("新问题")
            set(service, "dialogueTurnId", oldTurn)
            val old = epoch.get()
            begin(service, "d-old")
            advance(service, "d-old", old)
            call(service, "invalidate", Boolean::class.javaPrimitiveType!!, false)
            assertEquals("cancelled", chat.snapshot().getJSONObject(1).getString("state"))
            assertEquals("pending", chat.snapshot().getJSONObject(3).getString("state"))
            assertEquals(0, (field(service, "interactionFeedback") as JSONObject).length())
            begin(service, "d-new")
            advance(service, "d-new", old) // Matching id cannot make an old epoch current.
            finish(service, "d-old")
        }
    }

    private fun setupTask(service: StandaloneService): StandaloneEngine {
        set(service, "profile", StandaloneProfile("https://example.invalid/v1", "fixture-only-model-key", "fixture-planner", "fixture-monitor",
            toolsBaseUrl = "http://127.0.0.1:8766", toolsToken = "fixture-only-tools-token-0123456789", manualId = "tl-sg105-v4"))
        val engine = field(service, "engine") as StandaloneEngine
        engine.setGoal("隔离组件测试：保留原教室配置任务", true, SystemClock.elapsedRealtime())
        return engine
    }

    private fun say(service: StandaloneService, text: String) {
        set(service, "pendingRecognitionTrace", JSONObject().put("automatic", true))
        call(service, "onText", String::class.java, text)
    }

    private fun assertTaskUnchanged(before: JSONObject, after: JSONObject) {
        assertEquals(before.getString("task_instance_id"), after.getString("task_instance_id"))
        assertEquals(before.getJSONObject("goal").toString(), after.getJSONObject("goal").toString())
        assertEquals(before.getLong("revision"), after.getLong("revision"))
        assertEquals(before.getString("phase"), after.getString("phase"))
        assertEquals(before.opt("paused_reason"), after.opt("paused_reason"))
    }

    private fun events(dir: File): List<JSONObject> = File(dir, "events.jsonl").let { file ->
        if (file.isFile) file.readLines().filter(String::isNotBlank).map(::JSONObject) else emptyList()
    }

    private fun isolated(afterMain: ((StandaloneService, File) -> Unit)? = null, block: (StandaloneService, File) -> Unit) {
        val root = File(target.cacheDir, "classroom-runtime-${UUID.randomUUID()}").apply { check(mkdir()) }
        val dir = File(root, "run-fixture").apply { check(mkdir()) }
        val context = object : ContextWrapper(target) {
            override fun getFilesDir(): File = root
        }
        var service: StandaloneService? = null
        val production = onMain { StandaloneService.instance }
        try {
            onMain {
                val isolated = StandaloneService()
                ContextWrapper::class.java.getDeclaredMethod("attachBaseContext", Context::class.java)
                    .apply { isAccessible = true }.invoke(isolated, context)
                service = isolated
                assertEquals(root.canonicalFile, isolated.filesDir.canonicalFile)
                set(isolated, "directory", dir)
                set(isolated, "lastFolded", false)
                set(isolated, "lastPresence", null)
                for (name in listOf("api", "dialogueApi", "voice", "speech", "sharedMicrophone", "videoRecorder", "screenRecorder"))
                    assertNull("isolated fixture must not own $name", field(isolated, name))
                block(isolated, dir)
                assertSame("fixture must never replace the production service singleton", production, StandaloneService.instance)
            }
            afterMain?.invoke(service!!, dir)
        } finally {
            onMain {
                service?.let {
                    set(it, "closed", true)
                    (field(it, "dialogueApi") as? StandaloneApi)?.close()
                    (field(it, "main") as Handler).removeCallbacksAndMessages(null)
                    (field(it, "worker") as ExecutorService).shutdownNow()
                    (field(it, "videoPublisher") as ExecutorService).shutdownNow()
                }
            }
            check(root.canonicalFile.parentFile == target.cacheDir.canonicalFile)
            root.deleteRecursively()
        }
    }

    private fun field(receiver: Any, name: String): Any? = receiver.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.get(receiver)

    private fun set(receiver: Any, name: String, value: Any?) = receiver.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.set(receiver, value)

    private fun call(service: StandaloneService, name: String, type: Class<*>, value: Any) {
        try { service.javaClass.getDeclaredMethod(name, type).apply { isAccessible = true }.invoke(service, value) }
        catch (failure: InvocationTargetException) { throw failure.targetException }
    }

    private fun <T> onMain(block: () -> T): T {
        val task = FutureTask(Callable<T> { block() })
        instrumentation.runOnMainSync(task)
        return try { task.get() } catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }
}
