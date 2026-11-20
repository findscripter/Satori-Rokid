package com.satori.glasses.capture

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.provider.Settings
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import java.util.concurrent.ExecutionException
import java.util.concurrent.FutureTask

/** One explicitly authorized ordinary-app probe, not a production firmware adapter.
 * The harness must freshly verify global idle and no vendor tmp before opting in. The vendor has
 * no session-scoped stop, so this test sends only ON and leaves termination to its bounded timer.
 */
@RunWith(AndroidJUnit4::class)
class RokidNativePairProbeDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext
    private val now get() = SystemClock.elapsedRealtime()
    private val vendor = ComponentName("com.rokid.os.master.screenstream",
        "com.rokid.os.master.screenstream.receiver.ScreenRecordReceiver")
    private val taskKeys = listOf("task_instance_id", "goal", "phase", "revision", "paused_reason",
        "active_desire_id", "active_step_id", "desires", "pending")

    @Suppress("DEPRECATION")
    @Test(timeout = 110_000) fun ordinaryAppRecordsFirstPersonAlongsideNativeScreen() {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue("use -e satoriNativePairProbe true for this real camera/microphone and system-screen probe",
            arguments.getString("satoriNativePairProbe") == "true")
        val id = "${System.currentTimeMillis()}-${UUID.randomUUID()}"
        val screenName = "satori-acceptance-$id.mp4"
        val screenPath = "/sdcard/ScreenRecorder/$screenName"
        val samples = JSONArray()
        val report = JSONObject().put("schema", "satori.rokid-native-pair-probe/1")
            .put("source_control", "instrumented_app_not_real_remote")
            .put("physical_task_success", false).put("media_verification", "actual media files still require decode")
            .put("production_firmware_adapter_implemented", false).put("presence_simulated", false)
            .put("network_simulated", false).put("vendor_stop_sent", false)
            .put("vendor_start_sent", false).put("expectedVendorScreenPath", screenPath)
            .put("vendor_max_time_extra_ms", 8_000L).put("samples", samples).put("passed", false)
        var service: StandaloneService? = null
        var currentVideoPath: String? = null
        var beforeRecording: JSONObject? = null
        var vendorDeadline = 0L
        var failure: Throwable? = null
        var stage = "preflight"
        try {
            val uid = Process.myUid()
            report.put("caller_uid", uid).put("target_uid", target.applicationInfo.uid)
                .put("vendor_component", vendor.flattenToString()).put("preflight_at", clock())
            assertEquals("must run as the target app UID", target.applicationInfo.uid, uid)
            assertTrue("must be an ordinary app UID", uid % 100_000 >= 10_000)
            val info = target.packageManager.getReceiverInfo(vendor, PackageManager.MATCH_DISABLED_COMPONENTS)
            report.put("receiver", JSONObject().put("enabled", info.enabled).put("exported", info.exported)
                .put("application_enabled", info.applicationInfo.enabled).put("uid", info.applicationInfo.uid)
                .put("permission", info.permission ?: JSONObject.NULL))
            assertEquals(vendor.packageName, info.packageName)
            assertEquals(vendor.className, info.name)
            assertTrue("vendor receiver must already be enabled and exported", info.enabled && info.applicationInfo.enabled && info.exported)
            assertNull("this probe only covers the audited receiver without a sender permission", info.permission)
            for (permission in listOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
                assertEquals("permission must already be granted: $permission", PackageManager.PERMISSION_GRANTED,
                    target.checkSelfPermission(permission))
            // shortcut: ordinary-app scoped storage cannot prove tmp absence; use only a fresh controlled harness preflight.
            assertEquals("harness must freshly verify no vendor tmp before this one probe", "true",
                arguments.getString("preflightNoVendorTmp"))
            report.put("preflight_no_vendor_tmp", true).put("tmp_check_source", "external_read_only_harness")
            assertEquals("vendor must be idle before starting anything", 0, vendorState())
            assertFalse("never overwrite a visible existing output", File(screenPath).exists())
            requireNoImports()
            val initial = main {
                assertNull("starting Standalone must not stop an existing CaptureService", CaptureService.instance)
                StandaloneService.instance
            }
            val original = initial?.let { main { it.snapshot() } } ?: readSavedTask()
            requirePausedTask(original)
            if (initial != null) requireIdleRecorders(original)
            report.put("original_task", fields(original, taskKeys))

            stage = "start_real_activity"
            requireNoImports()
            instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            await("production service did not start", 8_000) { main { StandaloneService.instance != null } }
            val live = main { StandaloneService.instance!! }
            service = live
            if (initial != null) assertSame("must retain the pre-existing service", initial, live)
            val before = main { live.snapshot() }
            beforeRecording = fields(before, taskKeys)
            report.put("before_recording", snapshot(before))
            requirePausedTask(before)
            requireIdleRecorders(before)
            assertEquals("the real glasses must already be unfolded", true, before.opt("legs_open"))
            assertTrue("production logging must be healthy", before.getBoolean("data_logging_ok"))
            for (key in taskKeys - "revision")
                assertEquals("startup changed original task field $key", original.opt(key)?.toString(), before.opt(key)?.toString())

            stage = "start_first_person"
            main {
                val fresh = live.snapshot()
                requireIdleRecorders(fresh)
                assertTaskUnchanged(beforeRecording!!, fresh)
                try { live.command("record_start") } finally {
                    currentVideoPath = live.snapshot().getJSONObject("video").optString("path").takeIf { it.isNotBlank() }
                }
            }
            val path = checkNotNull(currentVideoPath) { "production record_start did not create a video session" }
            report.put("currentVideoPath", path).put("first_person_requested_at", clock())
            assertEquals(File(target.filesDir, "first-person-videos").canonicalFile, File(path).canonicalFile.parentFile)
            await("first-person recorder did not become ready", 12_000) {
                val video = main { live.snapshot().getJSONObject("video") }
                assertEquals(path, video.getString("path"))
                assertNotEquals("first-person startup failed", "error", video.getString("state"))
                video.getString("state") == "recording" && video.optLong("video_input_frames") >= 3 && video.optLong("audio_samples") > 0
            }
            val ready = main { live.snapshot() }
            val readyFrames = ready.getJSONObject("video").getLong("video_input_frames")
            await("first-person frames did not grow before vendor start", 3_000) {
                main { live.snapshot().getJSONObject("video").optLong("video_input_frames") > readyFrames }
            }

            stage = "start_vendor_screen"
            requireNoImports()
            assertEquals("vendor became busy; never send a competing ON", 0, vendorState())
            assertFalse("never overwrite a visible existing output", File(screenPath).exists())
            sample(samples, "before_vendor_on", live)
            val intent = Intent("com.rokid.yodaos.action.SCREENRECORD_ON").setComponent(vendor)
                .putExtra("FileName", screenName).putExtra("MaxTime", 8_000L)
            assertEquals(vendor, intent.component)
            assertTrue(screenName.matches(Regex("satori-acceptance-[0-9]+-[a-f0-9-]+\\.mp4")))
            vendorDeadline = now + 45_000
            report.put("vendor_send_at", clock())
            target.sendBroadcast(intent) // SecurityException is retained and rethrown, without privilege fallback.
            report.put("vendor_start_sent", true)
            await("vendor never reported recording", 6_000) {
                requireOwnRecording(live, path, beforeRecording!!)
                vendorState() == 1
            }
            report.put("vendor_on_observed_at", clock())
            var checkpoint = main { live.snapshot().getJSONObject("video") }
            val firstConcurrent = JSONObject(checkpoint.toString())
            var checkpointAt = now
            var nextSample = 0L
            sample(samples, "vendor_on_observed", live)
            stage = "wait_vendor_automatic_stop"
            while (vendorState() == 1 && now < vendorDeadline) {
                val current = requireOwnRecording(live, path, beforeRecording!!)
                if (now >= nextSample) { sample(samples, "concurrent", live); nextSample = now + 1_000 }
                if (now - checkpointAt >= 3_000) {
                    assertMediaGrowing(checkpoint, current)
                    checkpoint = current
                    checkpointAt = now
                }
                SystemClock.sleep(100)
            }
            assertEquals("vendor did not automatically return to idle within 45 seconds; never send OFF", 0, vendorState())
            report.put("vendor_off_observed_at", clock())
            val afterConcurrent = requireOwnRecording(live, path, beforeRecording!!)
            assertMediaGrowing(firstConcurrent, afterConcurrent)
            report.put("concurrent_video_input_frames", afterConcurrent.getLong("video_input_frames") - firstConcurrent.getLong("video_input_frames"))
                .put("concurrent_audio_samples", afterConcurrent.getLong("audio_samples") - firstConcurrent.getLong("audio_samples"))
            sample(samples, "vendor_off_observed", live)

            stage = "save_own_first_person"
            stopOwnRecording(live, path)
            val finished = main { live.snapshot() }
            assertTaskUnchanged(beforeRecording!!, finished)
            report.put("after_recording", snapshot(finished)).put("task_preserved_during_recording", true)
            val video = finished.getJSONObject("video")
            assertTrue("first-person finalization must succeed", video.getBoolean("finalized"))
            assertTrue("first-person file must remain available", File(path).isFile && File(path).length() > 0)
            report.put("first_person_file_bytes", File(path).length()).put("passed", true)
        } catch (caught: Throwable) {
            failure = caught
            report.put("failure_type", caught.javaClass.name)
        } finally {
            report.put("last_stage", stage)
            fun clean(block: () -> Unit) {
                try { block() } catch (cleanup: Throwable) {
                    report.put("cleanup_failure_type", cleanup.javaClass.name)
                    if (failure == null) failure = cleanup else failure!!.addSuppressed(cleanup)
                    report.put("passed", false)
                }
            }
            val live = service
            if (live != null && currentVideoPath != null) clean { stopOwnRecording(live, currentVideoPath!!) }
            // Even after a failure, observe only until the existing deadline; never stop a global vendor session.
            if (report.optBoolean("vendor_start_sent")) clean {
                while (vendorState() == 1 && now < vendorDeadline) SystemClock.sleep(200)
                report.put("vendor_final_state", vendorState()).put("vendor_final_observed_at", clock())
            }
            if (live != null) clean {
                val last = main { live.snapshot() }
                report.put("last_app_snapshot", snapshot(last))
                beforeRecording?.let { assertTaskUnchanged(it, last) }
            }
            clean {
                val folder = File(target.filesDir, "diagnostics").apply { check(mkdirs() || isDirectory) }
                val destination = File(folder, "rokid-native-pair-$id.json")
                check(destination.createNewFile())
                destination.writeText(report.toString(2))
                instrumentation.sendStatus(0, Bundle().apply { putString("rokid_native_pair_report", destination.absolutePath) })
                println("rokid_native_pair_report=${destination.absolutePath}")
            }
        }
        failure?.let { throw it }
    }

    private fun requireNoImports() {
        for (name in listOf("standalone-import.json", "standalone-tools-import.json"))
            assertFalse("never consume a pending private import: $name", File(target.filesDir, name).exists())
    }

    private fun readSavedTask(): JSONObject {
        val file = File(target.filesDir, "standalone-state.json")
        assertTrue("an existing paused task is required", file.isFile)
        return JSONObject(file.readText())
    }

    private fun requirePausedTask(state: JSONObject) {
        assertTrue("preserve an existing real task", state.optString("task_instance_id").isNotBlank() && state.optJSONObject("goal") != null)
        assertEquals("task must already be explicitly paused", "user", state.optString("paused_reason"))
        assertNotEquals("restoring this phase would change pending task state", "awaiting_step_confirmation", state.optString("phase"))
    }

    private fun requireIdleRecorders(state: JSONObject) {
        for (key in listOf("video", "screen")) assertEquals("existing $key must be idle", "idle", state.getJSONObject(key).getString("state"))
    }

    private fun assertTaskUnchanged(before: JSONObject, after: JSONObject) {
        for (key in taskKeys) assertEquals("task field changed: $key", before.opt(key)?.toString(), after.opt(key)?.toString())
    }

    private fun requireOwnRecording(service: StandaloneService, path: String, task: JSONObject): JSONObject = main {
        assertSame("production service was replaced", service, StandaloneService.instance)
        val state = service.snapshot()
        assertTaskUnchanged(task, state)
        assertEquals(true, state.opt("legs_open"))
        assertEquals("idle", state.getJSONObject("screen").getString("state"))
        state.getJSONObject("video").also {
            assertEquals("never claim another video session", path, it.getString("path"))
            assertEquals("first-person recording stopped during vendor screen capture", "recording", it.getString("state"))
        }
    }

    private fun stopOwnRecording(service: StandaloneService, path: String) {
        main {
            assertSame("never stop a replacement service", service, StandaloneService.instance)
            val video = service.snapshot().getJSONObject("video")
            assertEquals("never stop another video session", path, video.getString("path"))
            if (video.optString("state") in setOf("starting", "recording")) service.command("record_stop")
        }
        await("own first-person recording did not save", 12_000) {
            val video = main { service.snapshot().getJSONObject("video") }
            assertEquals(path, video.getString("path"))
            assertNotEquals("own first-person recording failed", "error", video.getString("state"))
            video.getString("state") == "saved"
        }
    }

    private fun assertMediaGrowing(before: JSONObject, after: JSONObject) {
        for (key in listOf("video_input_frames", "frames", "audio_samples"))
            assertTrue("$key stopped growing while native screen was recording", after.getLong(key) > before.getLong(key))
    }

    private fun vendorState() = Settings.Global.getInt(target.contentResolver, "rokid_yodaos_screen_record", -1)
    private fun clock() = JSONObject().put("unix_ms", System.currentTimeMillis()).put("elapsed_ns", SystemClock.elapsedRealtimeNanos())
    private fun fields(source: JSONObject, keys: List<String>) = JSONObject().apply {
        for (key in keys) if (source.has(key)) put(key, source.get(key))
    }
    private fun snapshot(source: JSONObject) = fields(source, taskKeys + listOf("worn", "legs_open", "capturing",
        "frames", "network", "busy", "listening", "voice_busy", "speech_busy", "data_logging_ok", "run", "version",
        "video", "screen", "microphone"))
    private fun sample(samples: JSONArray, label: String, service: StandaloneService) {
        samples.put(clock().put("label", label).put("vendor_state", vendorState())
            .put("app", main { snapshot(service.snapshot()) }))
    }
    private fun await(message: String, timeoutMs: Long, condition: () -> Boolean) {
        val end = now + timeoutMs
        while (!condition()) {
            assertTrue(message, now < end)
            SystemClock.sleep(100)
        }
    }
    private fun <T> main(block: () -> T): T {
        val task = FutureTask<T> { block() }
        instrumentation.runOnMainSync(task)
        return try { task.get() } catch (failure: ExecutionException) { throw (failure.cause ?: failure) }
    }
}
