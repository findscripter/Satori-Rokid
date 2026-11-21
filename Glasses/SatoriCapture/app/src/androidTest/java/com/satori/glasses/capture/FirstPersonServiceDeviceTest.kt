package com.satori.glasses.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.os.Handler
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
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Explicitly selected LOCAL real-camera/microphone sample. Test presence is simulated and JPEG/flow
 * demand is forced while the model is idle. This is not a human-wearing/subjective-audibility test.
 * All HTTPS is disabled before capture; actual images/audio stay in app-private recording/evidence
 * files and the production local MediaStore export. No goal or completion decision is fabricated.
 */
@RunWith(AndroidJUnit4::class)
class FirstPersonServiceDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext
    private val now get() = SystemClock.elapsedRealtime()
    private val forbiddenHttp = AtomicInteger()
    private lateinit var service: StandaloneService
    private lateinit var evidence: File
    private lateinit var camera: CameraStreamer
    private lateinit var microphone: SharedMicrophone
    private lateinit var voice: StandaloneVoice
    private var clipDeadline = 0L
    private var nextSnapshot = 0L
    private val acousticPhrase = "这是第一视角有声录像，本机环境收音测试。"

    @Test(timeout = 90_000) fun realServiceRecordsCameraAndEnvironmentWhileSharingPttAndAnalyzingJpegs() {
        val selected = InstrumentationRegistry.getArguments().getString("class").orEmpty().split(',').map { it.trim().substringBefore('#') }
        assumeTrue("real camera/microphone acceptance requires explicit class/method selection", javaClass.name in selected)
        val originals = preserveState()
        val videoPrefs = target.getSharedPreferences("video_recordings", 0)
        val originalPrefs = HashMap(videoPrefs.all)
        evidence = File(target.filesDir, "first-person-service-tests/sample-${System.currentTimeMillis()}")
        check(evidence.mkdirs())
        val summary = JSONObject().put("kind", "real_camera_shared_microphone_service_integration")
            .put("presence_simulated", true).put("human_wearing_verified", false).put("audibility_verified", false)
            .put("vision_api_disabled", true).put("simulated_analysis_demand", true).put("cloud_audio_disabled", true)
        var activity: Activity? = null
        var publisher: ExecutorService? = null
        var passed = false
        try {
            main { target.stopService(Intent(target, StandaloneService::class.java)) }
            await("original idle service did not stop", 8_000) { main { StandaloneService.instance == null } }
            originals.keys.drop(1).forEach { check(!it.exists() || it.delete()) }
            originals.keys.first().writeText(StandaloneEngine().snapshot(now).toString())
            activity = instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            await("production foreground service did not start", 8_000) { main { StandaloneService.instance != null } }
            service = main { StandaloneService.instance!! }
            preparePrivateLocalCapture()
            publisher = read(service, "videoPublisher")
            await("production local speech backend did not bind", 7_000) {
                main { val speech = read<StandaloneSpeech>(service, "speech"); read<StandaloneVendorSpeech>(speech, "vendor").ready || read<Boolean>(speech, "local") }
            }
            main { service.command("record_start"); camera.setAnalysisEnabled(true) }
            val requested = main { service.snapshot() }.getJSONObject("video")
            assertTrue("production record_start did not accept the request", read<FirstPersonRecorder>(service, "videoRecorder").wantsFrames)
            val epochMs = requested.getLong("start_elapsed_ns") / 1_000_000
            val path = requested.getString("path")
            summary.put("path", path).put("requested_at_ms", epochMs)
            note("record_start", main { service.snapshot() })
            await("real camera/AAC recorder did not become ready", 7_000) {
                main { service.snapshot().getJSONObject("video").optString("state") == "recording" }
            }
            val recordingReadyAt = now
            clipDeadline = recordingReadyAt + 15_000
            summary.put("recording_ready_at_ms", recordingReadyAt).put("startup_wait_ms", recordingReadyAt - epochMs)
                .put("sample_window", "13 seconds after real recording state, excluding camera cold-start")
            assertEquals(1, microphone.subscriberCount)
            assertEquals("first recording must start one hardware source", 1L, microphone.sourceStarts)
            val initialFrames = recorderStats().getLong("video_input_frames")
            main { service.command("pause"); camera.setAnalysisEnabled(true) }
            assertTrue("pausing the AI task must not stop independent video", read<FirstPersonRecorder>(service, "videoRecorder").wantsFrames)
            await("video frames stopped after task pause", 1_500) { recorderStats().getLong("video_input_frames") >= initialFrames + 3 }
            note("video_survived_task_pause", main { service.snapshot() })
            val afterPause = recorderStats().getLong("video_input_frames")
            main { service.command("stop"); camera.setAnalysisEnabled(true) }
            assertTrue("stopping the AI task must not stop independent video", read<FirstPersonRecorder>(service, "videoRecorder").wantsFrames)
            await("video frames stopped after task stop", 1_500) { recorderStats().getLong("video_input_frames") >= afterPause + 3 }
            note("video_survived_task_stop", main { service.snapshot() })

            // The real local speaker supplies a known non-private acoustic signal for raw-channel capture.
            main { read<StandaloneSpeech>(service, "speech").speak(acousticPhrase) }
            main { voice.start() }
            await("PTT did not attach to the existing shared microphone", 5_000) { microphone.subscriberCount == 2 }
            assertEquals("PTT must not open a second AudioRecord source", 1L, microphone.sourceStarts)
            note("ptt_and_video_shared_source", microphone.stats())
            val beforePtt = recorderStats().getLong("video_input_frames")
            val holdUntil = now + 1_500
            while (now < holdUntil && now < clipDeadline) { sample(); SystemClock.sleep(50) }
            main { voice.stop() }
            await("PTT stop must retain only the video subscription", 2_000) { microphone.subscriberCount == 1 }
            assertEquals(1L, microphone.sourceStarts)
            assertTrue("PTT must not interrupt camera recording", recorderStats().getLong("video_input_frames") > beforePtt)
            note("ptt_released_video_continues", microphone.stats())
            await("offline PTT recognition did not settle", 3_000) { !voice.busy }
            assertEquals("local recognition must not dispatch a new user goal", "stopped", main { service.snapshot() }.getString("phase"))

            while (now < recordingReadyAt + 13_000 && now < clipDeadline) {
                assertRecordingHealthy()
                sample()
                SystemClock.sleep(50)
            }
            assertTrue("sample exceeded the requested 10-15 second window", now <= clipDeadline)
            assertTrue("known local acoustic phrase did not finish while the microphone was recording", acousticFixturePlayed())
            main { service.command("record_stop") }
            clipDeadline = 0 // finalization and media verification have their own bounded deadlines
            await("production service did not receive a successful video_saved", 10_000) {
                val state = recorderStats().optString("state")
                if (state == "error") fail("real service recorder failed: ${recorderStats().optString("reason")}")
                state == "saved" && events().any { it.optString("name") == "video_saved" && it.optJSONObject("data")?.optString("path") == path }
            }
            val finished = recorderStats()
            assertTrue(finished.getBoolean("finalized"))
            assertTrue("effective recording duration must exclude camera preheat", finished.getLong("captured_duration_ms") in 10_000..15_500)
            assertTrue("real sensor stream must encode a sustained clip", finished.getLong("frames") >= 90)
            assertTrue("real environment PCM must reach the AAC encoder", finished.getLong("audio_samples") >= 150_000)
            assertEquals("PTT/video source must be reused throughout", 1L, microphone.sourceStarts)
            await("all subscriptions must release after video stop", 2_000) { microphone.subscriberCount == 0 }
            val file = File(path)
            assertTrue(file.isFile && file.length() > 10_000)
            val sidecar = JSONObject(File(file.parentFile, file.nameWithoutExtension + ".json").readText())
            assertTrue(sidecar.getBoolean("finalized"))
            val verification = verifyMedia(file, finished)
            val jpegLines = File(evidence, "frames.jsonl").takeIf { it.isFile }?.readLines().orEmpty()
            val jpegFrames = jpegLines.map { JSONObject(it) }
            assertTrue("real JPEG analysis must run concurrently with video", jpegFrames.size >= 3)
            assertTrue("real JPEG frames must run through the optical-flow gate", jpegFrames.any {
                it.optJSONObject("flow")?.optBoolean("available") == true && it.optJSONObject("flow")?.optString("method") == "lk_ransac_farneback"
            })
            assertTrue("no visual/model request may be issued", events().none { it.optString("name") == "request_start" })
            assertEquals("no recorded image or audio may leave the device", 0, forbiddenHttp.get())
            assertTrue("production local TTS must actually report completed playback", events().any { entry ->
                entry.optString("name") == "audio_done" && entry.optJSONObject("data")?.let { data ->
                    data.optString("provider") == "rokid_local" || data.optString("source") == "local_tts"
                } == true
            })
            summary.put("video", finished).put("media_verification", verification).put("microphone", microphone.stats())
                .put("jpeg_frames", jpegFrames.size).put("jpeg_counter", camera.framesEncoded).put("http_requests", forbiddenHttp.get())
                .put("known_local_acoustic_phrase_played_during_recording", true)
                .put("analysis_note", "camera JPEG/flow enabled manually to simulate AI analysis demand; model remained idle/stopped")
            passed = true
            println(JSONObject().put("first_person_service_verified", true).put("path", path)
                .put("duration_ms", finished.getLong("duration_ms")).put("captured_duration_ms", finished.getLong("captured_duration_ms"))
                .put("frames", finished.getLong("frames"))
                .put("jpeg_frames", jpegFrames.size).put("audio_rms", verification.getDouble("audio_rms")).toString())
        } catch (failure: Throwable) {
            summary.put("failure_type", failure.javaClass.simpleName)
                .put("failure", if (failure is AssertionError) failure.message.orEmpty().take(300) else "local service media verification failed")
            throw failure
        } finally {
            clipDeadline = 0
            summary.put("passed", passed).put("forbidden_http_attempts", forbiddenHttp.get())
            if (::service.isInitialized) runCatching { summary.put("last_snapshot", main { service.snapshot() }) }
            // Best-effort pre-cleanup checkpoint; an evidence I/O failure must not leave sensors running.
            runCatching { File(evidence, "service-media-verification.json").writeText(summary.toString(2)) }
            try {
                if (::service.isInitialized) main {
                    if (::voice.isInitialized) runCatching { voice.cancel() }
                    runCatching { service.command("record_stop") }
                }
                main { target.stopService(Intent(target, StandaloneService::class.java)); activity?.finish() }
                await("foreground test service did not stop", 8_000) { main { StandaloneService.instance == null } }
                publisher?.awaitTermination(8, TimeUnit.SECONDS)
                instrumentation.waitForIdleSync() // finish any publisher preference update before restoring preferences
                val published = videoPrefs.getString("last_uri", null)
                summary.put("published_uri", published ?: JSONObject.NULL)
                File(evidence, "service-media-verification.json").writeText(summary.toString(2))
            } finally {
                originals.forEach { (file, bytes) -> if (bytes == null) check(!file.exists() || file.delete()) else file.writeBytes(bytes) }
                restorePrefs(videoPrefs, originalPrefs)
            }
        }
    }

    private fun preparePrivateLocalCapture() = main {
        val handler = read<Handler>(service, "main")
        handler.removeCallbacks(read(service, "tick"))
        read<GlassesInputReceiver?>(service, "receiver")?.let { service.unregisterReceiver(it) }
        write(service, "receiver", null)
        read<StandaloneApi?>(service, "api")?.close()
        write(service, "api", null)
        read<StandaloneSpeech?>(service, "speech")?.close()
        read<StandaloneVoice?>(service, "voice")?.close()
        read<StandaloneApi?>(service, "speechApi")?.close()
        read<StandaloneApi?>(service, "voiceApi")?.close()
        val dummy = StandaloneProfile("https://example.invalid/v1", "local-media-test-no-real-credential", "disabled", "disabled")
        val deniedSpeech = StandaloneApi(dummy) { forbiddenHttp.incrementAndGet(); throw ApiFailure(message = "local-only media test") }
        val deniedVoice = StandaloneApi(dummy) { forbiddenHttp.incrementAndGet(); throw ApiFailure(message = "local-only media test") }
        val event = service.javaClass.getDeclaredMethod("event", String::class.java, JSONObject::class.java).apply { isAccessible = true }
        microphone = read(service, "sharedMicrophone")
        camera = read(service, "camera")
        write(service, "speechApi", deniedSpeech)
        write(service, "voiceApi", deniedVoice)
        write(service, "speech", StandaloneSpeech(service, deniedSpeech) { name, data -> event.invoke(service, name, data) })
        voice = StandaloneVoice(service, deniedVoice,
            { text -> note("ptt_text_intercepted_to_protect_idle_state", JSONObject().put("text_length", text.length)) },
            { _ -> note("ptt_result", JSONObject().put("recognized", false)) },
            { _ -> note("ptt_offline_trace", JSONObject().put("received", true)) }, microphone)
        write(service, "voice", voice)
        write(service, "directory", evidence)
        write(service, "worn", true); write(service, "legsOpen", true); write(service, "lastPresence", null)
        write(service, "lastFolded", false)
        service.javaClass.getDeclaredMethod("applyPresence").apply { isAccessible = true }.invoke(service)
        handler.removeCallbacks(read(service, "tick"))
    }

    private fun assertRecordingHealthy() {
        val state = recorderStats()
        assertEquals("production recording ended before record_stop", "recording", state.optString("state"))
        assertEquals(1, microphone.subscriberCount)
        assertEquals(1L, microphone.sourceStarts)
        assertEquals(0, forbiddenHttp.get())
        assertEquals("stopped", main { service.snapshot() }.getString("phase"))
    }

    private fun sample() {
        if (::service.isInitialized && now >= nextSnapshot) {
            note("snapshot", main { service.snapshot() })
            nextSnapshot = now + 1_000
        }
    }
    private fun recorderStats(): JSONObject = main { service.snapshot().getJSONObject("video") }
    private fun events(): List<JSONObject> = File(evidence, "events.jsonl").takeIf { it.isFile }?.readLines()
        ?.mapNotNull { runCatching { JSONObject(it) }.getOrNull() } ?: emptyList()
    private fun acousticFixturePlayed(): Boolean {
        val recorded = events()
        val ids = recorded.filter { it.optString("name") == "speech_requested" && it.optJSONObject("data")?.optString("text") == acousticPhrase }
            .mapNotNull { it.optJSONObject("data")?.optString("id") }.toSet()
        return ids.any { id ->
            recorded.any { it.optString("name") == "audio_started" && it.optJSONObject("data")?.optString("id") == id } &&
                recorded.any { it.optString("name") == "audio_done" && it.optJSONObject("data")?.optString("id") == id &&
                    it.optJSONObject("data")?.let { data -> data.optString("provider") == "rokid_local" || data.optString("source") == "local_tts" } == true }
        }
    }
    private fun note(name: String, data: JSONObject) = synchronized(this) {
        File(evidence, "service-test.jsonl").appendText(JSONObject().put("at_ms", now).put("name", name).put("data", data).toString() + "\n")
    }

    private fun preserveState(): Map<File, ByteArray?> {
        assumeTrue("legacy capture must not be interrupted", main { CaptureService.instance == null })
        val existing = main { StandaloneService.instance?.snapshot() }
        val originalService = main { StandaloneService.instance }
        if (existing != null) {
            assumeTrue("an active user task must be preserved", safeState(existing))
            assumeTrue("an existing recording must be preserved", existing.optJSONObject("video")?.optString("state") !in listOf("starting", "recording", "finalizing"))
            assumeTrue("user voice capture must be preserved", !existing.optBoolean("recording") && !existing.optBoolean("busy"))
            assumeTrue("pending user speech recognition must be preserved", originalService == null ||
                main { read<StandaloneVoice?>(originalService, "voice")?.busy != true })
        }
        assumeTrue("pending private profile import must be preserved", !File(target.filesDir, "standalone-import.json").exists())
        assumeTrue("camera permission must be granted by install harness", target.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        assumeTrue("microphone permission must be granted by install harness", target.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        val state = File(target.filesDir, "standalone-state.json")
        return listOf(state, File(state.path + ".bak"), File(state.path + ".new")).associateWith { file ->
            if (!file.exists()) null else file.readBytes().also { bytes ->
                val saved = runCatching { JSONObject(String(bytes, Charsets.UTF_8)) }.getOrNull()
                assumeTrue("unreadable original state must be preserved", saved != null)
                assumeTrue("saved active user task must be preserved", safeState(saved!!))
            }
        }
    }
    private fun safeState(state: JSONObject): Boolean = state.optString("phase") in listOf("idle", "stopped", "complete")

    private fun restorePrefs(prefs: SharedPreferences, values: Map<String, *>) {
        val edit = prefs.edit().clear()
        values.forEach { (key, value) -> when (value) {
            is String -> edit.putString(key, value)
            is Int -> edit.putInt(key, value)
            is Long -> edit.putLong(key, value)
            is Boolean -> edit.putBoolean(key, value)
            is Float -> edit.putFloat(key, value)
            is Set<*> -> { @Suppress("UNCHECKED_CAST") val strings = value as Set<String>; edit.putStringSet(key, strings) }
        } }
        check(edit.commit())
    }

    private fun verifyMedia(file: File, recording: JSONObject): JSONObject {
        val result = JSONObject()
        val extractor = MediaExtractor()
        val bounds = linkedMapOf<String, Pair<Long, Long>>()
        var audioFormat: MediaFormat? = null
        var audioTrack = -1
        try {
            extractor.setDataSource(file.absolutePath)
            assertEquals("real sample must have video and audio", 2, extractor.trackCount)
            for (index in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(index)
                val mime = format.getString(MediaFormat.KEY_MIME).orEmpty()
                val type = if (mime.startsWith("video/")) "video" else "audio"
                if (type == "video") {
                    assertEquals(MediaFormat.MIMETYPE_VIDEO_AVC, mime)
                    assertEquals(recording.getInt("width"), format.getInteger(MediaFormat.KEY_WIDTH))
                    assertEquals(recording.getInt("height"), format.getInteger(MediaFormat.KEY_HEIGHT))
                } else {
                    assertEquals(MediaFormat.MIMETYPE_AUDIO_AAC, mime)
                    assertEquals(16_000, format.getInteger(MediaFormat.KEY_SAMPLE_RATE))
                    assertEquals(1, format.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
                    audioFormat = format; audioTrack = index
                }
                extractor.selectTrack(index); extractor.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                var first = -1L; var last = -1L; var samples = 0
                while (extractor.sampleTime >= 0) {
                    val pts = extractor.sampleTime
                    assertTrue("track PTS must increase strictly", pts > last)
                    if (first < 0) first = pts
                    last = pts; samples++
                    if (!extractor.advance()) break
                }
                assertTrue(samples > 0)
                bounds[type] = first to last
                result.put("${type}_first_pts_us", first).put("${type}_last_pts_us", last).put("${type}_packets", samples)
                extractor.unselectTrack(index)
            }
            val video = bounds.getValue("video"); val audio = bounds.getValue("audio")
            assertTrue("container video must begin near zero without a camera warm-up hole", abs(video.first) <= 20_000)
            result.put("start_coverage_gap_us", abs(video.first - audio.first)).put("end_coverage_gap_us", abs(video.second - audio.second))
            // Initial camera bring-up may precede its first image; both tracks use the same elapsed clock.
            assertTrue("camera/audio startup coverage gap is too large", abs(video.first - audio.first) <= 1_000_000)
            assertTrue("camera/audio end timestamps diverged", abs(video.second - audio.second) <= 400_000)
            assertNotNull(audioFormat)
            extractor.selectTrack(audioTrack); extractor.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
            result.put("audio_rms", decodedAudioRms(extractor, audioFormat!!))
            assertTrue("environment microphone track must contain non-silent decoded audio", result.getDouble("audio_rms") > 40.0)
        } finally { extractor.release() }
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(file.absolutePath)
            val containerDuration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
                ?: throw AssertionError("final MP4 must report a container duration")
            result.put("container_duration_ms", containerDuration)
                .put("captured_duration_ms", recording.getLong("captured_duration_ms"))
            assertTrue("container retained camera warm-up instead of the effective recording duration",
                abs(containerDuration - recording.getLong("captured_duration_ms")) <= 750)
            assertEquals(recording.getInt("rotation").toString(), retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION))
            result.put("rotation", retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION))
            val image = retriever.getFrameAtTime(1_000_000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            assertNotNull("real camera MP4 must decode a frame", image)
            image?.let { result.put("decoded_width", it.width).put("decoded_height", it.height); it.recycle() }
        } finally { retriever.release() }
        return result
    }

    private fun decodedAudioRms(extractor: MediaExtractor, format: MediaFormat): Double {
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        try {
            codec.configure(format, null, null, 0); codec.start()
            val info = MediaCodec.BufferInfo()
            var inputEos = false; var outputEos = false; var square = 0.0; var samples = 0L
            val end = now + 10_000
            while (!outputEos && now < end) {
                if (!inputEos) {
                    val index = codec.dequeueInputBuffer(0)
                    if (index >= 0) {
                        val input = codec.getInputBuffer(index)!!; input.clear()
                        val count = extractor.readSampleData(input, 0)
                        if (count < 0) { codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputEos = true }
                        else { codec.queueInputBuffer(index, 0, count, extractor.sampleTime, 0); extractor.advance() }
                    }
                }
                val output = codec.dequeueOutputBuffer(info, 1_000)
                if (output >= 0) {
                    try {
                        if (info.size > 0) {
                            val pcm = codec.getOutputBuffer(output)!!.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                            pcm.position(info.offset); pcm.limit(info.offset + info.size)
                            while (pcm.remaining() >= 2) { val value = pcm.short.toDouble(); square += value * value; samples++ }
                        }
                        outputEos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    } finally { codec.releaseOutputBuffer(output, false) }
                }
            }
            assertTrue("recorded AAC did not decode through EOS", outputEos)
            assertTrue(samples > 16_000)
            return sqrt(square / samples)
        } finally { runCatching { codec.stop() }; codec.release() }
    }

    private fun await(message: String, timeout: Long, condition: () -> Boolean) {
        val end = minOf(now + timeout, if (clipDeadline > 0) clipDeadline else Long.MAX_VALUE)
        while (!condition() && now < end) { sample(); SystemClock.sleep(50) }
        assertTrue(message, condition())
    }
    private fun <T> main(action: () -> T): T {
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() }
        catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }
    private fun write(owner: Any, name: String, value: Any?) = owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(owner, value)
    private fun <T> read(owner: Any, name: String): T {
        @Suppress("UNCHECKED_CAST")
        return owner.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(owner) as T
    }
}
