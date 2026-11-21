package com.satori.glasses.capture

import android.graphics.Color
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.DataInputStream
import java.io.File
import java.nio.ByteOrder
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt

/** Opt-in, read-only inspection of one saved pair. No Service, recorder, sensor, network or task access. */
@RunWith(AndroidJUnit4::class)
class ClassroomMediaDeviceTest {
    private data class Recording(val file: File, val sidecar: File, val metadata: JSONObject)
    private data class Track(val index: Int, val mime: String, val first: Long, val last: Long, val packets: Int)
    private val identityKeys = listOf("run_session_id", "data_run_id", "capture_group_id", "boot_id", "clock_domain")

    @Test(timeout = 120_000) fun inspectExplicitSavedCaptureGroup() {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue("use -e satoriMediaCheck true after both recordings are saved",
            arguments.getString("satoriMediaCheck") == "true")
        val group = arguments.getString("captureGroupId").orEmpty()
        assertTrue("an explicit captureGroupId is required", group.matches(Regex("capture-[A-Za-z0-9_-]{1,120}")))
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val files = instrumentation.targetContext.filesDir
        val report = JSONObject().put("schema", "satori.classroom-media-device-check/1")
            .put("capture_group_id", group).put("passed", false).put("source_files_unchanged", false)
            .put("max_clip_duration_us", 60_000_000L).put("physical_sync_accuracy_verified", false)
            .put("full_video_decode_verified", false).put("speech_recognition_evaluated", false)
            .put("scope", "saved recording media only; no independent task-success evaluation")
            .put("sync_limit", "Visible SYNC frames require manual inspection; encoder receipt time is not capture time")
        val media = JSONArray()
        report.put("media", media)
        var stage = "match_saved_pair"
        var failureKind: String? = null
        try {
            val first = findSaved(files, "first-person-videos", "recording", group)
            val screen = findSaved(files, "screen-recordings", "screen", group)
            for (key in identityKeys) {
                val value = first.metadata.getString(key)
                assertTrue("invalid identity field", value.matches(Regex("[A-Za-z0-9_.-]{1,160}")))
                assertEquals("two recordings must share identity", value, screen.metadata.getString(key))
                report.put(key, value)
            }
            assertEquals("android.elapsedRealtimeNanos", first.metadata.getString("clock_domain"))
            assertNotEquals("boot identity must be known", "unknown", first.metadata.getString("boot_id"))
            assertEquals("first_person", first.metadata.getString("media_role"))
            assertEquals("screen", screen.metadata.getString("media_role"))
            assertTrue(first.metadata.getBoolean("with_audio"))
            assertFalse(screen.metadata.getBoolean("with_audio"))
            assertEquals("satori.screen-recording/1", screen.metadata.getString("schema"))
            assertEquals("manual_sync_required", screen.metadata.getString("timeline_mapping"))
            assertTrue(screen.metadata.has("capture_start_elapsed_ns") && screen.metadata.isNull("capture_start_elapsed_ns"))
            val runId = first.metadata.getString("data_run_id")
            assertTrue("invalid run identity", runId.matches(Regex("run-[0-9]{1,20}")))
            val runs = checkedDirectory(files, "standalone-runs")
            val run = checkedDirectory(runs, runId)
            val sessionFile = checkedFile(run, "session.json")
            val session = readMetadata(sessionFile)
            assertEquals("satori.session/1", session.getString("schema"))
            assertEquals(first.metadata.getString("run_session_id"), session.getString("session_id"))
            for (key in listOf("boot_id", "clock_domain")) assertEquals(first.metadata.getString(key), session.getString(key))
            val inputs = listOf(first.file, first.sidecar, screen.file, screen.sidecar, sessionFile)
            val before = inputs.map { it.length() to it.lastModified() }
            for ((role, recording) in listOf("first_person" to first, "screen" to screen)) {
                stage = "inspect_$role"
                val item = JSONObject().put("media_role", role).put("file", recording.file.name)
                    .put("bytes", recording.file.length())
                media.put(item)
                val tracks = inspectTracks(recording.file, item)
                val video = tracks.filter { it.mime == MediaFormat.MIMETYPE_VIDEO_AVC }
                val audio = tracks.filter { it.mime == MediaFormat.MIMETYPE_AUDIO_AAC }
                assertEquals("exactly one H264 track is required", 1, video.size)
                assertEquals("unexpected track type or count", if (role == "first_person") 2 else 1, tracks.size)
                assertEquals("screen must be silent and first person must contain AAC", if (role == "first_person") 1 else 0, audio.size)
                assertEquals("sidecar frame count differs from MP4", recording.metadata.getLong("frames"), video.single().packets.toLong())
                stage = "video_frames_$role"
                item.put("retrieved_frames", inspectVideoFrames(recording.file, video.single()))
                if (audio.isNotEmpty()) {
                    stage = "decode_aac"
                    val decoded = decodeAudio(recording.file, audio.single())
                    item.put("decoded_audio", decoded)
                    assertTrue("AAC decoded no PCM samples", decoded.getLong("scalar_samples") > 0)
                    assertTrue("decoded AAC contains no signal above 32 PCM16 units", decoded.getLong("non_silent_scalar_samples") > 0)
                }
            }
            stage = "source_stability"
            assertEquals("source file changed during inspection", before, inputs.map { it.length() to it.lastModified() })
            report.put("source_files_unchanged", true).put("passed", true)
        } catch (failure: Throwable) {
            failureKind = failure.javaClass.simpleName
            report.put("failure_kind", failureKind).put("failure_stage", stage)
        } finally {
            instrumentation.sendStatus(0, Bundle().apply { putString("classroom_media_report", report.toString()) })
            println(report.toString())
        }
        if (failureKind != null) throw AssertionError("Read-only media check failed at $stage ($failureKind); inspect classroom_media_report")
    }

    @Test(timeout = 180_000) fun inspectOfficialRawPair() {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue("use -e satoriOfficialMediaCheck true for the copied official recordings",
            arguments.getString("satoriOfficialMediaCheck") == "true")
        val expected = mapOf("screen" to arguments.getString("expectedScreenSha256").orEmpty(),
            "first_person" to arguments.getString("expectedFirstPersonSha256").orEmpty())
        assertTrue("both explicit source SHA256 values are required", expected.values.all { it.matches(Regex("[a-f0-9]{64}")) })
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val report = JSONObject().put("schema", "satori.official-media-device-check/1")
            .put("source", "rokid_official_raw_pair").put("passed", false)
            .put("pair_identity", "same vendor filename plus preverified source hashes")
            .put("physical_sync_verified", false).put("full_video_decode_verified", false)
            .put("speech_recognition_evaluated", false).put("max_clip_duration_us", 180_000_000L)
            .put("scope", "actual video tracks, sampled images and complete AAC decoding of two verified cache copies")
        val media = JSONArray()
        report.put("media", media)
        var stage = "verify_cache_copies"
        var failureKind: String? = null
        try {
            val directory = checkedDirectory(instrumentation.targetContext.cacheDir, "official-acceptance")
            val inputs = linkedMapOf("screen" to checkedFile(directory, "screen.mp4"),
                "first_person" to checkedFile(directory, "first-person.mp4"))
            for ((role, file) in inputs) assertEquals("cache copy differs from approved source", expected.getValue(role), sha256(file))
            report.put("source_hashes_verified", true)
            for ((role, file) in inputs) {
                stage = "inspect_official_$role"
                val item = JSONObject().put("media_role", role).put("file", file.name)
                    .put("bytes", file.length()).put("sha256", expected.getValue(role))
                media.put(item)
                val tracks = inspectTracks(file, item, maxDurationUs = 180_000_000L, allowVideoReordering = true)
                val video = tracks.filter { it.mime.startsWith("video/") }
                val audio = tracks.filter { it.mime.startsWith("audio/") }
                item.put("actual_audio_track_count", audio.size).put("actual_audio_present", audio.isNotEmpty())
                assertEquals("one actual video track is required", 1, video.size)
                assertTrue("at most one audio track is supported", audio.size <= 1)
                stage = "official_video_frames_$role"
                item.put("retrieved_frames", inspectVideoFrames(file, video.single()))
                val decodedAudio = JSONArray()
                item.put("decoded_audio", decodedAudio)
                for (track in audio) {
                    stage = "official_aac_$role"
                    val decoded = decodeAudio(file, track, maxDurationUs = 180_000_000L, requireSharedMicrophone = false)
                    decodedAudio.put(decoded)
                    assertTrue("AAC decoded no PCM samples", decoded.getLong("scalar_samples") > 0)
                    item.put("audio_signal_above_threshold", decoded.getLong("non_silent_scalar_samples") > 0)
                }
                if (role == "first_person") {
                    stage = "official_first_person_audio_contract"
                    item.put("expected_aac_track_present", audio.size == 1 && audio.single().mime == MediaFormat.MIMETYPE_AUDIO_AAC)
                    assertTrue("official first person has no verified AAC track", item.getBoolean("expected_aac_track_present"))
                    assertTrue("official first-person AAC has no observed signal above threshold", item.getBoolean("audio_signal_above_threshold"))
                }
            }
            stage = "official_copy_stability"
            for ((role, file) in inputs) assertEquals("cache copy changed during inspection", expected.getValue(role), sha256(file))
            report.put("cache_copies_unchanged", true).put("passed", true)
        } catch (failure: Throwable) {
            failureKind = failure.javaClass.simpleName
            report.put("failure_kind", failureKind).put("failure_stage", stage)
        } finally {
            instrumentation.sendStatus(0, Bundle().apply { putString("official_media_report", report.toString()) })
            println(report.toString())
        }
        if (failureKind != null) throw AssertionError("Official read-only media check failed at $stage ($failureKind); inspect official_media_report")
    }

    private fun sha256(file: File): String {
        assertTrue("official copy must be nonempty and at most 256 MiB", file.length() in 1..268_435_456L)
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(256 * 1024)
            var bytes = 0L
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                bytes += count
                check(bytes <= 268_435_456L)
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }

    private fun checkedDirectory(parent: File, name: String): File = File(parent, name).also {
        check(!Files.isSymbolicLink(it.toPath()) && it.isDirectory && it.canonicalFile.parentFile == parent.canonicalFile)
    }

    private fun checkedFile(parent: File, name: String): File = File(parent, name).also {
        check(!Files.isSymbolicLink(it.toPath()) && it.isFile && it.canonicalFile.parentFile == parent.canonicalFile)
    }

    private fun readMetadata(file: File): JSONObject {
        val size = file.length()
        check(size in 1..65_536)
        val bytes = ByteArray(size.toInt())
        DataInputStream(file.inputStream()).use { input ->
            input.readFully(bytes)
            check(input.read() == -1)
        }
        return JSONObject(bytes.toString(Charsets.UTF_8))
    }

    private fun findSaved(files: File, folder: String, prefix: String, group: String): Recording {
        val directory = checkedDirectory(files, folder)
        val candidates = checkNotNull(directory.listFiles())
        check(candidates.size <= 2_000)
        val matched = candidates.filter { it.name.matches(Regex("$prefix-[0-9]{1,20}\\.json")) }.mapNotNull { candidate ->
            val sidecar = checkedFile(directory, candidate.name)
            val metadata = readMetadata(sidecar)
            if (metadata.optString("capture_group_id") != group) null else {
                assertEquals("selected recording is not saved", "saved", metadata.getString("state"))
                assertTrue("selected recording is not finalized", metadata.getBoolean("finalized"))
                val stem = sidecar.nameWithoutExtension
                for (suffix in listOf(".pending.mp4", ".failed.mp4", ".pending.json", ".json.new", ".json.bak")) {
                    assertFalse("selected recording still has unfinished artifacts", File(directory, stem + suffix).exists())
                }
                val file = checkedFile(directory, "$stem.mp4")
                assertTrue("selected media must be nonempty and bounded", file.length() in 1..134_217_728L)
                Recording(file, sidecar, metadata)
            }
        }
        assertEquals("capture group must match exactly one saved recording per role", 1, matched.size)
        return matched.single()
    }

    private fun inspectTracks(file: File, report: JSONObject, maxDurationUs: Long = 60_000_000L,
                              allowVideoReordering: Boolean = false): List<Track> {
        val extractor = MediaExtractor()
        val tracks = mutableListOf<Track>()
        val summaries = JSONArray()
        report.put("tracks", summaries)
        try {
            extractor.setDataSource(file.absolutePath)
            assertTrue("unexpected number of tracks", extractor.trackCount in 1..2)
            val deadline = SystemClock.elapsedRealtime() + 10_000
            for (index in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(index)
                val mime = format.getString(MediaFormat.KEY_MIME).orEmpty()
                val summary = JSONObject().put("index", index).put("mime", mime)
                summaries.put(summary)
                assertTrue("unsupported media track", mime in listOf(MediaFormat.MIMETYPE_VIDEO_AVC, MediaFormat.MIMETYPE_VIDEO_HEVC, MediaFormat.MIMETYPE_AUDIO_AAC))
                val duration = format.getLong(MediaFormat.KEY_DURATION)
                summary.put("duration_us", duration)
                assertTrue("clip exceeds the explicit duration limit", duration in 1..maxDurationUs)
                for (key in listOf(MediaFormat.KEY_WIDTH, MediaFormat.KEY_HEIGHT, MediaFormat.KEY_SAMPLE_RATE, MediaFormat.KEY_CHANNEL_COUNT)) {
                    if (format.containsKey(key)) summary.put(key, format.getInteger(key))
                }
                extractor.selectTrack(index)
                extractor.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                var first = -1L
                var last = -1L
                var minimum = Long.MAX_VALUE
                var maximum = -1L
                var duplicates = 0
                var decreases = 0
                var packets = 0
                var bytes = 0L
                while (extractor.sampleTime >= 0) {
                    check(SystemClock.elapsedRealtime() < deadline && packets < maxDurationUs / 1_000_000 * 200)
                    val pts = extractor.sampleTime
                    if (pts == last) duplicates++
                    if (pts < last) decreases++
                    // Official H264/HEVC may reorder pictures; packet order is not presentation order.
                    if (!allowVideoReordering || !mime.startsWith("video/")) {
                        assertTrue("MP4 track PTS must increase strictly", pts > last)
                    }
                    if (first < 0) first = pts
                    minimum = minOf(minimum, pts)
                    maximum = maxOf(maximum, pts)
                    assertTrue("sample span exceeds the explicit duration limit", maximum - minimum <= maxDurationUs)
                    assertTrue("empty MP4 packet", extractor.sampleSize > 0)
                    bytes += extractor.sampleSize
                    last = pts
                    packets++
                    if (!extractor.advance()) break
                }
                assertTrue("MP4 track contains no packets", packets > 0)
                summary.put("first_pts_us", first).put("last_pts_us", last)
                    .put("minimum_pts_us", minimum).put("maximum_pts_us", maximum)
                    .put("packet_pts_decreases", decreases).put("duplicate_adjacent_pts", duplicates)
                    .put("packets", packets).put("encoded_bytes", bytes).put("strictly_increasing_pts", duplicates == 0 && decreases == 0)
                tracks += Track(index, mime, minimum, maximum, packets)
                extractor.unselectTrack(index)
            }
        } finally { extractor.release() }
        return tracks
    }

    private fun inspectVideoFrames(file: File, video: Track): JSONArray {
        val retriever = MediaMetadataRetriever()
        val frames = JSONArray()
        var anyNonempty = false
        try {
            retriever.setDataSource(file.absolutePath)
            for (at in listOf(video.first, video.first + (video.last - video.first) / 2, video.last).distinct()) {
                val bitmap = checkNotNull(retriever.getFrameAtTime(at, MediaMetadataRetriever.OPTION_CLOSEST))
                try {
                    var count = 0
                    var nonblack = 0
                    var lowest = 255
                    var highest = 0
                    for (y in 0 until bitmap.height step maxOf(1, bitmap.height / 64)) {
                        for (x in 0 until bitmap.width step maxOf(1, bitmap.width / 64)) {
                            val color = bitmap.getPixel(x, y)
                            val level = (Color.red(color) + Color.green(color) + Color.blue(color)) / 3
                            lowest = minOf(lowest, level)
                            highest = maxOf(highest, level)
                            if (level > 0) nonblack++
                            count++
                        }
                    }
                    anyNonempty = anyNonempty || nonblack > 0
                    frames.put(JSONObject().put("requested_pts_us", at).put("width", bitmap.width).put("height", bitmap.height)
                        .put("sampled_pixels", count).put("nonblack_pixels", nonblack).put("min_level", lowest)
                        .put("max_level", highest).put("pixels_nonempty", nonblack > 0))
                } finally { bitmap.recycle() }
            }
            assertTrue("all retrieved video frames were black", anyNonempty)
        } finally { retriever.release() }
        return frames
    }

    private fun decodeAudio(file: File, track: Track, maxDurationUs: Long = 60_000_000L,
                            requireSharedMicrophone: Boolean = true): JSONObject {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(file.absolutePath)
            extractor.selectTrack(track.index)
            val format = extractor.getTrackFormat(track.index)
            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            assertTrue("unsupported AAC sample rate", sampleRate in 8_000..192_000)
            assertTrue("unsupported AAC channel count", channels in 1..8)
            if (requireSharedMicrophone) {
                assertEquals("shared microphone sample rate", 16_000, sampleRate)
                assertEquals("shared microphone channel count", 1, channels)
            }
            format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
            val decoder = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
            codec = decoder
            decoder.configure(format, null, null, 0)
            decoder.start()
            var inputEos = false
            var outputEos = false
            var inputPackets = 0
            var outputBuffers = 0
            var samples = 0L
            var nonSilent = 0L
            var sum = 0.0
            var firstPts = -1L
            var lastPts = -1L
            val info = MediaCodec.BufferInfo()
            val deadline = SystemClock.elapsedRealtime() + 45_000
            while (!outputEos && SystemClock.elapsedRealtime() < deadline) {
                if (!inputEos) {
                    val index = decoder.dequeueInputBuffer(0)
                    if (index >= 0) {
                        val input = checkNotNull(decoder.getInputBuffer(index))
                        input.clear()
                        val size = extractor.readSampleData(input, 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEos = true
                        } else {
                            check(++inputPackets <= maxDurationUs / 1_000_000 * 200)
                            decoder.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val output = decoder.dequeueOutputBuffer(info, 1_000)
                if (output == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val actual = decoder.outputFormat
                    if (actual.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                        assertEquals("AAC decoder must output requested PCM16", AudioFormat.ENCODING_PCM_16BIT, actual.getInteger(MediaFormat.KEY_PCM_ENCODING))
                    }
                    assertEquals(sampleRate, actual.getInteger(MediaFormat.KEY_SAMPLE_RATE))
                    assertEquals(channels, actual.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
                } else if (output >= 0) {
                    try {
                        if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                            check(++outputBuffers <= maxDurationUs / 1_000_000 * 200 && info.size % 2 == 0)
                            assertTrue("decoded PCM PTS must increase", info.presentationTimeUs > lastPts)
                            if (firstPts < 0) firstPts = info.presentationTimeUs
                            lastPts = info.presentationTimeUs
                            val pcm = checkNotNull(decoder.getOutputBuffer(output)).duplicate().order(ByteOrder.LITTLE_ENDIAN)
                            pcm.position(info.offset)
                            pcm.limit(info.offset + info.size)
                            while (pcm.remaining() >= 2) {
                                val value = pcm.short.toInt()
                                sum += value.toDouble() * value
                                if (abs(value) > 32) nonSilent++
                                samples++
                            }
                            check(samples <= sampleRate.toLong() * channels * (maxDurationUs / 1_000_000 + 5))
                        }
                        outputEos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    } finally { decoder.releaseOutputBuffer(output, false) }
                }
            }
            assertTrue("AAC decoder did not reach end of stream", outputEos)
            assertEquals("AAC decoder did not consume every extracted packet", track.packets, inputPackets)
            val rms = if (samples > 0) sqrt(sum / samples) else 0.0
            return JSONObject().put("eos_reached", true).put("input_packets", inputPackets).put("output_buffers", outputBuffers)
                .put("pcm_encoding", "PCM16").put("sample_rate", sampleRate).put("channels", channels)
                .put("scalar_samples", samples).put("pcm_frames", samples / channels)
                .put("decoded_duration_ms", samples * 1000.0 / channels / sampleRate)
                .put("first_output_pts_us", firstPts).put("last_output_pts_us", lastPts)
                .put("rms_pcm16", rms).put("rms_dbfs", if (rms > 0) 20 * log10(rms / 32768) else JSONObject.NULL)
                .put("non_silent_threshold_pcm16", 32).put("non_silent_scalar_samples", nonSilent)
                .put("interpretation", "decoded microphone signal only; not speaker identity, intelligibility or recognition accuracy")
        } finally {
            codec?.let { runCatching { it.stop() }; it.release() }
            extractor.release()
        }
    }
}
