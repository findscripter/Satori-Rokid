package com.satori.glasses.capture

import android.graphics.Color
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

/** Real device codecs, synthetic moving I420 + 440 Hz PCM only. No camera, microphone or network. */
@RunWith(AndroidJUnit4::class)
class FirstPersonRecorderDeviceTest {
    private data class Bounds(val first: Long, val last: Long, val count: Int)
    private data class Decoded(val buffers: Int, val rms: Double)

    @Test(timeout = 45_000) fun realH264AndAacProduceDecodableRotatedMp4WithAlignedPts() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val file = output("av")
        val terminal = CountDownLatch(1)
        val saved = AtomicReference<JSONObject?>(null)
        val error = AtomicReference("")
        val recorder = FirstPersonRecorder(context) { name, data ->
            if (name == "video_saved") { saved.set(data); terminal.countDown() }
            if (name == "video_error") { error.set(data.optString("reason")); terminal.countDown() }
        }
        try {
            recorder.start(file, withAudio = true)
            assertTrue("start must request frames immediately", recorder.wantsFrames)
            val epoch = recorder.snapshot().getLong("start_elapsed_ns")
            var audioChunk = 0
            for (frame in 0 until 48) {
                val at = epoch + frame * 1_000_000_000L / 15
                waitUntil(at)
                recorder.offerVideo(RecordingVideoFrame(640, 480, 90, at, movingI420(frame, 640, 480)))
                while (audioChunk * 100_000_000L <= at - epoch) {
                    recorder.offerAudio(epoch + audioChunk * 100_000_000L, tone(audioChunk * 1600, 1600))
                    audioChunk++
                }
                assertTrue("video input queue exceeded its three-frame bound", recorder.snapshot().getInt("queued_video_frames") <= 3)
            }
            while (audioChunk < 32) {
                recorder.offerAudio(epoch + audioChunk * 100_000_000L, tone(audioChunk * 1600, 1600))
                audioChunk++
            }
            waitUntil(epoch + 3_200_000_000L)
            recorder.stop()
            assertFalse("stop must refuse new frames immediately", recorder.wantsFrames)
            assertTrue("encoder finalization did not finish", terminal.await(10, TimeUnit.SECONDS))
            assertTrue("real codec recording failed: ${error.get()}", saved.get() != null)
            val result = saved.get()!!
            assertTrue(result.getBoolean("finalized"))
            assertEquals(file.absolutePath, result.getString("path"))
            assertTrue(file.isFile && file.length() > 1000)
            assertFalse("pending file must disappear only after successful finalization", File(file.parentFile, file.nameWithoutExtension + ".pending.mp4").exists())
            val sidecar = JSONObject(File(file.parentFile, file.nameWithoutExtension + ".json").readText())
            assertTrue(sidecar.getBoolean("finalized"))
            assertEquals(90, sidecar.getInt("rotation"))
            assertTrue("video encoder lost too many paced frames", result.getLong("frames") >= 20)
            assertTrue("shared PCM must reach the AAC encoder", result.getLong("audio_samples") >= 48_000)

            val tracks = inspect(file)
            assertEquals("exactly one H264 and one AAC track are required", 2, tracks.size)
            val video = tracks.getValue("video/")
            val audio = tracks.getValue("audio/")
            assertTrue("initial audio/video timestamps diverged", abs(video.first - audio.first) <= 150_000)
            assertTrue("audio/video end timestamps diverged", abs(video.last - audio.last) <= 250_000)
            val decodedVideo = decode(file, "video/")
            val decodedAudio = decode(file, "audio/")
            assertTrue("H264 must decode actual image buffers", decodedVideo.buffers >= 10)
            assertTrue("AAC must decode actual PCM buffers", decodedAudio.buffers >= 10)
            assertTrue("decoded AAC must contain the synthetic audible tone", decodedAudio.rms > 500)

            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(file.absolutePath)
                assertEquals("MP4 orientation metadata was not preserved", "90", retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION))
                val image = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                assertNotNull("platform retriever cannot decode the finalized MP4", image)
                image!!.let { bitmap ->
                    try {
                        var lowest = 255
                        var highest = 0
                        for (y in 0 until bitmap.height step 8) for (x in 0 until bitmap.width step 8) {
                            val pixel = bitmap.getPixel(x, y)
                            val level = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                            lowest = minOf(lowest, level); highest = maxOf(highest, level)
                        }
                        assertTrue("decoded image must contain the generated moving block", highest - lowest > 20)
                        result.put("retriever_width", bitmap.width).put("retriever_height", bitmap.height)
                    } finally { bitmap.recycle() }
                }
            } finally { retriever.release() }
            result.put("video_first_pts_us", video.first).put("video_last_pts_us", video.last)
                .put("audio_first_pts_us", audio.first).put("audio_last_pts_us", audio.last)
                .put("decoded_audio_rms", decodedAudio.rms)
            File(file.parentFile, "media-verification.json").writeText(result.toString(2))
            println(JSONObject().put("recording_media_verified", true).put("path", file.absolutePath)
                .put("video_samples", video.count).put("audio_samples", audio.count).put("audio_rms", decodedAudio.rms).toString())
        } finally { recorder.close() }
    }

    @Test(timeout = 15_000) fun missingSharedAudioCannotBeReportedAsASuccessfulAvRecording() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val file = output("missing-audio")
        val terminal = CountDownLatch(1)
        val outcome = AtomicReference("")
        val recorder = FirstPersonRecorder(context) { name, _ ->
            if (name in listOf("video_saved", "video_error")) { outcome.set(name); terminal.countDown() }
        }
        try {
            recorder.start(file, withAudio = true)
            val epoch = recorder.snapshot().getLong("start_elapsed_ns")
            repeat(4) { index ->
                recorder.offerVideo(RecordingVideoFrame(640, 480, 0, epoch + index * 1_000L, movingI420(index, 640, 480)))
            }
            recorder.stop()
            assertFalse(recorder.wantsFrames)
            assertTrue("missing audio must terminate instead of waiting forever for a track", terminal.await(10, TimeUnit.SECONDS))
            assertEquals("video_error", outcome.get())
            assertFalse("an AV failure must never publish final MP4", file.exists())
            assertFalse(recorder.snapshot().getBoolean("finalized"))
            assertTrue("failed pending media must remain available for diagnosis",
                File(file.parentFile, file.nameWithoutExtension + ".pending.mp4").exists())
        } finally { recorder.close() }
    }

    @Test(timeout = 15_000) fun startWithoutAFirstFrameHasABoundedStartupFailure() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val terminal = CountDownLatch(1)
        val outcome = AtomicReference("")
        val recorder = FirstPersonRecorder(context) { name, data ->
            if (name == "video_error") { outcome.set(data.optString("reason")); terminal.countDown() }
        }
        try {
            recorder.start(output("no-frame"), withAudio = false)
            assertTrue(recorder.wantsFrames)
            assertTrue("no-frame startup must fail within its ten-second budget", terminal.await(12, TimeUnit.SECONDS))
            assertEquals("no_video_frame", outcome.get())
            assertFalse(recorder.active)
            assertFalse(recorder.wantsFrames)
        } finally { recorder.close() }
    }

    private fun inspect(file: File): Map<String, Bounds> {
        val result = linkedMapOf<String, Bounds>()
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
            assertEquals(2, extractor.trackCount)
            for (track in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(track)
                val mime = format.getString(MediaFormat.KEY_MIME).orEmpty()
                val prefix = if (mime.startsWith("video/")) "video/" else "audio/"
                if (prefix == "video/") {
                    assertEquals(MediaFormat.MIMETYPE_VIDEO_AVC, mime)
                    assertEquals(640, format.getInteger(MediaFormat.KEY_WIDTH))
                    assertEquals(480, format.getInteger(MediaFormat.KEY_HEIGHT))
                } else {
                    assertEquals(MediaFormat.MIMETYPE_AUDIO_AAC, mime)
                    assertEquals(16_000, format.getInteger(MediaFormat.KEY_SAMPLE_RATE))
                    assertEquals(1, format.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
                }
                extractor.selectTrack(track)
                extractor.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                var first = -1L
                var last = -1L
                var count = 0
                while (extractor.sampleTime >= 0) {
                    val pts = extractor.sampleTime
                    assertTrue("MP4 timestamps must be strictly increasing per track", pts > last)
                    if (first < 0) first = pts
                    last = pts; count++
                    if (!extractor.advance()) break
                }
                assertTrue("MP4 track has no media samples", count > 0)
                result[prefix] = Bounds(first, last, count)
                extractor.unselectTrack(track)
            }
        } finally { extractor.release() }
        return result
    }

    private fun decode(file: File, prefix: String): Decoded {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(file.absolutePath)
            val track = (0 until extractor.trackCount).first { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)!!.startsWith(prefix) }
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            codec = decoder
            decoder.configure(format, null, null, 0)
            decoder.start()
            var inputEos = false
            var outputEos = false
            var buffers = 0
            var sum = 0.0
            var samples = 0L
            val info = MediaCodec.BufferInfo()
            val deadline = SystemClock.elapsedRealtime() + 10_000
            while (!outputEos && SystemClock.elapsedRealtime() < deadline) {
                if (!inputEos) {
                    val index = decoder.dequeueInputBuffer(0)
                    if (index >= 0) {
                        val input = decoder.getInputBuffer(index)!!
                        input.clear()
                        val count = extractor.readSampleData(input, 0)
                        if (count < 0) {
                            decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEos = true
                        } else {
                            decoder.queueInputBuffer(index, 0, count, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val output = decoder.dequeueOutputBuffer(info, 1_000)
                if (output >= 0) {
                    try {
                        if (prefix == "audio/" && info.size > 0) {
                            val pcm = decoder.getOutputBuffer(output)!!.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                            pcm.position(info.offset); pcm.limit(info.offset + info.size)
                            while (pcm.remaining() >= 2) { val value = pcm.short.toDouble(); sum += value * value; samples++ }
                            buffers++
                        } else if (prefix == "video/") {
                            val image = runCatching { decoder.getOutputImage(output) }.getOrNull()
                            if (image != null) { buffers++; image.close() } else if (info.size > 0) buffers++
                        }
                        outputEos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    } finally { decoder.releaseOutputBuffer(output, false) }
                }
            }
            assertTrue("real decoder did not reach EOS", outputEos)
            return Decoded(buffers, if (samples > 0) sqrt(sum / samples) else 0.0)
        } finally {
            codec?.let { runCatching { it.stop() }; it.release() }
            extractor.release()
        }
    }

    private fun output(label: String): File {
        val directory = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir,
            "recording-media-test-$label-${UUID.randomUUID()}")
        check(directory.mkdirs())
        return File(directory, "recording-${System.currentTimeMillis()}.mp4")
    }

    private fun waitUntil(atNs: Long) {
        while (SystemClock.elapsedRealtimeNanos() < atNs) {
            val remaining = atNs - SystemClock.elapsedRealtimeNanos()
            if (remaining > 2_000_000) SystemClock.sleep((remaining / 1_000_000).coerceAtMost(10)) else Thread.yield()
        }
    }

    private fun tone(firstSample: Int, samples: Int): ByteArray = ByteArray(samples * 2).also { bytes ->
        for (index in 0 until samples) {
            val value = (sin(2.0 * Math.PI * 440 * (firstSample + index) / 16_000) * 12_000).toInt()
            bytes[index * 2] = value.toByte(); bytes[index * 2 + 1] = (value shr 8).toByte()
        }
    }

    private fun movingI420(frame: Int, width: Int, height: Int): ByteArray {
        val luma = width * height
        return ByteArray(luma * 3 / 2).also { bytes ->
            java.util.Arrays.fill(bytes, 0, luma, 45.toByte())
            java.util.Arrays.fill(bytes, luma, luma + luma / 4, 90.toByte())
            java.util.Arrays.fill(bytes, luma + luma / 4, bytes.size, 200.toByte())
            val left = width / 20 + frame * maxOf(2, width / 80) % (width - width / 4 - width / 10)
            for (y in height / 4 until height * 3 / 4) for (x in left until left + width / 4) bytes[y * width + x] = 205.toByte()
        }
    }
}
