package com.satori.glasses.capture

import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.security.MessageDigest

/** Explicitly selected, local-only diagnosis. Records recognition failures; never rewrites expected text. */
@RunWith(AndroidJUnit4::class)
class StandaloneAsrDiagnosticDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test(timeout = 180_000) fun recordEightFixtureAlternativesWithoutChangingExpectations() {
        val selected = InstrumentationRegistry.getArguments().getString("class").orEmpty()
            .split(',').map { it.trim().substringBefore('#') }
        assumeTrue("diagnosis requires explicit class/method selection", javaClass.name in selected)
        assumeTrue("do not compete with an active standalone service", StandaloneService.instance == null)
        assumeTrue("do not compete with legacy capture", CaptureService.instance == null)
        val manifest = instrumentation.context.assets.open("voice-commands/source-manifest.json")
            .bufferedReader().use { JSONObject(it.readText()) }
        val output = JSONObject().put("schema", "satori.asr-fixture-diagnostic/1")
            .put("purpose", "diagnosis only; mismatched 1-best is a failure of command accuracy, not accepted or rewritten")
            .put("input", "unmodified offline SAPI PCM fixtures, not a microphone or live-user recording")
            .put("cloud_requests", 0).put("chunk_bytes", 3200).put("trailing_silence_ms", 500)
            .put("fixture_source", manifest)
        val records = JSONArray()
        val folder = File(target.filesDir, "standalone-asr-diagnostics").apply { check(mkdirs() || isDirectory) }
        val file = File(folder, "eight-fixtures-${System.currentTimeMillis()}.json")
        val model = Model(StandaloneOfflineModel.prepare(target).absolutePath)
        try {
            val fixtures = manifest.getJSONArray("fixtures")
            for (index in 0 until fixtures.length()) {
                val entry = fixtures.getJSONObject(index)
                val bytes = instrumentation.context.assets.open("voice-commands/${entry.getString("file")}").use { it.readBytes() }
                val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
                assertEquals("do not diagnose a changed fixture", entry.getString("sha256"), hash)
                val pcm = WavData.parse(bytes)
                assertEquals(16000, pcm.sampleRate)
                assertEquals(1, pcm.channels)
                val baseline = decode(model, bytes, pcm, alternatives = 0)
                val topFive = decode(model, bytes, pcm, alternatives = 5)
                val candidates = combineCandidates(topFive)
                val actual = StandaloneVoice.normalizeOfflineText((0 until baseline.length())
                    .map { baseline.getJSONObject(it).getJSONObject("result").optString("text") }.filter { it.isNotBlank() }.joinToString(" "))
                val expected = entry.getString("spoken_text")
                records.put(JSONObject().put("id", entry.getString("id"))
                    .put("fixture_role", entry.optString("role", "primary_acceptance_fixture"))
                    .put("expected_spoken_text", expected)
                    .put("actual_1best", actual).put("one_best_matches_expected", actual == expected)
                    .put("pcm_offset", pcm.offset).put("pcm_bytes", pcm.length)
                    .put("baseline_segments", baseline).put("nbest_segments", topFive)
                    .put("combined_candidates", candidates)
                    .put("expected_in_candidates", (0 until candidates.length()).any { candidates.getJSONObject(it).getString("text") == expected }))
                // Preserve progress even if a later native/model problem stops the diagnostic run.
                file.writeText(output.put("fixtures", records).put("complete", false).toString(2))
            }
            output.put("complete", true)
        } finally {
            model.close()
            file.writeText(output.put("fixtures", records).toString(2))
            instrumentation.sendStatus(0, Bundle().apply { putString("diagnostic_file", "standalone-asr-diagnostics/${file.name}") })
        }
        assertEquals("all eight fixtures must be diagnosed", 8, records.length())
        // Deliberately no ASR-accuracy assertion: this diagnostic preserves failures for review.
        // StandaloneVoiceCommandDeviceTest remains the strict, unchanged acceptance test.
    }

    private fun decode(model: Model, wav: ByteArray, pcm: WavData, alternatives: Int): JSONArray {
        val recognizer = Recognizer(model, 16000f)
        try {
            recognizer.setWords(true)
            if (alternatives > 0) recognizer.setMaxAlternatives(alternatives)
            val results = JSONArray()
            fun feed(bytes: ByteArray) {
                if (recognizer.acceptWaveForm(bytes, bytes.size)) results.put(JSONObject()
                    .put("kind", "endpoint").put("result", JSONObject(recognizer.result)))
            }
            var offset = pcm.offset
            val end = pcm.offset + pcm.length
            while (offset < end) {
                val count = minOf(3200, end - offset)
                feed(wav.copyOfRange(offset, offset + count))
                offset += count
            }
            repeat(5) { feed(ByteArray(3200)) }
            results.put(JSONObject().put("kind", "final").put("result", JSONObject(recognizer.finalResult)))
            return results
        } finally { recognizer.close() }
    }

    private fun combineCandidates(segments: JSONArray): JSONArray {
        data class Candidate(val text: String, val score: Double)
        var combinations = listOf(Candidate("", 0.0))
        for (i in 0 until segments.length()) {
            val result = segments.getJSONObject(i).getJSONObject("result")
            val alternatives = result.optJSONArray("alternatives") ?: continue
            val choices = (0 until alternatives.length()).map { alternatives.getJSONObject(it) }
                .filter { it.optString("text").isNotBlank() }
                .map { Candidate(it.getString("text"), it.optDouble("confidence", 0.0).takeIf { score -> score.isFinite() } ?: 0.0) }
            if (choices.isEmpty()) continue
            combinations = combinations.flatMap { earlier -> choices.map { next ->
                Candidate(StandaloneVoice.normalizeOfflineText("${earlier.text} ${next.text}"), earlier.score + next.score)
            } }.distinctBy { it.text }.sortedByDescending { it.score }.take(5)
        }
        return JSONArray(combinations.filter { it.text.isNotBlank() }.map { JSONObject().put("text", it.text)
            .put("vosk_score_sum_not_probability", it.score) })
    }
}
