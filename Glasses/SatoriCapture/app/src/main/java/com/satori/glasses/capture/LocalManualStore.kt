package com.satori.glasses.capture

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.security.MessageDigest
import java.util.Locale

/** A bounded snapshot of bundled source bytes; its hashes do not claim remote verification. */
class LocalManualStore(documents: List<ByteArray>) {
    private data class Section(val record: JSONObject, val heading: Set<String>, val body: Set<String>)
    private val manuals = sortedMapOf<String, JSONObject>()
    private val sections = ArrayList<Section>()

    init {
        require(documents.size in 1..64 && documents.sumOf { it.size.toLong() } <= 16L * MAX_MANUAL_BYTES)
        for (raw in documents) {
            require(raw.size in 1..MAX_MANUAL_BYTES)
            val text = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(raw)).toString()
            val parser = JSONTokener(text)
            val value = parser.nextValue() as? JSONObject ?: throw IllegalArgumentException("Manual must be an object")
            require(parser.nextClean() == '\u0000') { "Trailing manual data" }
            keys(value, setOf("schema_version", "manual_id", "title", "model", "version", "source_url", "synthetic", "sections"))
            require(value.get("schema_version") is Int && value.getInt("schema_version") == 1)
            require(value.get("synthetic") is Boolean)
            val id = text(value.get("manual_id"), 80, true)
            require(id !in manuals) { "Duplicate manual identifier" }
            val metadata = JSONObject().put("manual_id", id).put("synthetic", value.getBoolean("synthetic"))
            for ((name, limit) in mapOf("title" to 240, "model" to 160, "version" to 120, "source_url" to 2000)) {
                metadata.put(name, text(value.get(name), limit))
            }
            val source = metadata.getString("source_url")
            val uri = URI(source)
            require(uri.scheme == "https" && !uri.host.isNullOrBlank() && uri.rawUserInfo == null &&
                uri.rawFragment == null && source.none { it.isWhitespace() })
            metadata.put("sha256", MessageDigest.getInstance("SHA-256").digest(raw).joinToString("") { "%02x".format(it) })
            val entries = value.getJSONArray("sections")
            require(entries.length() in 1..200 && sections.size + entries.length() <= 4096)
            val ids = HashSet<String>()
            for (i in 0 until entries.length()) {
                val entry = entries.getJSONObject(i)
                keys(entry, setOf("section_id", "title", "page", "text"), setOf("keywords"))
                val sectionId = text(entry.get("section_id"), 80, true)
                require(ids.add(sectionId)) { "Duplicate section identifier" }
                val title = text(entry.get("title"), 240)
                val body = text(entry.get("text"), 16000)
                val page = entry.get("page")
                require(page == JSONObject.NULL || page is Int && page in 1..100000)
                val words = if (entry.has("keywords")) entry.getJSONArray("keywords") else JSONArray()
                require(words.length() <= 40)
                val keywords = (0 until words.length()).map { text(words.get(it), 80) }
                val record = copy(metadata).put("section_id", sectionId).put("section_title", title).put("page", page).put("text", body)
                val heading = listOf(metadata.getString("title"), metadata.getString("model"), title) + keywords
                sections.add(Section(record, tokens(heading.joinToString(" ")), tokens(body)))
            }
            manuals[id] = metadata.put("section_count", entries.length())
        }
    }

    fun list(): JSONObject = JSONObject().put("manuals", JSONArray(manuals.values.map { copy(it) }))

    fun lookup(manualId: String): JSONObject? {
        text(manualId, 80, true)
        return manuals[manualId]?.let { copy(it) }
    }

    fun getSection(manualId: String, sectionId: String): JSONObject? {
        text(manualId, 80, true)
        text(sectionId, 80, true)
        return sections.firstOrNull { it.record.getString("manual_id") == manualId && it.record.getString("section_id") == sectionId }
            ?.let { copy(it.record) }
    }

    fun search(manualId: String, query: String, limit: Int = 4): JSONObject {
        text(manualId, 80, true)
        text(query, 500)
        require(limit in 1..4)
        val terms = tokens(query)
        val matches = sections.asSequence().filter { it.record.getString("manual_id") == manualId }
            .map { it to (3 * terms.intersect(it.heading).size + terms.intersect(it.body).size) }
            .filter { it.second > 0 }
            .sortedWith(compareByDescending<Pair<Section, Int>> { it.second }.thenBy { it.first.record.getString("section_id") })
            .take(limit).map { copy(it.first.record).put("score", it.second) }.toList()
        return JSONObject().put("matches", JSONArray(matches))
    }

    companion object {
        const val MAX_MANUAL_BYTES = 1_048_576
        private val identifier = Regex("[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
        private val latin = Regex("[a-z0-9]+(?:[-_.][a-z0-9]+)*")
        private val chinese = Regex("[\\u3400-\\u9fff]+")

        private fun copy(value: JSONObject) = JSONObject(value.toString())

        private fun keys(value: JSONObject, required: Set<String>, optional: Set<String> = emptySet()) {
            val actual = value.keys().asSequence().toSet()
            require(actual.containsAll(required) && actual.all { it in required || it in optional })
        }

        private fun text(value: Any, maximum: Int, id: Boolean = false): String {
            require(value is String && value.isNotBlank() && value.length <= maximum)
            require(value.none { it.code < 32 && it != '\n' && it != '\t' })
            require(!id || identifier.matches(value))
            return value
        }

        private fun tokens(value: String): Set<String> {
            // shortcut: this corpus is Chinese/ASCII English; add full casefold when importing other alphabets.
            val result = latin.findAll(value.lowercase(Locale.ROOT)).map { it.value }.toMutableSet()
            for (match in chinese.findAll(value)) {
                val phrase = match.value
                if (phrase.length == 1) result.add(phrase)
                else for (i in 0 until phrase.length - 1) result.add(phrase.substring(i, i + 2))
            }
            return result
        }
    }
}
