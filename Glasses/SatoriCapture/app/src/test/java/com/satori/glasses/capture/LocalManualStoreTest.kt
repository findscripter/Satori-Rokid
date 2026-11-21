package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class LocalManualStoreTest {
    private val assets = File("src/main/assets/manuals")
    private fun store() = LocalManualStore(listOf("classroom-demo.json", "tl-sg105-v4.json").map { File(assets, it).readBytes() })
    private fun manual() = JSONObject(File(assets, "tl-sg105-v4.json").readText(Charsets.UTF_8))
    private fun rejects(block: () -> Unit) {
        try { block(); fail("Accepted invalid manual input") } catch (_: IllegalArgumentException) { }
    }

    @Test fun bundledManualBytesRetainOriginalHashesAndProvenance() {
        val store = store()
        assertEquals(2, store.list().getJSONArray("manuals").length())
        val real = store.lookup("tl-sg105-v4")!!
        assertEquals("0a1152f2bc22c2f9295eb5ba0b11ef305cc2ebbde1a63c1f0060a17ac4831258", real.getString("sha256"))
        assertFalse(real.getBoolean("synthetic"))
        val demo = store.lookup("classroom-demo")!!
        assertEquals("9016943f30056191785dfde1ebade68a6de596906e733d8b217a82972944d80c", demo.getString("sha256"))
        assertTrue(demo.getBoolean("synthetic"))
        assertTrue(store.getSection("tl-sg105-v4", "applicability")!!.getString("section_title").contains("研究整理"))
        assertTrue(store.getSection("tl-sg105-v4", "verification-boundary")!!.getString("section_title").contains("研究约束"))
    }

    @Test fun rankingMatchesPythonManualStoreFixtures() {
        // Generated with Research/Backend/satori_repro/manual_tools.py against the original JSON bytes.
        val cases = listOf(
            Triple("tl-sg105-v4", "网线应该插哪个接口", "port-connection:4,applicability:1"),
            Triple("tl-sg105-v4", "Power Link/Act", "indicator-meaning:12,link-troubleshooting:6,verification-boundary:2"),
            Triple("tl-sg105-v4", "型号 铭牌 版本", "applicability:12,verification-boundary:1"),
            Triple("tl-sg105-v4", "网线 不亮", "indicator-meaning:3,link-troubleshooting:3,port-connection:3"),
            Triple("tl-sg105-v4", "火星登陆", ""),
            Triple("tl-sg105-v4", "TL-SG105", "applicability:4,indicator-meaning:4,link-troubleshooting:3,port-connection:3"),
            Triple("classroom-demo", "网线 LAN RJ45", "ethernet-connection:12,link-check:4,scope:4,troubleshoot:4"),
            Triple("classroom-demo", "联网 IP DNS", "link-check:12,troubleshoot:2")
        )
        val store = store()
        for ((id, query, expected) in cases) {
            val matches = store.search(id, query).getJSONArray("matches")
            val actual = (0 until matches.length()).joinToString(",") {
                val row = matches.getJSONObject(it)
                assertEquals(id, row.getString("manual_id"))
                "${row.getString("section_id")}:${row.getInt("score")}"
            }
            assertEquals(query, expected, actual)
            assertEquals(matches.length(), StandaloneManuals.context(id, query, store.search(id, query)).getJSONArray("passages").length())
        }
    }

    @Test fun unknownQueriesAndModelsNeverFallBackToAnotherManual() {
        val store = store()
        assertEquals(0, store.search("missing-model", "网线 LAN").getJSONArray("matches").length())
        assertEquals(0, store.search("tl-sg105-v4", "???").getJSONArray("matches").length())
        assertNull(store.lookup("missing-model"))
        assertNull(store.getSection("tl-sg105-v4", "ethernet-connection"))
        assertEquals(1, store.search("tl-sg105-v4", "Link", 1).getJSONArray("matches").length())
    }

    @Test fun returnedJsonCannotMutateTheSnapshot() {
        val store = store()
        store.lookup("tl-sg105-v4")!!.put("synthetic", true)
        store.getSection("tl-sg105-v4", "indicator-meaning")!!.put("text", "tampered")
        store.search("tl-sg105-v4", "Power").getJSONArray("matches").getJSONObject(0).put("manual_id", "wrong")
        assertFalse(store.lookup("tl-sg105-v4")!!.getBoolean("synthetic"))
        val section = store.search("tl-sg105-v4", "Power").getJSONArray("matches").getJSONObject(0)
        assertEquals("tl-sg105-v4", section.getString("manual_id"))
        assertNotEquals("tampered", section.getString("text"))
    }

    @Test fun inputBoundsAndMetadataTypesAreValidated() {
        val raw = File(assets, "tl-sg105-v4.json").readBytes()
        rejects { LocalManualStore(emptyList()) }
        rejects { LocalManualStore(List(65) { raw }) }
        rejects { LocalManualStore(listOf(ByteArray(LocalManualStore.MAX_MANUAL_BYTES + 1))) }
        rejects { LocalManualStore(listOf(raw, raw)) }
        for (bad in listOf(manual().put("synthetic", "false"), manual().put("schema_version", true),
            manual().put("source_url", "http://example.org/manual.pdf"), manual().put("source_url", "https://user:pass@example.org/manual.pdf"),
            manual().put("source_url", "https://example.org/manual.pdf#fragment"), manual().put("extra", true))) {
            rejects { LocalManualStore(listOf(bad.toString().toByteArray(Charsets.UTF_8))) }
        }
        val badSection = manual()
        badSection.getJSONArray("sections").getJSONObject(0).put("page", true)
        rejects { LocalManualStore(listOf(badSection.toString().toByteArray(Charsets.UTF_8))) }
        val store = store()
        rejects { store.search("../tl-sg105-v4", "网线") }
        rejects { store.search("tl-sg105-v4", " ") }
        rejects { store.search("tl-sg105-v4", "x".repeat(501)) }
        rejects { store.search("tl-sg105-v4", "网线\u0000") }
        rejects { store.search("tl-sg105-v4", "网线", 5) }
    }

    @Test fun pdfIndexKeepsOriginalPdfHashSeparateFromImportedJsonHash() {
        val sources = JSONObject(File(assets, "source-index.json").readText(Charsets.UTF_8)).getJSONArray("sources")
        val source = sources.getJSONObject(0)
        assertEquals("tl-sg105-v4", source.getString("manual_id"))
        assertEquals("3ea5dddb376c52ed80dcfdd2af3431c7aaa76e45d4b0c5ad421ecac60dcb0457", source.getString("sha256"))
        assertNotEquals(store().lookup("tl-sg105-v4")!!.getString("sha256"), source.getString("sha256"))
        assertTrue(source.getInt("document_bytes") <= source.getInt("max_bytes"))
        assertTrue(sources.getJSONObject(1).getBoolean("synthetic"))
        assertFalse(sources.getJSONObject(1).has("source_url"))
    }
}
