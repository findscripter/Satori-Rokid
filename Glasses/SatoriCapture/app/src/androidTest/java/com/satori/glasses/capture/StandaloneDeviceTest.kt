package com.satori.glasses.capture

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.security.KeyStore
import java.util.Random
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Device-local acceptance only: synthetic images, dummy secrets, and idle/paused service states. */
@RunWith(AndroidJUnit4::class)
class StandaloneDeviceTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test fun isolatedKeystoreSupportsNonExportableEncryptedPreferences() {
        // Production Settings has a fixed alias. Never save/clear it in an acceptance test.
        // This exercises the actual device keystore with a disposable, independently named key.
        val suffix = UUID.randomUUID().toString()
        val alias = "satori.device-test.$suffix"
        val prefsName = "standalone_device_test_$suffix"
        val prefs = target.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val dummy = "device-test-only-not-an-api-credential"
        try {
            val profile = StandaloneProfile("https://example.invalid/v1", dummy, "test-planner", "test-monitor")
            profile.validate()
            assertFalse("profile string must redact credentials", profile.toString().contains(dummy))
            val key = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
                init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256).setRandomizedEncryptionRequired(true).build())
            }.generateKey()
            assertNull("keystore key must not be exportable", key.encoded)
            val encrypt = Cipher.getInstance("AES/GCM/NoPadding")
            encrypt.init(Cipher.ENCRYPT_MODE, key)
            encrypt.updateAAD(alias.toByteArray(Charsets.UTF_8))
            val envelope = JSONObject().put("iv", Base64.encodeToString(encrypt.iv, Base64.NO_WRAP))
                .put("data", Base64.encodeToString(encrypt.doFinal(dummy.toByteArray(Charsets.UTF_8)), Base64.NO_WRAP))
            assertTrue("test preferences commit failed", prefs.edit().putString("envelope", envelope.toString()).commit())
            assertFalse("preferences must not contain plaintext", prefs.all.values.any { it.toString().contains(dummy) })
            val persisted = JSONObject(prefs.getString("envelope", "")!!)
            val loadedKey = store.getKey(alias, null) as SecretKey
            val decrypt = Cipher.getInstance("AES/GCM/NoPadding")
            decrypt.init(Cipher.DECRYPT_MODE, loadedKey,
                GCMParameterSpec(128, Base64.decode(persisted.getString("iv"), Base64.NO_WRAP)))
            decrypt.updateAAD(alias.toByteArray(Charsets.UTF_8))
            val restored = String(decrypt.doFinal(Base64.decode(persisted.getString("data"), Base64.NO_WRAP)), Charsets.UTF_8)
            assertTrue("isolated encrypted round trip failed", restored == dummy)
        } finally {
            // Only UUID-scoped test data is deleted. The production alias/preferences are untouched.
            prefs.edit().clear().commit()
            target.deleteSharedPreferences(prefsName)
            if (store.containsAlias(alias)) store.deleteEntry(alias)
        }
    }

    @Test fun realOpenCvTracksStaticTextureAndKeepsAHeartbeat() {
        val gate = StandaloneFlowGate()
        try {
            val jpeg = image()
            assertEquals("start", checked(gate.analyze(jpeg, 0)).getString("reason"))
            var result = JSONObject()
            for (index in 1..25) {
                result = checked(gate.analyze(jpeg, index * 200L))
                assertEquals("identical textured frames must be still", "still", result.getString("state"))
                assertTrue("stationary scene must have tracked points", result.getInt("tracks") >= 12)
            }
            assertEquals("heartbeat", result.getString("reason"))
            assertTrue(result.getBoolean("key"))
        } finally { gate.close() }
    }

    @Test fun realOpenCvSeparatesWholeViewTranslationAndViewSettlement() {
        val gate = StandaloneFlowGate()
        try {
            checked(gate.analyze(image(), 0))
            val shifted = image(shiftX = 12)
            val motion = checked(gate.analyze(shifted, 200))
            assertEquals("whole-view translation should be camera motion", "head", motion.getString("state"))
            assertTrue("translation must produce nonzero camera flow", motion.getDouble("head_rate") > .07)
            assertEquals("still", checked(gate.analyze(shifted, 400)).getString("state"))
            assertEquals("view_settled", checked(gate.analyze(shifted, 800)).getString("reason"))
        } finally { gate.close() }
    }

    @Test fun realOpenCvRetainsLocalObjectMovementAndHandSettlement() {
        val gate = StandaloneFlowGate()
        try {
            checked(gate.analyze(image(patchX = 24), 0))
            val moved = image(patchX = 168)
            val motion = checked(gate.analyze(moved, 200))
            assertEquals("moving local object should not be a head turn", "hands", motion.getString("state"))
            assertEquals("hands_start", motion.getString("reason"))
            assertTrue("local movement needs dense residual or compensated difference",
                motion.getDouble("local_rate") >= .12 || motion.getDouble("changed") >= .06)
            checked(gate.analyze(moved, 400))
            assertEquals("hands_settled", checked(gate.analyze(moved, 800)).getString("reason"))
        } finally { gate.close() }
    }

    @Test fun realOpenCvReportsTexturelessFramesAsUnknownAndResetsAfterGap() {
        val gate = StandaloneFlowGate()
        try {
            val black = image(black = true)
            checked(gate.analyze(black, 0))
            var unclear = false
            for (index in 1..11) {
                val sample = checked(gate.analyze(black, index * 200L))
                assertEquals("unknown", sample.getString("state"))
                if (sample.optString("reason") == "unclear") unclear = true
            }
            assertTrue("untrackable view needs periodic unclear keyframes", unclear)
            assertEquals("gap", checked(gate.analyze(image(), 4_000)).getString("reason"))
            gate.reset()
            assertEquals("start", checked(gate.analyze(image(), 4_200)).getString("reason"))
        } finally { gate.close() }
    }

    @Test fun activityStartsIdleStandaloneWithoutGatewayOrCapture() {
        val existing = onMain { StandaloneService.instance }
        if (existing != null) {
            val initial = onMain { existing.snapshot() }
            assumeTrue("an existing user task must be preserved", initial.optString("phase") == "idle")
            assertIdleService()
            return
        }
        withRestorableState(StandaloneEngine().snapshot(SystemClock.elapsedRealtime())) { assertIdleService() }
    }

    @Test fun restoredTaskRequiresExplicitResumeBeforeAnyCaptureOrModelRequest() {
        assumeTrue("stop the existing service before isolated restore acceptance", onMain { StandaloneService.instance } == null)
        val engine = StandaloneEngine()
        engine.setGoal("设备本地验收：不执行此目标", nowMs = SystemClock.elapsedRealtime())
        withRestorableState(engine.snapshot(SystemClock.elapsedRealtime())) {
            val first = currentSnapshot()
            assertEquals("standalone", first.getString("mode"))
            assertEquals("decomposing", first.getString("phase"))
            assertEquals("user", first.getString("paused_reason"))
            assertFalse(first.getBoolean("capturing"))
            assertFalse(first.getBoolean("busy"))
            SystemClock.sleep(2_200)
            val later = currentSnapshot()
            assertFalse("restored state must never auto-resume capture", later.getBoolean("capturing"))
            assertFalse("restored state must never start an API request", later.getBoolean("busy"))
            assertEquals("user", later.getString("paused_reason"))
            assertNull("PC capture service must remain stopped", onMain { CaptureService.instance })
        }
    }

    private fun assertIdleService() {
        val first = currentSnapshot()
        assertEquals("standalone", first.getString("mode"))
        assertEquals("idle", first.getString("phase"))
        assertFalse(first.getBoolean("capturing"))
        assertFalse(first.getBoolean("busy"))
        assertFalse(first.getBoolean("recording"))
        assertNull("standalone must not start the PC gateway capture path", onMain { CaptureService.instance })
        SystemClock.sleep(2_200)
        val later = currentSnapshot()
        assertFalse(later.getBoolean("capturing"))
        assertFalse(later.getBoolean("busy"))
        assertEquals("idle must not encode any frames", first.getLong("frames"), later.getLong("frames"))
    }

    /** Preserves existing state bytes; refuses to replace any saved user goal or pending profile import. */
    private fun withRestorableState(snapshot: JSONObject, block: () -> Unit) {
        assumeTrue("legacy PC capture must not be interrupted", onMain { CaptureService.instance } == null)
        assumeTrue("existing standalone service must not be interrupted", onMain { StandaloneService.instance } == null)
        assumeTrue("do not consume a pending private profile import", !File(target.filesDir, "standalone-import.json").exists())
        assumeTrue("camera permission must be granted by the installation harness",
            target.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        assumeTrue("microphone permission must be granted by the installation harness",
            target.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
        val state = File(target.filesDir, "standalone-state.json")
        val files = listOf(state, File(state.path + ".bak"), File(state.path + ".new"))
        val backups = files.associateWith { if (it.exists()) it.readBytes() else null }
        for (file in files) if (file.exists()) {
            val original = runCatching { JSONObject(file.readText()) }.getOrNull()
            assumeTrue("saved state is not safe to replace", original != null)
            assumeTrue("saved user goal must be preserved", original!!.isNull("goal"))
        }
        var activity: Activity? = null
        try {
            files.drop(1).forEach { check(!it.exists() || it.delete()) }
            state.writeText(snapshot.toString())
            activity = instrumentation.startActivitySync(Intent(target, StandaloneActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            await("standalone service did not start") { onMain { StandaloneService.instance } != null }
            block()
        } finally {
            try {
                onMain { target.stopService(Intent(target, StandaloneService::class.java)); activity?.finish() }
                await("standalone service did not stop") { onMain { StandaloneService.instance } == null }
            } finally {
                for ((file, original) in backups) {
                    if (original == null) check(!file.exists() || file.delete()) else file.writeBytes(original)
                }
            }
        }
    }

    private fun currentSnapshot(): JSONObject = onMain { StandaloneService.instance!!.snapshot() }

    private fun <T> onMain(action: () -> T): T {
        val task = java.util.concurrent.FutureTask(java.util.concurrent.Callable<T> { action() })
        instrumentation.runOnMainSync(task)
        return try { task.get() }
        catch (failure: java.util.concurrent.ExecutionException) { throw (failure.cause ?: failure) }
    }

    private fun await(message: String, condition: () -> Boolean) {
        val deadline = SystemClock.elapsedRealtime() + 10_000
        while (!condition() && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(50)
        assertTrue(message, condition())
    }

    private fun checked(sample: JSONObject): JSONObject {
        assertTrue("native OpenCV unavailable: ${sample.optString("error")}", sample.getBoolean("available"))
        assertEquals("lk_ransac_farneback", sample.getString("method"))
        assertTrue(sample.getInt("analysis_width") in 1..160)
        assertTrue(sample.getInt("analysis_height") in 1..120)
        listOf("head_rate", "local_rate", "local_area", "changed").forEach { field ->
            assertTrue("non-finite native flow metric", sample.getDouble(field).isFinite())
        }
        return sample
    }

    private fun image(shiftX: Int = 0, patchX: Int? = null, black: Boolean = false): ByteArray {
        val width = 320
        val height = 240
        val random = Random(718L)
        val tiles = IntArray((width / 8) * (height / 8)) { 40 + random.nextInt(176) }
        val pixels = IntArray(width * height) { index ->
            val x = index % width
            val y = index / width
            val sourceX = x - shiftX
            val tone = when {
                black || sourceX !in 0 until width -> 0
                patchX != null && x in patchX until patchX + 72 && y in 80 until 152 ->
                    if (((x - patchX) / 6 + (y - 80) / 6) % 2 == 0) 255 else 0
                else -> tiles[(y / 8) * (width / 8) + sourceX / 8]
            }
            Color.rgb(tone, tone, tone)
        }
        val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        return try {
            ByteArrayOutputStream().use { stream ->
                check(bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream))
                stream.toByteArray()
            }
        } finally { bitmap.recycle() }
    }
}
