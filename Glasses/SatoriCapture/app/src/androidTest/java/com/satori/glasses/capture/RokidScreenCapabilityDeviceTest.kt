package com.satori.glasses.capture

import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Process
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Opt-in metadata/no-recipient probe. Never addresses the vendor receiver or requests recording.
 * A normal send return only means the system did not reject this caller/action; it proves no capture capability.
 */
@RunWith(AndroidJUnit4::class)
class RokidScreenCapabilityDeviceTest {
    @Suppress("DEPRECATION")
    @Test fun inspectWithoutStartingRecording() {
        assumeTrue("use -e satoriRokidCapability true for the no-recording capability probe",
            InstrumentationRegistry.getArguments().getString("satoriRokidCapability") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val packages = context.packageManager
        val uid = Process.myUid()
        val vendor = ComponentName("com.rokid.os.master.screenstream",
            "com.rokid.os.master.screenstream.receiver.ScreenRecordReceiver")
        val absent = ComponentName(context.packageName,
            "${context.packageName}.SatoriRokidCapabilityAbsentReceiver")
        val probes = JSONArray()
        val report = JSONObject().put("schema", "satori.rokid-screen-capability/1")
            .put("caller_uid", uid).put("target_uid", context.applicationInfo.uid)
            .put("target_package", context.packageName).put("probe_component", absent.flattenToString())
            .put("vendor_component", vendor.flattenToString()).put("probes", probes)
            .put("vendor_broadcast_sent", false).put("recording_requested", false)
            .put("recording_verified", false).put("completed", false)
            .put("scope", "receiver metadata and no-recipient broadcast rejection only")
        var stage = "verify_caller_uid"
        try {
            assertEquals("must run as the target app UID", context.applicationInfo.uid, uid)
            // Android assigns each user's apps app IDs >= 10000; exclude root/system/shell in every user.
            assertTrue("must be an ordinary app UID", uid % 100_000 >= 10_000)

            stage = "query_vendor_receiver"
            val receiver = JSONObject()
            report.put("receiver", receiver)
            try {
                val info = packages.getReceiverInfo(vendor, PackageManager.MATCH_DISABLED_COMPONENTS)
                assertEquals(vendor.packageName, info.packageName)
                assertEquals(vendor.className, info.name)
                receiver.put("query_status", "resolved").put("enabled", info.enabled)
                    .put("application_enabled", info.applicationInfo.enabled).put("exported", info.exported)
                    .put("uid", info.applicationInfo.uid).put("permission", info.permission ?: JSONObject.NULL)
                info.permission?.let { permission ->
                    receiver.put("caller_has_permission",
                        context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED)
                }
            } catch (_: PackageManager.NameNotFoundException) {
                // Package visibility can hide a real receiver from this ordinary app.
                receiver.put("query_status", "not_visible_or_missing")
            } catch (failure: SecurityException) {
                receiver.put("query_status", "security_exception").put("exception_type", failure.javaClass.name)
            }

            stage = "verify_absent_receiver"
            assertEquals(context.packageName, absent.packageName)
            assertNotEquals(vendor, absent)
            var absentConfirmed = false
            try {
                packages.getReceiverInfo(absent, PackageManager.MATCH_DISABLED_COMPONENTS)
            } catch (_: PackageManager.NameNotFoundException) {
                absentConfirmed = true
            }
            assertTrue("probe must never reach an installed receiver", absentConfirmed)
            report.put("probe_receiver_absent", true)

            stage = "send_only_to_absent_receiver"
            for (action in listOf("com.rokid.yodaos.action.SCREENRECORD_ON",
                "com.rokid.yodaos.action.SCREENRECORD_OFF")) {
                val intent = Intent(action).setComponent(absent).setPackage(context.packageName)
                assertEquals(absent, intent.component)
                assertEquals(context.packageName, intent.`package`)
                val result = JSONObject().put("action", action)
                probes.put(result)
                try {
                    context.sendBroadcast(intent)
                    result.put("result", "not_rejected_no_recipient")
                } catch (failure: SecurityException) {
                    result.put("result", "security_exception").put("exception_type", failure.javaClass.name)
                }
            }
            report.put("completed", true)
        } finally {
            report.put("last_stage", stage)
            instrumentation.sendStatus(0, Bundle().apply { putString("rokid_screen_capability_report", report.toString()) })
            println("rokid_screen_capability_report=$report")
        }
    }
}
