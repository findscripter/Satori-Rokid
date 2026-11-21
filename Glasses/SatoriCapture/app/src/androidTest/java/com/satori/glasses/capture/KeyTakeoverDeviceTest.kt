package com.satori.glasses.capture

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.UUID

/**
 * Synthetic, per-case random unprotected action aliases only. Constructor injection changes this
 * fixture receiver's action matching; production defaults/system ACTIONS are not modified. Never
 * send a real ACTION_AI_START or physical LONG_PRESS: this firmware protects them even from shell.
 * Every Intent is also package-limited to Satori, so Rokid's real assistant/power receivers cannot
 * receive these aliases even if an assertion fails.
 * The low-priority receiver is an isolated surrogate, not the native assistserver. No hardware
 * key, Activity navigation, microphone, camera or model is started; this is not physical-key proof.
 */
@RunWith(AndroidJUnit4::class)
class KeyTakeoverDeviceTest {
    private data class Signal(val action: String, val ordered: Boolean, val aborted: Boolean)
    private data class Result(val signals: List<Signal>, val observedButtons: List<String>, val tailCount: Int)
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val target get() = instrumentation.targetContext

    @Test fun orderedAiStartStopsTheLowerPrioritySamePackageReceiver() {
        val result = broadcastCase(GlassesInputReceiver.AI_START, ordered = true, capturePhysical = true)
        assertEquals(listOf(Signal(GlassesInputReceiver.AI_START, true, true)), result.signals)
        assertEquals(0, result.tailCount)
    }

    @Test fun orderedPhysicalLongPressCanBeExplicitlyTakenOver() {
        val result = broadcastCase(PHYSICAL_LONG_PRESS, ordered = true, capturePhysical = true)
        assertEquals(listOf(Signal(PHYSICAL_LONG_PRESS, true, true)), result.signals)
        assertEquals(0, result.tailCount)
    }

    @Test fun unorderedAiStartIsObservedButNeverReportedAsAborted() {
        val result = broadcastCase(GlassesInputReceiver.AI_START, ordered = false, capturePhysical = true)
        assertEquals(listOf(Signal(GlassesInputReceiver.AI_START, false, false)), result.signals)
        assertEquals("ordinary broadcasts remain visible to other receivers", 1, result.tailCount)
    }

    @Test fun unorderedPhysicalLongPressCannotBeClaimedAsIntercepted() {
        val result = broadcastCase(PHYSICAL_LONG_PRESS, ordered = false, capturePhysical = true)
        assertEquals(listOf(Signal(PHYSICAL_LONG_PRESS, false, false)), result.signals)
        assertEquals(1, result.tailCount)
    }

    @Test fun legacyDefaultDoesNotCapturePhysicalLongPress() {
        val result = broadcastCase(PHYSICAL_LONG_PRESS, ordered = true, capturePhysical = null)
        assertEquals(1, result.tailCount)
        assertTrue(result.signals.none { it.aborted })
        assertEquals(listOf(PHYSICAL_LONG_PRESS), result.observedButtons)
    }

    @Test fun disabledTakeoverPreservesTheOrderedAiEventForOtherConsumers() {
        val result = broadcastCase(GlassesInputReceiver.AI_START, ordered = true, capturePhysical = true, intercept = false)
        assertEquals(listOf(Signal(GlassesInputReceiver.AI_START, true, false)), result.signals)
        assertEquals(1, result.tailCount)
    }

    @Test fun syntheticAiBroadcastAndProgBlueRouteShareOneGestureGate() {
        val clock = AtomicLong(1000)
        val gate = VoiceKeyGate { clock.get() }
        val accepted = AtomicInteger()
        val result = broadcastCase(GlassesInputReceiver.AI_START, ordered = true, capturePhysical = true) { signal ->
            if (signal.aborted && gate.accept("broadcast:${signal.action}")) accepted.incrementAndGet()
        }
        assertEquals(0, result.tailCount)
        // A synthetic source-route signal exercises the same pure gate as KEYCODE_PROG_BLUE.
        // It does not inject a real hardware key or claim Activity.dispatchKeyEvent coverage.
        clock.set(1020)
        if (gate.accept("key:186")) accepted.incrementAndGet()
        assertEquals("one physical gesture's two routes must not start then immediately stop PTT", 1, accepted.get())
        clock.set(1500)
        assertTrue("a later independent route must remain usable", gate.accept("key:186"))
    }

    private fun broadcastCase(action: String, ordered: Boolean, capturePhysical: Boolean?, intercept: Boolean = true,
                              onSignal: (Signal) -> Unit = {}): Result {
        var noExistingReceivers = false
        instrumentation.runOnMainSync {
            noExistingReceivers = StandaloneService.instance == null && CaptureService.instance == null
        }
        assumeTrue("existing application receivers/user tasks must not be replaced or compete with the fixture", noExistingReceivers)
        val suffix = UUID.randomUUID().toString().replace("-", "")
        val aiAlias = "com.satori.glasses.capture.TEST_AI_START_$suffix"
        val physicalAlias = "com.satori.glasses.capture.TEST_PHYSICAL_LONG_PRESS_$suffix"
        val selectedAlias = when (action) {
            GlassesInputReceiver.AI_START -> aiAlias
            PHYSICAL_LONG_PRESS -> physicalAlias
            else -> throw IllegalArgumentException("unsupported synthetic contract")
        }
        Log.i("KeyTakeoverDeviceTest", "alias=$selectedAlias ordered=$ordered native_delivery=false package=${target.packageName}")
        val signals = CopyOnWriteArrayList<Signal>()
        val buttons = CopyOnWriteArrayList<String>()
        val observed = CountDownLatch(2)
        val finished = CountDownLatch(1)
        val tailCount = AtomicInteger()
        val signal: (String, Boolean, Boolean) -> Unit = { seenAction, isOrdered, aborted ->
            Signal(seenAction, isOrdered, aborted).also {
                signals += it
                // Route only the expected alias. Unexpected callback actions fail below on the test
                // thread instead of throwing an AssertionError in Android's receiver/main thread.
                if (seenAction == selectedAlias) onSignal(it.copy(action = action))
            }
            observed.countDown()
        }
        val button: (String) -> Unit = { buttons += it; observed.countDown() }
        val receiver = if (capturePhysical == null) {
            GlassesInputReceiver(interceptAiKey = { intercept }, onAiKey = { _, _ -> },
                onWear = {}, onLegs = {}, onButton = button, onAiSignal = signal,
                aiStartAction = aiAlias, physicalLongPressAction = physicalAlias)
        } else {
            GlassesInputReceiver(interceptAiKey = { intercept }, onAiKey = { _, _ -> },
                onWear = {}, onLegs = {}, onButton = button, capturePhysicalLongPress = capturePhysical,
                onAiSignal = signal, aiStartAction = aiAlias, physicalLongPressAction = physicalAlias)
        }
        val tail = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                tailCount.incrementAndGet()
                observed.countDown()
            }
        }
        val resultReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) { finished.countDown() }
        }
        var registered = false
        var tailRegistered = false
        try {
            ContextCompat.registerReceiver(target, receiver, IntentFilter(selectedAlias).apply { priority = 1000 }, ContextCompat.RECEIVER_EXPORTED)
            registered = true
            ContextCompat.registerReceiver(target, tail, IntentFilter(selectedAlias).apply { priority = -1000 }, ContextCompat.RECEIVER_EXPORTED)
            tailRegistered = true
            val intent = Intent(selectedAlias).setPackage(target.packageName).addFlags(Intent.FLAG_RECEIVER_REGISTERED_ONLY)
            assertEquals("test broadcasts must never be system-wide", target.packageName, intent.`package`)
            assertEquals(selectedAlias, intent.action)
            if (ordered) target.sendOrderedBroadcast(intent, null, resultReceiver, Handler(Looper.getMainLooper()), 0, null, null)
            else target.sendBroadcast(intent)
            if (ordered) assertTrue("synthetic ordered chain did not finish", finished.await(5, TimeUnit.SECONDS))
            else assertTrue("both ordinary receivers must observe the non-abortable event", observed.await(5, TimeUnit.SECONDS))
            assertTrue("callbacks must carry the fixture alias, not a fabricated real system action", signals.all { it.action == selectedAlias })
            assertTrue("unhandled legacy callbacks must carry the same alias", buttons.all { it == selectedAlias })
            return Result(signals.map { it.copy(action = action) }, buttons.map { action }, tailCount.get())
        } finally {
            if (tailRegistered) target.unregisterReceiver(tail)
            if (registered) target.unregisterReceiver(receiver)
        }
    }

    companion object {
        private const val PHYSICAL_LONG_PRESS = "com.android.action.ACTION_SPRITE_BUTTON_LONG_PRESS"
    }
}
