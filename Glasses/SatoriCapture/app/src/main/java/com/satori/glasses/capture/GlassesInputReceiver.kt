package com.satori.glasses.capture

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * Rokid system broadcasts documented for bare-metal apps (only delivered to dynamically registered receivers).
 * Touchpad AI_START and opt-in physical long-press can be intercepted only when ordered. Ordinary
 * broadcasts are never described as blocked. Default physical-key behavior stays unchanged for the legacy client.
 */
class GlassesInputReceiver(
    private val interceptAiKey: () -> Boolean,
    private val onAiKey: (ordered: Boolean, aborted: Boolean) -> Unit,
    private val onWear: (Boolean) -> Unit,
    private val onLegs: (Boolean) -> Unit,
    private val onButton: (String) -> Unit,
    private val capturePhysicalLongPress: Boolean = false,
    private val onAiSignal: ((action: String, ordered: Boolean, aborted: Boolean) -> Unit)? = null,
    private val aiStartAction: String = AI_START,
    private val physicalLongPressAction: String = PHYSICAL_LONG_PRESS,
) : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (val action = intent.action) {
            aiStartAction, physicalLongPressAction -> {
                if (action == physicalLongPressAction && !capturePhysicalLongPress) { onButton(action); return }
                val ordered = isOrderedBroadcast
                val abort = ordered && interceptAiKey()
                if (abort) abortBroadcast()
                if (onAiSignal != null) onAiSignal.invoke(action, ordered, abort) else onAiKey(ordered, abort)
            }
            TAKE_STATUS -> intent.getStringExtra("glasses_take_state")?.let { onWear(it == "1") }
            LEG_STATUS -> intent.getStringExtra("glasses_leg_state")?.let { onLegs(it == "1") }
            null -> Unit
            else -> onButton(action)
        }
    }

    companion object {
        const val AI_START = "com.android.action.ACTION_AI_START"
        const val PHYSICAL_LONG_PRESS = "com.android.action.ACTION_SPRITE_BUTTON_LONG_PRESS"
        const val TAKE_STATUS = "com.rokid.sprite.ACTION_TAKE_STATUS_CHANGED"
        const val LEG_STATUS = "com.rokid.sprite.ACTION_LEG_STATUS_CHANGED"
        val ACTIONS = listOf(
            AI_START,
            PHYSICAL_LONG_PRESS,
            TAKE_STATUS,
            LEG_STATUS,
            "com.android.action.ACTION_SPRITE_BUTTON_CLICK",
            "com.android.action.ACTION_SPRITE_BUTTON_DOUBLE_CLICK",
            "com.android.action.ACTION_TWO_FINGER_DOUBLE_TAP",
            "com.android.action.ACTION_TWO_FINGER_SWIPE_FORWARD",
            "com.android.action.ACTION_TWO_FINGER_SWIPE_BACK",
        )
    }
}
