package com.satori.glasses.capture

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.BatteryManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicLong

/**
 * Foreground service that owns the camera, IMU, prompt player and push-to-talk recorder, and streams to the
 * PC gateway. Capture runs only while the gateway is connected, streaming is enabled, and (unless disabled)
 * the glasses are worn and unfolded, so nothing drains the 210 mAh battery without a receiver.
 */
class CaptureService : LifecycleService(), GatewayClient.Listener {
    private val main = Handler(Looper.getMainLooper())
    private val framesSent = AtomicLong()
    private val framesOffline = AtomicLong()
    private val framesOversize = AtomicLong()
    private val imuOffline = AtomicLong()
    private val eventsOffline = AtomicLong()
    private var foregroundTypes = 0
    private val startedNs = SystemClock.elapsedRealtimeNanos()
    private lateinit var prefs: SharedPreferences
    private lateinit var gateway: GatewayClient
    private lateinit var camera: CameraStreamer
    private lateinit var imu: ImuStreamer
    private lateinit var player: PromptPlayer
    private lateinit var voice: VoiceRecorder
    private var receiver: GlassesInputReceiver? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var capturing = false
    @Volatile private var worn: Boolean? = null // written on main, read by helloHeader on the gateway thread
    @Volatile private var legsOpen: Boolean? = null

    @Volatile private var config = CaptureConfig()

    private val statusTask = object : Runnable {
        override fun run() {
            sendStatus()
            main.postDelayed(this, STATUS_MS)
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        prefs = getSharedPreferences("capture", MODE_PRIVATE)
        config = CaptureConfig.fromJson(prefs.getString("config", null))
        startAsForeground()
        gateway = GatewayClient({ config.host to config.port }, this)
        camera = CameraStreamer(this, this, ::onFrame, ::emitEvent)
        imu = ImuStreamer(this) { header, payload -> if (!gateway.send(header, payload)) imuOffline.incrementAndGet() }
        player = PromptPlayer(this, ::emitEvent)
        voice = VoiceRecorder({ header, payload -> gateway.send(header, payload) }, ::emitEvent)
        registerInputs()
        readSystemWearState("service_created")
        gateway.start()
        main.postDelayed(statusTask, STATUS_MS)
        Log.i(TAG, "service created with ${config.toJson()} worn=$worn legs_open=$legsOpen")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (intent == null) {
            // Restarted by the system after a kill: keep the saved config. Camera/mic access may be denied for a
            // background restart; that surfaces as camera_error / ptt_error events and `up` relaunches cleanly.
            Log.i(TAG, "service restarted by the system")
            return START_STICKY
        }
        val updated = CaptureConfig.fromIntent(config, intent)
        val volumeRequested = intent.extras?.containsKey("volume") == true
        if (updated != config || volumeRequested) applyConfig(updated, "intent", volumeRequested)
        return START_STICKY
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        receiver?.let { unregisterReceiver(it) }
        voice.stop("service_stop")
        player.shutdown()
        camera.shutdown()
        imu.stop()
        gateway.stop()
        releaseWakeLock()
        instance = null
        Log.i(TAG, "service destroyed")
        super.onDestroy()
    }

    /** Key events seen by MainActivity while it is in front (started with --ez stay true). */
    fun onKey(keyCode: Int, action: String) {
        emitEvent("key", JSONObject().put("code", keyCode).put("action", action))
        if (keyCode == android.view.KeyEvent.KEYCODE_ENTER && action == "up" && voice.active) voice.stop("tap")
    }

    // region GatewayClient.Listener (called on gateway threads)

    override fun helloHeader(): JSONObject = JSONObject()
        .put("proto", Wire.PROTOCOL_VERSION)
        .put("app_version", BuildConfig.VERSION_NAME)
        .put(
            "device",
            JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
                .put("sdk", Build.VERSION.SDK_INT).put("fingerprint", Build.FINGERPRINT),
        )
        .put("boot_count", Settings.Global.getInt(contentResolver, Settings.Global.BOOT_COUNT, -1))
        .put("boot_ns", SystemClock.elapsedRealtimeNanos())
        .put("wall_ms", System.currentTimeMillis())
        .put("config", config.toJson())
        .put("worn", worn ?: JSONObject.NULL)
        .put("legs_open", legsOpen ?: JSONObject.NULL)

    override fun onConnected() {
        main.post {
            readSystemWearState("gateway_connected")
            updateCapture("gateway_connected")
        }
    }

    /**
     * Main thread only. The take/leg broadcasts report changes only, so until one arrives the state is unknown, and an
     * app started with the glasses off would capture. The Rokid system keeps the current state in vendor properties
     * (vendor.rkd.glasses.is_take_on / is_spread, "1" or "0"; not in the official docs, seen on firmware 1.27.009):
     * fill whatever is still unknown from them. A broadcast always wins afterwards.
     */
    private fun readSystemWearState(reason: String) {
        if (worn != null && legsOpen != null) return
        val take = if (worn == null) systemFlag("vendor.rkd.glasses.is_take_on") else null
        val spread = if (legsOpen == null) systemFlag("vendor.rkd.glasses.is_spread") else null
        if (take != null) worn = take
        if (spread != null) legsOpen = spread
        Log.i(TAG, "wear state from system properties ($reason): worn=$take legs_open=$spread")
        if (take != null || spread != null) {
            emitEvent(
                "wear_initial",
                JSONObject().put("worn", take ?: JSONObject.NULL).put("legs_open", spread ?: JSONObject.NULL)
                    .put("source", "system_property").put("reason", reason),
            )
        }
    }

    private fun systemFlag(name: String): Boolean? = try {
        val get = Class.forName("android.os.SystemProperties").getMethod("get", String::class.java, String::class.java)
        when (get.invoke(null, name, "") as String) {
            "1" -> true
            "0" -> false
            else -> null // missing, or not readable by apps (SELinux)
        }
    } catch (e: Exception) {
        Log.w(TAG, "cannot read $name: $e")
        null
    }

    override fun onDisconnected(reason: String) {
        main.post {
            voice.stop("gateway_disconnected")
            updateCapture("gateway_disconnected")
        }
    }

    override fun onMessage(message: WireMessage) {
        val header = message.header
        when (message.type) {
            "ping" -> {
                val received = SystemClock.elapsedRealtimeNanos()
                gateway.sendUrgent(
                    JSONObject().put("type", "pong").put("pc_ns", header.optLong("pc_ns")).put("rx_ns", received)
                        .put("tx_ns", SystemClock.elapsedRealtimeNanos()),
                )
            }
            "config" -> main.post {
                val updated = try {
                    config.merged(header)
                } catch (e: IllegalArgumentException) {
                    emitEvent("config_rejected", JSONObject().put("error", e.message ?: e.toString()))
                    return@post
                }
                applyConfig(updated, "gateway", volumeRequested = header.has("volume"))
            }
            "play_audio" -> {
                if (voice.active) voice.stop("prompt")
                player.play(header.optString("id"), message.payload, header.optString("mode") == "immediate")
            }
            "stop_audio" -> player.stop()
            "ptt_stop" -> voice.stop("gateway")
            "hello_ack" -> Log.i(TAG, "gateway session ${header.optString("session_id")}")
            else -> Log.w(TAG, "ignored message type ${message.type}")
        }
    }

    // endregion

    private fun onFrame(header: JSONObject, jpeg: ByteArray) {
        if (jpeg.size > Wire.MAX_PAYLOAD - 64 * 1024) {
            // Would make Wire.write throw and drop the link; lower width/height/jpeg_quality instead.
            if (framesOversize.incrementAndGet() % 50 == 1L) {
                emitEvent("frame_too_large", JSONObject().put("bytes", jpeg.size).put("limit", Wire.MAX_PAYLOAD))
            }
            return
        }
        if (gateway.sendFrame(header, jpeg)) framesSent.incrementAndGet() else framesOffline.incrementAndGet()
    }

    private fun emitEvent(name: String, data: JSONObject) {
        Log.i(TAG, "event $name $data")
        val sent = gateway.send(
            JSONObject().put("type", "event").put("name", name).put("ts_ns", SystemClock.elapsedRealtimeNanos())
                .put("data", data),
        )
        if (!sent) eventsOffline.incrementAndGet()
    }

    private fun shouldCapture(): Boolean =
        gateway.connected && config.streaming && !(config.pauseWhenRemoved && (worn == false || legsOpen == false))

    /** Main thread only. */
    private fun updateCapture(reason: String) {
        val want = shouldCapture()
        if (want == capturing) return
        capturing = want
        if (want) {
            camera.start(config)
            imu.start(config.imuHz)
            acquireWakeLock()
        } else {
            camera.stop()
            imu.stop()
            releaseWakeLock()
        }
        emitEvent(
            if (want) "capture_resumed" else "capture_paused",
            JSONObject().put("reason", reason).put("worn", worn ?: JSONObject.NULL)
                .put("legs_open", legsOpen ?: JSONObject.NULL),
        )
    }

    /** Main thread only. */
    private fun applyConfig(updated: CaptureConfig, source: String, volumeRequested: Boolean = false) {
        val old = config
        config = updated
        prefs.edit().putString("config", updated.toJson().toString()).apply()
        if (capturing) {
            if (updated.width != old.width || updated.height != old.height) {
                camera.stop()
                camera.start(updated)
            } else {
                camera.applyRate(updated)
            }
            if (updated.imuHz != old.imuHz) imu.start(updated.imuHz)
        }
        if (updated.host != old.host || updated.port != old.port) gateway.reconnect()
        // Only an explicit volume request touches the stream volume (compared with the live level, which the temple
        // keys may have changed); other config changes never undo the wearer's own volume choice.
        if (volumeRequested && updated.volume >= 0) applyVolume(updated.volume)
        updateCapture("config_$source")
        emitEvent("config_applied", updated.toJson().put("source", source))
    }

    private fun applyVolume(level: Int) {
        val audio = getSystemService(AudioManager::class.java)
        val target = level.coerceIn(0, audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
        if (audio.getStreamVolume(AudioManager.STREAM_MUSIC) == target) return
        try {
            audio.setStreamVolume(AudioManager.STREAM_MUSIC, target, 0)
        } catch (e: SecurityException) { // e.g. Do Not Disturb policy
            emitEvent("volume_error", JSONObject().put("error", e.toString()))
        }
    }

    private fun registerInputs() {
        val r = GlassesInputReceiver(
            // Without a PC link our push-to-talk has nowhere to go, so leave the long press to the Rokid assistant.
            interceptAiKey = { config.interceptAiKey && gateway.connected },
            onAiKey = { ordered, aborted -> main.post { onAiKey(ordered, aborted) } },
            onWear = { value ->
                main.post {
                    worn = value
                    emitEvent("wear", JSONObject().put("worn", value))
                    if (!value) voice.stop("removed")
                    updateCapture("wear")
                }
            },
            onLegs = { value ->
                main.post {
                    legsOpen = value
                    emitEvent("legs", JSONObject().put("open", value))
                    updateCapture("legs")
                }
            },
            onButton = { action -> emitEvent("button", JSONObject().put("action", action)) },
        )
        val filter = IntentFilter().apply {
            priority = IntentFilter.SYSTEM_HIGH_PRIORITY
            GlassesInputReceiver.ACTIONS.forEach(::addAction)
        }
        ContextCompat.registerReceiver(this, r, filter, ContextCompat.RECEIVER_EXPORTED)
        receiver = r
    }

    private fun onAiKey(ordered: Boolean, aborted: Boolean) {
        emitEvent(
            "ai_key",
            JSONObject().put("ordered", ordered).put("aborted", aborted).put("intercept", config.interceptAiKey)
                .put("ptt_active", voice.active),
        )
        if (!aborted) return // the Rokid assistant handles this press
        if (voice.active) {
            voice.stop("ai_key")
        } else if (gateway.connected) {
            ensureMicrophoneForeground()
            player.stop()
            voice.start("ai_key")
        }
    }

    /** RECORD_AUDIO may have been granted after onCreate; the foreground type must include microphone to record. */
    private fun ensureMicrophoneForeground() {
        if (foregroundTypes and ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE != 0) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startAsForeground()
        }
    }

    private fun sendStatus() {
        if (capturing) acquireWakeLock() // the 4 h wake-lock timeout would otherwise end long sessions silently
        val audio = getSystemService(AudioManager::class.java)
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val power = getSystemService(PowerManager::class.java)
        gateway.send(
            JSONObject().put("type", "status").put("ts_ns", SystemClock.elapsedRealtimeNanos())
                .put("uptime_s", (SystemClock.elapsedRealtimeNanos() - startedNs) / 1e9)
                .put("battery_pct", if (level >= 0 && scale > 0) level * 100 / scale else JSONObject.NULL)
                .put("battery_temp_c", (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0)
                .put("plugged", battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0)
                .put("thermal", power.currentThermalStatus)
                .put("capturing", capturing)
                .put("worn", worn ?: JSONObject.NULL)
                .put("legs_open", legsOpen ?: JSONObject.NULL)
                .put("frames_encoded", camera.framesEncoded)
                .put("frames_sent", framesSent.get())
                .put("frames_offline", framesOffline.get())
                .put("frames_oversize", framesOversize.get())
                .put("imu_batches_offline", imuOffline.get())
                .put("events_offline", eventsOffline.get())
                .put("volume", audio.getStreamVolume(AudioManager.STREAM_MUSIC))
                .put("volume_max", audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
                .put("imu", imu.stats())
                .put("link", gateway.stats())
                .put("ptt_active", voice.active)
                .put("playing", player.playing)
                .put("config", config.toJson()),
        )
    }

    private fun startAsForeground() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Satori Capture", NotificationManager.IMPORTANCE_LOW),
        )
        val notification = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.presence_video_online)
            .setContentTitle("Satori capture")
            .setOngoing(true)
            .build()
        var types = ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, types)
        foregroundTypes = types
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SatoriCapture:capture")
            .apply {
                setReferenceCounted(false)
                acquire(WAKE_LOCK_TIMEOUT_MS)
            }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    companion object {
        const val ACTION_STOP = "com.satori.glasses.capture.STOP"
        private const val TAG = "SatoriCapture"
        private const val CHANNEL_ID = "capture"
        private const val NOTIFICATION_ID = 47800
        private const val STATUS_MS = 2000L
        private const val WAKE_LOCK_TIMEOUT_MS = 4 * 60 * 60 * 1000L

        @Volatile var instance: CaptureService? = null
            private set
    }
}
