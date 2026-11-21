package com.satori.glasses.capture

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Binder
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Parcel
import org.json.JSONObject
import java.io.Closeable
import java.util.UUID
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Future
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Adapter for the exported, device-local Rokid TtsService, verified against this device's APK.
 * Only play (transaction 1) and stop-by-UUID (transaction 2) are used; vendor settings are never changed.
 * Binding happens on main, blocking speech on the caller's background worker, Binder IPC on one
 * bounded daemon worker. A blocked Binder never blocks main or creates an unbounded thread pool.
 * The owner serializes utterances; late callbacks are accepted only for the current unique UUID.
 */
class StandaloneVendorSpeech(context: Context, private val onEvent: (String, JSONObject) -> Unit) : Closeable {
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val remote = AtomicReference<IBinder?>(null)
    private val current = AtomicReference<Utterance?>(null)
    private val closed = AtomicBoolean()
    private val disabled = AtomicBoolean()
    private val speaking = AtomicBoolean()
    private val generation = AtomicInteger()
    private val ipc = ThreadPoolExecutor(1, 1, 30, TimeUnit.SECONDS, ArrayBlockingQueue<Runnable>(1),
        { task -> Thread(task, "rokid-local-tts-ipc").apply { isDaemon = true } }, ThreadPoolExecutor.AbortPolicy())
        .apply { allowCoreThreadTimeOut(true) }
    @Volatile private var bindFailure = ""
    private var bound = false // main-thread only

    val ready: Boolean get() = !closed.get() && !disabled.get() && remote.get()?.isBinderAlive == true

    private class Utterance(val id: String, val uuid: String, val binder: IBinder) {
        val started = AtomicBoolean()
        val stopped = AtomicBoolean()
        val cancelled = AtomicBoolean()
        val stopScheduled = AtomicBoolean()
        val finished = CountDownLatch(1)
        @Volatile var submitted = false
        @Volatile var command: Future<Boolean>? = null
        @Volatile var failure = ""
    }

    private val listener = object : Binder() {
        override fun onTransact(code: Int, data: Parcel, reply: Parcel?, flags: Int): Boolean {
            if (code == IBinder.INTERFACE_TRANSACTION) { reply?.writeString(LISTENER); return true }
            if (code != 1 && code != 2) return super.onTransact(code, data, reply, flags)
            data.enforceInterface(LISTENER)
            val uuid = data.readString()
            reply?.writeNoException()
            val utterance = current.get()
            if (utterance == null || uuid != utterance.uuid || utterance.cancelled.get() || closed.get()) return true
            if (code == 1) {
                if (utterance.started.compareAndSet(false, true)) emit("audio_started", utterance)
            } else {
                utterance.stopped.set(true)
                utterance.finished.countDown()
            }
            return true
        }
    }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, service: IBinder) {
            if (closed.get()) { unbind(); return }
            bindFailure = ""
            remote.set(service)
        }
        override fun onServiceDisconnected(name: ComponentName) = disconnected("binder_disconnected")
        override fun onBindingDied(name: ComponentName) = disconnected("binding_died")
        override fun onNullBinding(name: ComponentName) = disconnected("null_binding")
    }

    init {
        main.post {
            if (!closed.get()) {
                try {
                    bound = app.bindService(Intent(ACTION).setComponent(ComponentName(PACKAGE, SERVICE)), connection, Context.BIND_AUTO_CREATE)
                    if (!bound) bindFailure = "bind_rejected"
                } catch (_: Exception) { bindFailure = "bind_failed" }
            }
        }
    }

    /** Returns only after start+stop callbacks, cancellation, failure, or a 45-second total deadline. */
    fun speakBlocking(text: String, id: String, shouldContinue: () -> Boolean): Boolean {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            error(id, "main_thread_call")
            return false
        }
        if (text.isBlank() || text.length > 2_000 || id.isBlank() || id.length > 200) {
            error(id.take(200), "invalid_request")
            return false
        }
        if (closed.get() || !speaking.compareAndSet(false, true)) return false
        val ticket = generation.get()
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(45)
        fun permitted(): Boolean = !closed.get() && ticket == generation.get() &&
            try { shouldContinue() } catch (_: Exception) { false }
        var utterance: Utterance? = null
        var success = false
        try {
            val bindDeadline = minOf(deadline, System.nanoTime() + TimeUnit.SECONDS.toNanos(3))
            while (!ready && permitted() && !disabled.get() && bindFailure.isEmpty() && System.nanoTime() < bindDeadline) {
                Thread.sleep(25)
            }
            if (!permitted()) return false
            if (!ready) { error(id, bindFailure.ifEmpty { if (disabled.get()) "ipc_disabled" else "bind_timeout" }); return false }
            val binder = remote.get() ?: return false
            val item = Utterance(id, UUID.randomUUID().toString(), binder)
            utterance = item
            current.set(item)
            if (!permitted()) { item.cancelled.set(true); return false }
            val command = try {
                ipc.submit(Callable {
                    if (item.cancelled.get() || closed.get()) return@Callable false
                    transact(item.binder, 1) { parcel ->
                        parcel.writeString(text)
                        parcel.writeString(item.uuid)
                        parcel.writeStrongBinder(listener)
                    }
                })
            } catch (_: RejectedExecutionException) {
                item.failure = "ipc_busy"
                return false
            }
            item.submitted = true
            item.command = command
            while (permitted() && !item.cancelled.get() && System.nanoTime() < deadline) {
                try {
                    if (!command.get(waitMillis(deadline), TimeUnit.MILLISECONDS)) return false
                    break
                } catch (_: TimeoutException) { /* poll cancellation without blocking the owner's worker */ }
                catch (interrupted: InterruptedException) { throw interrupted }
                catch (_: Exception) { item.failure = "ipc_failed"; return false }
            }
            if (!permitted() || item.cancelled.get()) return false
            while (!item.stopped.get() && item.failure.isEmpty() && permitted() && System.nanoTime() < deadline) {
                item.finished.await(waitMillis(deadline), TimeUnit.MILLISECONDS)
            }
            if (System.nanoTime() >= deadline) {
                item.failure = "ipc_or_playback_timeout"
                disabled.set(true)
                return false
            }
            success = permitted() && !item.cancelled.get() && item.failure.isEmpty() && item.started.get() && item.stopped.get()
            if (success) emit("audio_done", item, "vendor_onTtsStop")
            return success
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
            utterance?.cancelled?.set(true)
            return false
        } finally {
            utterance?.let { item ->
                if (!success) {
                    item.cancelled.set(true)
                    item.command?.cancel(true)
                    ipc.purge()
                    requestStop(item)
                    if (item.failure.isNotEmpty()) error(item.id, item.failure) else emit("audio_stopped", item)
                }
                current.compareAndSet(item, null)
            }
            speaking.set(false)
        }
    }

    /** Cancels only this adapter's current UUID. Never calls vendor parameter/configuration methods. */
    fun stop() {
        generation.incrementAndGet()
        current.get()?.let { item ->
            item.cancelled.set(true)
            item.finished.countDown()
            item.command?.cancel(true)
            ipc.purge()
            requestStop(item)
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        stop()
        remote.set(null)
        main.post { unbind() }
        // Drain a queued stop after an in-flight play transaction; a stuck Binder uses one daemon only.
        ipc.shutdown()
    }

    private fun requestStop(item: Utterance) {
        if (!item.submitted || !item.stopScheduled.compareAndSet(false, true)) return
        try { ipc.execute { runCatching { transact(item.binder, 2) { it.writeString(item.uuid) } } } }
        catch (_: RejectedExecutionException) { /* bounded/stopped executor: no replacement thread is created */ }
    }

    private fun transact(binder: IBinder, code: Int, write: (Parcel) -> Unit): Boolean {
        val data = Parcel.obtain()
        val reply = Parcel.obtain()
        return try {
            data.writeInterfaceToken(SERVER)
            write(data)
            if (!binder.transact(code, data, reply, 0)) return false
            reply.readException()
            true
        } finally { reply.recycle(); data.recycle() }
    }

    private fun disconnected(reason: String) {
        remote.set(null)
        bindFailure = reason
        current.get()?.let { it.failure = reason; it.finished.countDown() }
    }

    private fun unbind() {
        if (bound) { bound = false; runCatching { app.unbindService(connection) } }
    }

    private fun waitMillis(deadline: Long) = TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime()).coerceIn(1, 100)

    private fun emit(name: String, item: Utterance, signal: String? = null) {
        val data = JSONObject().put("id", item.id).put("provider", "rokid_local")
        if (signal != null) data.put("completion_signal", signal)
        runCatching { onEvent(name, data) }
    }

    private fun error(id: String, reason: String) {
        runCatching { onEvent("audio_error", JSONObject().put("id", id).put("provider", "rokid_local")
            .put("reason", reason).put("error", "本机语音服务暂不可用")) }
    }

    companion object {
        private const val PACKAGE = "com.rokid.os.sprite.assistserver"
        private const val SERVICE = "com.rokid.os.sprite.tts.TtsService"
        private const val ACTION = "com.rokid.os.sprite.tts.TTS_SERVICE"
        private const val SERVER = "com.rokid.os.sprite.tts.ITtsServer"
        private const val LISTENER = "com.rokid.os.sprite.tts.ITtsListener"
    }
}
