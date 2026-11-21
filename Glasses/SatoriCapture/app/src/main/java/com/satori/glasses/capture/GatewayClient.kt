package com.satori.glasses.capture

import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.LinkedBlockingDeque
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/**
 * TCP link to the PC gateway. One thread connects and writes, one thread reads.
 * Camera frames use a single "latest" slot (an unsent older frame is replaced, never queued);
 * IMU batches, events, status and audio are queued in order. Nothing is buffered while offline.
 * Exactly one onDisconnected follows each onConnected, whichever thread notices the loss first.
 */
class GatewayClient(
    private val endpoint: () -> Pair<String, Int>,
    private val listener: Listener,
) {
    interface Listener {
        fun helloHeader(): JSONObject
        fun onConnected()
        fun onDisconnected(reason: String)
        fun onMessage(message: WireMessage)
    }

    private val control = LinkedBlockingDeque<WireMessage>()
    private val latestFrame = AtomicReference<WireMessage?>(null)
    private val linked = AtomicBoolean(false)
    private val sequence = AtomicLong()
    private val framesReplaced = AtomicLong()
    private val controlDropped = AtomicLong()
    private val connections = AtomicLong()

    @Volatile private var running = false
    @Volatile private var socket: Socket? = null
    @Volatile private var linkedSocket: Socket? = null
    @Volatile private var lastReceivedMs = 0L
    @Volatile private var acknowledged = false
    @Volatile var lastError = ""
        private set
    private var worker: Thread? = null

    val connected: Boolean get() = linked.get()

    fun start() {
        if (running) return
        running = true
        worker = Thread(::runLoop, "gateway-link").apply {
            isDaemon = true
            start()
        }
    }

    fun stop() {
        running = false
        closeSocket()
        worker?.interrupt()
        worker = null
    }

    /** Drops the current connection so the next attempt uses a changed endpoint. */
    fun reconnect() = closeSocket()

    /** Queues a non-frame message. Returns false when offline (the message is dropped). */
    fun send(header: JSONObject, payload: ByteArray = WireMessage.EMPTY): Boolean {
        if (!linked.get()) return false
        if (control.size >= MAX_CONTROL) {
            control.pollFirst()
            controlDropped.incrementAndGet()
        }
        control.offerLast(stamp(WireMessage(header, payload)))
        return true
    }

    /** Puts a message at the head of the queue (clock-sync replies). */
    fun sendUrgent(header: JSONObject): Boolean {
        if (!linked.get()) return false
        control.offerFirst(stamp(WireMessage(header)))
        return true
    }

    /** Offers a camera frame; an older frame that has not been written yet is replaced. */
    fun sendFrame(header: JSONObject, jpeg: ByteArray): Boolean {
        if (!linked.get()) return false
        if (latestFrame.getAndSet(stamp(WireMessage(header, jpeg))) != null) framesReplaced.incrementAndGet()
        return true
    }

    fun stats(): JSONObject = JSONObject()
        .put("connected", linked.get())
        .put("connections", connections.get())
        .put("queued", control.size)
        .put("frames_replaced", framesReplaced.get())
        .put("control_dropped", controlDropped.get())
        .put("last_error", lastError)

    private fun stamp(message: WireMessage): WireMessage {
        message.header.put("seq", sequence.incrementAndGet())
        return message
    }

    private fun runLoop() {
        var backoffMs = INITIAL_BACKOFF_MS
        while (running) {
            val (host, port) = endpoint()
            var reason = "closed"
            var linkedThisRound = false
            acknowledged = false
            val s = Socket()
            socket = s // registered before connect() so a failed attempt is closed too (no descriptor leak)
            try {
                s.tcpNoDelay = true
                s.keepAlive = true
                s.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MS)
                // The gateway pings every 2 s: a read that waits longer means a stalled link. Closing the socket on
                // that timeout also unblocks a writer stuck on a full send buffer.
                s.soTimeout = READ_TIMEOUT_MS
                val output = DataOutputStream(BufferedOutputStream(s.getOutputStream(), 64 * 1024))
                val input = DataInputStream(BufferedInputStream(s.getInputStream(), 64 * 1024))
                control.clear()
                latestFrame.set(null)
                lastReceivedMs = SystemClock.elapsedRealtime()
                Wire.write(output, stamp(WireMessage(listener.helloHeader().put("type", "hello"))))
                linkedSocket = s
                linked.set(true)
                linkedThisRound = true
                connections.incrementAndGet()
                Log.i(TAG, "connected to $host:$port, waiting for hello_ack")
                val reader = Thread({ readLoop(input, s) }, "gateway-read").apply {
                    isDaemon = true
                    start()
                }
                writeLoop(output)
                reader.join(1000)
            } catch (e: InterruptedException) {
                reason = "interrupted"
            } catch (e: Exception) {
                reason = "${e.javaClass.simpleName}: ${e.message ?: ""}"
            } finally {
                if (linkedThisRound) {
                    markDown(reason, s)
                } else {
                    lastError = reason
                    closeQuietly(s)
                }
                if (socket === s) socket = null
            }
            if (!running) break
            // A gateway that accepted us (hello_ack) gets a quick retry; a refusing or absent one backs off to 5 s.
            if (acknowledged) backoffMs = INITIAL_BACKOFF_MS
            try {
                Thread.sleep(backoffMs)
            } catch (_: InterruptedException) {
                break
            }
            backoffMs = (backoffMs * 2).coerceAtMost(MAX_BACKOFF_MS)
        }
    }

    private fun writeLoop(output: DataOutputStream) {
        while (running && linked.get()) {
            // The gateway pings every 2 s; prolonged silence means a stalled link that TCP may never report.
            if (SystemClock.elapsedRealtime() - lastReceivedMs > SILENCE_TIMEOUT_MS) {
                throw IOException("no message from the gateway for ${SILENCE_TIMEOUT_MS / 1000} s")
            }
            val message = control.pollFirst(20, TimeUnit.MILLISECONDS)
            if (message != null) {
                // Stamp clock-sync replies at the last moment so queueing delay does not bias the offset.
                if (message.type == "pong") message.header.put("tx_ns", SystemClock.elapsedRealtimeNanos())
                Wire.write(output, message)
                continue
            }
            val frame = latestFrame.getAndSet(null) ?: continue
            Wire.write(output, frame)
        }
    }

    private fun readLoop(input: DataInputStream, s: Socket) {
        var reason = "closed by gateway"
        try {
            while (running) {
                val message = Wire.read(input)
                lastReceivedMs = SystemClock.elapsedRealtime()
                if (message.type == "hello_ack" && !acknowledged) {
                    acknowledged = true
                    // Capture starts only once the gateway accepted this link; a refused attempt never opens the camera.
                    listener.onConnected()
                }
                listener.onMessage(message)
            }
        } catch (e: Exception) {
            reason = "read: ${e.javaClass.simpleName} ${e.message ?: ""}"
        } finally {
            markDown(reason, s)
        }
    }

    /**
     * Single place that ends a link. It only closes the socket it is given, so a late reader of an old connection
     * cannot close the next one, and only the first caller for the current link notifies the listener.
     */
    private fun markDown(reason: String, s: Socket) {
        closeQuietly(s)
        if (linkedSocket === s && linked.compareAndSet(true, false)) {
            linkedSocket = null
            lastError = reason
            Log.i(TAG, "disconnected: $reason")
            listener.onDisconnected(reason)
        }
    }

    private fun closeSocket() {
        val s = socket ?: return
        socket = null
        closeQuietly(s)
    }

    private fun closeQuietly(s: Socket) {
        try {
            s.close()
        } catch (_: Exception) {
        }
    }

    companion object {
        private const val TAG = "SatoriCapture"
        private const val CONNECT_TIMEOUT_MS = 3000
        private const val INITIAL_BACKOFF_MS = 500L
        private const val MAX_BACKOFF_MS = 5000L
        private const val MAX_CONTROL = 4000
        private const val SILENCE_TIMEOUT_MS = 12_000L
        private const val READ_TIMEOUT_MS = 15_000
    }
}
