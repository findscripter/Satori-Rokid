package com.satori.glasses.capture

import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.nio.charset.StandardCharsets

/** One protocol message: a JSON header (always has "type") plus an optional binary payload. */
class WireMessage(val header: JSONObject, val payload: ByteArray = EMPTY) {
    val type: String get() = header.optString("type")

    companion object {
        val EMPTY = ByteArray(0)
    }
}

/**
 * Framing shared with Research/Backend/satori_repro/glasses/protocol.py:
 * `"SATG"` magic, u32 header length, u32 payload length (both big-endian), UTF-8 JSON header, payload bytes.
 */
object Wire {
    const val PROTOCOL_VERSION = 1
    const val MAX_HEADER = 64 * 1024
    const val MAX_PAYLOAD = 8 * 1024 * 1024
    private val MAGIC = "SATG".toByteArray(StandardCharsets.US_ASCII)

    fun write(output: DataOutputStream, message: WireMessage) {
        val header = message.header.toString().toByteArray(StandardCharsets.UTF_8)
        require(header.size <= MAX_HEADER) { "header too large: ${header.size}" }
        require(message.payload.size <= MAX_PAYLOAD) { "payload too large: ${message.payload.size}" }
        output.write(MAGIC)
        output.writeInt(header.size)
        output.writeInt(message.payload.size)
        output.write(header)
        output.write(message.payload)
        output.flush()
    }

    fun read(input: DataInputStream): WireMessage {
        val magic = ByteArray(MAGIC.size)
        input.readFully(magic)
        if (!magic.contentEquals(MAGIC)) throw IOException("bad magic")
        val headerLength = input.readInt()
        val payloadLength = input.readInt()
        if (headerLength !in 2..MAX_HEADER || payloadLength !in 0..MAX_PAYLOAD) {
            throw IOException("bad lengths $headerLength/$payloadLength")
        }
        val header = ByteArray(headerLength)
        input.readFully(header)
        val payload = if (payloadLength == 0) WireMessage.EMPTY else ByteArray(payloadLength).also { input.readFully(it) }
        return WireMessage(JSONObject(String(header, StandardCharsets.UTF_8)), payload)
    }
}
