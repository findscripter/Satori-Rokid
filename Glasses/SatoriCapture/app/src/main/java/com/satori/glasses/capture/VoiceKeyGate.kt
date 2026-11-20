package com.satori.glasses.capture

/** One gesture can arrive as both a vendor broadcast and PROG_BLUE. Rejected duplicates do not extend the gate. */
class VoiceKeyGate(private val clock: () -> Long) {
    private var lastAt: Long? = null
    private var lastSource: String? = null

    @Synchronized fun accept(source: String): Boolean {
        require(source.isNotBlank() && source.length <= 160)
        val now = clock()
        val previous = lastAt
        if (previous != null) {
            val gap = now - previous
            val limit = if (source == lastSource) 150 else 500
            if (gap >= 0 && gap < limit) return false
        }
        lastAt = now
        lastSource = source
        return true
    }
}

/** Phone remotes can collapse a held key to a short key pair; two releases remain distinguishable. */
internal class RecordingKeyGate {
    private var previous: Long? = null
    fun reset() { previous = null }
    fun press(atMs: Long): Boolean {
        val gap = previous?.let { atMs - it }
        if (gap != null && gap in 100..700) { reset(); return true }
        if (gap == null || gap < 0 || gap > 700) previous = atMs
        return false
    }
}
