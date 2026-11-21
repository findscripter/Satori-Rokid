package com.satori.glasses.capture

data class StandaloneFrame(val id: String, val atMs: Long, val jpeg: ByteArray, val key: String? = null)

/** Short, bounded RAM window. Recording on disk is independent and never deleted by this ring. */
class StandaloneFrameBuffer(private val capacity: Int = 180) {
    private val frames = ArrayDeque<StandaloneFrame>()
    val size: Int @Synchronized get() = frames.size

    @Synchronized fun add(frame: StandaloneFrame) {
        if (frames.lastOrNull()?.let { it.atMs >= frame.atMs } == true) return
        frames.addLast(frame)
        while (frames.size > capacity) frames.removeFirst()
    }

    @Synchronized fun clear() = frames.clear()

    fun awaitFresh(afterMs: Long, deadlineMs: Long, nowMs: () -> Long, cancelled: () -> Boolean,
                   pause: (Long) -> Unit = { Thread.sleep(it) }): List<StandaloneFrame> {
        // Wait outside the buffer monitor so the camera producer can publish its first frame.
        while (!cancelled()) {
            val at = nowMs()
            if (at >= deadlineMs) return emptyList()
            val available = select(afterMs, at, 2)
            if (available.isNotEmpty()) return if (cancelled()) emptyList() else available
            pause(minOf(50L, deadlineMs - at))
        }
        return emptyList()
    }

    @Synchronized fun select(afterMs: Long, nowMs: Long, limit: Int = 12): List<StandaloneFrame> {
        require(limit in 1..12)
        val recent = frames.filter { it.atMs > afterMs && it.atMs >= nowMs - 90_000 && it.atMs <= nowMs }
        if (recent.size <= limit) return recent
        val selected = linkedSetOf(recent.lastIndex)
        // Spread keyframes as well: repeated hands_ongoing must not crowd out the whole window.
        val priorities = listOf("hands_settled", "hands_start", "view_settled", "hands_ongoing")
        for (key in priorities) {
            val indices = recent.indices.filter { recent[it].key == key }
            if (indices.isNotEmpty() && selected.size < limit / 2 + 1) selected.add(indices.last())
        }
        if (limit > 1) selected.add(0)
        val uniform = (0 until limit).map { (it * recent.lastIndex.toDouble() / (limit - 1).coerceAtLeast(1)).toInt() }
        for (index in uniform + recent.indices) {
            if (selected.size >= limit) break
            selected.add(index)
        }
        return selected.sorted().map { recent[it] }
    }
}
