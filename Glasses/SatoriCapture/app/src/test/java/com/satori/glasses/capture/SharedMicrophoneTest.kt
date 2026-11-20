package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.Semaphore

class SharedMicrophoneTest {
    private class FakeSource(val pcm: (JSONObject, ByteArray) -> Unit,
        val raw: (JSONObject, ByteArray) -> Unit, val event: (String, JSONObject) -> Unit) : SharedMicrophone.Source {
        var starts = 0
        var stops = 0
        var fail = false
        override fun start(): Int { starts++; if (fail) throw IllegalStateException("fake source unavailable"); return starts }
        override fun stop(reason: String) { stops++ }
        fun emit(value: Byte = 1, channels: Int = 8) {
            val header = JSONObject().put("utterance", starts).put("ts_ns", 123_000_000L).put("source_channels", channels)
            pcm(header, byteArrayOf(value, 0, value, 0))
            raw(header, byteArrayOf(if (channels == 8) 9 else value, 0, 9, 0))
        }
    }

    @Test fun twoSubscribersShareOneSourceAndOneCancellationDoesNotStopTheOther() {
        lateinit var source: FakeSource
        val created = AtomicInteger()
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { pcm, raw, event ->
            created.incrementAndGet(); FakeSource(pcm, raw, event).also { source = it }
        })
        val first = CountDownLatch(1); val second = CountDownLatch(2)
        val one = microphone.subscribe({ _, _ -> first.countDown() }, { _, _ -> })
        val two = microphone.subscribe({ _, _ -> second.countDown() }, { _, _ -> })
        try {
            assertEquals(1, created.get()); assertEquals(1, source.starts)
            source.emit()
            assertTrue(first.await(2, TimeUnit.SECONDS))
            one.close(); assertEquals(0, source.stops); assertEquals(1, microphone.subscriberCount)
            source.emit()
            assertTrue(second.await(2, TimeUnit.SECONDS))
            assertEquals(4L, microphone.pcmSamples)
            two.close(); assertEquals(1, source.stops); assertEquals(0, microphone.subscriberCount)
            assertEquals(1L, microphone.sourceStarts)
        } finally { one.close(); two.close(); microphone.close() }
    }

    @Test fun rawAndProcessedChannelsUseTheSameSourceAndMonoFallbackIsExplicit() {
        lateinit var source: FakeSource
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { pcm, raw, event -> FakeSource(pcm, raw, event).also { source = it } })
        val processed = ArrayList<Byte>(); val raw = ArrayList<Byte>(); val done = CountDownLatch(4)
        val one = microphone.subscribe({ ts, bytes -> assertEquals(123_000_000L, ts); synchronized(processed) { processed.add(bytes[0]) }; done.countDown() }, { _, _ -> })
        val two = microphone.subscribe({ _, bytes -> synchronized(raw) { raw.add(bytes[0]) }; done.countDown() }, { _, _ -> }, raw = true)
        try {
            source.emit(1, 8); source.emit(2, 1)
            assertTrue(done.await(2, TimeUnit.SECONDS))
            assertEquals(listOf<Byte>(1, 2), processed); assertEquals(listOf<Byte>(9, 2), raw)
            assertEquals("mono_fallback", microphone.stats().getString("audio_mode"))
            assertFalse(microphone.stats().getBoolean("raw_available"))
            assertEquals(1, source.starts)
        } finally { one.close(); two.close(); microphone.close() }
    }

    @Test fun startFailureIsDeliveredWithoutLeakingHardwareOrSubscriptions() {
        val error = CountDownLatch(1)
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { pcm, raw, event -> FakeSource(pcm, raw, event).apply { fail = true } })
        val subscription = microphone.subscribe({ _, _ -> fail("failed source emitted PCM") }, { name, _ -> if (name == "ptt_error") error.countDown() })
        try { assertTrue(error.await(2, TimeUnit.SECONDS)); assertEquals(1L, microphone.sourceStarts) }
        finally { subscription.close(); microphone.close() }
        assertEquals(0, microphone.subscriberCount)
    }

    @Test fun blockedConsumerDoesNotBlockOtherConsumerOrCreateUnboundedQueues() {
        lateinit var source: FakeSource
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { pcm, raw, event -> FakeSource(pcm, raw, event).also { source = it } }, callbackCapacity = 2)
        val entered = CountDownLatch(1); val release = CountDownLatch(1); val other = Semaphore(0); val error = CountDownLatch(1)
        val one = microphone.subscribe({ _, _ -> entered.countDown(); release.await(2, TimeUnit.SECONDS) }, { name, data ->
            if (name == "ptt_error" && data.optString("reason") == "backpressure") error.countDown()
        })
        val two = microphone.subscribe({ _, _ -> other.release() }, { _, _ -> })
        try {
            source.emit(); assertTrue(entered.await(2, TimeUnit.SECONDS)); assertTrue(other.tryAcquire(2, TimeUnit.SECONDS))
            repeat(4) { source.emit(); assertTrue(other.tryAcquire(2, TimeUnit.SECONDS)) }
            assertEquals(1, microphone.subscriberCount)
            assertEquals(0, source.stops)
            assertTrue(microphone.stats().getLong("callback_overflows") > 0)
            release.countDown()
            assertTrue("backpressure must be an explicit terminal error", error.await(2, TimeUnit.SECONDS))
        } finally { release.countDown(); one.close(); two.close(); microphone.close() }
    }

    @Test fun latePacketsFromPriorStartCannotEnterTheNewSubscription() {
        lateinit var source: FakeSource
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { pcm, raw, event -> FakeSource(pcm, raw, event).also { source = it } })
        microphone.subscribe({ _, _ -> }, { _, _ -> }).close()
        val received = CountDownLatch(1); val count = AtomicInteger()
        val current = microphone.subscribe({ _, _ -> count.incrementAndGet(); received.countDown() }, { _, _ -> })
        try {
            source.pcm(JSONObject().put("utterance", 1).put("ts_ns", 1), byteArrayOf(1, 0))
            source.emit()
            assertTrue(received.await(2, TimeUnit.SECONDS)); assertEquals(1, count.get()); assertEquals(2, source.starts)
        } finally { current.close(); microphone.close() }
    }

    @Test fun hardwareFailureTerminatesBothSubscriptionsWithAnExplicitError() {
        lateinit var source: FakeSource
        val microphone = SharedMicrophone(SharedMicrophone.SourceFactory { pcm, raw, event -> FakeSource(pcm, raw, event).also { source = it } })
        val errors = CountDownLatch(2)
        val one = microphone.subscribe({ _, _ -> fail("failed source must not send PCM") }, { name, data ->
            if (name == "ptt_error" && data.optString("reason") == "source_error") errors.countDown()
        })
        val two = microphone.subscribe({ _, _ -> fail("failed source must not send PCM") }, { name, data ->
            if (name == "ptt_error" && data.optString("reason") == "source_error") errors.countDown()
        }, raw = true)
        try {
            source.event("ptt_error", JSONObject().put("utterance", source.starts).put("error", "fake read error"))
            assertTrue(errors.await(2, TimeUnit.SECONDS))
            assertEquals(0, microphone.subscriberCount)
            assertEquals(1, source.stops)
        } finally { one.close(); two.close(); microphone.close() }
    }
}
