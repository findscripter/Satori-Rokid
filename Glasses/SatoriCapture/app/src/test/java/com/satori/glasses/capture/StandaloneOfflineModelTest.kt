package com.satori.glasses.capture

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

class StandaloneOfflineModelTest {
    private val emptyHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    @Test fun officialEmptyOnlineCmvnConfigIsCopiedAndVerified() {
        val output = ByteArrayOutputStream()
        StandaloneOfflineModel.copyVerified(ByteArrayInputStream(byteArrayOf()), output, 0, emptyHash)
        assertEquals(0, output.size())
    }

    @Test fun zeroByteManifestCannotHideNonemptyDataOrWrongHash() {
        rejected(byteArrayOf(1), 0, emptyHash)
        rejected(byteArrayOf(), 0, "0".repeat(64))
        rejected(byteArrayOf(), 1, emptyHash)
    }

    @Test fun impossibleManifestFileSizesAreRejectedBeforeCopy() {
        rejected(byteArrayOf(), -1, emptyHash)
        rejected(byteArrayOf(), 100_000_001, emptyHash)
    }

    private fun rejected(bytes: ByteArray, expected: Long, hash: String) {
        try {
            StandaloneOfflineModel.copyVerified(ByteArrayInputStream(bytes), ByteArrayOutputStream(), expected, hash)
            fail("accepted inconsistent model asset")
        } catch (_: IllegalArgumentException) { }
    }
}
