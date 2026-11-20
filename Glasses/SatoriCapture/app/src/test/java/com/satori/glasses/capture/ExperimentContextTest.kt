package com.satori.glasses.capture

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ExperimentContextTest {
    @Test fun absentParticipantIsExplicitlyUnknownAndSecretFieldsAreRejected() {
        val valid = ExperimentContext.validate(ExperimentContext.defaults().put("trial_label", "pilot-01"))
        assertTrue(valid.isNull("participant_code"))
        for (bad in listOf(valid.put("api_key", "secret"), ExperimentContext.defaults().put("participant_code", "Person Name"))) {
            try { ExperimentContext.validate(bad); fail("unbounded or secret metadata") } catch (_: IllegalArgumentException) { }
        }
    }
}
