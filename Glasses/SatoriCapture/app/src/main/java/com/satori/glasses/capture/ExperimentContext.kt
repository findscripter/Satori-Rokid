package com.satori.glasses.capture

import org.json.JSONObject

/** Pseudonymous trial labels only; secrets and free-form participant identities are rejected. */
object ExperimentContext {
    fun validate(value: JSONObject): JSONObject {
        val fields = setOf("schema", "participant_code", "protocol_id", "trial_label", "task_variant")
        require(value.keys().asSequence().all { it in fields })
        require(value.optString("schema") == "satori.experiment-context/1")
        val result = JSONObject().put("schema", "satori.experiment-context/1")
        for (key in fields - "schema") {
            val text = value.opt(key)
            if (text == null || text == JSONObject.NULL) result.put(key, JSONObject.NULL)
            else {
                require(text is String && text.matches(Regex("[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")))
                result.put(key, text)
            }
        }
        return result
    }

    fun defaults() = JSONObject().put("schema", "satori.experiment-context/1")
        .put("participant_code", JSONObject.NULL).put("protocol_id", "classroom-demo-v1")
        .put("trial_label", JSONObject.NULL).put("task_variant", JSONObject.NULL)
}
