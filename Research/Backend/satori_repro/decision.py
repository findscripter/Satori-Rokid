"""Conservative, inspectable interruption policy for the offline baseline.

This deliberately small rule policy does not claim to reproduce Satori's learned
or prompted decision model. Model text remains data; no output is executed.
"""

from .models import _timestamp
from .state import CONFIDENCE_THRESHOLD


def decide(state, previous_decision=None, *, timestamp_ms, cooldown_ms=15000):
    """Return a display decision; callers retain only actual non-observe prompts."""
    timestamp_ms = _timestamp(timestamp_ms)
    cooldown_ms = _timestamp(cooldown_ms, label="cooldown_ms")
    if previous_decision is not None:
        previous_time = _timestamp(previous_decision.get("timestamp_ms"))
        if previous_time > timestamp_ms:
            raise ValueError("previous_decision cannot come from the future")

    evidence = list(state.get("evidence_frame_ids", []))
    result = {"kind": "observe", "message": "", "reason": "no_help_needed",
              "evidence_frame_ids": evidence, "timestamp_ms": timestamp_ms}
    steps = state.get("task_step_ids", [])
    confirmed = set(state.get("confirmed_completed_step_ids", []))
    if not state.get("task_confirmed", False):
        result.update(kind="clarify", message=f"请确认：你正在进行“{state.get('task_title', state['task_id'])}”吗？",
                      reason="task_unconfirmed")
    elif steps and set(steps) <= confirmed:
        result.update(kind="complete", message="所有步骤都已由你确认完成。", reason="all_steps_user_confirmed")
    elif state.get("uncertain", True) or state.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
        result.update(kind="clarify", message="当前画面不足以可靠判断进度，请确认你正在进行的步骤。",
                      reason="uncertain_observation")
    elif state.get("needs_help", False):
        help_text = state.get("help_text", "").strip()
        if evidence and help_text:
            result.update(kind="guide", message=help_text, reason="evidenced_help_needed")
        else:
            result.update(kind="clarify", message="需要更多画面或步骤确认，才能给出具体帮助。",
                          reason="insufficient_guidance_evidence")

    if result["kind"] == "observe" or previous_decision is None:
        return result
    repeated = (previous_decision.get("kind") == result["kind"]
                and previous_decision.get("message") == result["message"])
    if repeated:
        result.update(kind="observe", message="", reason="repeat_suppressed")
    elif result["kind"] != "complete" and timestamp_ms - previous_time < cooldown_ms:
        result.update(kind="observe", message="", reason="cooldown_suppressed")
    return result
