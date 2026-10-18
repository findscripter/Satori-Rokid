"""Pure CPU output-accounting checks; never loads weights or runs a model."""
import pytest

from run_offline_smoke import summarize_new_tokens


class FakeSequence(list):
    def tolist(self):
        return list(self)


class FakeTokenizer:
    all_special_ids = [8, 9]
    names = {7: "visible action", 8: "<keep_silent>", 9: "<|im_end|>"}

    def convert_ids_to_tokens(self, token):
        return self.names[token]

    def batch_decode(self, sequences, skip_special_tokens):
        return ["".join(self.names[token] for token in sequences[0]
                        if not skip_special_tokens or token not in self.all_special_ids)]


@pytest.mark.parametrize("ids,silent,finish,count", [
    ([7, 9], False, "stop", 2), ([8, 7, 9], True, "stop", 3), ([7, 7], False, "length", 2),
])
def test_counts_actual_new_tokens_and_retains_silence(ids, silent, finish, count):
    result = summarize_new_tokens(FakeTokenizer(), [FakeSequence(ids)], 9, [9], "<keep_silent>")
    assert result["completion_tokens"] == count
    assert result["generated_token_ids"] == ids
    assert result["keep_silent_emitted"] is silent
    assert result["finish_reason"] == finish
    assert result["actual_eos_observed"] is (finish == "stop")
    if silent:
        assert result["decoded_without_special_tokens"] == "visible action"
        assert "<keep_silent>" in result["raw_response_with_special_tokens"]
        assert any(item["token"] == "<keep_silent>" for item in result["raw_special_tokens"])
