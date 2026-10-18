"""CPU-only protocol tests. No model weights, credentials, or network are read."""

import base64
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("mage_local_server", Path(__file__).with_name("server.py"))
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


def decode_image_for_test(raw, mime):
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image")
    return image


def image_url():
    ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def request_body():
    return {"model": "mage-local", "stream": False, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "Return JSON."},
                         {"role": "user", "content": [
                             {"type": "text", "text": "Frame ID: v000; timestamp_ms: 5500"},
                             {"type": "image_url", "image_url": {"url": image_url()}},
                             {"type": "text", "text": "Where am I and what am I doing?"},
                         ]}]}


def parse(body):
    return server.parse_chat_request(body, "mage-local", image_decoder=decode_image_for_test)


def test_accepts_stage_provider_messages_and_keeps_frame_text():
    parsed = parse(request_body())
    assert parsed.json_mode is True and parsed.max_tokens == 1024
    assert len(parsed.images) == 1
    assert "Frame ID: v000; timestamp_ms: 5500" in str(parsed.messages)
    assert parsed.messages[1]["content"][1] == {"type": "image"}
    assert "base64" not in str(parsed.messages)


@pytest.mark.parametrize("url", ["https://example.org/a.png", "file:///tmp/a.png", "/tmp/a.png",
                                "data:text/plain;base64,YWJj", "data:image/png;base64,not-base64!"])
def test_rejects_external_urls_and_nonimages(url):
    body = request_body()
    body["messages"][1]["content"][1]["image_url"]["url"] = url
    with pytest.raises(server.APIError):
        parse(body)


@pytest.mark.parametrize("patch", [
    {"stream": True}, {"model": "other"}, {"max_tokens": 1025},
    {"max_tokens": True}, {"max_tokens": 0}, {"tools": []},
    {"tool_choice": "auto"}, {"response_format": {"type": "json_schema"}},
])
def test_rejects_unsupported_requests_without_model_execution(patch):
    body = request_body()
    body.update(patch)
    with pytest.raises(server.APIError):
        parse(body)


def test_limits_images_and_individual_decoded_bytes(monkeypatch):
    body = request_body()
    part = body["messages"][1]["content"][1]
    body["messages"][1]["content"] = [part] * 33
    with pytest.raises(server.APIError):
        parse(body)
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 8)
    with pytest.raises(server.APIError):
        parse(request_body())


def test_valid_magic_with_corrupt_pixels_is_rejected():
    body = request_body()
    body["messages"][1]["content"][1]["image_url"]["url"] = (
        "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nbroken").decode())
    with pytest.raises(server.APIError):
        parse(body)


def test_body_size_nonfinite_and_duplicate_json_keys_are_rejected():
    with pytest.raises(server.APIError):
        server.decode_request_json(b" " * 11, max_bytes=10)
    with pytest.raises(server.APIError):
        server.decode_request_json(b'{"x":NaN}')
    with pytest.raises(server.APIError):
        server.decode_request_json(b'{"x":1,"x":2}')


def test_auth_compares_only_bearer_token():
    token = "test-only-token-01234567890123456789"
    assert server.valid_authorization("Bearer " + token, token) is True
    assert server.valid_authorization("Bearer wrong", token) is False
    assert server.valid_authorization(None, token) is False


def test_json_response_is_honest_about_no_constrained_decoding_and_usage():
    result = server.GenerationResult('{"answer":"visible table"}', 50, 8, "stop")
    response = server.build_completion(parse(request_body()), result, "mage-local", "revision-test", 123.5)
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"] == {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58}
    assert response["server_metadata"]["constrained_decoding"] is False
    assert response["server_metadata"]["revision"] == "revision-test"
    assert response["server_metadata"]["latency_ms"] == 123.5


@pytest.mark.parametrize("text,reason", [
    ("not json", "stop"), ("[]", "stop"), ('{"answer":', "length"),
    ('{"answer":"valid but truncated generation"}', "length"),
])
def test_invalid_or_truncated_json_returns_502_without_output_echo(text, reason):
    result = server.GenerationResult(text, 50, 8, reason)
    with pytest.raises(server.APIError) as caught:
        server.build_completion(parse(request_body()), result, "mage-local", "revision-test", 3)
    assert caught.value.status_code == 502
    assert caught.value.finish_reason == reason
    assert text not in str(caught.value)


def test_plain_text_mode_reports_length_instead_of_claiming_stop():
    body = request_body()
    body.pop("response_format")
    parsed = parse(body)
    result = server.GenerationResult("partial text", 50, 8, "length")
    response = server.build_completion(parsed, result, "mage-local", "revision-test", 3)
    assert response["choices"][0]["finish_reason"] == "length"


def test_token_file_validation_uses_only_test_owned_file(tmp_path):
    path = tmp_path / "token"
    path.write_text("test-token-01234567890123456789\n")
    assert server.read_token(path) == "test-token-01234567890123456789"
    path.write_text("invalid token with spaces")
    with pytest.raises(ValueError, match="token"):
        server.read_token(path)


def test_model_adapter_uses_official_bounded_processor_and_bf16_inputs(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    calls = {}
    class PixelTensor:
        def __init__(self):
            self.dtype = "float32"
        def to(self, device=None, *, dtype=None):
            if device is not None:
                calls.setdefault("pixel_devices", []).append(device)
            if dtype is not None:
                self.dtype = dtype
            return self
    pixel = PixelTensor()
    class Processor:
        tokenizer = SimpleNamespace(model_max_length=131072)
        image_processor = SimpleNamespace(min_pixels=1, max_pixels=9999999,
                                          size={"shortest_edge":1,"longest_edge":9999999})
        def apply_chat_template(self, messages, **kwargs):
            calls["template_kwargs"] = kwargs
            return "prompt"
        def __call__(self, **kwargs):
            calls["processor_kwargs"] = kwargs
            return {"input_ids": np.array([[1,2]]), "pixel_values": pixel}
        def batch_decode(self, sequence, **kwargs):
            return ["{}"]
    processor = Processor()
    class Model:
        config = SimpleNamespace(_commit_hash=None, text_config=SimpleNamespace(max_position_embeddings=262144))
        generation_config = SimpleNamespace(eos_token_id=9)
        device = "cuda:0"
        dtype = "bfloat16"
        def eval(self): return self
        def to(self, device): return self
        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return np.array([[1,2,3,9]])
    model = Model()
    class ProcessorFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["processor_load_kwargs"] = kwargs
            return processor
    class ModelFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_load_kwargs"] = kwargs
            return model
    torch = SimpleNamespace(__version__="test", bfloat16="bfloat16", inference_mode=nullcontext,
                            cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
                                                 empty_cache=lambda:None))
    class LocalConfig:
        model_type = "mage_vl"
    class ConfigFactory:
        @staticmethod
        def register(name, config_class, **kwargs):
            calls["registered_config"] = (name, config_class, kwargs)
    def load_local_config(class_reference, path, **kwargs):
        calls["dynamic_config"] = (class_reference, path, kwargs)
        return LocalConfig
    transformers = SimpleNamespace(__version__="test", AutoConfig=ConfigFactory,
                                   AutoProcessor=ProcessorFactory, AutoModelForCausalLM=ModelFactory)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils",
                        SimpleNamespace(get_class_from_dynamic_module=load_local_config))
    config = server.ServerConfig(tmp_path,"mage-local","test-revision",tmp_path/"unused-token")
    backend = server.ModelBackend(config)
    assert calls["dynamic_config"] == ("configuration_mage_vl.MageVLConfig", str(tmp_path), {"local_files_only":True})
    assert calls["registered_config"] == ("mage_vl", LocalConfig, {"exist_ok":True})
    assert calls["processor_load_kwargs"]["min_pixels"] == 3136
    assert calls["processor_load_kwargs"]["max_pixels"] == 150000
    assert processor.image_processor.min_pixels == 3136
    assert processor.image_processor.max_pixels == 150000
    assert processor.image_processor.size == {"shortest_edge":3136,"longest_edge":150000}
    assert calls["model_load_kwargs"]["dtype"] == "bfloat16"
    assert calls["model_load_kwargs"]["device_map"] == {"":0}
    assert backend.metadata["observed_successful_input_tokens"] == 0
    result = backend.generate(parse(request_body()))
    assert backend.metadata["observed_successful_input_tokens"] == 2
    assert backend.metadata["validated_max_input_tokens"] is None
    assert calls["template_kwargs"]["enable_thinking"] is False
    assert pixel.dtype == model.dtype
    assert calls["pixel_devices"] == ["cuda:0"]
    assert calls["generate_kwargs"]["do_sample"] is False
    assert result.prompt_tokens == 2 and result.completion_tokens == 2
    assert result.finish_reason == "stop"


def test_official_size_settings_replace_conflicting_legacy_dimensions():
    from types import SimpleNamespace
    processor = SimpleNamespace(image_processor=SimpleNamespace(
        size={"shortest_edge":1,"longest_edge":9999999,"height":4096,"width":4096}))
    server.configure_image_limits(processor)
    assert processor.image_processor.size == {"shortest_edge":3136,"longest_edge":150000}


def test_pixel_bounds_support_frozen_hf_size_dict_without_min_max_attributes():
    from dataclasses import dataclass
    from types import SimpleNamespace
    @dataclass(frozen=True)
    class SizeDict:
        height: int | None = None
        width: int | None = None
        longest_edge: int | None = None
        shortest_edge: int | None = None
        max_height: int | None = None
        max_width: int | None = None
    original = SizeDict(height=4096,width=4096,longest_edge=9999999,shortest_edge=1)
    processor = SimpleNamespace(image_processor=SimpleNamespace(size=original))
    server.configure_image_limits(processor)
    configured = processor.image_processor.size
    assert isinstance(configured, SizeDict)
    assert configured.shortest_edge == 3136 and configured.longest_edge == 150000
    assert configured.height is None and configured.width is None
    assert configured.max_height is None and configured.max_width is None
    assert original.height == 4096 and original.longest_edge == 9999999


def expanded_config(tmp_path, **patch):
    settings = dict(max_context_tokens=131072,max_new_tokens=8192,default_new_tokens=4096,
                    max_text_chars=2_000_000,max_images=128)
    settings.update(patch)
    return server.ServerConfig(tmp_path,"mage-local","revision-test",tmp_path/"token",**settings)


def test_configured_admission_limits_reach_parser_and_preserve_old_defaults(tmp_path):
    config = expanded_config(tmp_path)
    parsed = server.parse_chat_request(request_body(),"mage-local",config=config,image_decoder=decode_image_for_test)
    assert parsed.max_tokens == 4096
    body = request_body()
    body["max_tokens"] = 8192
    assert server.parse_chat_request(body,"mage-local",config=config,image_decoder=decode_image_for_test).max_tokens == 8192
    body["max_tokens"] = 8193
    with pytest.raises(server.APIError):
        server.parse_chat_request(body,"mage-local",config=config,image_decoder=decode_image_for_test)
    assert parse(request_body()).max_tokens == 1024
    body = request_body()
    part = body["messages"][1]["content"][1]
    body["messages"][1]["content"] = [part] * 33
    assert len(server.parse_chat_request(body,"mage-local",config=config,image_decoder=decode_image_for_test).images) == 33
    body["messages"][1]["content"] = [part] * 129
    with pytest.raises(server.APIError):
        server.parse_chat_request(body,"mage-local",config=config,image_decoder=decode_image_for_test)


@pytest.mark.parametrize("patch",[
    {"max_context_tokens":1_000_000},{"max_context_tokens":True},{"max_new_tokens":8193},
    {"default_new_tokens":8193},{"max_text_chars":2_000_001},{"max_images":129},
    {"max_images":0},{"max_new_tokens":0},
])
def test_unsafe_configurations_are_rejected_before_model_load(tmp_path,patch):
    with pytest.raises(ValueError):
        expanded_config(tmp_path,**patch)


def test_context_metadata_separates_declared_operational_and_validated_capacity(tmp_path):
    from types import SimpleNamespace
    config = expanded_config(tmp_path)
    metadata = server.context_metadata(config,SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144)),
                                       SimpleNamespace(model_max_length=131072))
    assert metadata["model_context_window"] == 262144
    assert metadata["tokenizer_context_window"] == 131072
    assert metadata["operational_context_tokens"] == 131072
    assert metadata["validated_max_input_tokens"] is None
    assert metadata["one_million_supported"] is False
    assert metadata["default_new_tokens"] == 4096 and metadata["max_new_tokens"] == 8192


def test_cpu_token_budget_counts_reserved_output_and_actual_model_window(tmp_path):
    config = expanded_config(tmp_path,max_context_tokens=16,max_new_tokens=4,default_new_tokens=4)
    budget = server.check_context_budget(12,4,config,model_context_window=32,tokenizer_context_window=32)
    assert budget["input_tokens"] == 12 and budget["reserved_output"] == 4
    with pytest.raises(server.APIError) as caught:
        server.check_context_budget(13,4,config,model_context_window=32,tokenizer_context_window=32)
    assert caught.value.code == "context_length_exceeded"
    assert caught.value.budget["max_context"] == 16
    assert caught.value.usage is None
    with pytest.raises(server.APIError):
        server.check_context_budget(10,4,config,model_context_window=12,tokenizer_context_window=32)


def test_budget_rejection_precedes_any_gpu_transfer_or_generate(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import threading
    moves = []
    class Inputs(dict):
        def to(self,*args,**kwargs):
            moves.append("gpu")
            return self
    class Processor:
        def apply_chat_template(self,*args,**kwargs): return "rendered"
        def __call__(self,**kwargs):
            assert kwargs["truncation"] is False
            return Inputs(input_ids=np.zeros((1,14),dtype=np.int64))
    backend=server.ModelBackend.__new__(server.ModelBackend)
    backend.config=expanded_config(tmp_path,max_context_tokens=16,max_new_tokens=4,default_new_tokens=4)
    backend.model_context_window=32
    backend.tokenizer_context_window=32
    backend.processor=Processor()
    backend.torch=SimpleNamespace(inference_mode=nullcontext)
    backend._lock=threading.Lock()
    def generate(**kwargs):
        moves.append("generate")
        raise AssertionError("must not generate")
    backend.model=SimpleNamespace(device="cuda:0",dtype="bfloat16",generate=generate)
    spec=server.ChatRequest("mage-local",[],[],4,0.0,False)
    with pytest.raises(server.APIError,match="context"):
        backend.generate(spec)
    assert moves==[]


def test_configured_text_limit_is_used_instead_of_old_module_constant(tmp_path):
    body=request_body()
    body["messages"][0]["content"]="x"*100001
    config=expanded_config(tmp_path)
    server.parse_chat_request(body,"mage-local",config=config,image_decoder=decode_image_for_test)
    with pytest.raises(server.APIError):
        parse(body)
    body["messages"][0]["content"]="x"*2_000_001
    with pytest.raises(server.APIError):
        server.parse_chat_request(body,"mage-local",config=config,image_decoder=decode_image_for_test)


def test_launch_script_explicitly_selects_new_profile_without_claiming_one_million():
    script=Path(__file__).with_name("run_server.sh").read_text(encoding="utf-8")
    for flag in ("--max-context-tokens","--max-new-tokens","--default-new-tokens","--max-text-chars","--max-images"):
        assert flag in script
    assert "96000" in script and "8192" in script and "4096" in script and "2000000" in script


def test_context_metadata_rejects_limits_beyond_actual_model_or_tokenizer(tmp_path):
    from types import SimpleNamespace
    config=expanded_config(tmp_path)
    with pytest.raises(ValueError):
        server.context_metadata(config,SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=65536)),
                                SimpleNamespace(model_max_length=131072))
    with pytest.raises(ValueError):
        server.context_metadata(config,SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144)),
                                SimpleNamespace(model_max_length=65536))
    metadata=server.context_metadata(config,SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144)),
                                     SimpleNamespace(model_max_length=10**30))
    assert metadata["tokenizer_context_window"] is None
    assert metadata["one_million_supported"] is False


def test_cli_passes_explicit_limits_to_config_without_starting_a_model(monkeypatch,tmp_path):
    from types import SimpleNamespace
    captured={}
    monkeypatch.setattr(server,"create_app",lambda config:config)
    monkeypatch.setitem(sys.modules,"uvicorn",SimpleNamespace(run=lambda app,**kwargs:captured.update(config=app,run=kwargs)))
    server.main(["--model-dir",str(tmp_path),"--served-model-name","mage-local",
                 "--revision","test-revision","--token-file",str(tmp_path/"unused-token"),
                 "--max-context-tokens","131072","--max-new-tokens","8192","--default-new-tokens","4096",
                 "--max-text-chars","2000000","--max-images","128"])
    assert captured["config"].max_context_tokens==131072
    assert captured["config"].max_new_tokens==8192 and captured["config"].default_new_tokens==4096
    assert captured["config"].max_text_chars==2_000_000 and captured["config"].max_images==128
    assert captured["run"]["workers"]==1


def test_http_context_budget_error_remains_structured_without_fake_usage(monkeypatch,tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace
    class App:
        def __init__(self,**kwargs):
            self.lifespan=kwargs["lifespan"];self.routes={};self.handlers={}
        def exception_handler(self,kind):
            def register(fn):self.handlers[kind]=fn;return fn
            return register
        def get(self,path):
            def register(fn):self.routes[path]=fn;return fn
            return register
        post=get
    token="test-only-budget-auth-token-12345678"
    body=json.dumps(request_body()).encode("utf-8")
    class Request:
        headers={"authorization":"Bearer "+token,"content-type":"application/json","content-length":str(len(body))}
        async def stream(self):
            yield body
    class Response:
        def __init__(self,content,status_code,headers=None):
            self.content,self.status_code,self.headers=content,status_code,headers
    monkeypatch.setitem(sys.modules,"fastapi",SimpleNamespace(FastAPI=App,Request=Request))
    monkeypatch.setitem(sys.modules,"fastapi.responses",SimpleNamespace(JSONResponse=Response))
    monkeypatch.setattr(server,"_pillow_decode",decode_image_for_test)
    config=expanded_config(tmp_path)
    config.token_file.write_text(token,encoding="ascii")
    seen=[]
    class Backend:
        def __init__(self,settings):
            self.metadata=server.context_metadata(settings,
                SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144)),
                SimpleNamespace(model_max_length=131072))
        def generate(self,spec):
            seen.append(spec.max_tokens)
            return server.check_context_budget(131000,spec.max_tokens,config,
                         model_context_window=262144,tokenizer_context_window=131072)
        def close(self):pass
    app=server.create_app(config,backend_factory=Backend)
    async def invoke():
        async with app.lifespan(app):
            health=await app.routes["/health"](Request())
            assert health["operational_context_tokens"]==131072
            assert health["validated_max_input_tokens"] is None and health["one_million_supported"] is False
            with pytest.raises(server.APIError) as caught:
                await app.routes["/v1/chat/completions"](Request())
            response=await app.handlers[server.APIError](Request(),caught.value)
            assert response.status_code==400
            assert response.content["error"]["code"]=="context_length_exceeded"
            assert response.content["error"]["budget"]["input_tokens"]==131000
            assert response.content["error"]["budget"]["reserved_output"]==4096
            assert "usage" not in response.content
    asyncio.run(invoke())
    assert seen==[4096]
