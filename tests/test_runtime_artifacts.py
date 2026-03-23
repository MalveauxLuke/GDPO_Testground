from __future__ import annotations

from types import SimpleNamespace

from dapo_lab.verl_adapter.runtime_artifacts import probe_runtime_artifacts


class DummyTokenizer:
    pass


class DummyProcessor:
    pass


def test_probe_runtime_artifacts_skips_processor_for_text_only_models(monkeypatch) -> None:
    log_messages: list[str] = []
    processor_calls: list[str] = []

    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._copy_to_local",
        lambda model_path, use_shm: "/tmp/text-only-model",
    )
    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._hf_tokenizer",
        lambda name_or_path, trust_remote_code: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._auto_config",
        lambda name_or_path, trust_remote_code: SimpleNamespace(
            model_type="qwen2",
            architectures=["Qwen2ForCausalLM"],
        ),
    )

    def _processor(name_or_path, trust_remote_code):
        processor_calls.append(name_or_path)
        return DummyProcessor()

    monkeypatch.setattr("dapo_lab.verl_adapter.runtime_artifacts._hf_processor", _processor)

    artifacts = probe_runtime_artifacts(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=False,
        log=log_messages.append,
    )

    assert artifacts.local_path == "/tmp/text-only-model"
    assert artifacts.tokenizer_class == "DummyTokenizer"
    assert artifacts.model_type == "qwen2"
    assert artifacts.processor is None
    assert artifacts.processor_mode == "skipped_text_only"
    assert processor_calls == []
    assert log_messages == [
        "before copy_to_local(model_path='Qwen/Qwen2.5-0.5B-Instruct', use_shm=False)",
        "after copy_to_local -> '/tmp/text-only-model'",
        "before hf_tokenizer",
        "after hf_tokenizer -> DummyTokenizer",
        "before AutoConfig.from_pretrained",
        "after AutoConfig.from_pretrained -> model_type=qwen2",
        "text-only model detected (qwen2); skipping hf_processor",
    ]


def test_probe_runtime_artifacts_loads_processor_for_multimodal_models(monkeypatch) -> None:
    log_messages: list[str] = []

    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._copy_to_local",
        lambda model_path, use_shm: "/tmp/multimodal-model",
    )
    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._hf_tokenizer",
        lambda name_or_path, trust_remote_code: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._auto_config",
        lambda name_or_path, trust_remote_code: SimpleNamespace(
            model_type="qwen2_vl",
            architectures=["Qwen2VLForConditionalGeneration"],
            vision_config=object(),
        ),
    )
    monkeypatch.setattr(
        "dapo_lab.verl_adapter.runtime_artifacts._hf_processor",
        lambda name_or_path, trust_remote_code: DummyProcessor(),
    )

    artifacts = probe_runtime_artifacts(
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=False,
        log=log_messages.append,
    )

    assert artifacts.local_path == "/tmp/multimodal-model"
    assert artifacts.tokenizer_class == "DummyTokenizer"
    assert artifacts.model_type == "qwen2_vl"
    assert artifacts.processor_class == "DummyProcessor"
    assert artifacts.processor_mode == "loaded"
    assert log_messages[-2:] == [
        "before hf_processor",
        "after hf_processor -> DummyProcessor",
    ]
