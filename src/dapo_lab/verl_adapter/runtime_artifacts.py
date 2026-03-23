from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class RuntimeArtifacts:
    local_path: str
    tokenizer: Any
    processor: Any | None
    model_type: str
    tokenizer_class: str
    processor_class: str | None
    processor_mode: str


def _copy_to_local(model_path: str, *, use_shm: bool) -> str:
    from verl.utils.fs import copy_to_local  # type: ignore

    return copy_to_local(model_path, use_shm=use_shm)


def _hf_tokenizer(name_or_path: str, *, trust_remote_code: bool) -> Any:
    from verl.utils import hf_tokenizer  # type: ignore

    return hf_tokenizer(name_or_path, trust_remote_code=trust_remote_code)


def _hf_processor(name_or_path: str, *, trust_remote_code: bool) -> Any | None:
    from verl.utils import hf_processor  # type: ignore

    return hf_processor(name_or_path, trust_remote_code=trust_remote_code)


def _auto_config(name_or_path: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)


def _emit(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def _looks_multimodal(model_config: Any) -> bool:
    attrs = (
        "vision_config",
        "vision_tower",
        "visual",
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "video_token_index",
        "mm_projector_type",
        "image_seq_length",
        "video_seq_length",
    )
    for attr in attrs:
        if getattr(model_config, attr, None) is not None:
            return True

    model_type = str(getattr(model_config, "model_type", "") or "").lower()
    multimodal_tokens = ("vl", "vision", "llava", "mllama", "glm4v", "idefics", "paligemma")
    if any(token in model_type for token in multimodal_tokens):
        return True

    architectures = getattr(model_config, "architectures", None) or []
    arch_text = " ".join(str(value).lower() for value in architectures)
    if any(token in arch_text for token in multimodal_tokens):
        return True

    return False


def probe_runtime_artifacts(
    *,
    model_path: str,
    trust_remote_code: bool,
    use_shm: bool = False,
    log: Callable[[str], None] | None = None,
) -> RuntimeArtifacts:
    _emit(log, f"before copy_to_local(model_path={model_path!r}, use_shm={use_shm})")
    try:
        local_path = _copy_to_local(model_path, use_shm=use_shm)
    except Exception as error:  # pragma: no cover - exercised with live verl runtime
        raise RuntimeError(f"copy_to_local failed for {model_path!r}: {error}") from error
    _emit(log, f"after copy_to_local -> {local_path!r}")

    _emit(log, "before hf_tokenizer")
    try:
        tokenizer = _hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    except Exception as error:  # pragma: no cover - exercised with live verl runtime
        raise RuntimeError(f"hf_tokenizer failed for {local_path!r}: {error}") from error
    tokenizer_class = tokenizer.__class__.__name__
    _emit(log, f"after hf_tokenizer -> {tokenizer_class}")

    _emit(log, "before AutoConfig.from_pretrained")
    try:
        model_config = _auto_config(local_path, trust_remote_code=trust_remote_code)
    except Exception as error:  # pragma: no cover - exercised with live transformers runtime
        raise RuntimeError(f"AutoConfig.from_pretrained failed for {local_path!r}: {error}") from error
    model_type = str(getattr(model_config, "model_type", "unknown"))
    _emit(log, f"after AutoConfig.from_pretrained -> model_type={model_type}")

    if not _looks_multimodal(model_config):
        _emit(log, f"text-only model detected ({model_type}); skipping hf_processor")
        return RuntimeArtifacts(
            local_path=local_path,
            tokenizer=tokenizer,
            processor=None,
            model_type=model_type,
            tokenizer_class=tokenizer_class,
            processor_class=None,
            processor_mode="skipped_text_only",
        )

    _emit(log, "before hf_processor")
    try:
        processor = _hf_processor(local_path, trust_remote_code=trust_remote_code)
    except Exception as error:  # pragma: no cover - exercised with live verl runtime
        raise RuntimeError(f"hf_processor failed for {local_path!r}: {error}") from error

    if processor is None:
        _emit(log, f"hf_processor returned None for {model_type}; treating model as text-only")
        return RuntimeArtifacts(
            local_path=local_path,
            tokenizer=tokenizer,
            processor=None,
            model_type=model_type,
            tokenizer_class=tokenizer_class,
            processor_class=None,
            processor_mode="returned_none",
        )

    processor_class = processor.__class__.__name__
    _emit(log, f"after hf_processor -> {processor_class}")
    return RuntimeArtifacts(
        local_path=local_path,
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        tokenizer_class=tokenizer_class,
        processor_class=processor_class,
        processor_mode="loaded",
    )
