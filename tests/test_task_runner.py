from __future__ import annotations

from dapo_lab.verl_adapter.runtime_artifacts import RuntimeArtifacts
from dapo_lab.verl_adapter.task_runner import _prepare_runtime_trainer

from .conftest import build_config


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(item) from error


def _attrdict(payload):
    if isinstance(payload, dict):
        return AttrDict({key: _attrdict(value) for key, value in payload.items()})
    if isinstance(payload, list):
        return [_attrdict(value) for value in payload]
    return payload


def test_prepare_runtime_trainer_emits_stage_markers(monkeypatch) -> None:
    log_messages: list[str] = []
    call_order: list[str] = []
    config = build_config("grpo")
    upstream_config = _attrdict(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "trust_remote_code": False,
                    "use_shm": False,
                }
            },
            "data": {
                "train_files": ["train.jsonl"],
                "val_files": ["val.jsonl"],
                "train_max_samples": -1,
                "val_max_samples": -1,
            },
        }
    )

    monkeypatch.setattr(
        "dapo_lab.verl_adapter.task_runner.probe_runtime_artifacts",
        lambda **kwargs: RuntimeArtifacts(
            local_path="/tmp/model",
            tokenizer=object(),
            processor=None,
            model_type="qwen2",
            tokenizer_class="FakeTokenizer",
            processor_class=None,
            processor_mode="skipped_text_only",
        ),
    )

    def _init_resource_pool_mgr(_upstream_config):
        call_order.append("init_resource_pool_mgr")
        return "pool"

    def _create_rl_dataset(data_files, data_config, tokenizer, processor, *, is_train, max_samples):
        call_order.append(f"create_rl_dataset({'train' if is_train else 'val'})")
        return {"kind": "train" if is_train else "val", "files": list(data_files)}

    def _create_rl_sampler(data_config, dataset):
        call_order.append("create_rl_sampler")
        return "sampler"

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.init_workers_called = False

        def init_workers(self) -> None:
            call_order.append("trainer.init_workers")
            self.init_workers_called = True

    trainer = _prepare_runtime_trainer(
        experiment_config=config,
        upstream_config=upstream_config,
        collate_fn="collate",
        role_worker_mapping={"actor": "worker"},
        ray_worker_group_cls="ray-group",
        init_resource_pool_mgr=_init_resource_pool_mgr,
        create_rl_dataset=_create_rl_dataset,
        create_rl_sampler=_create_rl_sampler,
        trainer_cls=FakeTrainer,
        log=log_messages.append,
    )

    assert isinstance(trainer, FakeTrainer)
    assert trainer.init_workers_called is True
    assert call_order == [
        "init_resource_pool_mgr",
        "create_rl_dataset(train)",
        "create_rl_dataset(val)",
        "create_rl_sampler",
        "trainer.init_workers",
    ]
    assert log_messages == [
        "before init_resource_pool_mgr",
        "after init_resource_pool_mgr",
        "before create_rl_dataset(train)",
        "after create_rl_dataset(train)",
        "before create_rl_dataset(val)",
        "after create_rl_dataset(val)",
        "before create_rl_sampler",
        "after create_rl_sampler",
        "before trainer.init_workers",
        "after trainer.init_workers",
    ]
