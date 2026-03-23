from __future__ import annotations

from dapo_lab.config_schema import ExperimentConfig

from .compat import CompatibilityReport
from .config_bridge import build_verl_config
from .contract import audit_bridge_config
from .trainer import ResearchTrainer


def launch_with_verl(config: ExperimentConfig, compatibility: CompatibilityReport) -> None:
    if not compatibility.importable:
        raise RuntimeError(compatibility.message)

    bridge_config = build_verl_config(config)
    contract_audit = audit_bridge_config(bridge_config)
    if not contract_audit.ok:
        problems: list[str] = []
        if contract_audit.missing_top_level:
            problems.append(f"missing top-level: {', '.join(contract_audit.missing_top_level)}")
        if contract_audit.unexpected_top_level:
            problems.append(f"unexpected top-level: {', '.join(contract_audit.unexpected_top_level)}")
        if contract_audit.missing_paths:
            problems.append(f"missing paths: {', '.join(contract_audit.missing_paths)}")
        if contract_audit.missing_target_paths:
            problems.append(f"missing _target_ paths: {', '.join(contract_audit.missing_target_paths)}")
        problems.extend(contract_audit.semantic_errors)
        raise RuntimeError("Pinned verl contract audit failed before runtime launch:\n" + "\n".join(problems))
    try:
        import ray  # type: ignore
        from omegaconf import OmegaConf  # type: ignore
        from verl.experimental.reward_loop import migrate_legacy_reward_impl  # type: ignore
        from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo  # type: ignore
        from verl.trainer.ppo.utils import need_critic, need_reference_policy  # type: ignore
        from verl.utils import hf_processor, hf_tokenizer  # type: ignore
        from verl.utils.config import validate_config  # type: ignore
        from verl.utils.dataset.rl_dataset import collate_fn  # type: ignore
        from verl.utils.device import auto_set_device  # type: ignore
        from verl.utils.fs import copy_to_local  # type: ignore
    except Exception as error:
        raise RuntimeError(
            "verl was importable, but the full training stack was not available. "
            "Install ray, omegaconf, and the standard verl runtime dependencies in your training image."
        ) from error

    class ResearchTaskRunner(TaskRunner):  # pragma: no cover - requires a live verl installation
        def run(self, upstream_config):
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(upstream_config)
            if need_critic(upstream_config):
                self.add_critic_worker(upstream_config)
            self.add_reward_model_resource_pool(upstream_config)
            self.add_ref_policy_worker(upstream_config, actor_rollout_cls)

            validate_config(
                config=upstream_config,
                use_reference_policy=need_reference_policy(upstream_config),
                use_critic=need_critic(upstream_config),
            )

            local_path = copy_to_local(
                upstream_config.actor_rollout_ref.model.path,
                use_shm=upstream_config.actor_rollout_ref.model.get("use_shm", False),
            )
            tokenizer = hf_tokenizer(local_path, trust_remote_code=upstream_config.actor_rollout_ref.model.trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=upstream_config.actor_rollout_ref.model.trust_remote_code)
            resource_pool_manager = self.init_resource_pool_mgr(upstream_config)

            train_dataset = create_rl_dataset(
                upstream_config.data.train_files,
                upstream_config.data,
                tokenizer,
                processor,
                is_train=True,
                max_samples=upstream_config.data.get("train_max_samples", -1),
            )
            val_dataset = create_rl_dataset(
                upstream_config.data.val_files,
                upstream_config.data,
                tokenizer,
                processor,
                is_train=False,
                max_samples=upstream_config.data.get("val_max_samples", -1),
            )
            train_sampler = create_rl_sampler(upstream_config.data, train_dataset)

            trainer = ResearchTrainer(
                experiment_config=config,
                config=upstream_config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
            trainer.init_workers()
            trainer.fit()

    upstream_config = OmegaConf.create(bridge_config)
    auto_set_device(upstream_config)
    upstream_config = migrate_legacy_reward_impl(upstream_config)
    validate_config(
        config=upstream_config,
        use_reference_policy=need_reference_policy(upstream_config),
        use_critic=need_critic(upstream_config),
    )
    run_ppo(upstream_config, task_runner_class=ray.remote(num_cpus=1)(ResearchTaskRunner))
