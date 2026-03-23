from __future__ import annotations

from dataclasses import asdict

from dapo_lab.config_schema import ExperimentConfig


def _reward_model_rollout(config: ExperimentConfig) -> dict:
    return {
        "_target_": "verl.workers.config.RolloutConfig",
        "name": config.algorithm.rollout_behavior.backend,
        "dtype": "bfloat16",
        "gpu_memory_utilization": config.verl.rollout.gpu_memory_utilization,
        "enforce_eager": config.verl.rollout.enforce_eager,
        "cudagraph_capture_sizes": None,
        "free_cache_engine": True,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
        "tensor_model_parallel_size": config.verl.rollout.tensor_model_parallel_size,
        "max_num_batched_tokens": 8192,
        "max_model_len": None,
        "max_num_seqs": 1024,
        "load_format": "auto",
        "engine_kwargs": {},
        "limit_images": None,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "disable_log_stats": True,
        "skip_tokenizer_init": False,
        "prompt_length": config.data.max_prompt_length,
        "response_length": config.data.max_response_length,
    }


def _reward_model_block(config: ExperimentConfig) -> dict:
    return {
        "enable": False,
        "enable_resource_pool": False,
        "n_gpus_per_node": 0,
        "nnodes": 0,
        "model_path": None,
        "rollout": _reward_model_rollout(config),
    }


def _legacy_reward_model_block() -> dict:
    return {
        "num_workers": None,
        "reward_manager": None,
        "enable": None,
        "enable_resource_pool": None,
        "n_gpus_per_node": None,
        "nnodes": None,
        "reward_loop_source": None,
        "reward_loop_module_path": None,
        "reward_loop_class_name": None,
        "model": {
            "path": None,
            "external_lib": None,
            "trust_remote_code": None,
        },
        "rollout": {
            "name": None,
            "dtype": None,
            "gpu_memory_utilization": None,
            "enforce_eager": None,
            "cudagraph_capture_sizes": None,
            "free_cache_engine": None,
            "data_parallel_size": None,
            "expert_parallel_size": None,
            "tensor_model_parallel_size": None,
            "max_num_batched_tokens": None,
            "max_model_len": None,
            "max_num_seqs": None,
            "load_format": None,
            "engine_kwargs": None,
            "limit_images": None,
            "enable_chunked_prefill": None,
            "enable_prefix_caching": None,
            "disable_log_stats": None,
            "skip_tokenizer_init": None,
            "prompt_length": None,
            "response_length": None,
        },
    }


def _reward_manager_name(config: ExperimentConfig) -> str:
    if config.algorithm.variant == "gdpo":
        return "gdpo"
    if config.algorithm.variant == "dapo":
        return "dapo"
    return "naive"


def build_verl_config(config: ExperimentConfig) -> dict:
    policy = config.algorithm.policy_loss
    filtering = config.algorithm.group_filtering
    overlong = config.reward.overlong
    adv_estimator = "gdpo" if config.algorithm.variant == "gdpo" else config.algorithm.advantage.mode
    return {
        "data": {
            "tokenizer": None,
            "use_shm": False,
            "train_files": config.data.train_files,
            "val_files": config.data.val_files,
            "train_max_samples": -1,
            "val_max_samples": -1,
            "train_batch_size": config.data.train_batch_size,
            "val_batch_size": None,
            "gen_batch_size": config.data.gen_batch_size,
            "prompt_key": config.data.prompt_key,
            "max_prompt_length": config.data.max_prompt_length,
            "max_response_length": config.data.max_response_length,
            "trust_remote_code": config.verl.trust_remote_code,
            "return_raw_chat": True,
            "shuffle": True,
            "validation_shuffle": False,
            "dataloader_num_workers": 1,
            "filter_overlong_prompts": False,
            "truncation": "error",
        },
        "algorithm": {
            "adv_estimator": adv_estimator,
            "norm_adv_by_std_in_grpo": config.algorithm.advantage.normalize_by_std,
            "use_kl_in_reward": config.algorithm.kl.enabled,
            "filter_groups": {
                "enable": filtering.enabled,
                "metric": filtering.metric,
                "max_num_gen_batches": filtering.max_num_gen_batches,
            },
            "gdpo_reward_keys": config.algorithm.gdpo.component_keys,
            "gdpo_reward_weights": config.algorithm.gdpo.component_weights,
        },
        "custom_reward_function": {
            "path": None,
            "name": None,
        },
        "reward_model": _legacy_reward_model_block(),
        "sandbox_fusion": {
            "url": None,
            "max_concurrent": None,
            "memory_limit_mb": None,
        },
        "reward": {
            "num_workers": 0,
            "custom_reward_function": {
                "path": None,
                "name": "compute_score",
            },
            "reward_manager": {
                "_target_": "verl.workers.config.reward_model.RewardManagerConfig",
                "source": "register",
                "name": _reward_manager_name(config),
                "module": {
                    "_target_": "verl.trainer.config.config.ModuleConfig",
                    "path": None,
                    "name": "custom_reward_manager",
                },
            },
            "reward_model": _reward_model_block(config),
            "sandbox_fusion": {
                "url": None,
                "max_concurrent": 64,
                "memory_limit_mb": 1024,
            },
            "reward_kwargs": {
                "overlong_buffer_cfg": {
                    "enable": overlong.enabled,
                    "len": overlong.buffer_length,
                    "penalty_factor": overlong.penalty_factor,
                    "log": overlong.log,
                },
                "max_resp_len": config.data.max_response_length,
            },
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": config.verl.model_path,
                "trust_remote_code": config.verl.trust_remote_code,
                "use_shm": False,
            },
            "actor": {
                "_target_": "verl.workers.config.FSDPActorConfig",
                "strategy": "fsdp",
                "rollout_n": config.data.rollout_n,
                "ppo_mini_batch_size": config.data.train_batch_size * config.data.rollout_n,
                "ppo_micro_batch_size": None,
                "use_kl_loss": False,
                "use_dynamic_bsz": False,
                "ppo_max_token_len_per_gpu": 16384,
                "ppo_micro_batch_size_per_gpu": config.verl.actor.ppo_micro_batch_size_per_gpu,
                "grad_clip": config.verl.actor.grad_clip,
                "ppo_epochs": config.verl.actor.ppo_epochs,
                "clip_ratio": policy.clip_ratio,
                "clip_ratio_low": policy.clip_ratio_low,
                "clip_ratio_high": policy.clip_ratio_high,
                "clip_ratio_c": policy.clip_ratio_c,
                "loss_agg_mode": policy.loss_agg_mode,
                "loss_scale_factor": None,
                "entropy_coeff": 0.0,
                "calculate_entropy": False,
                "use_prefix_grouper": False,
                "use_torch_compile": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": config.algorithm.kl.penalty,
                "shuffle": False,
                "data_loader_seed": config.experiment.seed,
            },
            "rollout": {
                "_target_": "verl.workers.config.RolloutConfig",
                "name": config.algorithm.rollout_behavior.backend,
                "mode": "async",
                "nnodes": 0,
                "n_gpus_per_node": 1,
                "dtype": "bfloat16",
                "n": config.data.rollout_n,
                "temperature": config.algorithm.rollout_behavior.temperature,
                "top_p": config.algorithm.rollout_behavior.top_p,
                "top_k": config.algorithm.rollout_behavior.top_k,
                "do_sample": config.algorithm.rollout_behavior.do_sample,
                "prompt_length": config.data.max_prompt_length,
                "response_length": config.data.max_response_length,
                "tensor_model_parallel_size": config.verl.rollout.tensor_model_parallel_size,
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "gpu_memory_utilization": config.verl.rollout.gpu_memory_utilization,
                "enforce_eager": config.verl.rollout.enforce_eager,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_use_dynamic_bsz": False,
                "log_prob_max_token_len_per_gpu": 16384,
                "val_kwargs": {
                    "_target_": "verl.workers.config.SamplingConfig",
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                    "n": 1,
                    "do_sample": False,
                },
            },
        },
        "critic": {
            "enable": config.verl.critic,
            "strategy": "fsdp",
        },
        "trainer": {
            "project_name": config.experiment.name,
            "experiment_name": config.experiment.name,
            "logger": ["console"],
            "nnodes": 1,
            "n_gpus_per_node": 1,
            "save_freq": config.trainer.save_freq,
            "test_freq": config.trainer.test_freq,
            "val_before_train": config.trainer.val_before_train,
            "device": "cuda",
            "use_legacy_worker_impl": "auto",
        },
        "global_profiler": {
            "tool": None,
            "steps": None,
            "profile_continuous_steps": False,
            "save_path": "outputs/profile",
            "global_tool_config": {
                "nsys": {
                    "controller_nsight_options": {},
                    "worker_nsight_options": {},
                },
                "torch_memory": {
                    "kw_args": {},
                },
            },
        },
        "transfer_queue": {
            "enable": False,
        },
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": 1,
            },
            "timeline_json_file": None,
        },
        "dapo_lab": asdict(config),
    }
