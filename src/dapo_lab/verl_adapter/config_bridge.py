from __future__ import annotations

from dataclasses import asdict

from dapo_lab.config_schema import ExperimentConfig

from .contract import load_pinned_scaffold


def _reward_manager_name(config: ExperimentConfig) -> str:
    if config.algorithm.variant == "gdpo":
        return "gdpo"
    if config.algorithm.variant == "dapo":
        return "dapo"
    return "naive"


def _reward_model_rollout(config: ExperimentConfig) -> dict:
    return {
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


def _kl_settings(config: ExperimentConfig) -> tuple[bool, bool]:
    use_kl_in_reward = config.algorithm.kl.enabled and config.algorithm.kl.source == "reward"
    use_kl_loss = config.algorithm.kl.enabled and config.algorithm.kl.source == "loss"
    return use_kl_in_reward, use_kl_loss


def build_verl_config(config: ExperimentConfig) -> dict:
    bridged = load_pinned_scaffold()

    policy = config.algorithm.policy_loss
    filtering = config.algorithm.group_filtering
    overlong = config.reward.overlong
    adv_estimator = "gdpo" if config.algorithm.variant == "gdpo" else config.algorithm.advantage.mode
    use_kl_in_reward, use_kl_loss = _kl_settings(config)

    bridged["data"].update(
        {
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
            "seed": config.experiment.seed,
            "trust_remote_code": config.verl.trust_remote_code,
            "return_raw_chat": True,
            "shuffle": True,
            "validation_shuffle": False,
            "dataloader_num_workers": 1,
            "filter_overlong_prompts": False,
            "truncation": "error",
        }
    )

    bridged["algorithm"].update(
        {
            "adv_estimator": adv_estimator,
            "norm_adv_by_std_in_grpo": config.algorithm.advantage.normalize_by_std,
            "use_kl_in_reward": use_kl_in_reward,
            "kl_penalty": config.algorithm.kl.penalty,
            "filter_groups": {
                "enable": filtering.enabled,
                "metric": filtering.metric,
                "max_num_gen_batches": filtering.max_num_gen_batches,
            },
            "gdpo_reward_keys": config.algorithm.gdpo.component_keys,
            "gdpo_reward_weights": config.algorithm.gdpo.component_weights,
        }
    )

    bridged["custom_reward_function"].update({"path": None, "name": None})
    bridged["reward_model"].update(
        {
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
    )
    bridged["sandbox_fusion"].update({"url": None, "max_concurrent": None, "memory_limit_mb": None})

    bridged["reward"].update(
        {
            "num_workers": 0,
            "custom_reward_function": {"path": None, "name": "compute_score"},
            "reward_kwargs": {
                "overlong_buffer_cfg": {
                    "enable": overlong.enabled,
                    "len": overlong.buffer_length,
                    "penalty_factor": overlong.penalty_factor,
                    "log": overlong.log,
                },
                "max_resp_len": config.data.max_response_length,
            },
        }
    )
    bridged["reward"]["reward_manager"].update(
        {
            "source": "register",
            "name": _reward_manager_name(config),
        }
    )
    bridged["reward"]["reward_manager"]["module"].update({"path": None, "name": "custom_reward_manager"})
    bridged["reward"]["reward_model"].update(
        {
            "enable": False,
            "enable_resource_pool": False,
            "n_gpus_per_node": 0,
            "nnodes": 0,
            "model_path": None,
        }
    )
    bridged["reward"]["reward_model"]["rollout"].update(_reward_model_rollout(config))
    bridged["reward"]["sandbox_fusion"].update({"url": None, "max_concurrent": 64, "memory_limit_mb": 1024})

    bridged["actor_rollout_ref"].update({"hybrid_engine": True, "nccl_timeout": 600})
    bridged["actor_rollout_ref"]["model"].update(
        {
            "path": config.verl.model_path,
            "trust_remote_code": config.verl.trust_remote_code,
            "external_lib": None,
            "use_shm": False,
            "use_remove_padding": True,
            "use_fused_kernels": False,
        }
    )

    bridged["actor_rollout_ref"]["actor"].update(
        {
            "strategy": "fsdp",
            "rollout_n": config.data.rollout_n,
            "ppo_mini_batch_size": config.data.train_batch_size,
            "ppo_micro_batch_size": None,
            "ppo_micro_batch_size_per_gpu": config.verl.actor.ppo_micro_batch_size_per_gpu,
            "use_dynamic_bsz": False,
            "use_kl_loss": use_kl_loss,
            "ppo_max_token_len_per_gpu": 16384,
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
            "calculate_sum_pi_squared": False,
            "use_prefix_grouper": False,
            "use_torch_compile": True,
            "kl_loss_coef": 0.001,
            "kl_loss_type": config.algorithm.kl.penalty,
            "shuffle": False,
            "data_loader_seed": config.experiment.seed,
        }
    )
    bridged["actor_rollout_ref"]["actor"]["optim"].update({"clip_grad": config.verl.actor.grad_clip})
    bridged["actor_rollout_ref"]["actor"]["fsdp_config"].update(
        {
            "strategy": "fsdp",
            "dtype": "bfloat16",
            "seed": config.experiment.seed,
            "use_torch_compile": True,
        }
    )

    bridged["actor_rollout_ref"]["ref"].update(
        {
            "strategy": "${actor_rollout_ref.actor.strategy}",
            "rollout_n": "${oc.select:actor_rollout_ref.rollout.n,1}",
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": config.verl.actor.ppo_micro_batch_size_per_gpu,
            "log_prob_use_dynamic_bsz": "${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}",
            "log_prob_max_token_len_per_gpu": "${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu,16384}",
            "ulysses_sequence_parallel_size": "${oc.select:actor_rollout_ref.actor.ulysses_sequence_parallel_size,1}",
            "use_torch_compile": "${oc.select:actor_rollout_ref.actor.use_torch_compile,true}",
        }
    )
    bridged["actor_rollout_ref"]["ref"]["fsdp_config"].update(
        {
            "strategy": "fsdp",
            "dtype": "bfloat16",
            "seed": config.experiment.seed,
            "use_torch_compile": True,
            "forward_only": True,
        }
    )

    bridged["actor_rollout_ref"]["rollout"].update(
        {
            "name": config.algorithm.rollout_behavior.backend,
            "mode": "async",
            "nnodes": 0,
            "n_gpus_per_node": "${oc.select:trainer.n_gpus_per_node,8}",
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
            "calculate_log_probs": False,
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": config.verl.actor.ppo_micro_batch_size_per_gpu,
            "log_prob_use_dynamic_bsz": False,
            "log_prob_max_token_len_per_gpu": 16384,
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
            "free_cache_engine": True,
            "cudagraph_capture_sizes": None,
        }
    )
    bridged["actor_rollout_ref"]["rollout"]["val_kwargs"].update(
        {
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0.0,
            "n": 1,
            "do_sample": False,
        }
    )
    bridged["actor_rollout_ref"]["rollout"]["trace"].update(
        {
            "project_name": "${oc.select:trainer.project_name,null}",
            "experiment_name": "${oc.select:trainer.experiment_name,null}",
        }
    )
    bridged["actor_rollout_ref"]["rollout"]["prometheus"].update(
        {"served_model_name": "${oc.select:actor_rollout_ref.model.path,null}"}
    )

    bridged["critic"].update(
        {
            "enable": config.verl.critic,
            "strategy": "fsdp",
            "ppo_mini_batch_size": "${oc.select:actor_rollout_ref.actor.ppo_mini_batch_size,256}",
            "ppo_epochs": "${oc.select:actor_rollout_ref.actor.ppo_epochs,1}",
            "loss_agg_mode": "${oc.select:actor_rollout_ref.actor.loss_agg_mode,token-mean}",
        }
    )

    bridged["distillation"].update({"enabled": False, "num_workers": 8})
    bridged["distillation"]["teacher_model"].update({"model_path": None, "enable_resource_pool": False})
    bridged["distillation"]["teacher_model"]["inference"].update(
        {
            "name": "${oc.select:actor_rollout_ref.rollout.name,null}",
            "tensor_model_parallel_size": config.verl.rollout.tensor_model_parallel_size,
        }
    )

    bridged["trainer"].update(
        {
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
        }
    )

    bridged["global_profiler"].update(
        {
            "tool": None,
            "steps": None,
            "profile_continuous_steps": False,
            "save_path": "outputs/profile",
        }
    )
    bridged["global_profiler"]["global_tool_config"]["nsys"].update(
        {
            "discrete": False,
            "controller_nsight_options": {},
            "worker_nsight_options": {},
        }
    )
    bridged["global_profiler"]["global_tool_config"]["torch_memory"].update(
        {
            "trace_alloc_max_entries": 100000,
            "stack_depth": 32,
            "kw_args": {},
        }
    )

    bridged["transfer_queue"].update({"enable": False})
    bridged["ray_kwargs"].update({"timeline_json_file": None})
    bridged["ray_kwargs"]["ray_init"].update({"num_cpus": 1})

    bridged["dapo_lab"] = asdict(config)
    return bridged
