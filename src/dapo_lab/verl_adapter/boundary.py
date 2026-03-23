from __future__ import annotations

LOCAL_OWNERSHIP = [
    "reward composition",
    "advantage dispatch",
    "policy loss and clipping",
    "loss aggregation mode",
    "dynamic sampling and group filtering",
    "overlong shaping and filtering",
    "trainer-loop ordering",
    "rollout request behavior",
    "config validation",
    "diagnostics hooks",
]

UPSTREAM_DELEGATION = [
    "Ray orchestration",
    "worker classes",
    "distributed controller plumbing",
    "FSDP and Megatron internals",
    "checkpoint internals",
    "cluster-specific runtime behavior",
]
