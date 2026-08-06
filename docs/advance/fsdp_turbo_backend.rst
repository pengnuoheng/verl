FSDP-Turbo backend
==================

Last updated: 08/03/2026.

FSDP-Turbo (``fsdp_turbo``) is a high-performance FSDP training backend built
on top of the `fsdp-turbo <https://gitcode.com/Ascend/FSDPTurbo.git>`_ library.
It extends verl's FSDP engine family (``fsdp``, ``fsdp2``) with native support
for hybrid parallelism that combines fully-sharded data parallelism with expert
parallelism (EP) and context parallelism (CP) on a single ``DeviceMesh``.

The backend is primarily designed for **Ascend NPU** and has been validated
with Qwen3.5-27B (dense) and Qwen3.5-35B-A3B (MoE). It also works on NVIDIA
GPUs where the ``fsdp_turbo`` python path is exported.

Configuration
-------------

Select the backend by setting ``strategy=fsdp_turbo``:

.. code-block:: bash

   actor_rollout_ref.actor.strategy=fsdp_turbo
   actor_rollout_ref.ref.strategy=fsdp_turbo

The turbo-specific parallelism and memory options live under
``fsdp_config.turbo_config``.  The full schema is defined in
``verl/trainer/config/engine/fsdp.yaml``.

Distributed plan
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   turbo_config:
     distributed:
       fully_shard_parallel_size: 16   # FSDP group size
       tensor_parallel_size: 1         # reserved, keep 1
       expert_parallel_size: 8         # EP group size (MoE only)
       expert_fully_shard_parallel_size: 1
       ulysses_parallel_size: 1        # CP / Ulysses SP size
       fsdp_plan:
         param_dtype: bf16
         reduce_dtype: fp32
         output_dtype: bf16
         fsdp_implementation: native
         num_to_forward_prefetch: 1
         num_to_backward_prefetch: 1

Key fields:

* ``fully_shard_parallel_size`` — number of ranks that share a single set of
  parameter shards (the FSDP mesh).
* ``expert_parallel_size`` — number of ranks across which MoE experts are
  distributed.  Set to ``1`` for dense models.
* ``ulysses_parallel_size`` — context-parallel size.  When greater than 1,
  sequences are split along the head dimension and gathered with all-to-all.

Module-level sharding plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSDP-Turbo accepts explicit module-glob patterns to control which submodules
are sharded, hooked, or re-computed.  Patterns use ``{*}`` for integer indices:

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules='{model.language_model.layers.{*}:{},lm_head:{}}'
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.hook_modules="['model.language_model.layers.{*}']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_modules="['model.language_model.layers.{*}.mlp.experts']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=fused

* ``fsdp_plan.apply_modules`` — modules to wrap with FSDP.
* ``fsdp_plan.hook_modules`` — modules whose forward hooks are managed by
  Turbo (needed for recompute overlap).
* ``ep_plan.apply_modules`` — modules whose parameters are split across the EP
  group (typically the MoE expert container).
* ``ep_plan.dispatcher`` — ``eager`` (default) or ``fused``.  Use ``fused``
  on NPU for better all-to-all overlap.

Memory plan
~~~~~~~~~~~

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
   +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan="['model.language_model.layers.{*}','model.visual.blocks.{*}']"

When ``recompute=True``, activation checkpointing is applied to every module
listed in ``recompute_plan``.

Module patches and CP function patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some Hugging Face model implementations need targeted patches so that
FSDP-Turbo's CP path can intercept attention and loss computation:

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches="[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.module_patches="[{target_function:...,patch_function:examples.qwen3_5_moe.modeling_qwen3_5_moe.qwen3_5_moe_gated_delta_net_forward}]"

Each entry in ``module_patches`` replaces a target function with a custom
patch function at engine initialization time.

Offload policy
~~~~~~~~~~~~~

FSDP-Turbo reuses verl's existing offload flags:

.. code-block:: bash

   actor_rollout_ref.actor.fsdp_config.offload_policy=True
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

When ``offload_policy`` is ``True``, Turbo applies ``CPUOffloadPolicy``
internally and disables verl's separate param/optimizer offload paths to avoid
double offloading.

Context parallelism constraints
-------------------------------

FSDP-Turbo CP has two hard requirements:

1. ``actor_rollout_ref.model.use_remove_padding`` must be ``True``.  verl's
   existing CP output path relies on remove-padding to gather local log-probs.
2. verl's own Ulysses sequence parallelism
   (``ulysses_sequence_parallel_size > 1``) must **not** be enabled
   simultaneously.  Use ``turbo_config.distributed.ulysses_parallel_size``
   instead.

Violating either constraint raises ``ValueError`` at engine initialization.

Gradient clipping with mixed meshes
-----------------------------------

When EP is enabled, dense and expert parameters may reside on different
``DeviceMesh`` objects.  PyTorch's stock ``clip_grad_norm_`` stacks per-gradient
DTensor scalars and fails when the meshes disagree.

FSDP-Turbo patches ``torch.nn.utils.clip_grad._get_total_norm`` and
``_clip_grads_with_norm_`` (see
``verl/workers/engine/fsdp/utils.py:apply_clip_grad_norm_patch``) to:

* Group norm scalars by ``(mesh, placement, device, dtype)`` before
  materialization, reducing the number of mesh-wide collectives from one per
  parameter to one per compatible group.
* Materialize each group's scalar via ``full_tensor()`` so that private
  ``_NormPartial`` placements complete their mesh reduction.
* Scale local gradient shards directly using a Python-float clip coefficient,
  avoiding ``foreach`` operations across incompatible meshes.

The patch is applied automatically when ``expert_parallel_size > 1`` and is
idempotent.  It delegates to the original PyTorch implementation when no
DTensor gradients are present, so non-Turbo code paths are unaffected.

Example scripts
---------------

Two GRPO example scripts are provided:

.. code-block:: bash

   # Qwen3.5-27B (dense)
   bash examples/grpo_trainer/run_qwen3_5_27b_fsdp_turbo.sh

   # Qwen3.5-35B-A3B (MoE with EP)
   bash examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh

Both scripts auto-detect NPU vs. GPU, configure HCCL environment variables on
NPU, and pass the full Turbo plan via Hydra command-line overrides.

For NPU nightly CI, a minimal smoke-test script is available:

.. code-block:: bash

   bash tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_5_2b_fsdp_turbo_npu.sh

Source of truth
---------------

* ``verl/workers/config/engine.py``: ``FSDPEngineConfig`` with the
  ``turbo_config`` field and ``strategy`` validation.
* ``verl/workers/engine/fsdp/transformer_impl.py``:
  ``FSDPTurboEngineWithLMHead`` — mesh initialization, module building, and
  CP validation.
* ``verl/workers/engine/fsdp/utils.py``:
  ``apply_clip_grad_norm_patch`` — mixed-mesh gradient clipping.
* ``verl/trainer/config/engine/fsdp.yaml``: default Turbo config schema.
* ``examples/grpo_trainer/run_qwen3_5_*_fsdp_turbo.sh``: runnable examples.
