# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "silu_mul_block_quant_fp8 Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

# from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_silu_mul_block_quant_fp8_inputs() -> dict[str, tuple[Any, ...]]:
    hidden_sizes = [64, 2880, 4096, 8192, 11008, 14336]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    in_dtype = torch.bfloat16
    out_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float32

    inputs = {}
    for num_tokens in num_tokens_list:
        for hidden_size in hidden_sizes:
            input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
            out = torch.empty(
                (num_tokens, hidden_size // 2), device=input.device, dtype=out_dtype
            )
            block_size = 16
            scales = torch.empty(
                (num_tokens, hidden_size // (block_size * 2)),
                device=input.device,
                dtype=scale_dtype,
            )
            scale_ub = torch.mean(input.abs()).to(scale_dtype).item()

            config_key = f"intermediate_{hidden_size}_numtokens_{num_tokens}"
            inputs[config_key] = (input, out, scales, block_size, scale_ub)

    return inputs


def pick_silu_mul_block_quant_fp8_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that intermediate_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "intermediate_{int}_numtokens_{int}".
    """
    if not config_keys:
        return None

    input_tensor: torch.Tensor = args[0]
    intermediate_size = input_tensor.shape[-1] // 2
    num_tokens = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"intermediate_(\d+)_numtokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'intermediate_{{int}}_numtokens_{{int}}'"
            )
        isize_str, ntokens_str = match.groups()
        configs.setdefault(int(isize_str), []).append(int(ntokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_isize = min(configs, key=lambda s: abs(s - intermediate_size))
    available_ntokens = sorted(configs[best_isize])
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    return f"intermediate_{best_isize}_numtokens_{best_ntokens}"


fp8_max = torch.finfo(torch.float8_e4m3fn).max

"""
@register_kernel(
    mutates_args=["out", "scales"],
    helion_settings=helion.Settings(
        autotune_baseline_atol=0.2,
        autotune_baseline_rtol=0.1,     
        #TODO search._autotune_metrics.num_accuracy_failures for tolerance strictness
    ),
    config_picker=pick_silu_mul_block_quant_fp8_config,
    input_generator=generate_silu_mul_block_quant_fp8_inputs,
)"""


@helion.kernel(autotune_effort="none")
def silu_mul_block_quant_fp8(
    input: torch.Tensor,  # [num_tokens, 2 * hidden_size]
    out: torch.Tensor,  # [num_tokens, hidden_size]
    scales: torch.Tensor,  # [num_tokens, hidden_size // block_size]
    block_size: int,
    scale_ub: float | None = None,
    is_scale_transposed: bool = False,
) -> torch.Tensor:
    # This code assumes batch_dim and num_tokens are flattened
    group_size = hl.specialize(block_size)
    assert group_size == 16
    assert input.is_contiguous() and input.ndim == 2
    assert scales.is_contiguous() and scales.dtype == torch.float32
    assert out.is_contiguous() and out.dtype == torch.float8_e4m3fn
    assert out.shape[-1] == input.shape[-1] // 2

    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    m = hl.specialize(original_shape[0])

    input_2d = input.view(-1, original_shape[-1])

    min_scaling_factor = 1 / (fp8_max * 512.0)

    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]

    for tile_m, tile_d in hl.tile([m, d], block_size=[1, group_size]):
        a_vals = input_part_a[tile_m, tile_d]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_d]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        abs_val = torch.abs(result_f32.reshape(-1))
        abs_max = torch.amax(abs_val)
        block_scale = abs_max / fp8_max

        if scale_ub is not None:
            block_scale = torch.clamp(block_scale, max=scale_ub)

        block_scale = torch.clamp(block_scale, min=min_scaling_factor)
        inv_block_scale = 1.0 / block_scale

        out[tile_m, tile_d] = (result_f32 * inv_block_scale).to(torch.float8_e4m3fn)

        if is_scale_transposed:
            scales[tile_d.id, tile_m] = inv_block_scale
        else:
            scales[tile_m, tile_d.id] = inv_block_scale

    return out, scales


def silu_mul_block_quant_fp8_baseline(
    input: torch.Tensor,
    block_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> torch.Tensor:
    return torch.ops._C.silu_and_mul_per_block_quant(
        input, block_size, scale_ub, is_scale_transposed
    )
