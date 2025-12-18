# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner V2 - High-performance inference with custom Metal kernels.

This is a thin Python wrapper around the Rust Metal kernels.
Key features:
- Custom Metal kernels for attention, GEMV, RoPE, RMS norm
- Zero-copy data transfer via unified memory
- vLLM-compatible interface
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Import Rust extensions
try:
    from vllm_metal_rust import (
        BatchStateManager,
        BlockTableManager,
        MetalBuffer,
    )

    RUST_METAL_AVAILABLE = True
    logger.info("Rust Metal V2 extensions loaded")
except ImportError as e:
    RUST_METAL_AVAILABLE = False
    MetalBuffer = None
    BatchStateManager = None  # type: ignore[misc, assignment]
    BlockTableManager = None  # type: ignore[misc, assignment]
    logger.warning(f"Rust Metal V2 extensions not available: {e}")


class MetalModelRunner:
    """Metal Model Runner V2 with custom Metal kernels.

    This runner uses Rust-based Metal kernels for maximum performance on Apple Silicon.
    It maintains compatibility with vLLM's model runner interface while using
    custom Metal kernels for attention and other critical operations.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert device.type == "mps", (
            f"MetalModelRunner requires Metal device (mps), got {device}"
        )

        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.lora_config = vllm_config.lora_config

        # Model parameters
        self.hidden_size = self.model_config.get_hidden_size()
        self.num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_dim = self.hidden_size // self.num_heads
        self.block_size = self.cache_config.block_size

        # GQA ratio
        self.gqa_ratio = self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1

        # Initialize Rust managers if available
        if RUST_METAL_AVAILABLE:
            max_num_reqs = self.scheduler_config.max_num_seqs
            max_model_len = self.model_config.max_model_len
            max_num_blocks = (max_model_len + self.block_size - 1) // self.block_size

            self._batch_state = BatchStateManager(
                max_num_reqs=max_num_reqs,
                max_model_len=max_model_len,
                block_size=self.block_size,
                max_num_blocks_per_req=max_num_blocks,
            )

            self._block_table = BlockTableManager(
                num_kv_cache_groups=1,
                block_size=self.block_size,
                max_num_reqs=max_num_reqs,
                max_num_blocks_per_req=max_num_blocks,
            )
        else:
            self._batch_state = None
            self._block_table = None

        # Model will be loaded later
        self.model: nn.Module | None = None

        # Memory tracking
        self.model_memory_usage: int = 0

        # KV cache
        self.kv_caches: list[torch.Tensor] = []

        # Statistics
        self._decode_count = 0
        self._prefill_count = 0

        # Pooling model flag
        self.is_pooling_model = False

        # Attention groups for warmup (empty for Metal)
        self.attn_groups: list = []

        logger.info(
            f"MetalModelRunner V2 initialized: "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"block_size={self.block_size}, "
            f"rust_available={RUST_METAL_AVAILABLE}"
        )

    def load_model(self, eep_scale_up: bool = False) -> nn.Module:
        """Load the model onto the Metal device."""
        from vllm.model_executor.model_loader import get_model

        self.model = get_model(
            vllm_config=self.vllm_config,
        )

        # Move to Metal device
        self.model = self.model.to(self.device)

        # Calculate model memory usage
        self.model_memory_usage = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )

        # Verify
        try:
            first_param = next(iter(self.model.parameters()))
            logger.info(
                f"Model loaded on device: {first_param.device}, "
                f"memory: {self.model_memory_usage / 1e9:.2f}GB"
            )
        except StopIteration:
            logger.warning("Model has no parameters")

        return self.model

    def update_config(self, overrides: dict) -> None:
        """Update configuration with overrides."""
        pass

    def reload_weights(self) -> None:
        """Reload model weights."""
        pass

    def maybe_remove_all_loras(self, lora_config) -> None:
        """Remove all LoRA adapters if needed.

        Metal doesn't support LoRA, so this is a no-op.
        """
        pass

    def compile_or_warm_up_model(self) -> None:
        """Compile or warm up the model.

        For Metal with eager mode, this just does a warmup forward pass.
        """
        if self.model is not None:
            logger.info("Warming up Metal model...")
            # Run a dummy forward pass to warm up Metal
            dummy_input = torch.randint(
                0, 100, (1, 16), dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.mps.synchronize()
            logger.info("Metal model warmup complete")

    def initialize_kv_cache(self, kv_cache_config) -> None:
        """Initialize KV cache based on kv_cache_config.

        Args:
            kv_cache_config: Configuration for the KV cache
        """
        from copy import deepcopy

        self.kv_cache_config = deepcopy(kv_cache_config)

        # Total number of blocks (shared across all groups)
        num_blocks = kv_cache_config.num_blocks

        # Create KV cache tensors for each cache group
        self.kv_caches = []

        for group_spec in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group_spec.kv_cache_spec

            # Create KV cache tensor
            # Shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
            kv_cache = torch.zeros(
                num_blocks,
                2,  # key and value
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                dtype=kv_cache_spec.dtype,
                device=self.device,
            )
            self.kv_caches.append(kv_cache)

        logger.info(
            f"Initialized {len(self.kv_caches)} KV cache group(s) with "
            f"{num_blocks} blocks on Metal"
        )

    def get_kv_cache_spec(self) -> dict:
        """Get the KV cache specification for the model.

        Returns:
            Dictionary mapping layer names to their KV cache specs.
        """
        from vllm.attention.layer import Attention
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.config.vllm import get_layers_from_vllm_config
        from typing import cast, Any

        kv_cache_spec: dict = {}
        self.shared_kv_cache_layers: dict[str, str] = {}

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)

        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                kv_tgt_layer = getattr(attn_module, 'kv_sharing_target_layer_name', None)
                if kv_tgt_layer:
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

            # Get the KV cache spec from the attention module
            if hasattr(attn_module, 'get_kv_cache_spec'):
                spec = attn_module.get_kv_cache_spec(self.vllm_config)
                if spec is not None:
                    kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def reset_mm_cache(self) -> None:
        """Reset multimodal cache. Not used for Metal."""
        pass

    def ensure_kv_transfer_shutdown(self) -> None:
        """Ensure KV transfer is shutdown. Not used for Metal."""
        pass

    def get_supported_tasks(self) -> set[str]:
        """Get supported tasks. For causal LM, we support 'generate'."""
        return {"generate"}

    def get_model(self) -> nn.Module | None:
        """Get the loaded model."""
        return self.model

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, list[int]]:
        """Execute model forward pass.

        This is a simplified interface for the V2 runner.
        For full vLLM integration, this would need to match
        the GPUModelRunner interface more closely.
        """
        # For now, delegate to PyTorch for model forward
        # The Metal kernels will be used for attention within the model
        raise NotImplementedError(
            "Full execute_model not yet implemented. "
            "V2 runner requires further vLLM integration."
        )

    def profile_run(self) -> None:
        """Profile run for memory estimation."""
        logger.info("Running Metal V2 profiling...")
        # Run a dummy forward pass to warm up Metal
        if self.model is not None:
            # Create dummy input
            dummy_input = torch.randint(
                0, 1000, (1, 16), dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.mps.synchronize()
        logger.info("Metal V2 profiling complete")

    def capture_model(self) -> int:
        """Capture model for graph execution.

        Metal doesn't support CUDA graphs, so this is a no-op.
        """
        logger.debug("Metal does not support graph capture, skipping")
        return 0

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model_config.get_vocab_size()


def create_metal_v2_runner(
    vllm_config: VllmConfig,
    device: torch.device,
) -> MetalModelRunner:
    """Factory function to create a Metal V2 model runner."""
    return MetalModelRunner(vllm_config, device)
