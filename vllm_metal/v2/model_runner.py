# SPDX-License-Identifier: Apache-2.0
"""Metal V2 Model Runner - extends GPU model runner for Metal/MLX backend."""

from contextlib import contextmanager

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# ============================================================================
# Module-level patching for Metal (must happen before importing GPUModelRunner)
# ============================================================================


def _patched_bincount_metal(
    prefill_token_ids: torch.Tensor,
    prefill_len: int,
    prompt_len: int,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """PyTorch-based bincount replacement for Metal (no Triton)."""
    prompt_bin_mask.zero_()
    output_bin_counts.zero_()

    # Get the tokens in the range [prompt_len, prefill_len)
    if prefill_len > prompt_len:
        tokens = prefill_token_ids[prompt_len:prefill_len]
        tokens_cpu = tokens.cpu().to(torch.int64)
        vocab_size = output_bin_counts.shape[0]
        counts = torch.bincount(tokens_cpu, minlength=vocab_size)
        min_len = min(len(counts), vocab_size)
        output_bin_counts[:min_len] = counts[:min_len].to(output_bin_counts.device)

    # Set prompt_bin_mask for tokens in [0, prompt_len)
    if prompt_len > 0:
        prompt_tokens = prefill_token_ids[:prompt_len]
        prompt_tokens_cpu = prompt_tokens.cpu().to(torch.int64)
        vocab_size = prompt_bin_mask.shape[0]
        for token in prompt_tokens_cpu:
            if 0 <= token < vocab_size:
                prompt_bin_mask[token] = 1


# =============================================================================
# Patch BlockTables with Metal-compatible implementation (pure PyTorch)
# vLLM 0.11.0 uses vllm.v1.worker.block_table (not gpu.block_table)
# =============================================================================
try:
    import vllm.v1.worker.block_table as block_table_module

    from vllm_metal.v2.metal_block_table import MetalBlockTables

    # Replace the entire BlockTables class with our Metal implementation
    block_table_module.BlockTables = MetalBlockTables
    logger.debug("Patched BlockTables with MetalBlockTables for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch block_table module: {e}")


# =============================================================================
# Mock classes for CUDA compatibility on MPS
# =============================================================================


class _MockCudaStream:
    """Mock CUDA stream for MPS compatibility."""

    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        torch.mps.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _MockCudaEvent:
    """Mock CUDA event for MPS compatibility."""

    def __init__(self, *args, **kwargs):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        torch.mps.synchronize()

    def query(self):
        return True


class _MockCudaGraphManager:
    """Mock CudaGraphManager for MPS compatibility."""

    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device
        self.pool = None
        self.cudagraph_mode = None
        self.capture_sizes = []
        self.disabled = True

    def capture(self, *args, **kwargs):
        pass

    def replay(self, *args, **kwargs):
        return None

    def get_graph(self, *args, **kwargs):
        return None

    def should_use_cudagraph(self, *args, **kwargs):
        return False

    def get_cudagraph_size(self, *args, **kwargs):
        # Return None to indicate no cudagraph should be used
        return None

    def get_cudagraph(self, *args, **kwargs):
        return None


# Now import the rest of vLLM modules (they will get our patched functions)
# vLLM 0.11.0 module paths (no gpu. prefix)
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402
from vllm.v1.utils import CpuGpuBuffer  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402


@contextmanager
def _patch_cuda_for_mps():
    """Context manager to patch CUDA stream/event/graph for MPS compatibility.

    vLLM's GPUModelRunner.__init__ creates CUDA streams, events and graphs.
    We temporarily replace them with MPS-compatible mocks.
    """
    original_stream = torch.cuda.Stream
    original_event = torch.cuda.Event
    original_current_stream = torch.cuda.current_stream
    original_graph_pool = getattr(torch.cuda, "graph_pool_handle", None)

    try:
        torch.cuda.Stream = _MockCudaStream  # type: ignore[assignment]
        torch.cuda.Event = _MockCudaEvent  # type: ignore[assignment]
        torch.cuda.current_stream = lambda device=None: _MockCudaStream()  # type: ignore[assignment,return-value]
        torch.cuda.graph_pool_handle = lambda: None  # type: ignore[assignment,return-value]
        yield
    finally:
        torch.cuda.Stream = original_stream
        torch.cuda.Event = original_event
        torch.cuda.current_stream = original_current_stream
        if original_graph_pool is not None:
            torch.cuda.graph_pool_handle = original_graph_pool


class MetalModelRunner(GPUModelRunner):
    """Metal/MLX model runner that extends the GPU model runner.

    This class inherits all the complex input batch management, attention
    metadata building, and model execution from GPUModelRunner. It only
    overrides Metal-specific functionality like:
    - Disabling CUDA-specific features (pinned memory, CUDA graphs)
    - Using MPS/MLX synchronization instead of CUDA
    - Metal-specific device handling
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Patch CUDA stream/event to work with MPS before calling parent init
        with _patch_cuda_for_mps():
            super().__init__(vllm_config, device)

        # Override CUDA-specific settings
        self.pin_memory = False  # Metal uses unified memory
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace the mock streams with our MPS-compatible versions
        self.output_copy_stream = _MockCudaStream()

        # Replace GPU tensors with MPS equivalents
        self._postprocess_tensors()

        # Log initialization
        logger.info(
            f"MetalModelRunner V2 initialized: "
            f"hidden_size={self.model_config.get_hidden_size()}, "
            f"num_heads={self.model_config.get_num_attention_heads(self.parallel_config)}, "
            f"num_kv_heads={self.model_config.get_num_kv_heads(self.parallel_config)}, "
            f"head_dim={self.model_config.get_head_size()}, "
            f"block_size={self.cache_config.block_size}"
        )

    def _postprocess_tensors(self) -> None:
        """Replace GPU tensors with device tensors for Metal."""
        # For Metal, we don't need separate CPU and GPU buffers
        # since MPS/MLX uses unified memory
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

    def _init_device_properties(self) -> None:
        """Initialize device properties for Metal/MPS.

        Override parent's CUDA-specific implementation which calls
        torch.cuda.get_device_properties().
        """
        # Metal doesn't have CUDA device properties, so we create mock values
        # num_sms is used for some internal calculations in vLLM
        from vllm_metal.utils import get_metal_device_info

        info = get_metal_device_info()

        # Create a mock device properties object
        class MetalDeviceProperties:
            def __init__(self, name: str, total_memory: int):
                self.name = name
                self.total_memory = total_memory
                # Apple Silicon doesn't have SMs like NVIDIA GPUs
                # Use GPU cores / 128 as a rough approximation
                self.multi_processor_count = max(1, info.get("gpu_cores", 8) // 128)

        self.device_properties = MetalDeviceProperties(
            name=info.get("name", "Apple Silicon"),
            total_memory=info.get("total_memory", 0),
        )
        self.num_sms = self.device_properties.multi_processor_count

    def _sync_device(self) -> None:
        """Synchronize the MPS/MLX device instead of CUDA."""
        import mlx.core as mx

        mx.eval([])  # Force MLX evaluation
        torch.mps.synchronize()

    def load_model(self, *args, **kwargs) -> None:
        """Load the model to the MPS device."""
        logger.info("Loading model with MLX acceleration...")

        # Load model using standard vLLM loader
        self.model = get_model(
            vllm_config=self.vllm_config,
        )

        # Move model to MPS device
        self.model = self.model.to(self.device)

        logger.info("Model loaded successfully")

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache for Metal backend.

        Delegates to parent GPUModelRunner's implementation which handles
        attention backend and KV cache initialization in vLLM 0.11.0.
        """
        # Call parent implementation - it handles all initialization
        super().initialize_kv_cache(kv_cache_config)

        # Metal backend - log initialized backends
        if hasattr(self, "attn_backends") and self.attn_backends:
            logger.info(
                f"Metal attention backends initialized: {list(self.attn_backends.keys())}"
            )
