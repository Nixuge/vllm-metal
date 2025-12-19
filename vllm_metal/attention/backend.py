# SPDX-License-Identifier: Apache-2.0
"""Metal attention backend for vLLM."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_metal._compat import (
    AttentionBackend,
    AttentionMetadata,
    init_logger,
)

if TYPE_CHECKING:
    from vllm_metal.attention.metal_attention import MetalAttentionImpl

logger = init_logger(__name__)


@dataclass
class MetalAttentionMetadata(AttentionMetadata):
    """Metadata for Metal attention operations.

    This class holds the attention metadata needed for both prefill
    and decode phases on the Metal backend.
    """

    # Basic info
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0

    # Prefill metadata
    query_start_loc: torch.Tensor | None = None  # [num_seqs + 1]
    seq_lens: torch.Tensor | None = None  # [num_seqs]
    context_lens: torch.Tensor | None = None  # [num_seqs]

    # Decode metadata
    block_table: torch.Tensor | None = None  # [num_seqs, max_num_blocks]
    slot_mapping: torch.Tensor | None = None  # [num_tokens]

    # For paged attention
    max_num_blocks_per_seq: int = 0
    use_cuda_graph: bool = False  # Always False for Metal

    # Cascade attention (not supported on Metal)
    use_cascade: bool = False

    # Device
    device: torch.device = torch.device("mps")

    @property
    def is_all_prefill(self) -> bool:
        """Check if all requests are in prefill phase.

        In vLLM V1 with paged attention, block_table may always be present.
        Use max_query_len to distinguish prefill (>1) from decode (==1).
        """
        return self.max_query_len > 1

    @property
    def is_all_decode(self) -> bool:
        """Check if all requests are in decode phase.

        Decode phase processes exactly 1 token per sequence.
        """
        return self.max_query_len == 1 and self.block_table is not None

    def get_seq_lens(self) -> torch.Tensor:
        """Get sequence lengths."""
        if self.seq_lens is not None:
            return self.seq_lens
        return torch.zeros(0, dtype=torch.int32, device=self.device)


class MetalAttentionMetadataBuilder:
    """Builder for MetalAttentionMetadata.

    This class helps construct attention metadata from batch information.
    Compatible with vLLM V1 engine interface.
    """

    def __init__(
        self,
        kv_cache_spec: Any = None,
        layer_names: list[str] | None = None,
        vllm_config: Any = None,
        device: torch.device | None = None,
    ):
        """Initialize the metadata builder.

        Args:
            kv_cache_spec: KV cache specification from vLLM V1.
            layer_names: Names of attention layers in this group.
            vllm_config: vLLM configuration.
            device: Target device for tensors.
        """
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names or []
        self.vllm_config = vllm_config
        # Metal doesn't support batch reordering optimization
        self.reorder_batch_threshold: int | None = None
        # Handle device parameter - can be string or torch.device
        if device is None:
            self.device = torch.device("mps")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        """Build attention metadata from CommonAttentionMetadata.

        This is the main entry point called by vLLM V1's build_attn_metadata.

        Args:
            common_prefix_len: Length of common prefix across sequences.
            common_attn_metadata: CommonAttentionMetadata containing batch info.
            fast_build: Whether to skip expensive computations.

        Returns:
            MetalAttentionMetadata for this batch.
        """
        return MetalAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            device=self.device,
        )

    def build_prefill_metadata(
        self,
        seq_lens: list[int],
        query_lens: list[int],
        slot_mapping: torch.Tensor,
    ) -> MetalAttentionMetadata:
        """Build metadata for prefill phase.

        Args:
            seq_lens: Length of each sequence.
            query_lens: Query length for each sequence.
            slot_mapping: Slot mapping for KV cache.

        Returns:
            MetalAttentionMetadata for prefill.
        """
        num_seqs = len(seq_lens)
        total_tokens = sum(query_lens)

        # Compute query start locations
        query_start_loc = torch.zeros(
            num_seqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        for i, qlen in enumerate(query_lens):
            query_start_loc[i + 1] = query_start_loc[i] + qlen

        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int32,
            device=self.device,
        )

        return MetalAttentionMetadata(
            num_actual_tokens=total_tokens,
            max_query_len=max(query_lens) if query_lens else 0,
            max_seq_len=max(seq_lens) if seq_lens else 0,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_tensor,
            slot_mapping=slot_mapping.to(self.device),
            device=self.device,
        )

    def build_decode_metadata(
        self,
        seq_lens: list[int],
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> MetalAttentionMetadata:
        """Build metadata for decode phase.

        Args:
            seq_lens: Length of each sequence.
            block_table: Block table for paged attention.
            slot_mapping: Slot mapping for KV cache.

        Returns:
            MetalAttentionMetadata for decode.
        """
        num_seqs = len(seq_lens)

        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int32,
            device=self.device,
        )

        return MetalAttentionMetadata(
            num_actual_tokens=num_seqs,  # One token per sequence in decode
            max_query_len=1,
            max_seq_len=max(seq_lens) if seq_lens else 0,
            seq_lens=seq_lens_tensor,
            block_table=block_table.to(self.device),
            slot_mapping=slot_mapping.to(self.device),
            max_num_blocks_per_seq=block_table.shape[1] if block_table.dim() > 1 else 0,
            device=self.device,
        )


class MetalAttentionBackend(AttentionBackend):
    """Attention backend for Apple Metal using MLX.

    This backend provides attention computation using MLX's optimized
    scaled_dot_product_attention on Apple Silicon.
    """

    # Tell vLLM to reshape Q/K/V tensors to [num_tokens, num_heads, head_size]
    # before calling impl.forward(), and to reshape output back after.
    accept_output_buffer = True

    @staticmethod
    def get_name() -> str:
        """Get the name of this backend."""
        return "metal"

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        """Return fully qualified class name as (module, qualname)."""
        return (cls.__module__, cls.__qualname__)

    @staticmethod
    def get_impl_cls() -> type["MetalAttentionImpl"]:
        """Get the implementation class."""
        from vllm_metal.attention.metal_attention import MetalAttentionImpl

        return MetalAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[MetalAttentionMetadata]:
        """Get the metadata class."""
        return MetalAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[MetalAttentionMetadataBuilder]:
        """Get the metadata builder class."""
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Get the shape of the KV cache.

        vLLM V1 allocates K and V together, so the shape includes
        a dimension of 2 for the K/V split.

        Args:
            num_blocks: Number of cache blocks.
            block_size: Tokens per block.
            num_kv_heads: Number of key-value heads.
            head_size: Size of each head.
            cache_dtype_str: Cache dtype string (ignored for Metal).

        Returns:
            Shape tuple for the cache tensors (includes both K and V).
        """
        # Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
        # where 2 is for K and V
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap KV cache blocks.

        On Apple Silicon with unified memory, this is a simple copy.
        """
        for src, dst in src_to_dst.tolist():
            dst_kv_cache[dst].copy_(src_kv_cache[src])

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """Copy KV cache blocks.

        Args:
            kv_caches: List of [key_cache, value_cache] pairs.
            src_to_dists: Source to destination block mapping.
        """
        for src, dst in src_to_dists.tolist():
            for kv_cache in kv_caches:
                kv_cache[dst].copy_(kv_cache[src])
