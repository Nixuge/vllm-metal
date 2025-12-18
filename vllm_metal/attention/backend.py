# SPDX-License-Identifier: Apache-2.0
"""Metal attention backend for vLLM."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from vllm_metal._compat import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    init_logger,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@dataclass
class MetalAttentionMetadata(AttentionMetadata):
    """Metadata for Metal attention operations.

    This class stores the metadata needed for attention computation
    on Apple Metal backend.
    """

    # Sequence lengths for prefill
    seq_lens: list[int] | None = None
    # Sequence lengths as tensor
    seq_lens_tensor: torch.Tensor | None = None
    # Maximum sequence length in prefill
    max_prefill_seq_len: int = 0
    # Maximum sequence length in decode
    max_decode_seq_len: int = 0

    # Number of prefill tokens
    num_prefill_tokens: int = 0
    # Number of decode tokens
    num_decode_tokens: int = 0

    # Number of prefill sequences
    num_prefills: int = 0

    # Block tables for paged attention
    block_tables: torch.Tensor | None = None

    # Slot mapping for KV cache
    slot_mapping: torch.Tensor | None = None

    # Context lengths for each sequence
    context_lens_tensor: torch.Tensor | None = None

    # Query start locations for variable-length attention
    query_start_loc: torch.Tensor | None = None

    # Sequence start locations
    seq_start_loc: torch.Tensor | None = None

    # Attention mask for prefill (optional, for eager attention)
    attn_mask: torch.Tensor | None = None

    # Whether we're in prefill phase
    is_prompt: bool = False

    @property
    def prefill_metadata(self) -> Optional["MetalAttentionMetadata"]:
        """Get metadata for prefill phase."""
        if self.num_prefill_tokens == 0:
            return None

        return MetalAttentionMetadata(
            seq_lens=self.seq_lens,
            seq_lens_tensor=self.seq_lens_tensor,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            num_prefills=self.num_prefills,
            block_tables=None,
            slot_mapping=self.slot_mapping[: self.num_prefill_tokens]
            if self.slot_mapping is not None
            else None,
            context_lens_tensor=None,
            query_start_loc=self.query_start_loc,
            seq_start_loc=self.seq_start_loc,
            attn_mask=self.attn_mask,
            is_prompt=True,
        )

    @property
    def decode_metadata(self) -> Optional["MetalAttentionMetadata"]:
        """Get metadata for decode phase."""
        if self.num_decode_tokens == 0:
            return None

        return MetalAttentionMetadata(
            seq_lens=None,
            seq_lens_tensor=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=0,
            block_tables=self.block_tables,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens :]
            if self.slot_mapping is not None
            else None,
            context_lens_tensor=self.context_lens_tensor,
            query_start_loc=None,
            seq_start_loc=None,
            attn_mask=None,
            is_prompt=False,
        )


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for Metal attention metadata."""

    def __init__(self, input_builder, runner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_builder = input_builder
        self.runner = runner
        self.slot_mapping: list[int] = []
        self.prefill_seq_lens: list[int] = []
        self.context_lens: list[int] = []
        self.block_tables: list[list[int]] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

    def __call__(self, seq_group_metadata, seq_lens, query_lens, *args, **kwargs):
        """Process a sequence group."""
        is_prompt = seq_group_metadata.is_prompt
        block_tables = seq_group_metadata.block_tables

        for i, seq_id in enumerate(seq_group_metadata.seq_data):
            seq_len = seq_lens[i]
            query_len = query_lens[i]
            context_len = seq_len - query_len

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += query_len
                self.prefill_seq_lens.append(seq_len)

                # Compute slot mapping for prefill
                if block_tables:
                    block_table = block_tables[seq_id]
                    for j in range(query_len):
                        block_number = block_table[
                            (context_len + j) // self.runner.block_size
                        ]
                        block_offset = (context_len + j) % self.runner.block_size
                        slot = block_number * self.runner.block_size + block_offset
                        self.slot_mapping.append(slot)
            else:
                self.num_decode_tokens += 1
                self.context_lens.append(context_len)

                if block_tables:
                    block_table = block_tables[seq_id]
                    self.block_tables.append(list(block_table))

                    # Slot for single decode token
                    block_number = block_table[context_len // self.runner.block_size]
                    block_offset = context_len % self.runner.block_size
                    slot = block_number * self.runner.block_size + block_offset
                    self.slot_mapping.append(slot)

    def build(
        self,
        seq_lens: list[int],
        query_lens: list[int],
        cuda_graph_pad_size: int = 0,
        batch_size: int = 0,
    ) -> MetalAttentionMetadata:
        """Build the attention metadata."""
        device = self.runner.device

        slot_mapping_tensor = (
            torch.tensor(self.slot_mapping, dtype=torch.long, device=device)
            if self.slot_mapping
            else None
        )

        seq_lens_tensor = (
            torch.tensor(self.prefill_seq_lens, dtype=torch.int, device=device)
            if self.prefill_seq_lens
            else None
        )

        context_lens_tensor = (
            torch.tensor(self.context_lens, dtype=torch.int, device=device)
            if self.context_lens
            else None
        )

        # Build block tables tensor
        block_tables_tensor = None
        if self.block_tables:
            max_blocks = max(len(bt) for bt in self.block_tables)
            padded_block_tables = [
                bt + [0] * (max_blocks - len(bt)) for bt in self.block_tables
            ]
            block_tables_tensor = torch.tensor(
                padded_block_tables, dtype=torch.int, device=device
            )

        # Compute query start locations
        query_start_loc = None
        if query_lens:
            query_start_loc = torch.zeros(
                len(query_lens) + 1, dtype=torch.int32, device=device
            )
            query_start_loc[1:] = torch.cumsum(
                torch.tensor(query_lens, device=device), dim=0
            )

        # Compute sequence start locations
        seq_start_loc = None
        if self.prefill_seq_lens:
            seq_start_loc = torch.zeros(
                len(self.prefill_seq_lens) + 1, dtype=torch.int32, device=device
            )
            seq_start_loc[1:] = torch.cumsum(
                torch.tensor(self.prefill_seq_lens, device=device), dim=0
            )

        return MetalAttentionMetadata(
            seq_lens=self.prefill_seq_lens if self.prefill_seq_lens else None,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=max(self.prefill_seq_lens)
            if self.prefill_seq_lens
            else 0,
            max_decode_seq_len=max(self.context_lens) + 1 if self.context_lens else 0,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            block_tables=block_tables_tensor,
            slot_mapping=slot_mapping_tensor,
            context_lens_tensor=context_lens_tensor,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            is_prompt=self.num_prefills > 0,
        )


class MetalAttentionBackend(AttentionBackend):
    """Attention backend for Apple Metal."""

    # Whether this backend accepts an output buffer (for in-place operations)
    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "METAL"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        from vllm_metal.attention.metal_attention import MetalAttentionImpl

        return MetalAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[MetalAttentionMetadata]:
        return MetalAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[MetalAttentionMetadataBuilder]:
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Get the shape of the KV cache."""
        # Shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        # The '2' is for key and value
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap blocks between source and destination KV caches."""
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]
        dst_kv_cache[dst_indices] = src_kv_cache[src_indices]

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        """Copy blocks within KV caches."""
        for kv_cache in kv_caches:
            src_indices = src_to_dsts[:, 0]
            dst_indices = src_to_dsts[:, 1]
            kv_cache[dst_indices] = kv_cache[src_indices]

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        """Get supported attention head sizes."""
        return [64, 80, 96, 112, 128, 192, 256]

    @classmethod
    def get_required_kv_cache_layout(cls):
        """Return the required KV cache layout for this backend.

        Returns None to indicate no specific layout requirement,
        allowing the framework to use its default layout.
        """
        return None
