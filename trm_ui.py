"""
TRM-UI: TAOBAO-MM user-interest model built around Samsung's TRM core.

Directory layout expected:

    /content/
    ├── TinyRecursiveModels/      # Samsung TRM repository
    └── TRM-UI/                   # this project

This file intentionally imports the recursive reasoning blocks from the
sibling TinyRecursiveModels repository instead of copying/reimplementing
Samsung's attention/SwiGLU/RMSNorm implementation.

First-stage goal:
    TAOBAO-MM embeddings [B, T, 256]
        -> projection [B, T, 512]
        -> Samsung TRM recursive core
        -> user-interest representation z_H [B, 512]
        -> target projection [B, 512]
        -> binary click/no-click score

This is a forward/backward integration model, not the final training setup.
ACT/Q-halting and the original TRM LM head are deliberately not used here
because those components are specific to Samsung's puzzle-token task.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Locate the sibling Samsung TRM repository.
# ---------------------------------------------------------------------------

TRM_REPO = Path(__file__).resolve().parent.parent / "TinyRecursiveModels"

if not TRM_REPO.exists():
    raise FileNotFoundError(
        "Could not find the sibling TinyRecursiveModels repository.\n"
        f"Expected it at:\n  {TRM_REPO}\n\n"
        "Expected layout:\n"
        "  /content/TinyRecursiveModels\n"
        "  /content/TRM-UI\n"
    )

if str(TRM_REPO) not in sys.path:
    sys.path.insert(0, str(TRM_REPO))


from models.common import trunc_normal_init_  # noqa: E402
from models.recursive_reasoning.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
)
from models.layers import RotaryEmbedding


class TRMUI(nn.Module):
    """TRM-UI using Samsung's recursive reasoning core.

    The Samsung TRM recursion is retained:

        for H cycle:
            repeat L cycles:
                z_L <- TRM(z_L, z_H + input)
            z_H <- TRM(z_H, z_L)

    The task-specific input/output interface is changed from puzzle tokens to
    TAOBAO-MM user history and target-item representations.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_size: int = 512,
        num_heads: int = 8,
        expansion: float = 4.0,
        history_length: int = 50,
        H_cycles: int = 3,
        L_cycles: int = 6,
        L_layers: int = 2,
        forward_dtype: str = "float32",
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if H_cycles <= 0 or L_cycles <= 0 or L_layers <= 0:
            raise ValueError("H_cycles, L_cycles and L_layers must be positive")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # ---------------------------------------------------------------
        # 256 -> 512 interface for the Samsung TRM hidden dimension.
        # ---------------------------------------------------------------
        self.history_projection = nn.Linear(input_dim, hidden_size)
        self.target_projection = nn.Linear(input_dim, hidden_size)

        # ---------------------------------------------------------------
        # Build the configuration expected by Samsung's TRM blocks.
        # We do NOT instantiate Samsung's puzzle-specific embedding/lm/q
        # heads. We only reuse its recursive reasoning block.
        # ---------------------------------------------------------------
        trm_config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=1,
            seq_len=history_length,
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=1,
            vocab_size=1,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            H_layers=0,
            L_layers=L_layers,
            hidden_size=hidden_size,
            expansion=expansion,
            num_heads=num_heads,
            pos_encodings="rope",
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            halt_max_steps=H_cycles,
            halt_exploration_prob=0.0,
            forward_dtype=forward_dtype,
            mlp_t=False,
            puzzle_emb_len=0,
            no_ACT_continue=True,
        )

        # Samsung's official TRM uses the same L-level reasoning module for
        # both z_L and z_H updates. We preserve that design exactly here.
        self.reasoning_core = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(trm_config) for _ in range(L_layers)]
        )

        # Learnable initial latent states. Samsung's implementation stores
        # these as initialized buffers; making them parameters is useful for
        # this new recommendation task and will be explicitly documented as
        # a TRM-UI interface adaptation.
        self.register_buffer(
            "z_H_init",
            trunc_normal_init_(torch.empty(hidden_size), std=1.0),
        )
        self.register_buffer(
            "z_L_init",
            trunc_normal_init_(torch.empty(hidden_size), std=1.0),
        )

        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=history_length,
            base=10000.0,
        )

        # ---------------------------------------------------------------
        # Recommendation head.
        # The target is NOT injected into the recursive history stream.
        # This lets z_H represent user interest before target comparison.
        # ---------------------------------------------------------------
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def _recursive_reasoning(self, history: torch.Tensor) -> torch.Tensor:
        """Run Samsung's H/L recursive process and return final z_H."""

        batch_size = history.shape[0]

        z_H = self.z_H_init.unsqueeze(0).expand(batch_size, -1).clone()
        z_L = self.z_L_init.unsqueeze(0).expand(batch_size, -1).clone()

        # The Samsung block operates on [B, T, D], so broadcast the latent
        # state across the complete user-history sequence.
        z_H = z_H.unsqueeze(1).expand(-1, self.history_length, -1)
        z_L = z_L.unsqueeze(1).expand(-1, self.history_length, -1)

        # Samsung TRM uses RoPE. We obtain the positional cos/sin tensors from
        # the same RotaryEmbedding implementation used by the original core.
        # The reasoning block itself remains untouched.
        cos_sin = self.rotary_emb()

        # Match Samsung TRM's recursive schedule: all but the final H cycle
        # are run without autograd; the final cycle carries gradients.
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.reasoning_core(
                        z_L,
                        z_H + history,
                        cos_sin=cos_sin,
                    )
                z_H = self.reasoning_core(
                    z_H,
                    z_L,
                    cos_sin=cos_sin,
                )

        for _ in range(self.L_cycles):
            z_L = self.reasoning_core(
                z_L,
                z_H + history,
                cos_sin=cos_sin,
            )
        z_H = self.reasoning_core(
            z_H,
            z_L,
            cos_sin=cos_sin,
        )

        # Aggregate the final latent over history positions to obtain a single
        # user-interest vector.
        return z_H.mean(dim=1)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict click probability from history and target embeddings.

        Args:
            history_embeddings: [B, T, 256]
            target_embedding:   [B, 256]

        Returns:
            Dictionary containing:
                user_interest: [B, 512]
                target_repr:   [B, 512]
                logits:        [B]
                probability:   [B]
        """

        if history_embeddings.ndim != 3:
            raise ValueError(
                "history_embeddings must have shape [B, T, input_dim]"
            )
        if target_embedding.ndim != 2:
            raise ValueError(
                "target_embedding must have shape [B, input_dim]"
            )
        if history_embeddings.shape[0] != target_embedding.shape[0]:
            raise ValueError("History and target batch sizes must match")
        if history_embeddings.shape[1] != self.history_length:
            raise ValueError(
                f"Expected history length {self.history_length}, "
                f"got {history_embeddings.shape[1]}"
            )
        if history_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected history embedding dimension {self.input_dim}, "
                f"got {history_embeddings.shape[-1]}"
            )
        if target_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected target embedding dimension {self.input_dim}, "
                f"got {target_embedding.shape[-1]}"
            )

        history = self.history_projection(history_embeddings)
        target = self.target_projection(target_embedding)

        user_interest = self._recursive_reasoning(history)

        combined = torch.cat([user_interest, target], dim=-1)
        logits = self.scoring_head(combined).squeeze(-1)
        probability = torch.sigmoid(logits)

        return {
            "user_interest": user_interest,
            "target_repr": target,
            "logits": logits,
            "probability": probability,
        }


# ---------------------------------------------------------------------------
# Small standalone integration test.
# ---------------------------------------------------------------------------


def run_integration_test(device: str | None = None) -> None:
    """Run one real forward/backward pass using random 256-d inputs.

    This deliberately tests the model interface independently of the
    TAOBAO-MM dataloader. The next test can plug in the real embeddings.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("TRM-UI recursive-core integration test")
    print("=" * 60)
    print(f"Device: {device}")

    torch.manual_seed(42)

    model = TRMUI(
        input_dim=256,
        hidden_size=512,
        num_heads=8,
        expansion=4.0,
        history_length=50,
        H_cycles=3,
        L_cycles=6,
        L_layers=2,
        forward_dtype="float32",
    ).to(device)

    batch_size = 2
    history = torch.randn(batch_size, 50, 256, device=device)
    target = torch.randn(batch_size, 256, device=device)
    labels = torch.tensor([0.0, 1.0], device=device)

    outputs = model(history, target)
    loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["logits"], labels
    )

    print("\nForward output:")
    print(f"  user_interest: {tuple(outputs['user_interest'].shape)}")
    print(f"  target_repr:   {tuple(outputs['target_repr'].shape)}")
    print(f"  logits:        {tuple(outputs['logits'].shape)}")
    print(f"  probability:   {outputs['probability'].detach().cpu()}")
    print(f"  loss:           {loss.item():.6f}")

    loss.backward()

    finite_grads = True
    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient_count += 1
            if not torch.isfinite(parameter.grad).all():
                finite_grads = False
                print(f"  Non-finite gradient: {name}")

    print("\nBackward check:")
    print(f"  parameters with gradients: {gradient_count}")
    print(f"  all gradients finite:      {finite_grads}")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN/Inf")
    if not finite_grads:
        raise RuntimeError("At least one gradient contains NaN/Inf")
    if gradient_count == 0:
        raise RuntimeError("No gradients reached the model")

    print("\n✓ TRM recursive forward pass succeeded.")
    print("✓ Binary scoring head succeeded.")
    print("✓ Backward pass succeeded.")
    print("✓ Gradients are finite.")


if __name__ == "__main__":
    run_integration_test()
