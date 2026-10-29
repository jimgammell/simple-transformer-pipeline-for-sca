import torch
from torch import nn

from .bits_and_bytes import SoftGF256Inv, SoftGF256Mult, SoftSbox, SoftXOR


class ASCADv1Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_xor = SoftXOR()
        self.soft_sbox = SoftSbox()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, label_count, class_count = logits.shape
        assert label_count == 66
        assert class_count == 256
        r_logits = logits[:, 0:16, :]
        rin_logits = logits[:, 16:17, :]
        rout_logits = logits[:, 17:18, :]
        kwrin_logits = logits[:, 18:34, :]
        rsbox_logits = logits[:, 34:50, :]
        routsbox_logits = logits[:, 50:66, :]
        sbox_logits_0 = self.soft_xor(r_logits, rsbox_logits)
        sbox_logits_1 = self.soft_sbox(self.soft_xor(rin_logits, kwrin_logits))
        sbox_logits_2 = self.soft_xor(rout_logits, routsbox_logits)
        sbox_logits = torch.log_softmax(
            torch.stack([sbox_logits_0, sbox_logits_1, sbox_logits_2]), dim=-1
        ).sum(dim=0)
        return sbox_logits


class ASCADv2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_xor = SoftXOR()
        self.soft_gf256mult = SoftGF256Mult()
        self.soft_gf256inv = SoftGF256Inv()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if (
            logits.ndim != 3
            or logits.shape[1] not in (18, 34)
            or logits.shape[2] != 256
        ):
            raise ValueError(
                f"Expected [batch, 18 or 34, 256] logits, got {logits.shape}"
            )
        if logits.shape[1] == 34:
            recovered = []
            for chunk in logits.split(8):
                alpha_logits = chunk[:, 0:1, 1:]
                permutation_logits = chunk[:, 18:34, :16]
                if not torch.isfinite(torch.logsumexp(alpha_logits, dim=-1)).all():
                    raise ValueError("ASCADv2 alpha must have nonzero support")
                if not torch.isfinite(
                    torch.logsumexp(permutation_logits, dim=-1)
                ).all():
                    raise ValueError(
                        "ASCADv2 permutation indices must have support in 0..15"
                    )
                alpha_logits = torch.cat(
                    [torch.full_like(chunk[:, 0:1, :1], float("-inf")), alpha_logits],
                    dim=-1,
                )
                slot_logits = self._unmask(
                    torch.cat([alpha_logits, chunk[:, 1:18, :]], dim=1)
                )
                permutation_log_probs = torch.log_softmax(permutation_logits, dim=-1)
                if not torch.isfinite(
                    torch.logsumexp(permutation_log_probs, dim=1)
                ).all():
                    raise ValueError(
                        "Each ASCADv2 byte must have predicted permutation support"
                    )
                slot_log_weights = torch.log_softmax(permutation_log_probs, dim=1)
                recovered.append(
                    torch.logsumexp(
                        slot_log_weights.transpose(1, 2).unsqueeze(-1)
                        + slot_logits.unsqueeze(1),
                        dim=2,
                    )
                )
            return torch.cat(recovered, dim=0)
        return self._unmask(logits)

    def _unmask(self, logits: torch.Tensor) -> torch.Tensor:
        alpha_logits = logits[:, 0:1, :]
        beta_logits = logits[:, 1:2, :]
        masked_sbox_logits = logits[:, 2:18, :]
        logits = self.soft_gf256mult(
            self.soft_gf256inv(alpha_logits),
            self.soft_xor(masked_sbox_logits, beta_logits),
        )
        return logits
