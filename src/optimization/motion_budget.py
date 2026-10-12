"""
Motion-Aware Dynamic Budget Controller for SeedVR2
Analyzes inter-frame motion delta across video batch intervals to adaptively
modulate diffusion noise scale and sampling budgets:
- On low-motion / static scenes: reduces noise perturbation and latency for crisp detail.
- On high-motion scenes: allocates full smoothing budget to eliminate temporal artifacts.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


def compute_batch_motion_profile(
    input_images: torch.Tensor,
    batch_metadata: List[Tuple[int, int, bool]],
    delta: int = 1,
) -> List[float]:
    """
    Computes normalized motion intensity [0.0, 1.0] for each video batch.
    input_images: Tensor of shape (T, H, W, C) or (T, C, H, W).
    batch_metadata: List of (start_idx, end_idx, uniform_padding) intervals.
    """
    if not batch_metadata or input_images is None:
        return []

    T = input_images.shape[0]
    if T <= 1:
        return [0.0] * len(batch_metadata)

    # Normalize to (T, C, H, W) float
    if input_images.shape[-1] in (1, 3, 4):
        t_seq = input_images.permute(0, 3, 1, 2).float()
    else:
        t_seq = input_images.float()

    # Downsample spatially to 64x64 for instant computation (< 0.05s)
    with torch.no_grad():
        small = F.interpolate(t_seq, size=(64, 64), mode="bilinear", align_corners=False)
        frame_diffs = torch.abs(small[delta:] - small[:-delta]).mean(dim=(1, 2, 3))
        diff_len = frame_diffs.shape[0]

        batch_scores = []
        for start_idx, end_idx, _ in batch_metadata:
            s = max(0, start_idx)
            e = min(diff_len, end_idx - 1)
            if s < e:
                score = frame_diffs[s:e].mean().item()
            elif s < diff_len:
                score = frame_diffs[s].item()
            else:
                score = frame_diffs[-1].item() if diff_len > 0 else 0.0
            batch_scores.append(score)

        # Robust 90th percentile normalization
        scores_t = torch.tensor(batch_scores, dtype=torch.float32)
        min_v = scores_t.min()
        max_v = scores_t.max()
        spread = max_v - min_v

        if spread > 1e-6:
            q90 = torch.quantile(scores_t, 0.90)
            denom = max(1e-6, (q90 - min_v).item())
            norm_scores = torch.clamp((scores_t - min_v) / denom, 0.0, 1.0).tolist()
        else:
            norm_scores = [0.5] * len(batch_metadata)

    return norm_scores


class MotionBudgetController:
    """
    Dynamically modulates per-batch generation parameters based on scene motion.
    """

    PROFILES = {
        "detail": {
            "noise_floor": 0.80,
            "motion_weight": 0.20,
        },
        "balanced": {
            "noise_floor": 0.50,
            "motion_weight": 0.50,
        },
        "throughput": {
            "noise_floor": 0.25,
            "motion_weight": 0.75,
        },
    }

    def __init__(
        self,
        motion: Optional[List[float]] = None,
        quality_profile: str = "balanced",
        intensity: float = 1.0,
    ):
        self.motion = motion or []
        self.quality_profile = quality_profile.lower().strip()
        if self.quality_profile not in self.PROFILES:
            self.quality_profile = "balanced"
        self.intensity = max(0.0, float(intensity))

        params = self.PROFILES[self.quality_profile]
        self.noise_floor = params["noise_floor"]
        self.motion_weight = params["motion_weight"]

    def values_for_batch(self, batch_idx: int, base_noise_scale: float) -> float:
        """
        Calculates adapted latent_noise_scale for batch_idx.
        """
        if not self.motion or batch_idx >= len(self.motion):
            m = 0.5
        else:
            m = self.motion[batch_idx]

        eff_m = min(1.0, max(0.0, m * self.intensity))
        adapted_scale = base_noise_scale * (self.noise_floor + self.motion_weight * eff_m)
        return float(adapted_scale)

    def summary(self) -> str:
        avg_m = (sum(self.motion) / len(self.motion)) if self.motion else 0.5
        return (
            f"[Motion Budget] profile={self.quality_profile} "
            f"intensity={self.intensity:.2f} avg_motion={avg_m:.2f} "
            f"batches={len(self.motion)}"
        )
