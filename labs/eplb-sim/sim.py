"""Minimal reproduction of DeepInfer's EPLB placement policy.

Imports policy.py directly: the `deepinfer` package itself refuses to import
off Linux (`os.uname()`), while the policy is pure torch.

    python labs/eplb-sim/sim.py
"""

import importlib.util
from pathlib import Path

import torch

POLICY = (
    Path(__file__).resolve().parents[2]
    / ".research/DeepInfer/deepinfer/model_executor/eplb/policy.py"
)
spec = importlib.util.spec_from_file_location("eplb_policy", POLICY)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def layer_peak(counts, physical_to_logical, copies, per_rank):
    """Mean over steps of the slowest rank's traffic, per layer.

    A physical copy serves 1/copies of its logical expert's traffic.
    """
    steps, layers, _ = counts.shape
    peaks = []
    for layer in range(layers):
        per_step = torch.zeros(steps)
        for step in range(steps):
            ranks = [
                sum(
                    counts[step, layer, expert].item() / copies[layer, expert].item()
                    for expert in physical_to_logical[
                        layer, rank * per_rank : (rank + 1) * per_rank
                    ].tolist()
                )
                for rank in range(physical_to_logical.size(1) // per_rank)
            ]
            per_step[step] = max(ranks)
        peaks.append(per_step.mean().item())
    return [round(peak, 1) for peak in peaks]


def main() -> None:
    torch.manual_seed(0)
    steps, layers, logical, physical, per_rank = 8, 3, 8, 12, 2

    # Skewed traffic: a few hot experts, with per-step jitter so the policy
    # optimizes the worst step rather than the mean.
    hot = torch.tensor([300.0, 240.0, 180.0, 120.0, 60.0, 30.0, 15.0, 7.0])
    counts = (hot * (0.8 + 0.4 * torch.rand(steps, layers, logical))).floor()

    # Naive: every expert on its own slot, 4 extra copies of experts 0..3,
    # laid out round-robin.
    naive_map = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]] * layers)
    naive_copies = torch.ones(layers, logical, dtype=torch.int)
    naive_copies[:, :4] += 1

    balanced_map, _, balanced_copies = policy.rebalance_experts_global(
        counts, physical, per_rank
    )

    print("naive      peak per layer:", layer_peak(counts, naive_map, naive_copies, per_rank))
    print(
        "rebalanced peak per layer:",
        layer_peak(counts, balanced_map, balanced_copies, per_rank),
    )
    print("copies per expert (layer0):", balanced_copies[0].tolist())
    print("placement (layer0):", balanced_map[0].tolist())

    assert balanced_copies[0, 0] > balanced_copies[0, 7], "hot expert must get more copies"
    assert layer_peak(counts, balanced_map, balanced_copies, per_rank)[0] < layer_peak(
        counts, naive_map, naive_copies, per_rank
    )[0], "rebalancing must lower the peak"
    print("PASS hot expert replicated, peak lowered")


if __name__ == "__main__":
    main()
