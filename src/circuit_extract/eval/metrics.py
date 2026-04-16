"""Metrics for comparing an extracted netlist against ground truth.

Two axes of evaluation:

1. **Component detection** — did we find the right components?
   Measured by precision / recall / F1 on component matching. A predicted
   component matches a ground-truth component if they agree on type (and
   optionally id / value). We use a greedy best-match strategy.

2. **Net accuracy** — did we get the topology right?
   Measured by the Adjusted Rand Index (ARI) over the pin→net assignment.
   Each pin is identified by (component_id, pin_name). We build a clustering
   of pins for both predicted and ground-truth netlists, then compute ARI.
   ARI = 1.0 means perfect agreement; 0.0 means random.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from circuit_extract.schema import Netlist

# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------


@dataclass
class ComponentMetrics:
    """Precision / recall / F1 for component detection."""

    matched: int
    predicted: int
    ground_truth: int

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.ground_truth if self.ground_truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _normalise_type(t: str) -> str:
    """Collapse type variants so matching is less strict on polarity."""
    t = t.lower().strip()
    mapping = {
        "bjt_npn": "bjt",
        "bjt_pnp": "bjt",
        "nmos": "mosfet",
        "pmos": "mosfet",
    }
    return mapping.get(t, t)


def _match_components(predicted: Netlist, ground_truth: Netlist) -> int:
    """Greedy matching: each predicted component matches at most one GT."""
    gt_pool = list(range(len(ground_truth.components)))
    matched = 0

    for pred_comp in predicted.components:
        best_idx: int | None = None
        best_score = -1
        for i in gt_pool:
            gt_comp = ground_truth.components[i]
            score = 0
            if _normalise_type(pred_comp.type) == _normalise_type(gt_comp.type):
                score += 2
            if pred_comp.id.upper() == gt_comp.id.upper():
                score += 1
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None and best_score >= 2:
            gt_pool.remove(best_idx)
            matched += 1

    return matched


def component_metrics(predicted: Netlist, ground_truth: Netlist) -> ComponentMetrics:
    matched = _match_components(predicted, ground_truth)
    return ComponentMetrics(
        matched=matched,
        predicted=len(predicted.components),
        ground_truth=len(ground_truth.components),
    )


# ---------------------------------------------------------------------------
# Net metrics (Adjusted Rand Index)
# ---------------------------------------------------------------------------


PinKey = tuple[str, str]  # (component_id, pin)


def _pin_to_cluster(netlist: Netlist) -> dict[PinKey, int]:
    """Map each pin to a cluster id (the index of the net it belongs to)."""
    mapping: dict[PinKey, int] = {}
    for cluster_id, net in enumerate(netlist.nets):
        for ref in net.pins:
            key: PinKey = (ref.component_id.upper(), ref.pin.upper())
            mapping[key] = cluster_id
    return mapping


def _adjusted_rand_index(clusters_a: dict[PinKey, int], clusters_b: dict[PinKey, int]) -> float:
    """Compute the Adjusted Rand Index between two clusterings.

    Only considers pins that appear in *both* clusterings (the intersection).
    """
    common_keys = sorted(set(clusters_a) & set(clusters_b))
    n = len(common_keys)
    if n < 2:
        return 0.0

    labels_a = [clusters_a[k] for k in common_keys]
    labels_b = [clusters_b[k] for k in common_keys]

    # Count pair agreements
    tp = 0  # same in both
    tn = 0  # different in both
    fp = 0  # same in a, different in b
    fn = 0  # different in a, same in b

    for i, j in combinations(range(n), 2):
        same_a = labels_a[i] == labels_a[j]
        same_b = labels_b[i] == labels_b[j]
        if same_a and same_b:
            tp += 1
        elif not same_a and not same_b:
            tn += 1
        elif same_a and not same_b:
            fp += 1
        else:
            fn += 1

    # ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
    # Using the formula: ARI = 2(TP*TN - FN*FP) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
    numerator = 2 * (tp * tn - fn * fp)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)

    if denominator == 0:
        return 1.0 if (fp == 0 and fn == 0) else 0.0
    return numerator / denominator


@dataclass
class NetMetrics:
    """Net topology accuracy."""

    adjusted_rand_index: float
    common_pins: int
    predicted_pins: int
    ground_truth_pins: int


def net_metrics(predicted: Netlist, ground_truth: Netlist) -> NetMetrics:
    pred_clusters = _pin_to_cluster(predicted)
    gt_clusters = _pin_to_cluster(ground_truth)
    ari = _adjusted_rand_index(pred_clusters, gt_clusters)
    return NetMetrics(
        adjusted_rand_index=ari,
        common_pins=len(set(pred_clusters) & set(gt_clusters)),
        predicted_pins=len(pred_clusters),
        ground_truth_pins=len(gt_clusters),
    )


# ---------------------------------------------------------------------------
# Combined result
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Full evaluation result for a single image."""

    stem: str
    components: ComponentMetrics
    nets: NetMetrics


def evaluate(predicted: Netlist, ground_truth: Netlist, stem: str = "") -> EvalResult:
    """Run all metrics comparing predicted against ground truth."""
    return EvalResult(
        stem=stem,
        components=component_metrics(predicted, ground_truth),
        nets=net_metrics(predicted, ground_truth),
    )
