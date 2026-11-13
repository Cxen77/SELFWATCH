"""Causal, reversible associations across calibrated overlapping cameras."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import math

import numpy as np


@dataclass(frozen=True)
class LinkPolicy:
    radius_m: float = 1.10
    margin_m: float = .08
    min_cosine: float = .55
    pair_tolerance_s: float = .15
    confirmations: int = 2
    contradictions: int = 2
    max_gap_s: float = .75
    memory_s: float = 4.
    appearance_weight_m: float = 1.25

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in (
                self.radius_m, self.margin_m, self.pair_tolerance_s, self.max_gap_s,
                self.memory_s, self.appearance_weight_m))
                or not -1 <= self.min_cosine <= 1 or self.confirmations < 1
                or self.contradictions < 1 or self.memory_s < self.max_gap_s):
            raise ValueError("Invalid cross-camera link policy")


@dataclass(frozen=True)
class Sighting:
    # The caller provides a camera/epoch/local-ID tuple and a clean unit vector.
    key: tuple
    floor: object | None = None
    feature: np.ndarray | None = None


@dataclass
class Link:
    last_supported: float
    conflicts: int = 0


class OverlapMemory:
    """Keep independent local owners and revise their current aliases prospectively.

    Simultaneous sightings require mutually separated candidates. Accepted
    edges have a bounded lifetime and conflicting observations can revoke
    them. A split restores the previous camera-local ID without rewriting
    past outputs. Identity scores are evidence, not probabilities.
    """
    def __init__(self, rig, policy=LinkPolicy(), *, use_uncertainty=True, revise=True):
        self.rig, self.policy = rig, policy
        self.use_uncertainty, self.revise = use_uncertainty, revise
        self.last = -math.inf
        self.seen, self.ordinal, self.proposals, self.links = {}, {}, {}, {}
        self.next_id = 0

    def _distance(self, left, right):
        if not self.rig.shares_coverage(left.floor, right.floor, self.policy.pair_tolerance_s):
            return None
        a, b = left.floor.contact, right.floor.contact
        distance = math.dist(a.xy_m, b.xy_m)
        uncertainty = a.radius_m + b.radius_m if self.use_uncertainty else 0.
        return max(0., distance - uncertainty), distance + uncertainty

    def _matches(self, by_view):
        candidates, confirmed = [], {}
        for left, right in combinations(by_view.values(), 2):
            options = defaultdict(list)
            rows = []
            for a, b in product(left, right):
                bounds = self._distance(a, b)
                if bounds is None or bounds[0] > self.policy.radius_m:
                    continue
                # A geometry competitor stays in the comparison even when its
                # embedding is missing. A missing crop cannot rule someone out.
                lower, upper = bounds
                pair = tuple(sorted((a.key, b.key)))
                cosine = None if a.feature is None or b.feature is None else float(a.feature @ b.feature)
                # A crop with a low measured cosine is a weaker rival.
                # A missing cosine keeps its best-case cost.
                penalty = self.policy.appearance_weight_m * (1. - cosine) if cosine is not None else 0.
                row = dict(pair=pair, distance_lower_m=lower, distance_upper_m=upper, cosine=cosine,
                           cost_lower=lower + penalty, cost_upper=upper + penalty)
                options[a.key].append(row)
                options[b.key].append(row)
                rows.append(row)
            candidates.extend(rows)
            for row in rows:
                pair = row["pair"]
                upper = row["distance_upper_m"]
                alternative = min((other["cost_lower"] for key in pair
                                   for other in options[key] if other is not row), default=math.inf)
                row["separated"] = row["cost_upper"] + self.policy.margin_m <= alternative
                if (row["separated"] and upper <= self.policy.radius_m
                        and row["cosine"] is not None and row["cosine"] >= self.policy.min_cosine):
                    confirmed[pair] = row
        return candidates, confirmed

    def _components(self):
        groups = {key: {key} for key in self.seen}
        for a, b in self.links:
            if groups[a] is groups[b]:
                continue
            merged = groups[a] | groups[b]
            for key in merged:
                groups[key] = merged
        return groups

    def update(self, timestamp, sightings):
        """Process one time-ordered bundle and return IDs plus current evidence."""
        if not math.isfinite(timestamp) or timestamp <= self.last:
            raise ValueError("Expected strictly increasing capture bundles")
        current = {obs.key: obs for obs in sightings}
        if len(current) != len(sightings):
            raise ValueError("Duplicate local owner in bundle")
        by_view = defaultdict(list)
        for obs in sightings:
            camera, epoch, _ = obs.key
            if obs.floor is not None and (obs.floor.camera_id, obs.floor.stream_epoch) != (camera, epoch):
                raise ValueError("Foot contact does not belong to the local owner")
            by_view[camera].append(obs)
        if any(len({obs.key[1] for obs in view}) != 1 for view in by_view.values()):
            raise ValueError("Multiple stream epochs in one camera bundle")
        self.last = timestamp
        active_epochs = {camera: view[0].key[1] for camera, view in by_view.items()}
        expired = set()
        for key, last_seen in list(self.seen.items()):
            if (timestamp - last_seen > self.policy.memory_s
                    or active_epochs.get(key[0], key[1]) != key[1]):
                expired.add(key)
                del self.seen[key]
                del self.ordinal[key]
        for key in current:
            if key not in self.ordinal:
                self.next_id += 1
                self.ordinal[key] = self.next_id
            self.seen[key] = timestamp

        alternatives, supported = self._matches(by_view)
        events = []
        for pair, link in list(self.links.items()):
            a, b = pair
            if a in expired or b in expired:
                reason = "owner_expired"
            elif not self.revise:
                continue
            elif pair in supported:
                link.last_supported = timestamp
                link.conflicts = 0
                continue
            else:
                bounds = self._distance(current[a], current[b]) if a in current and b in current else None
                if bounds is not None:
                    link.conflicts = link.conflicts + 1 if bounds[0] > self.policy.radius_m else 0
                if link.conflicts >= self.policy.contradictions:
                    reason = "contradictory_geometry"
                elif timestamp - link.last_supported > self.policy.memory_s:
                    reason = "support_expired"
                else:
                    continue
            del self.links[pair]
            events.append(dict(event="revoke", pair=pair, reason=reason, time_s=timestamp))

        previous = self.proposals
        self.proposals = {}
        groups = self._components()
        for pair, evidence in sorted(supported.items(), key=lambda item: (
                item[1]["cost_upper"], -item[1]["cosine"], item[0])):
            a, b = pair
            if pair in self.links:
                continue
            count, seen_at = previous.get(pair, (0, -math.inf))
            count = count + 1 if timestamp - seen_at <= self.policy.max_gap_s else 1
            self.proposals[pair] = count, timestamp
            if count < self.policy.confirmations or groups[a] is groups[b]:
                continue
            left, right = groups[a], groups[b]
            if {key[0] for key in left} & {key[0] for key in right}:
                continue
            # Prevent a transitive link from hiding a measured incompatibility.
            if any(bounds is not None and bounds[0] > self.policy.radius_m
                   for p, q in product(left & current.keys(), right & current.keys())
                   if p[0] != q[0]
                   for bounds in (self._distance(current[p], current[q]),)):
                continue
            merged = left | right
            for key in merged:
                groups[key] = merged
            self.links[pair] = Link(timestamp)
            events.append(dict(evidence, event="link", time_s=timestamp, confirmations=count))
        ids = {key: min(self.ordinal[member] for member in groups[key]) for key in current}
        return ids, events, alternatives
