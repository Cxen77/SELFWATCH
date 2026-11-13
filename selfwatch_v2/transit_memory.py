"""Causal dormant-owner handoffs over configured camera transitions."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TransitPolicy:
    cosine: float = .74
    margin: float = .08
    confirmations: int = 2
    quiet_s: float = .45
    max_transit_s: float = 2.


class TransitMemory:
    """Associate new local owners with recently departed owners on allowed routes."""
    def __init__(self, routes, policy):
        self.routes, self.policy = frozenset(routes), policy
        self.last = -math.inf
        self.records, self.pending, self.unresolved, self.local_ids = {}, {}, {}, {}
        self.next_id = 0

    def update(self, timestamp, sightings):
        if timestamp <= self.last or not math.isfinite(timestamp):
            raise ValueError("Expected increasing capture bundles")
        self.last = timestamp
        active = {obs.key: obs.feature for obs in sightings}
        if len(active) != len(sightings):
            raise ValueError("Duplicate local owner")
        self.records = {key: record for key, record in self.records.items()
                        if timestamp - record[1] <= self.policy.max_transit_s}
        self.local_ids = {key: value for key, value in self.local_ids.items() if key in self.records}
        # If a supposedly departed source reappears with its destination,
        # revoke the conflicting alias. Prefer the original owner if present.
        claims = {}
        for key in sorted(active.keys() & self.records.keys(),
                          key=lambda key: (self.records[key][0] != self.local_ids[key], self.local_ids[key])):
            gid, seen, feature = self.records[key]
            if gid in claims:
                self.records[key] = self.local_ids[key], seen, feature
            else:
                claims[gid] = key
        self.unresolved = {key: born for key, born in self.unresolved.items()
                           if key in self.records and timestamp - born <= self.policy.max_transit_s}
        occupied = {self.records[key][0] for key in active.keys() & self.records.keys()
                    if key not in self.pending}
        proposals = {}
        for key, feature in active.items():
            if feature is None or (key in self.records and key not in self.unresolved):
                continue
            scores = {}
            for source, (gid, seen, prototype) in self.records.items():
                if (source in active or gid in occupied or prototype is None
                        or (source[0], key[0]) not in self.routes
                        or timestamp - seen < self.policy.quiet_s):
                    continue
                similarity = float(prototype @ feature)
                scores[gid] = max(similarity, scores.get(gid, -1.))
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            if ranked and ranked[0][1] >= self.policy.cosine:
                gid, similarity = ranked[0]
                rival = ranked[1][1] if len(ranked) > 1 else -1.
                if similarity - rival >= self.policy.margin:
                    proposals[key] = gid, similarity

        output, next_pending = {}, {}
        ranked = sorted(active, key=lambda key: (-proposals.get(key, (0, -2.))[1], key))
        # Reserve every current local owner first. A new handoff cannot steal
        # an ID from a different current owner while evidence is ambiguous.
        for key in ranked:
            if key not in self.records:
                self.next_id += 1
                self.records[key] = self.next_id, timestamp, active[key]
                self.local_ids[key] = self.next_id
                self.unresolved[key] = timestamp
        current_ids = {self.records[key][0] for key in active}
        for key in ranked:
            gid, _, previous_feature = self.records[key]
            if key in proposals:
                target, cosine = proposals[key]
                # Competing destinations with comparable evidence both wait.
                rival = max((value for other, (candidate, value) in proposals.items()
                             if other != key and candidate == target), default=-1.)
                if target not in current_ids and cosine - rival >= self.policy.margin:
                    earlier = self.pending.get(key)
                    count = earlier[1] + 1 if earlier and earlier[0] == target else 1
                    if count >= self.policy.confirmations:
                        current_ids.add(target)
                        gid = target
                        self.unresolved.pop(key, None)
                    else:
                        next_pending[key] = target, count
            self.records[key] = gid, timestamp, active[key] if active[key] is not None else previous_feature
            output[key] = gid
        self.pending = next_pending
        return output
