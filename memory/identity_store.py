"""
IdentityStore — Stage 1 of the Strangler-Fig identity-centric migration.

PURPOSE (this stage only):
    A PURE READ-ONLY SHADOW LAYER. It observes the live, track-centric
    pipeline and mirrors the durable per-identity state that today lives
    on STrack objects (and therefore DIES when a track dies) into
    gid-keyed Identity objects that OUTLIVE any individual track.

    It influences ZERO live decisions. Nothing in the pipeline reads from
    this module to make an assignment. Its only output is measurement:
    the shadow-divergence metric — how often the live system MINTED A NEW
    gid where this store, by durable appearance memory, would have
    RE-OWNED an existing identity.

    That divergence number is the evidence that justifies Stage 3+
    (flipping assignment authority). Until we have it, an authority flip
    would be a guess.

WHY THIS IS SAFE:
    - No live code path consumes Identity state for assignment.
    - observe() only READS tracker + GlobalIdentityManager fields.
    - Removing the 3 pipeline hooks + this file fully restores stable-id-v1.

WHAT IT MIRRORS (the gap that causes ownership churn):
    - appearance:  anchor embedding + EMA + small gallery  (today: on STrack)
    - kinematics:  center position + velocity                (today: on STrack)
    - trajectory:  rolling center history                    (today: nowhere durable)
    - activation:  total_hits / last_seen                    (today: partial in GIM)
    - ownership:   gid <-> local_id binding history          (today: implicit)
"""

import numpy as np
from collections import deque


# ── Shadow matching thresholds (MEASUREMENT ONLY — not live gates) ───────
# These decide what the store WOULD have done, purely for the divergence
# metric. They never gate a live assignment.
SHADOW_ANCHOR_SIM = 0.60      # cosine sim to anchor to claim "same identity"
SHADOW_EMA_SIM = 0.62         # cosine sim to EMA as fallback evidence
SHADOW_GALLERY_SIM = 0.58     # best-of-gallery cosine sim
SHADOW_MAX_COAST = 300        # frames an unobserved identity stays matchable

GALLERY_SIZE = 8
TRAJECTORY_SIZE = 64
ANCHOR_HITS = 100             # mirror STrack's anchor-snapshot maturity gate


def _normalize(vec):
    """Return an L2-normalized copy, or None if input is None/degenerate."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float32).copy()
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


class Identity:
    """
    Durable, gid-keyed identity record. OUTLIVES the tracks that feed it.

    In Stage 1 this is a passive mirror. In later stages it becomes the
    authority that owns trajectory/velocity/appearance. Keeping the field
    set stable now means later stages only flip WHO READS it, not its shape.
    """

    __slots__ = (
        "gid", "anchor_emb", "ema_emb", "gallery",
        "kin_center", "kin_vel", "trajectory",
        "total_hits", "first_seen_frame", "last_seen_frame", "coast_frames",
        "owning_lid", "ownership_history",
    )

    def __init__(self, gid, frame_count):
        self.gid = gid

        # ── Appearance (the durable memory that today dies with the track) ──
        self.anchor_emb = None              # frozen snapshot at maturity
        self.ema_emb = None                 # rolling EWMA, mirrors STrack.embedding
        self.gallery = deque(maxlen=GALLERY_SIZE)

        # ── Kinematics ──────────────────────────────────────────────────
        self.kin_center = None              # np.array([cx, cy])
        self.kin_vel = np.zeros(2, dtype=np.float32)
        self.trajectory = deque(maxlen=TRAJECTORY_SIZE)

        # ── Activation / lifecycle ──────────────────────────────────────
        self.total_hits = 0
        self.first_seen_frame = frame_count
        self.last_seen_frame = frame_count
        self.coast_frames = 0               # consecutive frames unobserved

        # ── Ownership binding ───────────────────────────────────────────
        self.owning_lid = None              # local_id currently carrying this gid
        self.ownership_history = deque(maxlen=32)  # (frame, local_id) transitions

    # ── Observation ingest (read-only mirror of live state) ─────────────

    def observe(self, local_id, center, vel, embedding, total_hits, frame_count):
        self.last_seen_frame = frame_count
        self.coast_frames = 0
        self.total_hits = max(self.total_hits, int(total_hits))

        if center is not None:
            self.kin_center = np.asarray(center, dtype=np.float32)
            self.trajectory.append((frame_count, float(center[0]), float(center[1])))
        if vel is not None:
            self.kin_vel = np.asarray(vel, dtype=np.float32)

        emb = _normalize(embedding)
        if emb is not None:
            if self.ema_emb is None:
                self.ema_emb = emb
            else:
                self.ema_emb = _normalize(0.8 * self.ema_emb + 0.2 * emb)
            self.gallery.append(emb)

            # Mirror STrack's anchor rule: snapshot once at maturity, never overwrite.
            if self.anchor_emb is None and self.total_hits >= ANCHOR_HITS:
                self.anchor_emb = self.ema_emb.copy()

        # Record ownership transitions (who carries this gid over time).
        if local_id != self.owning_lid:
            self.ownership_history.append((frame_count, local_id))
            self.owning_lid = local_id

    def coast(self):
        """Advance one frame while unobserved (dead-reckon center, age out)."""
        self.coast_frames += 1
        if self.kin_center is not None:
            self.kin_center = self.kin_center + self.kin_vel

    # ── Shadow appearance match (MEASUREMENT ONLY) ──────────────────────

    def match_score(self, embedding):
        """
        How strongly does `embedding` look like THIS identity, by durable
        appearance memory? Returns (best_sim, basis_str). Used only to decide
        what the store WOULD have done — never gates a live assignment.
        """
        emb = _normalize(embedding)
        if emb is None:
            return 0.0, "no_emb"

        best_sim = 0.0
        basis = "none"

        if self.anchor_emb is not None:
            s = float(np.dot(self.anchor_emb, emb))
            if s > best_sim:
                best_sim, basis = s, "anchor"
        if self.ema_emb is not None:
            s = float(np.dot(self.ema_emb, emb))
            if s > best_sim:
                best_sim, basis = s, "ema"
        if self.gallery:
            s = max(float(np.dot(g, emb)) for g in self.gallery)
            if s > best_sim:
                best_sim, basis = s, "gallery"

        return best_sim, basis

    def would_claim(self, embedding):
        """True if durable appearance memory is strong enough to RE-OWN."""
        sim, basis = self.match_score(embedding)
        if basis == "anchor":
            return sim >= SHADOW_ANCHOR_SIM, sim
        if basis == "ema":
            return sim >= SHADOW_EMA_SIM, sim
        if basis == "gallery":
            return sim >= SHADOW_GALLERY_SIM, sim
        return False, sim


class IdentityStore:
    """
    Shadow registry of durable Identity objects, keyed by gid.

    Stage 1 contract: observe the live pipeline, mirror durable state, and
    measure shadow-divergence. It NEVER feeds an assignment back into the
    live system. Read-only by construction.
    """

    def __init__(self):
        self._identities = {}        # gid -> Identity
        self._frame_count = 0

        # ── Shadow-divergence counters ──────────────────────────────────
        # The headline metric: live minted a NEW gid, but a dormant Identity
        # in this store would have RE-OWNED it by appearance.
        self._divergence_events = 0
        self._observations = 0
        self._new_gid_events = 0
        self._reclaim_events = 0     # store-confirmed re-own of an existing gid

    # ── Main ingest ─────────────────────────────────────────────────────

    def observe(self, tracks, global_id_manager, frame_count,
                new_gids_this_frame=None):
        """
        Read-only mirror of the settled per-frame state.

        Args:
            tracks: tracker.tracks (live STrack list)
            global_id_manager: live GlobalIdentityManager (for lid->gid)
            frame_count: current frame index
            new_gids_this_frame: optional set of gids the live system minted
                this frame (used for divergence detection). If None, inferred.
        """
        self._frame_count = frame_count
        seen_gids = set()

        # Infer freshly-minted gids if caller didn't supply them.
        inferred_new = set()

        for track in tracks:
            if track.time_since_update != 0:
                continue  # only ingest tracks actually observed this frame
            lid = track.local_id
            gid = global_id_manager.get_global_id(lid)
            if gid is None:
                continue

            box = getattr(track, "box", None)
            center = None
            if box is not None:
                center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            vel = getattr(track, "vel", None)
            emb = track.get_averaged_embedding()
            total_hits = getattr(track, "total_hits", 0)

            is_new = gid not in self._identities
            if is_new:
                self._identities[gid] = Identity(gid, frame_count)
                self._new_gid_events += 1
                inferred_new.add(gid)

                # ── Shadow-divergence test ──────────────────────────────
                # The live system just minted gid as NEW. Would any DORMANT
                # identity in the store have re-owned this observation by
                # appearance? If yes, that is exactly the churn we want to fix.
                claimant, claim_sim = self._best_dormant_claim(emb, gid)
                if claimant is not None:
                    self._divergence_events += 1
                    print(f"[SHADOW DIVERGENCE] live minted NEW gid={gid} "
                          f"but store would re-own gid={claimant.gid} "
                          f"(sim={claim_sim:.3f}, dormant={claimant.coast_frames}f)")
            else:
                # Store agrees this gid already exists — confirm appearance.
                ident = self._identities[gid]
                if ident.coast_frames > 0 and emb is not None:
                    ok, sim = ident.would_claim(emb)
                    if ok:
                        self._reclaim_events += 1

            self._identities[gid].observe(
                lid, center, vel, emb, total_hits, frame_count)
            seen_gids.add(gid)
            self._observations += 1

        # Coast every identity not observed this frame; prune the truly stale.
        stale = []
        for gid, ident in self._identities.items():
            if gid in seen_gids:
                continue
            ident.coast()
            if ident.coast_frames > SHADOW_MAX_COAST:
                stale.append(gid)
        for gid in stale:
            del self._identities[gid]

    def _best_dormant_claim(self, embedding, exclude_gid):
        """
        Among DORMANT (currently-unobserved) identities, find the strongest
        appearance claimant for `embedding`. Returns (Identity|None, sim).
        Measurement only.
        """
        if embedding is None:
            return None, 0.0
        best_ident = None
        best_sim = 0.0
        for gid, ident in self._identities.items():
            if gid == exclude_gid:
                continue
            if ident.coast_frames == 0:
                continue  # only dormant identities can be "re-owned"
            ok, sim = ident.would_claim(embedding)
            if ok and sim > best_sim:
                best_sim = sim
                best_ident = ident
        return best_ident, best_sim

    # ── Reporting (the whole point of Stage 1) ──────────────────────────

    def shadow_report(self):
        """One-line periodic divergence summary. Pure observability."""
        live_count = sum(1 for i in self._identities.values()
                         if i.coast_frames == 0)
        dormant = len(self._identities) - live_count
        print(f"[IDENTITY STORE] tracked_gids={len(self._identities)} "
              f"live={live_count} dormant={dormant} "
              f"obs={self._observations} new_gids={self._new_gid_events} "
              f"reclaims={self._reclaim_events} "
              f"DIVERGENCE={self._divergence_events}")

    # ── Read-only accessors (for later stages / debug; unused by live) ──

    def get(self, gid):
        return self._identities.get(gid)

    @property
    def count(self):
        return len(self._identities)

    @property
    def divergence_count(self):
        return self._divergence_events
