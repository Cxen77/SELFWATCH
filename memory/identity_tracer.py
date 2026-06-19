"""
IdentityTracer — Per-frame forensic trace for ONE global identity.

Usage:
    Set TRACE_GID = <gid_to_watch> in config.py, then run normally.
    Every stage of the pipeline will print a compact but complete trace line.

Output format per frame:
    [TRACE F#{frame}] {stage}: local={lid} gid={gid} ...fields...

Stages logged:
    PREDICT     — track's predicted box and velocity before association
    GATE        — WHY a detection-track pair was gated (GATE_VALUE applied)
    MATCH       — the final association decision (matched / unmatched)
    LOST        — track marked as lost this frame
    BIRTH       — new local track spawned (potential fragmentation)
    REBIND      — GlobalIdentityManager changed gid mapping
    THINKING    — gid entered thinking/latent state
    RECOVER     — gid recovered from thinking/memory
    SWAP        — post-freeze separation verifier fired
"""

import numpy as np


class IdentityTracer:
    """
    Attach to SelfWatchPipeline and call the hook methods at each stage.
    All methods are no-ops if the target gid/lid is not relevant.
    """

    def __init__(self, target_gid, global_id_manager):
        self.target_gid = target_gid
        self._gim = global_id_manager
        self._target_lid = None   # resolved each frame

    # ── Internal helpers ──────────────────────────────────────────────

    def _resolve_lid(self, tracks):
        """Find the local_id currently mapped to target_gid."""
        for t in tracks:
            if self._gim.get_global_id(t.local_id) == self.target_gid:
                self._target_lid = t.local_id
                return t.local_id
        return None

    @staticmethod
    def _box_str(b):
        if b is None:
            return "None"
        return f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"

    @staticmethod
    def _vel_str(v):
        if v is None:
            return "(0,0)"
        return f"({v[0]:+.1f},{v[1]:+.1f})"

    def _tag(self, frame, stage):
        return f"[TRACE F#{frame:05d}] {stage:<10}"

    # ── Stage hooks ───────────────────────────────────────────────────

    def log_predict(self, frame, tracks):
        """Call after STrack.predict() — log predicted box and velocity."""
        lid = self._resolve_lid(tracks)
        if lid is None:
            return
        for t in tracks:
            if t.local_id == lid:
                print(
                    f"{self._tag(frame, 'PREDICT')} "
                    f"local={lid} gid={self.target_gid} "
                    f"pred={self._box_str(t.predicted_box.tolist())} "
                    f"smooth={self._box_str(t.smooth_box.tolist())} "
                    f"vel={self._vel_str(t.vel)} "
                    f"tsu={t.time_since_update} age={t.age} hits={t.total_hits}"
                )
                return

    def log_cost_matrix(self, frame, tracks, det_boxes, det_embs, cost, iou, app_dist):
        """
        Call after cost matrix is built in _fused_associate.
        Prints the BEST and GATE status for every cell involving target track.
        """
        lid = self._resolve_lid(tracks)
        if lid is None:
            return

        GATE_VALUE = 1e5
        for j, t in enumerate(tracks):
            if t.local_id != lid:
                continue
            print(
                f"{self._tag(frame, 'COSTMATRIX')} "
                f"local={lid} gid={self.target_gid} "
                f"n_det={len(det_boxes)}"
            )
            for i in range(len(det_boxes)):
                gated = cost[i, j] >= GATE_VALUE
                tag = "GATE" if gated else "open"
                print(
                    f"  det[{i}]: {tag}  iou={iou[i,j]:.3f} "
                    f"app={app_dist[i,j]:.3f} "
                    f"cost={cost[i,j]:.4f}  "
                    f"det_box={self._box_str(det_boxes[i])}"
                )

    def log_match_result(self, frame, tracks, matched_t, matched_d,
                         unmatched_t, unmatched_d, stage_name):
        """Call after each association stage — log whether target was matched."""
        lid = self._resolve_lid(tracks)
        if lid is None:
            return
        for j, t in enumerate(tracks):
            if t.local_id != lid:
                continue
            if j in matched_t:
                di = matched_d[matched_t.index(j)]
                print(
                    f"{self._tag(frame, stage_name)} "
                    f"local={lid} gid={self.target_gid} "
                    f"MATCHED det[{di}]"
                )
            else:
                print(
                    f"{self._tag(frame, stage_name)} "
                    f"local={lid} gid={self.target_gid} "
                    f"UNMATCHED — will go LOST"
                )

    def log_lost(self, frame, track):
        """Call inside mark_lost — log when target track enters LOST."""
        if track.local_id == self._target_lid:
            print(
                f"{self._tag(frame, 'LOST')} "
                f"local={track.local_id} gid={self.target_gid} "
                f"tsu={track.time_since_update} vel={self._vel_str(track.vel)} "
                f"smooth={self._box_str(track.smooth_box.tolist())}"
            )

    def log_new_birth(self, frame, new_track, det_box, det_score, zero_emb):
        """
        Call when a new local track spawns near the target gid's region.
        Helps identify fragmentation births.
        """
        # We can't know which gid the new track will get yet;
        # log it whenever target_lid is None (target is currently lost)
        if self._target_lid is not None:
            return
        print(
            f"{self._tag(frame, 'BIRTH')} "
            f"new_local={new_track.local_id} conf={det_score:.2f} "
            f"zero_emb={zero_emb} "
            f"box={self._box_str(det_box)} "
            f"  [target_gid={self.target_gid} is currently LOST]"
        )

    def log_rebind(self, frame, lid, old_gid, new_gid, source, decision):
        """Call inside GlobalIdentityManager._log_rebind."""
        if lid == self._target_lid or old_gid == self.target_gid or new_gid == self.target_gid:
            print(
                f"{self._tag(frame, 'REBIND')} "
                f"local={lid} {old_gid} -> {new_gid} "
                f"src={source} decision={decision}"
            )

    def log_gid_assigned(self, frame, lid, new_gid, reason):
        """Call when a new GID is freshly allocated."""
        if new_gid == self.target_gid or lid == self._target_lid:
            print(
                f"{self._tag(frame, 'GID_ALLOC')} "
                f"local={lid} -> gid={new_gid} reason={reason}"
            )

    def log_thinking(self, frame, gid):
        """Call when gid enters THINKING/LATENT state."""
        if gid == self.target_gid:
            print(f"{self._tag(frame, 'THINKING')} gid={gid} entered latent state")

    def log_recovery(self, frame, gid, proposed_lid, source, sim):
        """Call when gid is proposed for recovery."""
        if gid == self.target_gid:
            print(
                f"{self._tag(frame, 'RECOVER')} "
                f"gid={gid} proposed_lid={proposed_lid} src={source} sim={sim:.3f}"
            )

    def log_sep_verify(self, frame, gid_a, gid_b, same, swap, decision):
        """Call from post-freeze separation verifier."""
        if self.target_gid in (gid_a, gid_b):
            print(
                f"{self._tag(frame, 'SEPVERIFY')} "
                f"pair=({gid_a},{gid_b}) "
                f"same={same:.3f} swap={swap:.3f} decision={decision}"
            )

    def log_freeze(self, frame, gid, partner_gid):
        """Call when target gid enters frozen state."""
        if gid == self.target_gid:
            print(
                f"{self._tag(frame, 'FREEZE')} "
                f"gid={gid} frozen with partner={partner_gid}"
            )

    def log_unfreeze(self, frame, gid):
        """Call when target gid exits frozen state."""
        if gid == self.target_gid:
            print(f"{self._tag(frame, 'UNFREEZE')} gid={gid} cooldown started")
