"""
Layer 2: Global Identity Manager — Trajectory-First Identity Rebinding

The SINGLE authority for mapping local_id -> global_id.

All recovery paths MUST satisfy trajectory consistency before rebinding.

Direction gating is enforced at EVERY entry point:
  - New track registration with proposals
  - Provisional track promotion
  - Committed track re-assignment
  - Inertia-voted proposals

Forensic logging: every rebinding decision is logged with:
  - direction agreement (dot product)
  - trajectory distance
  - rejection/acceptance reason
  - velocity data from both sides

Philosophy: it is ALWAYS better to create a new ID than to steal
an existing one incorrectly. Wait > wrong.
"""

import numpy as np
import math


class GlobalIdentityManager:
    # Simple, uniform inertia — no tiered momentum stacking
    INERTIA_BASE = 5         # Base frames before any proposal accepted
    COMMIT_DELAY = 5         # Frames before a new ID becomes committed
    FROZEN_INERTIA = 5       # Frames of consistent proposals required post-freeze

    def __init__(self):
        self._local_to_global = {}
        self._next_global_id = 1
        self._proposals = {}        # lid -> { gid, count, source }
        self._provisional = {}      # lid -> { gid, birth_frame, age }
        self._frame_count = 0

        # Gallery: gid -> { velocity, last_box, last_center }
        self._gallery_motion = {}

        # ACT-R: gid -> { total_age, total_hits, last_seen_frame }
        self._activation_data = {}

        # Forensic log of all rebinding decisions
        self._rebind_log = []       # list of dicts, last 100

    # ── ACT-R Activation-Based Persistence ─────────────────────────────

    def _update_activation(self, gid, age=None, hits=None):
        """Update ACT-R activation data for a global identity."""
        if gid not in self._activation_data:
            self._activation_data[gid] = {
                "total_age": 0,
                "total_hits": 0,
                "last_seen_frame": self._frame_count,
            }
        data = self._activation_data[gid]
        if age is not None:
            data["total_age"] = max(data["total_age"], age)
        if hits is not None:
            data["total_hits"] = max(data["total_hits"], hits)
        data["last_seen_frame"] = self._frame_count

    def get_thinking_window(self, gid):
        """
        ACT-R activation-based THINKING window.

        Long-established identities persist longer during occlusion.
        Short-lived tracks decay quickly. This replaces the fixed
        THINKING_WINDOW constant.

        activation = ln(age+1) + recency_bonus + frequency_bonus
          - age=500, hits=400: activation ≈ 6.2 → window=30
          - age=50,  hits=30:  activation ≈ 3.9 → window=19
          - age=5,   hits=3:   activation ≈ 1.8 → window=8

        Returns:
            int: Dynamic thinking window in frames [5, 45]
        """
        import math
        data = self._activation_data.get(gid)
        if data is None:
            return 15  # fallback to default

        age = max(1, data["total_age"])
        hits = data["total_hits"]

        # Base activation: logarithmic scaling of track lifetime
        base = math.log(age + 1)

        # Frequency bonus: ratio of hits to age (well-tracked = higher)
        freq_bonus = min(1.0, hits / max(age, 1)) * 0.5

        # Recency bonus: how recently was this identity active?
        frames_since = self._frame_count - data["last_seen_frame"]
        recency_bonus = max(0.0, 1.0 - frames_since * 0.05)

        activation = base + freq_bonus + recency_bonus

        # Map activation to window: clamp to [5, 45]
        window = int(activation * 4.5)
        return max(5, min(45, window))

    def update(self, tracks, proposals=None, frozen_gids=None, occlusion_manager=None):
        self._frame_count += 1
        proposals = proposals or []
        frozen_gids = frozen_gids or set()
        self._occlusion_manager = occlusion_manager  # Store reference for exit region checks
        active_lids = {t.local_id for t in tracks}



        # Build track lookup for direction gating
        track_by_lid = {}
        for t in tracks:
            track_by_lid[t.local_id] = t

        # Build proposal dict with normalized source
        proposal_dict = {}
        for p in proposals:
            if len(p) >= 4:
                lid, gid, conf, source = p[0], p[1], p[2], p[3]
            else:
                lid, gid, conf = p[0], p[1], p[2]
                source = "warm"
            base_source = source.split("(")[0] if "(" in source else source
            proposal_dict[lid] = (gid, conf, base_source)

        # ═══════════════════════════════════════════════════════════════
        #  1. REGISTER NEW TRACKS
        # ═══════════════════════════════════════════════════════════════
        for t in tracks:
            lid = t.local_id
            if lid not in self._local_to_global:
                entry = proposal_dict.get(lid)

                if entry is not None and entry[2] in ("thinking", "phantom"):
                    proposed_gid = entry[0]
                    ok, reason, dot_val, dist_val = self._full_trajectory_check(
                        proposed_gid, t)

                    if ok:
                        self._local_to_global[lid] = proposed_gid
                        proposal_dict.pop(lid)
                        self._log_rebind(
                            lid, None, proposed_gid, entry[2],
                            "ACCEPT_NEW", dot_val, dist_val)
                    else:
                        new_gid = self._allocate_id()
                        print(f"[INSTRUMENTATION] GLOBAL NEW GID ASSIGNED: local={lid} -> new_gid={new_gid} (REJECT_NEW: {reason})")
                        self._local_to_global[lid] = new_gid
                        self._provisional[lid] = {
                            "gid": new_gid,
                            "birth_frame": self._frame_count,
                            "age": 0,
                        }
                        proposal_dict.pop(lid, None)
                        self._log_rebind(
                            lid, None, proposed_gid, entry[2],
                            f"REJECT_NEW:{reason}", dot_val, dist_val)
                else:
                    new_gid = self._allocate_id()
                    print(f"[INSTRUMENTATION] GLOBAL NEW GID ASSIGNED: local={lid} -> new_gid={new_gid} (NO_PROPOSAL)")
                    self._local_to_global[lid] = new_gid
                    self._provisional[lid] = {
                        "gid": new_gid,
                        "birth_frame": self._frame_count,
                        "age": 0,
                    }

        # ═══════════════════════════════════════════════════════════════
        #  2. PROCESS PROPOSALS FOR EXISTING TRACKS
        # ═══════════════════════════════════════════════════════════════
        for lid in active_lids:
            entry = proposal_dict.get(lid)
            if entry is None:
                self._proposals.pop(lid, None)
                continue

            proposed_gid, conf, source = entry
            current_gid = self._local_to_global.get(lid)

            # Same ID — no change needed
            if current_gid == proposed_gid:
                self._proposals.pop(lid, None)
                continue

            # ── STICKY OWNERSHIP: committed tracks are PROTECTED ────────
            # If this track has passed the provisional commit window,
            # its ownership is STICKY. Reject proposals for committed
            # tracks — ownership changes only through natural lifecycle
            # (track dies, owner archived, impossible trajectory) OR
            # through Flexible Trajectory Reattachment (target GID is inactive).
            if lid not in self._provisional and current_gid is not None:
                # Is proposed_gid currently active elsewhere?
                proposed_active_elsewhere = False
                for other_lid, other_gid in self._local_to_global.items():
                    if other_lid != lid and other_gid == proposed_gid:
                        other_track = track_by_lid.get(other_lid)
                        if other_track is not None and other_track.is_confirmed and other_track.time_since_update == 0:
                            proposed_active_elsewhere = True
                            break

                if proposed_active_elsewhere:
                    self._proposals.pop(lid, None)
                    self._log_rebind(
                        lid, current_gid, proposed_gid, source,
                        "REJECT_COMMITTED_ACTIVE_ELSEWHERE", 0.0, 0.0)
                    continue

                # Otherwise, proposed_gid is inactive (ghosting/lost), so we allow
                # this committed track to rebind and inherit it!


            # FREEZE: reject ALL proposals for frozen identities
            if current_gid in frozen_gids:
                self._proposals.pop(lid, None)
                self._log_rebind(
                    lid, current_gid, proposed_gid, source,
                    "REJECT_FROZEN", 0.0, 0.0)
                continue

            # Also reject if the PROPOSED identity is frozen
            # (someone else owns that ID and is in an occlusion group)
            if proposed_gid in frozen_gids:
                self._proposals.pop(lid, None)
                self._log_rebind(
                    lid, current_gid, proposed_gid, source,
                    "REJECT_TARGET_FROZEN", 0.0, 0.0)
                continue

            # ── SOFT PENALTIES (cooldown/exit-region) ──────────────────
            # Instead of hard-blocking, we increase the inertia requirement.
            # This preserves recovery capability while biasing toward stability.
            soft_penalty = 0  # Extra inertia frames required
            soft_reasons = []

            if self._occlusion_manager is not None:
                if self._occlusion_manager.is_in_cooldown(current_gid):
                    soft_penalty += 2
                    soft_reasons.append("cooldown_src")
                if self._occlusion_manager.is_in_cooldown(proposed_gid):
                    soft_penalty += 2
                    soft_reasons.append("cooldown_tgt")

                # Exit region: if track is outside predicted exit zone,
                # add extra inertia rather than blocking
                exit_traj = self._occlusion_manager.get_exit_trajectory(proposed_gid)
                track = track_by_lid.get(lid)
                if exit_traj is not None and track is not None:
                    track_box = getattr(track, 'smooth_box', None)
                    if track_box is not None:
                        track_cx = float((track_box[0] + track_box[2]) / 2)
                        track_cy = float((track_box[1] + track_box[3]) / 2)
                        is_near, dist = self._occlusion_manager.is_near_exit_region(
                            proposed_gid, track_cx, track_cy)
                        if not is_near:
                            soft_penalty += 2
                            soft_reasons.append(f"exit_dist_{dist:.0f}px")
            else:
                track = track_by_lid.get(lid)

            # Full trajectory check (this stays as a HARD gate — physics-based)
            ok, reason, dot_val, dist_val = self._full_trajectory_check(
                proposed_gid, track)

            if not ok:
                self._proposals.pop(lid, None)
                self._log_rebind(
                    lid, current_gid, proposed_gid, source,
                    f"REJECT_TRAJ:{reason}", dot_val, dist_val)
                continue

            # ── PROVISIONAL: accept proposals (direction already checked) ──
            if lid in self._provisional:
                old_gid = self._local_to_global[lid]
                self._local_to_global[lid] = proposed_gid
                del self._provisional[lid]  # Now committed
                self._proposals.pop(lid, None)
                self._log_rebind(
                    lid, old_gid, proposed_gid, source,
                    "ACCEPT_PROVISIONAL", dot_val, dist_val)
                continue

            # ── Uniform inertia voting for ALL sources ────────────────
            if lid in self._proposals and self._proposals[lid]["gid"] == proposed_gid:
                self._proposals[lid]["count"] += 1
            else:
                self._proposals[lid] = {
                    "gid": proposed_gid, "count": 1, "source": source}


            required = self.INERTIA_BASE + soft_penalty

            # Historical Importance Weighting: Protect historically stable identities
            # from being easily replaced during transient/weak challenge.
            if current_gid in self._activation_data:
                act_data = self._activation_data[current_gid]
                stable_hits = act_data.get("total_hits", 0)
                if stable_hits > 50:
                    # Scale bonus dynamically: up to 15 extra frames of inertia
                    stability_bonus = min(15, int(stable_hits / 20))
                    required += stability_bonus

            if self._proposals[lid]["count"] >= required:
                old_gid = self._local_to_global[lid]
                self._local_to_global[lid] = proposed_gid
                self._proposals.pop(lid)
                self._log_rebind(
                    lid, old_gid, proposed_gid, source,
                    f"ACCEPT_{source}_{required}F", dot_val, dist_val)





        # ═══════════════════════════════════════════════════════════════
        #  3. AGE PROVISIONALS AND COMMIT (STRICT QUALITY GATE)
        # ═══════════════════════════════════════════════════════════════
        committed = []
        for lid, prov in self._provisional.items():
            track = track_by_lid.get(lid)
            if track is not None:
                # Require stronger evidence (sustained consecutive hits & conf)
                is_stably_tracked = track.consecutive_hits >= self.COMMIT_DELAY
                has_decent_score = track.score >= 0.55
                has_plausible_vel = np.linalg.norm(track.vel) < 20.0 if hasattr(track, 'vel') else True
                is_currently_active = track.time_since_update == 0

                if is_stably_tracked and has_decent_score and has_plausible_vel and is_currently_active:
                    committed.append(lid)
                else:
                    # Increment provisional age but don't commit yet; keep waiting for more evidence
                    prov["age"] += 1
            else:
                # Track is lost or dead before commitment, don't age it
                prov["age"] += 1

        for lid in committed:
            del self._provisional[lid]


        # ═══════════════════════════════════════════════════════════════
        #  4. UPDATE GALLERY MOTION DATA
        # ═══════════════════════════════════════════════════════════════
        for t in tracks:
            lid = t.local_id
            gid = self._local_to_global.get(lid)
            if gid is None:
                continue
            if not (t.is_confirmed and t.time_since_update == 0):
                continue

            # ACT-R: update activation data for this identity
            self._update_activation(
                gid,
                age=getattr(t, 'age', None),
                hits=getattr(t, 'total_hits', None),
            )

            vel = getattr(t, 'vel', None)
            box = getattr(t, 'smooth_box', None)
            if vel is not None and box is not None:
                speed = float(np.linalg.norm(vel))
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                if speed > 0.2:
                    self._gallery_motion[gid] = {
                        "velocity": vel.copy(),
                        "last_box": box.copy(),
                        "last_center": np.array([cx, cy], dtype=np.float32),
                        "frame": self._frame_count,
                    }

        # ═══════════════════════════════════════════════════════════════
        #  5. CLEANUP
        # ═══════════════════════════════════════════════════════════════
        dead = [lid for lid in self._proposals if lid not in active_lids]
        for lid in dead:
            del self._proposals[lid]
        dead_prov = [lid for lid in self._provisional if lid not in active_lids]
        for lid in dead_prov:
            del self._provisional[lid]



    # ── TRAJECTORY VALIDATION ──────────────────────────────────────────

    def _full_trajectory_check(self, proposed_gid, track):
        """
        Full trajectory consistency check for identity rebinding.

        Checks:
          1. Direction agreement:  dot(stored_vel, track_vel)
          2. Spatial continuity:   distance from predicted position
          3. Speed consistency:    speed ratio check

        Returns:
            (ok: bool, reason: str, dot_val: float, dist_val: float)
        """
        if track is None:
            return True, "no_track", 0.0, 0.0

        motion_data = self._gallery_motion.get(proposed_gid)
        if motion_data is None:
            # No stored motion data — allow with warning
            return True, "no_gallery_motion", 0.0, 0.0

        stored_vel = motion_data.get("velocity")
        stored_center = motion_data.get("last_center")
        stored_frame = motion_data.get("frame", 0)

        track_vel = getattr(track, 'vel', None)
        track_box = getattr(track, 'smooth_box', None)

        # Check if in collision cooldown to enable flexible reattachment
        is_flexible = False
        if self._occlusion_manager is not None and self._occlusion_manager.is_in_cooldown(proposed_gid):
            is_flexible = True

        # ── Check 1: Direction agreement ──────────────────────────────
        dot_val = 0.0
        if stored_vel is not None and track_vel is not None:
            sv = np.array(stored_vel, dtype=np.float32)
            tv = np.array(track_vel, dtype=np.float32)
            sv_speed = float(np.linalg.norm(sv))
            tv_speed = float(np.linalg.norm(tv))

            if sv_speed > 0.5 and tv_speed > 0.5:
                dot_val = float(np.dot(sv / sv_speed, tv / tv_speed))

                if not is_flexible:
                    # HARD GATE: opposite direction
                    if dot_val < 0:
                        return False, "opposite_direction", dot_val, 0.0

                    # HARD GATE: weak agreement for significant motion
                    if sv_speed > 2.0 and tv_speed > 2.0 and dot_val < 0.2:
                        return False, "weak_direction", dot_val, 0.0

                    # HARD GATE: speed mismatch (>4x different)
                    speed_ratio = max(sv_speed, tv_speed) / max(min(sv_speed, tv_speed), 0.1)
                    if speed_ratio > 4.0:
                        return False, f"speed_mismatch_{speed_ratio:.1f}x", dot_val, 0.0
                else:
                    # In flexible recovery, opposite direction or speed shift is allowed,
                    # but reject completely implausible trajectory angles (e.g. dot_val < -0.8)
                    if dot_val < -0.8:
                        return False, "implausible_opposite_turn", dot_val, 0.0

        # ── Check 2: Spatial continuity ───────────────────────────────
        dist_val = 0.0
        if stored_center is not None and track_box is not None:
            track_cx = float((track_box[0] + track_box[2]) / 2)
            track_cy = float((track_box[1] + track_box[3]) / 2)

            # Predict where the stored identity should be now
            frames_elapsed = max(1, self._frame_count - stored_frame)
            if stored_vel is not None:
                pred_cx = float(stored_center[0]) + float(stored_vel[0]) * frames_elapsed
                pred_cy = float(stored_center[1]) + float(stored_vel[1]) * frames_elapsed
            else:
                pred_cx = float(stored_center[0])
                pred_cy = float(stored_center[1])

            dist_val = math.hypot(track_cx - pred_cx, track_cy - pred_cy)

            # Allow larger radius for longer elapsed time, but cap it
            if is_flexible:
                # In flexible unfreeze, allow up to 250px for recovery
                max_dist = min(300.0, 150.0 + frames_elapsed * 20.0)
            else:
                max_dist = min(400.0, 80.0 + frames_elapsed * 15.0)

            if dist_val > max_dist:
                return False, f"too_far_{dist_val:.0f}px", dot_val, dist_val

        return True, "ok", dot_val, dist_val

    # ── FORENSIC LOGGING ───────────────────────────────────────────────

    def _log_rebind(self, lid, old_gid, proposed_gid, source,
                    decision, dot_val, dist_val):
        """Log every rebinding decision for forensic analysis."""
        entry = {
            "frame": self._frame_count,
            "local_id": lid,
            "old_gid": old_gid,
            "proposed_gid": proposed_gid,
            "source": source,
            "decision": decision,
            "direction_dot": round(dot_val, 3),
            "trajectory_dist": round(dist_val, 1),
        }
        self._rebind_log.append(entry)
        if len(self._rebind_log) > 100:
            self._rebind_log.pop(0)

        # Print rejections for immediate debugging
        if "REJECT" in decision:
            print(f"[REBIND] {decision}: local={lid} "
                  f"old_gid={old_gid} -> proposed_gid={proposed_gid} "
                  f"src={source} dot={dot_val:.3f} dist={dist_val:.0f}")
        elif "ACCEPT" in decision:
            print(f"[REBIND] {decision}: local={lid} "
                  f"gid={proposed_gid} src={source} "
                  f"dot={dot_val:.3f} dist={dist_val:.0f}")

    def get_rebind_log(self):
        """Return recent rebinding decisions for forensic overlay."""
        return list(self._rebind_log)

    # ── PUBLIC API ─────────────────────────────────────────────────────

    def get_global_id(self, local_id):
        return self._local_to_global.get(local_id, local_id)

    def is_provisional(self, local_id):
        return local_id in self._provisional

    def get_provisional_age(self, local_id):
        prov = self._provisional.get(local_id)
        return prov["age"] if prov else -1

    def _allocate_id(self):
        gid = self._next_global_id
        self._next_global_id += 1
        return gid
