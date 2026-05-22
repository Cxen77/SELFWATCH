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
    INERTIA_FULL = 5
    INERTIA_FAST = 3       # was 2 — slightly more cautious for phantom
    COMMIT_DELAY = 5       # Frames before a new ID becomes committed
    FROZEN_INERTIA = 5     # Frames of consistent proposals required post-freeze

    def __init__(self):
        self._local_to_global = {}
        self._next_global_id = 1
        self._proposals = {}        # lid -> { gid, count, source }
        self._provisional = {}      # lid -> { gid, birth_frame, age }
        self._frame_count = 0

        # Gallery: gid -> { velocity, last_box, last_center }
        self._gallery_motion = {}

        # ACT-R: gid -> { total_age, total_hits, last_seen_frame }
        # Used for activation-based persistence scoring
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

            # COOLDOWN: reject proposals for recently-unfrozen identities
            # This prevents rapid oscillation immediately after crowd separation
            if self._occlusion_manager is not None:
                if self._occlusion_manager.is_in_cooldown(current_gid):
                    self._proposals.pop(lid, None)
                    self._log_rebind(
                        lid, current_gid, proposed_gid, source,
                        "REJECT_COOLDOWN", 0.0, 0.0)
                    continue
                if self._occlusion_manager.is_in_cooldown(proposed_gid):
                    self._proposals.pop(lid, None)
                    self._log_rebind(
                        lid, current_gid, proposed_gid, source,
                        "REJECT_TARGET_COOLDOWN", 0.0, 0.0)
                    continue

                # EXIT REGION CHECK: if proposed_gid was recently frozen,
                # only allow rebinding if the track is near the predicted exit region
                exit_traj = self._occlusion_manager.get_exit_trajectory(proposed_gid)
                if exit_traj is not None and track is not None:
                    track_box = getattr(track, 'smooth_box', None)
                    if track_box is not None:
                        track_cx = float((track_box[0] + track_box[2]) / 2)
                        track_cy = float((track_box[1] + track_box[3]) / 2)
                        is_near, dist = self._occlusion_manager.is_near_exit_region(
                            proposed_gid, track_cx, track_cy)
                        if not is_near:
                            self._proposals.pop(lid, None)
                            self._log_rebind(
                                lid, current_gid, proposed_gid, source,
                                f"REJECT_EXIT_REGION_{dist:.0f}px", 0.0, dist)
                            continue

            # Full trajectory check
            track = track_by_lid.get(lid)
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

            # ── THINKING source: require more frames during crowd ambiguity ──
            if source == "thinking":
                if lid in self._proposals and self._proposals[lid]["gid"] == proposed_gid:
                    self._proposals[lid]["count"] += 1
                else:
                    self._proposals[lid] = {
                        "gid": proposed_gid, "count": 1, "source": source}

                # Use stronger inertia if the proposed identity was recently frozen
                required_frames = 2  # Default for thinking
                if (self._occlusion_manager is not None and
                        self._occlusion_manager.get_exit_trajectory(proposed_gid) is not None):
                    required_frames = self.FROZEN_INERTIA  # 5 frames for post-frozen

                if self._proposals[lid]["count"] >= required_frames:
                    old_gid = self._local_to_global[lid]
                    self._local_to_global[lid] = proposed_gid
                    self._proposals.pop(lid)
                    self._log_rebind(
                        lid, old_gid, proposed_gid, source,
                        f"ACCEPT_THINKING_{required_frames}F", dot_val, dist_val)
                continue

            # ── Inertia voting for phantom/warm proposals ──
            required = self.INERTIA_FAST if source == "phantom" else self.INERTIA_FULL

            if lid in self._proposals and self._proposals[lid]["gid"] == proposed_gid:
                self._proposals[lid]["count"] += 1
            else:
                self._proposals[lid] = {
                    "gid": proposed_gid, "count": 1, "source": source}

            if self._proposals[lid]["count"] >= required:
                old_gid = self._local_to_global[lid]
                self._local_to_global[lid] = proposed_gid
                self._proposals.pop(lid)
                self._log_rebind(
                    lid, old_gid, proposed_gid, source,
                    f"ACCEPT_INERTIA_{required}F", dot_val, dist_val)

        # ═══════════════════════════════════════════════════════════════
        #  3. AGE PROVISIONALS AND COMMIT
        # ═══════════════════════════════════════════════════════════════
        committed = []
        for lid, prov in self._provisional.items():
            prov["age"] += 1
            if prov["age"] >= self.COMMIT_DELAY:
                committed.append(lid)
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

        # ── Check 1: Direction agreement ──────────────────────────────
        dot_val = 0.0
        if stored_vel is not None and track_vel is not None:
            sv = np.array(stored_vel, dtype=np.float32)
            tv = np.array(track_vel, dtype=np.float32)
            sv_speed = float(np.linalg.norm(sv))
            tv_speed = float(np.linalg.norm(tv))

            if sv_speed > 0.5 and tv_speed > 0.5:
                dot_val = float(np.dot(sv / sv_speed, tv / tv_speed))

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
