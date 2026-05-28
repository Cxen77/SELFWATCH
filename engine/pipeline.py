"""
SELFWATCH Pipeline — 5-Layer Cognitive Architecture with Identity State Machine

Identity States:
  ACTIVE    — Currently detected, rendering normally
  THINKING  — Recently lost (< THINKING_WINDOW frames), rendering predicted box,
              holding identity, suppressing new ID assignment
  PHANTOM   — Lost beyond THINKING window, invisible trajectory prediction
  DEAD      — Expired, saved to warm memory

Key rules:
  - During THINKING, the identity is HELD — no new global_id is assigned
  - Phantoms are NEVER rendered (invisible predictors only)
  - One authoritative box per global_id
  - New tracks check THINKING/PHANTOM identities before getting new IDs
"""

import time
import cv2
import numpy as np

import config
from memory.global_identity import GlobalIdentityManager
from memory.active_memory import ActiveMemory
from memory.phantom import PhantomTracker
from memory.warm_memory import WarmMemory
from memory.reasoning import CognitiveReasoning
from memory.occlusion_groups import OcclusionGroupManager
from memory.ownership_arbitration import OwnershipArbitrationLayer
from memory.ownership_state import (
    OwnershipStateMachine, VISIBLE_ACTIVE as OSM_ACTIVE,
    VISIBLE_FROZEN as OSM_FROZEN, LATENT_CANDIDATE as OSM_LATENT,
    RECOVERING as OSM_RECOVERING, ARCHIVED as OSM_ARCHIVED,
)
from memory.event_log import CognitiveEventLogger
from memory.metrics import TrackingMetrics
from memory.debug_overlay import DebugOverlay, DebugLayerManager

PERSON_CLASS = 0
REID_INTERVAL = 12
_REID_CROP_H = 128
_REID_CROP_W = 128

# ── Identity State Machine ───────────────────────────────────────────
STATE_ACTIVE = 0
STATE_THINKING = 1
STATE_PHANTOM = 2
STATE_DEAD = 3

# How many frames to hold identity during uncertainty before going phantom
THINKING_WINDOW = 15   # ~3s at 5 FPS (was 35, too long for pose-change scenarios)
# Minimum lifetime to be worth preserving
MIN_PRESERVE_LIFETIME = 5

_DEBUG_MEMORY = False


def id_color(tid):
    np.random.seed(tid * 7)
    return tuple(int(c) for c in np.random.randint(100, 255, 3))


class SelfWatchPipeline:
    def __init__(self, detector, reid, tracker, enable_debug_overlay=True):
        # Layer 1
        self.detector = detector
        self.reid = reid
        self.tracker = tracker

        # Layer 2-5
        self.global_id_manager = GlobalIdentityManager()
        self.active_memory = ActiveMemory()
        self.phantom_tracker = PhantomTracker(
            max_phantom_age=config.PHANTOM_MAX_AGE,
            match_threshold=config.PHANTOM_MATCH_THRESHOLD
        )
        self.warm_memory = WarmMemory(max_size=config.MEMORY_MAX_WARM)
        self.reasoning = CognitiveReasoning()
        self.occlusion_manager = OcclusionGroupManager()
        self.ownership_arbiter = OwnershipArbitrationLayer()
        self.ownership_sm = OwnershipStateMachine()

        # Observability
        self.event_logger = CognitiveEventLogger(
            log_dir="logs", enabled=config.MEMORY_EVENT_LOGGING)
        self.metrics = TrackingMetrics()
        self.phantom_tracker._logger = self.event_logger
        self.phantom_tracker._metrics = self.metrics
        self.debug_overlay = DebugOverlay(enabled=enable_debug_overlay)
        self.layer_manager = DebugLayerManager()

        self.frame_count = 0

        # ── Identity state machine ───────────────────────────────────
        # gid -> { state, last_box, last_embedding, velocity, gallery,
        #          importance, entered_frame, owning_lid }
        self._id_states = {}
        self._prev_lid_to_gid = {}
        self._prev_lid_is_prov = {}

        self._prof = {
            "detect": 0.0, "reid": 0.0, "tracker": 0.0,
            "reasoning": 0.0, "identity": 0.0, "memory": 0.0,
            "draw": 0.0, "total": 0.0,
        }
        self._prof_count = 0

    # ══════════════════════════════════════════════════════════════════
    #  IDENTITY STATE HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _set_state(self, gid, state, **kwargs):
        """Transition a global identity to a new state."""
        prev = self._id_states.get(gid, {}).get("state")
        if prev == state:
            # Same state — update fields only
            self._id_states[gid].update(kwargs)
            return

        if gid not in self._id_states:
            self._id_states[gid] = {}

        self._id_states[gid]["state"] = state
        self._id_states[gid]["entered_frame"] = self.frame_count
        self._id_states[gid].update(kwargs)

        if _DEBUG_MEMORY:
            names = {0: "ACTIVE", 1: "THINKING", 2: "PHANTOM", 3: "DEAD"}
            pn = names.get(prev, "NEW")
            nn = names.get(state, "?")
            print(f"  [STATE] gid={gid}: {pn} -> {nn}")

    def _get_state(self, gid):
        entry = self._id_states.get(gid)
        return entry["state"] if entry else None

    def _get_thinking_identities(self):
        """Return dict of gid -> identity_data for all THINKING identities,
        enriched with embedding data from active memory."""
        result = {}
        for gid, data in self._id_states.items():
            if data["state"] != STATE_THINKING:
                continue
            # Get embedding from active memory (still alive during THINKING)
            am = self.active_memory.get_identity(gid)
            entry = dict(data)  # copy state data
            if am:
                entry["last_embedding"] = am.get("stable_embedding")
                if "last_box" not in entry or entry["last_box"] is None:
                    if am.get("trajectory"):
                        entry["last_box"] = am["trajectory"][-1].tolist()
                # Expose velocity for direction-consistency gating
                if "last_velocity" not in entry or entry["last_velocity"] is None:
                    entry["last_velocity"] = am.get("last_velocity")
            result[gid] = entry
        return result

    # ══════════════════════════════════════════════════════════════════
    #  MAIN FRAME PROCESSING
    # ══════════════════════════════════════════════════════════════════

    def process_frame(self, frame, frame_delta=1, frame_index=None, color_map=None):
        loop_start = time.perf_counter()
        
        # Optimize memory and rendering speed
        frame = cv2.resize(frame, (960, 540))
        
        frame_delta = max(1, int(frame_delta))
        self.frame_count += 1
        is_reid_frame = (
            (self.frame_count % REID_INTERVAL == 0) or (self.frame_count <= 2)
        )
        h, w = frame.shape[:2]

        DETECTOR_INTERVAL = getattr(config, 'DETECTOR_INTERVAL', 2)
        if self.frame_count % DETECTOR_INTERVAL == 0 or self.frame_count <= 2:
            t0 = time.perf_counter()
            det_result = self.detector.detect(
                frame, conf_threshold=0.35, target_classes=[PERSON_CLASS])
            raw_boxes = [list(b) for b in det_result.boxes]
            raw_scores = list(det_result.scores)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            raw_boxes = []
            raw_scores = []
            for track in self.tracker.tracks:
                if track.is_confirmed and track.time_since_update <= 1:
                    raw_boxes.append(track.smooth_box.tolist())
                    raw_scores.append(track.score)
            t1 = time.perf_counter()

        boxes, scores = [], []
        raw_crops = []  # Store crops for fingerprint color histograms
        for box, sc in zip(raw_boxes, raw_scores):
            bx1, by1, bx2, by2 = box
            if (by2 - by1) >= 60 and (bx2 - bx1) >= 25:
                boxes.append(box)
                scores.append(sc)
                # Extract raw crop for color histogram (CPU only)
                x1i, y1i = max(0, int(bx1)), max(0, int(by1))
                x2i, y2i = min(w, int(bx2)), min(h, int(by2))
                if x2i > x1i and y2i > y1i:
                    raw_crops.append(frame[y1i:y2i, x1i:x2i].copy())
                else:
                    raw_crops.append(None)

        # ── Layer 1B: ReID Embeddings ────────────────────────────────
        t_reid0 = time.perf_counter()
        if len(boxes) > 0:
            if is_reid_frame:
                crops = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crop = cv2.resize(
                            frame[y1:y2, x1:x2], (_REID_CROP_W, _REID_CROP_H))
                        crops.append(crop)
                    else:
                        crops.append(np.zeros(
                            (_REID_CROP_H, _REID_CROP_W, 3), dtype=np.uint8))
                embeddings = self.reid.extract_batch(crops)
            else:
                embeddings = self._reuse_embeddings(boxes)
        else:
            embeddings = np.zeros((0, 512), dtype=np.float32)
        t_reid1 = time.perf_counter()



        # ── Pre-Tracker Suppression Regions & Frozen / Cooldown LIDs ──
        # ── Pre-Tracker Suppression Regions & Frozen / Cooldown LIDs ──
        suppress_regions = []
        frozen_lids = set()
        cooldown_lids = set()
        exit_trajectories = {}
        collision_partners = {}

        for track in self.tracker.tracks:
            gid = self.global_id_manager.get_global_id(track.local_id)
            if gid is not None:
                if gid in self.occlusion_manager.frozen_gids:
                    suppress_regions.append(track.smooth_box.tolist())
                    frozen_lids.add(track.local_id)
                elif self.occlusion_manager.is_in_cooldown(gid):
                    cooldown_lids.add(track.local_id)

                # Cache exit trajectories and collision partners for low-level tracker
                traj = self.occlusion_manager.get_exit_trajectory(gid)
                if traj is not None:
                    exit_trajectories[track.local_id] = traj

                partners_gids = self.occlusion_manager._collision_partners.get(gid, set())
                partner_lids = set()
                for partner_gid in partners_gids:
                    for lid, val in self.global_id_manager._local_to_global.items():
                        if val == partner_gid:
                            partner_lids.add(lid)
                            break
                if partner_lids:
                    collision_partners[track.local_id] = partner_lids
            
            if track.is_lost and track.time_since_update <= THINKING_WINDOW:
                # Approximate predicted box since predict() hasn't run yet
                w = track.smooth_box[2] - track.smooth_box[0]
                h = track.smooth_box[3] - track.smooth_box[1]
                cx = (track.smooth_box[0] + track.smooth_box[2]) / 2 + track.vel[0] * frame_delta
                cy = (track.smooth_box[1] + track.smooth_box[3]) / 2 + track.vel[1] * frame_delta
                suppress_regions.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        # ── Layer 1C: StrongSORT ─────────────────────────────────────
        t_trk0 = time.perf_counter()
        active = self.tracker.update(
            boxes, scores, embeddings,
            frame_shape=frame.shape, frame_delta=frame_delta,
            suppress_regions=suppress_regions,
            frozen_lids=frozen_lids,
            cooldown_lids=cooldown_lids,
            exit_trajectories=exit_trajectories,
            collision_partners=collision_partners,
            frame_count=self.frame_count
        )


        t_trk1 = time.perf_counter()

        # ── Compute identity states BEFORE reasoning ─────────────────
        # Build set of currently confirmed+matched tracks
        confirmed_gids = set()
        confirmed_tracks_by_gid = {}
        all_tracks_by_lid = {}

        for track in self.tracker.tracks:
            gid = self.global_id_manager.get_global_id(track.local_id)
            all_tracks_by_lid[track.local_id] = track

            if track.is_confirmed and track.time_since_update == 0:
                confirmed_gids.add(gid)
                confirmed_tracks_by_gid[gid] = track

                # Flexible Trajectory Adaptation: if a track in cooldown successfully
                # reattaches/matches a detection, clear its post-freeze cooldown early
                # so it immediately unblocks and adapts to the new motion trajectory!
                if gid is not None and self.occlusion_manager.is_in_cooldown(gid):
                    self.occlusion_manager._post_freeze_cooldown.pop(gid, None)
                    self.occlusion_manager._exit_trajectories.pop(gid, None)
                    self.occlusion_manager._collision_partners.pop(gid, None)

        # ── Occlusion Group Detection ─────────────────────────────────
        # Build confirmed boxes + velocities for overlap check
        confirmed_boxes = {}
        confirmed_velocities = {}
        for gid, trk in confirmed_tracks_by_gid.items():
            confirmed_boxes[gid] = trk.smooth_box.tolist()
            confirmed_velocities[gid] = trk.vel.tolist()

        frozen_gids = self.occlusion_manager.update(
            confirmed_boxes, velocities=confirmed_velocities)

        # ── Ambiguity-Aware Identity Freezing ────────────────────────
        # Freeze identities that have weak/ambiguous association or
        # are provisional with low confidence.
        ambiguous_gids = set()
        for gid, trk in confirmed_tracks_by_gid.items():
            cost = getattr(trk, 'last_assoc_cost', 0.0)
            method = getattr(trk, 'last_assoc_method', 'NEW')
            # 1. High association cost (fused or BIoU)
            # 2. C-BIoU fallback matching
            # 3. Provisional (unconfirmed) but active track with low confidence
            is_cbiou = "C-BIoU" in method
            is_high_cost = cost > 0.65
            is_weak_provisional = self.global_id_manager.is_provisional(trk.local_id) and trk.score < 0.60
            
            if is_cbiou or is_high_cost or is_weak_provisional:
                ambiguous_gids.add(gid)

        # Combine physical occlusion frozen GIDs and ambiguous frozen GIDs
        frozen_gids = frozen_gids.union(ambiguous_gids)


        # Determine THINKING tracks: lost but within hold window
        # These are StrongSORT LOST tracks that still have predicted boxes
        thinking_tracks_by_gid = {}
        for track in self.tracker.tracks:
            gid = self.global_id_manager.get_global_id(track.local_id)
            if gid in confirmed_gids:
                continue  # Already active, skip
            # ACT-R: dynamic thinking window per identity
            dynamic_window = self.global_id_manager.get_thinking_window(gid)
            if track.is_lost and track.time_since_update <= dynamic_window:
                # Only hold if this identity was established (lived long enough)
                prev_state = self._get_state(gid)
                if prev_state in (STATE_ACTIVE, STATE_THINKING, None):
                    am_data = self.active_memory.get_identity(gid)
                    if am_data and am_data.get("age", 0) >= MIN_PRESERVE_LIFETIME:
                        thinking_tracks_by_gid[gid] = track

        # ── Auto-merge: cancel THINKING when a new active track overlaps ──
        # This handles pose-change scenarios (sitting→standing) where the
        # tracker creates a new local track for the same person.
        thinking_to_cancel = set()
        for tgid, ttrack in thinking_tracks_by_gid.items():
            tbox = ttrack.predicted_box
            if tbox is None:
                continue
            tcx = (tbox[0] + tbox[2]) / 2.0
            tcy = (tbox[1] + tbox[3]) / 2.0
            for cgid, ctrack in confirmed_tracks_by_gid.items():
                cbox = ctrack.smooth_box
                ccx = (cbox[0] + cbox[2]) / 2.0
                ccy = (cbox[1] + cbox[3]) / 2.0
                # Spatial proximity check (centers within ~60px)
                dist = ((tcx - ccx)**2 + (tcy - ccy)**2) ** 0.5
                if dist < 80:
                    # Check IoU overlap
                    ix1 = max(tbox[0], cbox[0])
                    iy1 = max(tbox[1], cbox[1])
                    ix2 = min(tbox[2], cbox[2])
                    iy2 = min(tbox[3], cbox[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_t = max(1, (tbox[2]-tbox[0]) * (tbox[3]-tbox[1]))
                    area_c = max(1, (cbox[2]-cbox[0]) * (cbox[3]-cbox[1]))
                    iou = inter / (area_t + area_c - inter + 1e-6)
                    if iou > 0.15 or dist < 50:
                        # New active track overlaps with THINKING box
                        # → same person changed pose, cancel THINKING
                        thinking_to_cancel.add(tgid)
                        # If the new track is provisional, absorb its identity
                        new_lid = ctrack.local_id
                        if self.global_id_manager.is_provisional(new_lid):
                            # Force the new track to inherit the THINKING identity
                            self.global_id_manager._local_to_global[new_lid] = tgid
                            self.global_id_manager._provisional.pop(new_lid, None)
                            print(f"[AUTO-MERGE] Provisional lid={new_lid} absorbed "
                                  f"THINKING gid={tgid} (dist={dist:.0f} iou={iou:.2f})")
                        break

        for tgid in thinking_to_cancel:
            del thinking_tracks_by_gid[tgid]
            self._set_state(tgid, STATE_ACTIVE)

        # ── Layer 5: Cognitive Reasoning ─────────────────────────────
        t_reas0 = time.perf_counter()
        proposals = self.reasoning.evaluate_tracks(
            self.tracker.tracks, self.phantom_tracker,
            self.active_memory, self.warm_memory,
            self._get_thinking_identities(),
            time.perf_counter(), frame_delta, self.metrics,
            restricted_gids=self.occlusion_manager.restricted_gids
        )
        t_reas1 = time.perf_counter()

        if _DEBUG_MEMORY and proposals:
            for p in proposals:
                lid, gid, conf, src = p
                print(f"  [PROPOSAL] local={lid} -> global={gid} "
                      f"conf={conf:.3f} src={src}")

        # ── Layer 2: Global Identity Manager ─────────────────────────
        t_id0 = time.perf_counter()
        self.global_id_manager.update(
            self.tracker.tracks, proposals, frozen_gids=frozen_gids,
            occlusion_manager=self.occlusion_manager)

        # ── Ghost Track Remediation (Rebind & Deactivate) ───────────
        # Target Point 3: detect detached ghost trajectories and transfer
        # identity ownership to consistent moving detections.
        # Avoid leaving original ID behind.
        from trackers.strack import STrack
        
        # 1. Find lost tracks that are "ghosting" (in cooldown or recently lost)
        ghost_tracks = []
        for track in self.tracker.tracks:
            gid = self.global_id_manager.get_global_id(track.local_id)
            if gid is not None and track.is_lost:
                if track.time_since_update > 0 and track.time_since_update <= 30:
                    ghost_tracks.append(track)
                    
        # 2. Find newly spawned tracks that might represent the real moving person
        active_new_tracks = []
        for track in self.tracker.tracks:
            if track.is_confirmed and track.time_since_update == 0:
                is_prov = self.global_id_manager.is_provisional(track.local_id)
                if track.age < 20 or is_prov:
                    active_new_tracks.append(track)
                    
        # 3. Pairwise check: if a new active track matches a lost ghost track's appearance
        # with overwhelming confidence, transfer GID and deactivate the ghost!
        for t_active in active_new_tracks:
            active_gid = self.global_id_manager.get_global_id(t_active.local_id)
            active_emb = t_active.get_averaged_embedding()
            if active_emb is None:
                continue
                
            best_ghost = None
            best_dist = 999.0
            
            for t_ghost in ghost_tracks:
                ghost_gid = self.global_id_manager.get_global_id(t_ghost.local_id)
                if ghost_gid == active_gid:
                    continue  # Already same ID
                    
                ghost_emb = t_ghost.get_averaged_embedding()
                if ghost_emb is None:
                    continue
                    
                # Cosine distance
                dist = 1.0 - float(np.dot(active_emb, ghost_emb))
                if dist < 0.18:  # Overwhelming visual similarity
                    if dist < best_dist:
                        best_dist = dist
                        best_ghost = t_ghost
                        
            if best_ghost is not None:
                ghost_gid = self.global_id_manager.get_global_id(best_ghost.local_id)
                
                # Double check that ghost_gid is not active elsewhere
                proposed_active_elsewhere = False
                for other_track in self.tracker.tracks:
                    if other_track.local_id != best_ghost.local_id and other_track.local_id != t_active.local_id:
                        other_gid = self.global_id_manager.get_global_id(other_track.local_id)
                        if other_gid == ghost_gid and other_track.is_confirmed and other_track.time_since_update == 0:
                            proposed_active_elsewhere = True
                            break
                            
                if not proposed_active_elsewhere:
                    # Let's transfer identity!
                    old_active_gid = active_gid
                    self.global_id_manager._local_to_global[t_active.local_id] = ghost_gid
                    t_active.global_id = ghost_gid
                    
                    # Remove active track from provisional since it now inherits a committed identity!
                    self.global_id_manager._provisional.pop(t_active.local_id, None)
                    
                    # Deactivate the ghost track to prevent it from persisting or being stolen!
                    best_ghost.state = STrack.STATE_LOST
                    best_ghost.time_since_update = 1000  # Instantly age out and die!
                    
                    # Clear any occlusion or cooldown trace
                    if self.occlusion_manager.is_in_cooldown(ghost_gid):
                        self.occlusion_manager._post_freeze_cooldown.pop(ghost_gid, None)
                        self.occlusion_manager._exit_trajectories.pop(ghost_gid, None)
                        self.occlusion_manager._collision_partners.pop(ghost_gid, None)
                        
                    print(f"[GHOST REMEDIATION] Transferred GID={ghost_gid} from ghost track local={best_ghost.local_id} "
                          f"to active track local={t_active.local_id} (old_gid={old_active_gid}, visual_dist={best_dist:.3f})")

        # Detect ID switches
        current_lid_to_gid = {}
        current_lid_is_prov = {}
        frame_id_switches = []
        for track in self.tracker.tracks:
            lid = track.local_id
            gid = self.global_id_manager.get_global_id(lid)
            is_prov = self.global_id_manager.is_provisional(lid)
            
            current_lid_to_gid[lid] = gid
            current_lid_is_prov[lid] = is_prov
            
            prev_gid = self._prev_lid_to_gid.get(lid)
            prev_is_prov = self._prev_lid_is_prov.get(lid, False)
            
            if prev_gid is not None and prev_gid != gid:
                # Do NOT count as an ID switch if the previous ID was merely provisional
                if not prev_is_prov:
                    self.metrics.record_id_switch(prev_gid, gid)
                    self.event_logger.log("id_switch", lid,
                                          old_gid=prev_gid, new_gid=gid)
                    frame_id_switches.append((lid, prev_gid, gid))
        t_id1 = time.perf_counter()

        # ── Layer 3 & 4: Memory + State Machine ─────────────────────
        t_mem0 = time.perf_counter()

        avg_conf = sum(scores) / max(1, len(scores)) if scores else 0.5
        self.warm_memory.set_difficulty(
            1.0 + len(active) * 0.15 + max(0.0, 1.0 - avg_conf)
        )

        # Recompute gids with updated identity manager
        confirmed_gids_final = set()
        for track in self.tracker.tracks:
            gid = self.global_id_manager.get_global_id(track.local_id)
            if track.is_confirmed and track.time_since_update == 0:
                confirmed_gids_final.add(gid)

        # ── STATE TRANSITIONS ────────────────────────────────────────
        # Tick the central ownership state machine
        self.ownership_sm.tick(self.frame_count)
        self.ownership_sm.update_frozen_set(frozen_gids)

        # 1. ACTIVE: currently detected tracks
        for gid in confirmed_gids_final:
            track = None
            for t in self.tracker.tracks:
                if (self.global_id_manager.get_global_id(t.local_id) == gid
                        and t.is_confirmed and t.time_since_update == 0):
                    track = t
                    break

            prev = self._get_state(gid)

            # Resurrect from warm/phantom if needed
            if prev == STATE_PHANTOM:
                if gid in self.phantom_tracker.phantoms:
                    self.phantom_tracker.remove(gid)
                self.metrics.resurrections += 1
                self.event_logger.log("phantom_resurrect", gid)
            elif prev == STATE_THINKING:
                # Recovered from THINKING — good, no phantom needed
                pass

            warm_data = self.warm_memory.resurrect(gid)
            if warm_data is not None:
                self.metrics.resurrections += 1
                self.event_logger.log("warm_resurrect", gid)

            self._set_state(gid, STATE_ACTIVE,
                            owning_lid=track.local_id if track else None)
            # Central SM: canonical ACTIVE state
            self.ownership_sm.activate(
                gid, owning_lid=track.local_id if track else None,
                box=track.smooth_box.tolist() if track else None,
                velocity=track.vel.tolist() if track else None)

            # Update active memory with fingerprint features
            if track:
                box_val = track.smooth_box.tolist()

                if gid in frozen_gids:
                    # FROZEN: do NOT update appearance/fingerprint
                    # Only update trajectory position (no embedding, no crop)
                    self.active_memory.update(
                        gid, None, box_val, track.score, frame_delta,
                        crop=None, velocity=track.vel.tolist())
                else:
                    # Normal: full fingerprint update with crop
                    track_crop = self._find_crop_for_track(
                        track, boxes, raw_crops)
                    # Use center crop if near other tracks (anti-pollution)
                    if track_crop is not None:
                        track_crop = self._center_crop(track_crop)
                    vel = track.vel.tolist()
                    self.active_memory.update(
                        gid, track.embedding, box_val, track.score,
                        frame_delta, crop=track_crop, velocity=vel)

        # 2. THINKING: recently lost, within hold window
        for gid, track in thinking_tracks_by_gid.items():
            if gid in confirmed_gids_final:
                continue  # Already active
            prev = self._get_state(gid)
            if prev in (STATE_ACTIVE, STATE_THINKING, None):
                # Store velocity for direction-consistency gating
                am_data = self.active_memory.get_identity(gid)
                last_vel = am_data.get("last_velocity", np.zeros(2)) if am_data else np.zeros(2)
                self._set_state(gid, STATE_THINKING,
                                last_box=track.predicted_box.tolist(),
                                owning_lid=track.local_id,
                                last_velocity=last_vel)
                # Central SM: THINKING = LATENT_CANDIDATE (not rendered)
                self.ownership_sm.to_latent(gid, reason="thinking")
                # Keep active memory alive during THINKING
                # (don't remove it — the identity is still "held")

        # 3. THINKING -> PHANTOM transition (exceeded hold window)
        for gid, data in list(self._id_states.items()):
            if data["state"] != STATE_THINKING:
                continue
            frames_thinking = self.frame_count - data.get("entered_frame", 0)
            # ACT-R: per-identity dynamic thinking window
            dynamic_window = self.global_id_manager.get_thinking_window(gid)
            if frames_thinking >= dynamic_window:
                # Transition to PHANTOM
                am_data = self.active_memory.remove(gid)
                if am_data and am_data.get("stable_embedding") is not None:
                    last_box = am_data["trajectory"][-1] if am_data["trajectory"] else None
                    vel = am_data.get("last_velocity", np.zeros(2))
                    lifetime = am_data.get("age", 0)
                    if last_box is not None and lifetime >= MIN_PRESERVE_LIFETIME:
                        self.phantom_tracker.spawn(
                            track_id=gid,
                            embedding=am_data["stable_embedding"],
                            last_position=last_box,
                            velocity=vel,
                            importance=am_data.get("importance", 1.0),
                            gallery=am_data.get("gallery", []))
                self._set_state(gid, STATE_PHANTOM)
                # Central SM: PHANTOM = still LATENT (already set)

        # 4. Tick phantoms, handle PHANTOM -> DEAD
        about_to_expire = {}
        for tid, phantom in self.phantom_tracker.phantoms.items():
            future_age = phantom.age_frames + frame_delta
            future_conf = phantom.confidence * (0.97 ** frame_delta)
            if future_age >= self.phantom_tracker.max_phantom_age or future_conf < 0.15:
                about_to_expire[tid] = phantom

        for gid, phantom in about_to_expire.items():
            data = {
                "stable_embedding": phantom.embedding,
                "recent_embedding": phantom.embedding,
                "best_embedding": phantom.embedding,
                "gallery": phantom.gallery,
                "importance": phantom.importance,
                "trajectory": [phantom.position],
                "last_velocity": phantom.velocity,
            }
            self.warm_memory.save_identity(gid, data, time.perf_counter())
            self.metrics.record_memory_save(gid, phantom.age_frames)
            self.event_logger.log("warm_save", gid)
            self._set_state(gid, STATE_DEAD)
            # Central SM: archive the identity
            self.ownership_sm.archive(gid, reason="phantom_expired")

        self.phantom_tracker.tick(frame_delta)

        # 5. Cleanup dead identities from state machine
        dead_gids = [gid for gid, d in self._id_states.items()
                     if d["state"] == STATE_DEAD]
        for gid in dead_gids:
            del self._id_states[gid]
            self.ownership_sm.remove(gid)

        # 6. Also handle identities that went directly lost
        #    (not in confirmed, not in thinking, not yet phantom)
        for gid, data in list(self._id_states.items()):
            if data["state"] == STATE_ACTIVE and gid not in confirmed_gids_final:
                # Track disappeared without going through THINKING
                # (e.g., transient track with age < MIN_PRESERVE_LIFETIME)
                am_data = self.active_memory.get_identity(gid)
                lifetime = am_data.get("age", 0) if am_data else 0
                if lifetime < MIN_PRESERVE_LIFETIME:
                    self.active_memory.remove(gid)
                    self._set_state(gid, STATE_DEAD)
                    self.ownership_sm.archive(gid, reason="transient")

        # Re-clean dead
        dead_gids = [gid for gid, d in self._id_states.items()
                     if d["state"] == STATE_DEAD]
        for gid in dead_gids:
            del self._id_states[gid]
            self.ownership_sm.remove(gid)

        # Decay warm memory
        self.warm_memory.decay(time.perf_counter())

        self._prev_lid_to_gid = current_lid_to_gid
        self._prev_lid_is_prov = current_lid_is_prov
        t_mem1 = time.perf_counter()

        # ── Draw ─────────────────────────────────────────────────────
        t_draw0 = time.perf_counter()

        # Build render list: ONE box per global_id
        display = {}  # gid -> (box, render_state)

        # ACTIVE tracks: normal rendering
        for local_id, tbox in active.items():
            if self.global_id_manager.is_provisional(local_id):
                continue  # Delay rendering provisional new IDs
            gid = self.global_id_manager.get_global_id(local_id)
            vel = None
            age = 0
            assoc_data = {}
            for t in self.tracker.tracks:
                if t.local_id == local_id:
                    vel = t.vel.tolist()
                    age = t.age
                    assoc_data = {
                        "cost": getattr(t, "last_assoc_cost", 0.0),
                        "method": getattr(t, "last_assoc_method", "NEW"),
                        "cbiou": getattr(t, "cbiou_buffer", 0)
                    }
                    break
            display[gid] = (tbox, STATE_ACTIVE, local_id, vel, age, assoc_data)

        # THINKING tracks: render predicted box (faded)
        for gid, data in self._id_states.items():
            if data["state"] == STATE_THINKING and gid not in display:
                box = data.get("last_box")
                if box:
                    am_data = self.active_memory.get_identity(gid)
                    vel = am_data.get("last_velocity", [0, 0]) if am_data else [0, 0]
                    age = am_data.get("age", 0) if am_data else 0
                    display[gid] = (box, STATE_THINKING, data.get("owning_lid", "?"), vel, age, {})

        # ── Visual Identity Metrics ──────────────────────────────────
        # Capture pre-arbitration display for forensic comparison
        raw_display = {}
        for gid, (tbox, state, lid, vel, age, assoc) in display.items():
            raw_display[gid] = {
                "box": list(tbox) if not isinstance(tbox, list) else tbox,
                "state": state,
            }

        # OWNERSHIP ARBITRATION: enforce single visual identity per region
        display, suppressed_entries = self.ownership_arbiter.arbitrate(
            display, frozen_gids=frozen_gids, frame_count=self.frame_count)

        if suppressed_entries:
            for sup_gid, sup_reason in suppressed_entries:
                self.event_logger.log("visual_suppress", sup_gid,
                                      reason=sup_reason, frame=self.frame_count)

        # Detect overlapping rendered boxes (human-perceived duplicates)
        display_boxes = [(gid, tbox) for gid, (tbox, *_) in display.items()]
        n_duplicates = 0
        for i in range(len(display_boxes)):
            for j in range(i + 1, len(display_boxes)):
                gid_a, box_a = display_boxes[i]
                gid_b, box_b = display_boxes[j]
                # Check IoU between rendered boxes
                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = max(1, (box_a[2]-box_a[0]) * (box_a[3]-box_a[1]))
                area_b = max(1, (box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))
                overlap = inter / (area_a + area_b - inter + 1e-6)
                if overlap > 0.3:
                    n_duplicates += 1
        self.metrics.record_duplicate_boxes(n_duplicates)

        # Update stability score
        n_active = sum(1 for _, (_, s, *_) in display.items() if s == STATE_ACTIVE)
        n_thinking = sum(1 for _, (_, s, *_) in display.items() if s == STATE_THINKING)
        n_phantom = self.phantom_tracker.count
        self.metrics.update_stability(n_active, n_thinking, n_duplicates)

        # ── Phase 3: Decoupled Render Metadata Generation ────────────────
        # Instead of drawing directly in the inference thread, collect all data needed by OverlayRenderer
        
        layer_flags = {
            "forensic": self.layer_manager.is_enabled("forensic"),
            "for_suppress": self.layer_manager.is_enabled("for_suppress"),
            "for_failure": self.layer_manager.is_enabled("for_failure"),
            "motion": self.layer_manager.is_enabled("motion"),
            "motion_prediction": self.layer_manager.is_enabled("motion_prediction"),
            "motion_velocity": self.layer_manager.is_enabled("motion_velocity"),
            "cognitive": self.layer_manager.is_enabled("cognitive"),
            "cog_frozen": self.layer_manager.is_enabled("cog_frozen"),
            "cog_thinking": self.layer_manager.is_enabled("cog_thinking"),
            "tracking": self.layer_manager.is_enabled("tracking"),
            "tracking_ids": self.layer_manager.is_enabled("tracking_ids"),
            "tracking_age": self.layer_manager.is_enabled("tracking_age"),
            "association": self.layer_manager.is_enabled("association"),
            "assoc_method": self.layer_manager.is_enabled("assoc_method"),
            "assoc_cost": self.layer_manager.is_enabled("assoc_cost"),
            "assoc_cbiou": self.layer_manager.is_enabled("assoc_cbiou"),
        }

        phantoms_meta = []
        if layer_flags["motion"] and layer_flags["motion_prediction"]:
            for pt in self.phantom_tracker.phantoms.values():
                cone = pt.get_cone_tip_and_edges()
                phantoms_meta.append({
                    "position": pt.position,
                    "track_id": pt.track_id,
                    "confidence": int(pt.confidence * 100),
                    "cone": cone,
                    "velocity": pt.initial_velocity
                })

        display_items = {}
        for gid, (tbox, state, lid, vel, age, assoc_data) in display.items():
            display_items[gid] = {
                "box": tbox,
                "state": state,
                "lid": lid,
                "vel": vel,
                "age": age,
                "assoc": assoc_data
            }

        recent_rejects_msgs = []
        if layer_flags["forensic"] and layer_flags["for_failure"]:
            rebind_log = self.global_id_manager.get_rebind_log()
            recent_rejects = [e for e in rebind_log if "REJECT" in e.get("decision", "") and e.get("frame", 0) >= self.frame_count - 3]
            for rej in recent_rejects[-3:]:
                recent_rejects_msgs.append(
                    f"BLOCKED: gid={rej['proposed_gid']} src={rej['source']} "
                    f"{rej['decision']} dot={rej['direction_dot']:.2f} dist={rej['trajectory_dist']:.0f}px"
                )

        hud_meta = {
            "tracked": len(display),
            "active": self.tracker.confirmed_count,
            "thinking": n_thinking,
            "phantom": self.phantom_tracker.count,
            "occlusion": self.occlusion_manager.count,
            "warm": self.warm_memory.count,
            "reid_tag": "ReID" if is_reid_frame else "cached",
            "frame_count": self.frame_count
        }

        rendering_metadata = {
            "layer_flags": layer_flags,
            "suppress_regions": suppress_regions if layer_flags["forensic"] and layer_flags["for_suppress"] else [],
            "phantoms": phantoms_meta,
            "frozen_gids": frozen_gids,
            "display_items": display_items,
            "color_map": color_map,
            "id_switches": len(frame_id_switches) if frame_id_switches else 0,
            "recent_rejects": recent_rejects_msgs,
            "debug_enabled": self.debug_overlay.enabled,
            "hud": hud_meta
        }

        t_draw1 = time.perf_counter()

        # ── Profiling ────────────────────────────────────────────────
        det_ms = (t1 - t0) * 1000.0
        reid_ms = (t_reid1 - t_reid0) * 1000.0
        trk_ms = (t_trk1 - t_trk0) * 1000.0
        total_ms = (t_draw1 - loop_start) * 1000.0

        self._prof["detect"] += det_ms
        self._prof["reid"] += reid_ms
        self._prof["tracker"] += trk_ms
        self._prof["reasoning"] += (t_reas1 - t_reas0) * 1000.0
        self._prof["identity"] += (t_id1 - t_id0) * 1000.0
        self._prof["memory"] += (t_mem1 - t_mem0) * 1000.0
        self._prof["draw"] += (t_draw1 - t_draw0) * 1000.0
        self._prof["total"] += total_ms
        self._prof_count += 1
        self.metrics.tick_frame()

        # Periodic status
        if self.frame_count % 60 == 0:
            s = self.metrics.get_summary()
            print(f"\n[STATUS F#{self.frame_count}] "
                  f"Active={len(confirmed_gids_final)} "
                  f"Thinking={n_thinking} "
                  f"Phantom={n_phantom} "
                  f"Warm={self.warm_memory.count} "
                  f"Saves={s['memory_saves']} "
                  f"Resurrections={s['resurrections']} "
                  f"IDswitches={s['id_switches']} "
                  f"SM={self.ownership_sm.get_summary()}\n")

        # Build states for UI
        track_states = {}
        for gid, (tbox, state, lid, vel, age, assoc_data) in display.items():
            track_states[gid] = {
                "box": list(tbox) if not isinstance(tbox, list) else tbox,
                "vel": [0, 0],
                "score": 0.0,
                "identity_state": state,
            }
        # Overlay real track data where available
        for track in self.tracker.tracks:
            gid = self.global_id_manager.get_global_id(track.local_id)
            if gid in track_states:
                track_states[gid]["vel"] = track.vel.tolist()
                track_states[gid]["score"] = float(track.score)

        return frame, {
            "det_ms": det_ms,
            "reid_ms": reid_ms,
            "trk_ms": trk_ms,
            "raw_detections": len(boxes),
            "active_tracks": len(display),
            "active_dict": {gid: box for gid, (box, _, _, _, _, _) in display.items()},
            "track_states": track_states,
            "raw_display": raw_display,
            "phantom_count": n_phantom,
            "thinking_count": n_thinking,
            "metrics": self.metrics,
            "ownership_sm_summary": self.ownership_sm.get_summary(),
            "processed_frame_index": self.frame_count,
            "raw_frame_index": frame_index,
            "frame_delta": frame_delta,
            "id_switches": frame_id_switches,
            "suppress_regions": suppress_regions,
            "frozen_gids": list(frozen_gids),
            "rendering_metadata": rendering_metadata,
        }

    # ── Crop Lookup ────────────────────────────────────────────────────

    def _find_crop_for_track(self, track, det_boxes, raw_crops):
        """Find the raw crop image matching a track via IoU to detections."""
        if not det_boxes or not raw_crops:
            return None
        from utils.iou import iou_matrix as _iou_matrix
        track_box = [track.smooth_box.tolist()]
        ious = _iou_matrix(track_box, det_boxes)  # (1, n_det)
        best_j = int(np.argmax(ious[0]))
        if ious[0, best_j] > 0.3 and best_j < len(raw_crops):
            return raw_crops[best_j]
        return None

    @staticmethod
    def _center_crop(crop):
        """Extract center 50% torso region to avoid edge/background pollution."""
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return crop
        # Horizontal: center 70%
        margin_x = int(w * 0.15)
        # Vertical: center 50% (skip head and feet)
        margin_top = int(h * 0.25)
        margin_bot = int(h * 0.25)
        center = crop[margin_top:h - margin_bot, margin_x:w - margin_x]
        if center.size == 0:
            return crop
        return center

    # ── Embedding Reuse ──────────────────────────────────────────────

    def _reuse_embeddings(self, boxes):
        from utils.iou import iou_matrix as _iou_matrix

        n_det = len(boxes)
        embeddings = np.zeros((n_det, 512), dtype=np.float32)
        confirmed = [
            t for t in self.tracker.tracks
            if t.is_confirmed and t.embedding is not None
        ]
        if not confirmed:
            return embeddings

        track_boxes = [t.predicted_box.tolist() for t in confirmed]
        iou = _iou_matrix(boxes, track_boxes)

        for i in range(n_det):
            best_j = int(np.argmax(iou[i]))
            if iou[i, best_j] > 0.3:
                embeddings[i] = confirmed[best_j].embedding.copy()

        return embeddings

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self):
        print("\n[PIPELINE] Exporting final metrics...")
        print("\n" + "=" * 60)
        print("  INTERNAL TRACKER METRICS (association-level)")
        print("  NOTE: These measure internal consistency, NOT visual output.")
        print("=" * 60)
        self.metrics.print_summary()
        self.metrics.export_csv("logs/metrics_summary.csv")

        # Ownership SM summary
        print("\n" + "=" * 60)
        print("  OWNERSHIP STATE MACHINE — Final State")
        print("=" * 60)
        sm_summary = self.ownership_sm.get_summary()
        for state_name, count in sm_summary.items():
            print(f"  {state_name:25s}: {count}")
        recent_transitions = self.ownership_sm.get_transition_log(last_n=20)
        if recent_transitions:
            print(f"\n  Recent transitions ({len(recent_transitions)}):")
            for t in recent_transitions[-10:]:
                print(f"    F#{t['frame']:>5d}: gid={t['gid']:>3d} "
                      f"{t['from']:>20s} -> {t['to']:<20s} [{t['reason']}]")
        print("=" * 60)

        n = max(self._prof_count, 1)
        print("\n" + "=" * 60)
        print("  SELFWATCH - Per-Subsystem Profiling (avg ms/frame)")
        print("=" * 60)
        for key in ["detect", "reid", "tracker", "reasoning",
                     "identity", "memory", "draw"]:
            avg = self._prof[key] / n
            tag = "GPU" if key in ("detect", "reid") else "CPU"
            print(f"  {key:14s}: {avg:7.2f} ms  [{tag}]")
        print(f"  {'-' * 40}")
        print(f"  {'FRAME total':14s}: {self._prof['total'] / n:7.2f} ms")
        print(f"  {'Frames':14s}: {n}")
        print("=" * 60)
        self.event_logger.close()
