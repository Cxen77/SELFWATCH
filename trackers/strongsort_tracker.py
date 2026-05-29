"""
StrongSORT+ Tracker for SELFWATCH — Observation-Centric Trajectory Continuity

Hybrid tracker combining:
  - StrongSORT fused cost (appearance + IoU) for primary association
  - ByteTrack-style 2-stage low-confidence association
  - OC-SORT-style observation coasting (via STrack)
  - C-BIoU buffered association for fast motion recovery
  - AMI (Ambiguous Match Improvement) to reject bad matches
  - IDSR (ID Switch Rectification) to heal track fragmentation
  - Track birth suppression during occlusion groups

Pipeline:
  Stage 1A: High-conf detections vs confirmed/tentative tracks (fused cost)
  Stage 1B: Remaining high-conf vs lost tracks (fused cost)
  Stage 2:  Low-conf detections vs remaining tracks (IoU only)
  Stage 3:  C-BIoU buffered IoU for remaining tracks
  Stage 4:  IDSR: Check new tracks against recently removed tracks
  Birth:    Suppressed if detection overlaps with occlusion/thinking regions

This module is PURELY spatial+appearance tracking.
It NEVER modifies track.id / global_id.
Identity mapping is handled EXCLUSIVELY by GlobalIDMapper.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .strack import STrack
from utils.iou import iou_matrix


class StrongSORTTracker:
    """
    StrongSORT+ multi-object tracker with observation-centric continuity.

    This is a PURE tracker — no brain, no phantom, no identity mutation.
    It outputs stable local_ids and leaves identity mapping to the
    GlobalIDMapper.
    """

    def __init__(
        self,
        appearance_weight=0.4,
        high_thresh=0.5,
        low_thresh=0.1,
        iou_thresh=0.25,
        max_cosine_dist=0.35,
        max_lost=150,
        confirm_threshold=4,
        embedding_history=10,
        min_quality_score=0.4,
        # New parameters
        biou_buffer=15,          # px: C-BIoU box expansion
        ambiguity_margin=0.0,    # AMI: disabled (crowd gating handles it)
    ):
        self.appearance_weight = appearance_weight
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_cosine_dist = max_cosine_dist
        self.max_lost = max(max_lost, 300)  # Ensure minimum 300 frames (~16s) survival
        self.confirm_threshold = confirm_threshold
        self.embedding_history = embedding_history
        self.min_quality_score = min_quality_score
        self.biou_buffer = biou_buffer
        self.ambiguity_margin = ambiguity_margin

        self.tracks = []   # All active tracks (confirmed + tentative + lost)

        # IDSR: recently removed tracks for post-hoc rectification
        self._recently_removed = []  # list of (box, vel, aspect, local_id, age)
        self._max_removed_history = 60  # Keep recently removed tracks longer for recovery

    # ─── Public API ──────────────────────────────────────────────────


    def update(self, boxes, scores, embeddings, crops=None, frame_shape=None,
               frame_delta=1, suppress_regions=None, frozen_lids=None, cooldown_lids=None,
               exit_trajectories=None, collision_partners=None, frame_count=None):
        """
        Run one frame of StrongSORT+ tracking.

        PURE TRACKER: no brain, no phantom, no identity mutation.
        Returns dict {local_id: [x1, y1, x2, y2]} for confirmed, active tracks.
        """
        frozen_lids = frozen_lids or set()
        cooldown_lids = cooldown_lids or set()
        exit_trajectories = exit_trajectories or {}
        collision_partners = collision_partners or {}
        frame_delta = max(1, int(frame_delta))


        # Predict all existing tracks forward
        for t in self.tracks:
            t.predict(frame_delta=frame_delta)

        # Handle empty frame
        if len(boxes) == 0:
            for t in self.tracks:
                t.mark_lost(frame_delta=frame_delta)
            self._cleanup()
            return self._output()

        boxes = [list(b) for b in boxes]
        scores = list(scores)

        # Extract frame height for pseudo-depth quantization
        frame_h = frame_shape[0] if frame_shape is not None else None

        # ── Split detections by confidence ───────────────────────────
        high_idx = [i for i, s in enumerate(scores) if s >= self.high_thresh]
        low_idx = [i for i, s in enumerate(scores)
                   if self.low_thresh <= s < self.high_thresh]

        high_boxes = [boxes[i] for i in high_idx]
        high_scores = [scores[i] for i in high_idx]
        high_embs = embeddings[high_idx] if len(high_idx) > 0 \
            else np.empty((0, 512), dtype=np.float32)

        low_boxes = [boxes[i] for i in low_idx]

        # ── Separate tracks by state ─────────────────────────────────
        confirmed_tracks = [t for t in self.tracks
                            if t.is_confirmed or t.is_tentative]
        lost_tracks = [t for t in self.tracks if t.is_lost]

        # Track-Perspective Priority: sort by total_hits (oldest first).
        # Established identities get matching priority over newer ones,
        # reducing the chance of a new track stealing an old identity's
        # detection during crowded scenes.
        confirmed_tracks.sort(key=lambda t: -t.total_hits)


        # ─────────────────────────────────────────────────────────────
        # STAGE 1A: High-conf detections vs CONFIRMED/TENTATIVE tracks
        # ─────────────────────────────────────────────────────────────
        matched_tc, matched_dc, unmatched_conf, unmatched_dets_c, fused_cost_c = \
            self._fused_associate(
                high_boxes, high_embs, high_scores, confirmed_tracks,
                frame_h=frame_h, frozen_lids=frozen_lids, cooldown_lids=cooldown_lids,
                exit_trajectories=exit_trajectories, collision_partners=collision_partners,
                frame_count=frame_count
            )


        # Update matched confirmed tracks
        for t_idx, d_idx in zip(matched_tc, matched_dc):
            trk = confirmed_tracks[t_idx]
            orig_i = high_idx[d_idx]
            crop = crops[orig_i] if crops is not None else None
            sibling_boxes = [boxes[high_idx[d]] for d in matched_dc
                             if d != d_idx]
            trk.update(boxes[orig_i], scores[orig_i], embeddings[orig_i],
                       crop=crop, min_quality_score=self.min_quality_score,
                       frame_shape=frame_shape, sibling_boxes=sibling_boxes,
                       frame_delta=frame_delta)
            trk.last_assoc_cost = float(fused_cost_c[d_idx, t_idx])
            trk.last_assoc_method = "Fused (App+IoU) [C]"
            trk.cbiou_buffer = 0

        # ─────────────────────────────────────────────────────────────
        # STAGE 1B: REMAINING high-conf detections vs LOST tracks
        # ─────────────────────────────────────────────────────────────
        remaining_det_idx = list(unmatched_dets_c)

        if remaining_det_idx and lost_tracks:
            rem_boxes = [high_boxes[i] for i in remaining_det_idx]
            rem_embs = high_embs[remaining_det_idx] if len(remaining_det_idx) > 0 \
                else np.empty((0, 512), dtype=np.float32)
            rem_scores = [high_scores[i] for i in remaining_det_idx]


            matched_tl, matched_dl, unmatched_lost, unmatched_dets_l, fused_cost_l = \
                self._fused_associate(
                    rem_boxes, rem_embs, rem_scores, lost_tracks,
                    frame_h=frame_h, frozen_lids=frozen_lids, cooldown_lids=cooldown_lids,
                    exit_trajectories=exit_trajectories, collision_partners=collision_partners,
                    frame_count=frame_count
                )


            for t_idx, d_idx in zip(matched_tl, matched_dl):
                trk = lost_tracks[t_idx]
                orig_d = remaining_det_idx[d_idx]
                orig_i = high_idx[orig_d]
                crop = crops[orig_i] if crops is not None else None
                trk.update(boxes[orig_i], scores[orig_i], embeddings[orig_i],
                           crop=crop, min_quality_score=self.min_quality_score,
                           frame_shape=frame_shape, sibling_boxes=[],
                           frame_delta=frame_delta)
                trk.last_assoc_cost = float(fused_cost_l[d_idx, t_idx])
                trk.last_assoc_method = "Fused (App+IoU)"
                trk.cbiou_buffer = 0

            unmatched_conf_tracks = [confirmed_tracks[i] for i in unmatched_conf]
            unmatched_lost_tracks = [lost_tracks[i] for i in unmatched_lost]
            final_unmatched_dets = [remaining_det_idx[i] for i in unmatched_dets_l]
        else:
            unmatched_conf_tracks = [confirmed_tracks[i] for i in unmatched_conf]
            unmatched_lost_tracks = lost_tracks
            final_unmatched_dets = remaining_det_idx

        # ─────────────────────────────────────────────────────────────
        # STAGE 2: Low-conf detections vs remaining unmatched tracks
        #          (ByteTrack-style: preserve continuity during partial
        #           occlusion/motion blur via low-confidence observations)
        # ─────────────────────────────────────────────────────────────
        remaining_tracks = unmatched_conf_tracks + unmatched_lost_tracks


        if low_boxes and remaining_tracks:
            matched_t2, matched_d2, unmatched_tracks_2, _, cost_iou_2 = \
                self._iou_associate(low_boxes, remaining_tracks, self.iou_thresh, frozen_lids=frozen_lids, cooldown_lids=cooldown_lids, collision_partners=collision_partners)


            for t_idx, d_idx in zip(matched_t2, matched_d2):
                trk = remaining_tracks[t_idx]
                orig_i = low_idx[d_idx]
                # Low-conf: update box but do NOT update embedding
                trk.update(boxes[orig_i], scores[orig_i], embedding=None,
                           min_quality_score=self.min_quality_score,
                           frame_delta=frame_delta)
                trk.last_assoc_cost = float(cost_iou_2[d_idx, t_idx])
                trk.last_assoc_method = "ByteTrack (Low Conf IoU)"
                trk.cbiou_buffer = 0

            still_unmatched = [remaining_tracks[i] for i in unmatched_tracks_2]
        else:
            still_unmatched = remaining_tracks

        # ─────────────────────────────────────────────────────────────
        # STAGE 3: C-BIoU — Buffered IoU for fast motion recovery
        #          Expand bounding boxes by buffer pixels and retry
        #          IoU matching for tracks that normal IoU missed.
        # ─────────────────────────────────────────────────────────────
        # Collect remaining high-conf detections
        remaining_high_dets = final_unmatched_dets  # indices into high_boxes
        if still_unmatched and remaining_high_dets:
            # Dynamic C-BIoU: scale buffer by velocity, CAPPED at 25px
            # Reduced from 40px to prevent cross-person matching
            biou_trk_boxes = []
            buffers_used = []
            for t in still_unmatched:
                vel_mag = np.linalg.norm(t.vel)
                # Scale buffer proportionally with velocity:
                # Slow tracks (vel<2): 8px buffer (tight, no cross-matching)
                # Moderate tracks (vel~5): ~18px buffer
                # Fast tracks (vel>10): up to 50px buffer to catch up
                dyn_buffer = max(8, min(50, int(vel_mag * 3.5)))
                biou_trk_boxes.append(self._expand_box(t.predicted_box.tolist(), dyn_buffer))
                buffers_used.append(dyn_buffer)

            # Use max dynamic buffer for detections
            max_buf = max(buffers_used) if buffers_used else 15
            biou_det_boxes = [self._expand_box(high_boxes[i], max_buf)
                              for i in remaining_high_dets]

            biou_iou = iou_matrix(biou_det_boxes, biou_trk_boxes)
            biou_cost = 1.0 - biou_iou

            if biou_cost.size > 0:
                row_idx, col_idx = linear_sum_assignment(biou_cost)
            else:
                row_idx, col_idx = [], []

            biou_matched_det = set()
            biou_matched_trk = set()
            for r, c in zip(row_idx, col_idx):
                if biou_iou[r, c] >= self.iou_thresh * 0.6:
                    orig_d = remaining_high_dets[r]
                    orig_i = high_idx[orig_d]
                    trk = still_unmatched[c]
                    det_box = boxes[orig_i]


                    # ── C-BIoU hard validation ────────────────────────
                    # C-BIoU is a fallback matcher. Validate before accepting.

                    # Check 0: Hard Track Locking during Collision & Recovery Lock
                    if trk.local_id in frozen_lids or trk.local_id in cooldown_lids:
                        continue # No fallback jumps allowed during collision freeze or recovery lock

                    # Check 1: Center distance must be reasonable

                    det_cx = (det_box[0] + det_box[2]) / 2
                    det_cy = (det_box[1] + det_box[3]) / 2
                    pred = trk.predicted_box
                    pred_cx = (pred[0] + pred[2]) / 2
                    pred_cy = (pred[1] + pred[3]) / 2
                    cdist = float(np.hypot(det_cx - pred_cx, det_cy - pred_cy))
                    if cdist > 200:
                        continue  # Too far — skip

                    # Check 2: Direction consistency for moving tracks
                    tv = trk.vel
                    tv_speed = float(np.linalg.norm(tv))
                    if tv_speed >= 1.5:
                        dv = np.array([det_cx - pred_cx, det_cy - pred_cy],
                                      dtype=np.float32)
                        dv_speed = float(np.linalg.norm(dv))
                        if dv_speed >= 1.0:
                            dot = float(np.dot(tv / tv_speed, dv / dv_speed))
                            if dot < -0.3:
                                continue  # Opposite direction — skip

                    # Check 3: Skip (appearance unreliable in C-BIoU scenarios)
                    # C-BIoU activates during occlusion/fast motion where
                    # appearance is most likely to be corrupted.

                    crop = crops[orig_i] if crops is not None else None
                    trk.update(boxes[orig_i], scores[orig_i], embeddings[orig_i],
                               crop=crop, min_quality_score=self.min_quality_score,
                               frame_shape=frame_shape, sibling_boxes=[],
                               frame_delta=frame_delta)
                    trk.last_assoc_cost = float(biou_cost[r, c])
                    trk.last_assoc_method = "C-BIoU"
                    trk.cbiou_buffer = buffers_used[c]
                    biou_matched_det.add(r)
                    biou_matched_trk.add(c)
                    print(f"  [TRACKER DEBUG] C-BIoU MATCH: local={trk.local_id} "
                          f"buffer={buffers_used[c]}px  cost={biou_cost[r, c]:.3f}")

            still_unmatched = [t for i, t in enumerate(still_unmatched)
                               if i not in biou_matched_trk]
            final_unmatched_dets = [d for i, d in enumerate(remaining_high_dets)
                                    if i not in biou_matched_det]

        # ── Mark unmatched tracks as lost ────────────────────────────
        for trk in still_unmatched:
            trk.mark_lost(frame_delta=frame_delta)

        # ── Initialize new tracks from unmatched high-conf dets ──────
        suppress_regions = suppress_regions or []
        new_tracks = []

        # Dynamic birth threshold: raise bar when scene is crowded
        active_count = sum(1 for t in self.tracks
                           if t.is_confirmed and t.time_since_update == 0)
        birth_thresh = self.high_thresh
        if active_count > 6:
            birth_thresh = min(0.7, self.high_thresh + 0.1)

        for d_idx in final_unmatched_dets:
            orig_i = high_idx[d_idx]
            if scores[orig_i] >= birth_thresh:
                det_box = boxes[orig_i]

                # Check suppression regions
                suppressed = False
                for region_box in suppress_regions:
                    if self._single_iou(det_box, region_box) > 0.15:
                        suppressed = True
                        break



                # Track-Aware NMS & Territorial Ownership:
                # Suppress births near active tracks or recently lost/frozen tracks' territorial corridors.
                if not suppressed:
                    for t in self.tracks:
                        # Case 1: Active established track
                        if t.is_confirmed and t.time_since_update == 0 and t.age > 5:
                            if self._single_iou(det_box, t.smooth_box.tolist()) > 0.30:
                                suppressed = True
                                print(f"  [TRACKER DEBUG] BIRTH SUPPRESSED: active track NMS (near local={t.local_id} age={t.age}). conf={scores[orig_i]:.2f}")
                                break
                        
                        # Case 2: Recently lost/frozen track with dynamic suppression window
                        elif t.is_confirmed and t.is_lost:
                            # 1. Dynamic suppression lifetime scaling (90-150 frames for historically stable tracks)
                            base_window = 45
                            history_bonus = min(80, int(t.total_hits / 3))  # up to 80 frames
                            rec_bonus = min(25, getattr(t, 'recovery_count', 0) * 10)  # up to 25 frames
                            conf_bonus = 25 if t.score >= 0.8 else 0
                            
                            suppression_window = max(45, min(150, base_window + history_bonus + rec_bonus + conf_bonus))
                            
                            if t.time_since_update <= suppression_window:
                                # 2. Territorial Ownership & Reverse Birth Assumption
                                # Define spatial region (smooth_box) and predicted trajectory corridor
                                ref_box = t.predicted_box.tolist() if t.predicted_box is not None else t.smooth_box.tolist()
                                
                                # A. Direct Spatial Box Overlap
                                iou = self._single_iou(det_box, ref_box)
                                
                                # B. Center Distance to Corridor
                                det_cx = (det_box[0] + det_box[2]) / 2.0
                                det_cy = (det_box[1] + det_box[3]) / 2.0
                                
                                # Estimate predicted track position along trajectory corridor
                                last_cx = (t.smooth_box[0] + t.smooth_box[2]) / 2.0
                                last_cy = (t.smooth_box[1] + t.smooth_box[3]) / 2.0
                                
                                vx, vy = t.vel[0], t.vel[1]
                                pred_cx = last_cx + vx * t.time_since_update
                                pred_cy = last_cy + vy * t.time_since_update
                                
                                import math
                                dist = math.hypot(det_cx - pred_cx, det_cy - pred_cy)
                                max_allowed_dist = min(350.0, 150.0 + 8.0 * t.time_since_update)
                                
                                # C. Motion Direction alignment: check if displacement vector matches velocity direction
                                disp_x = det_cx - last_cx
                                disp_y = det_cy - last_cy
                                disp_len = math.hypot(disp_x, disp_y)
                                vel_len = math.hypot(vx, vy)
                                
                                is_aligned = False
                                if vel_len > 0.5 and disp_len > 10:
                                    dot_product = (disp_x * vx + disp_y * vy) / (disp_len * vel_len)
                                    is_aligned = dot_product > 0.3  # Moving in the same general corridor direction
                                
                                # Decide if detection falls in territorial ownership zone
                                in_spatial_territory = iou > 0.12
                                in_trajectory_corridor = (dist < max_allowed_dist) and (is_aligned or vel_len <= 0.5)
                                
                                if in_spatial_territory or in_trajectory_corridor:
                                    suppressed = True
                                    print(f"  [TRACKER TERRITORY] BIRTH SUPPRESSED: inside lost local={t.local_id} "
                                          f"territory (hits={t.total_hits} rec={getattr(t, 'recovery_count', 0)} "
                                          f"window={suppression_window} frames={t.time_since_update} iou={iou:.2f} dist={dist:.1f}px). "
                                          f"Reverse birth assumption holds: prioritize recovery.")
                                    break

                if suppressed:
                    continue



                new_track = STrack(
                    det_box, scores[orig_i], embeddings[orig_i],
                    label="person",
                    confirm_threshold=self.confirm_threshold,
                    embedding_history_size=self.embedding_history,
                )
                new_tracks.append(new_track)
                is_zero = "YES" if np.sum(np.abs(embeddings[orig_i])) < 1e-6 else "NO"
                print(f"  [TRACKER DEBUG] NEW LOCAL TRACK SPAWNED: local={new_track.local_id} conf={scores[orig_i]:.2f} zero_emb={is_zero}")

        # ─────────────────────────────────────────────────────────────
        # STAGE 4: IDSR — ID Switch Rectification
        #          Recently removed track data is kept in history for
        #          the cognitive memory layer to use. Direct local_id
        #          reassignment is NOT done here — GlobalIdentityManager
        #          handles identity recovery at a higher layer.
        # ─────────────────────────────────────────────────────────────

        # ── IDSR: reuse old local_ids for new tracks ──────────────
        if new_tracks:
            self._idsr_rectify(new_tracks)

        self.tracks.extend(new_tracks)

        # ── Cleanup dead tracks ──────────────────────────────────────
        self._cleanup()
        return self._output()

    # ─── Fused Association (Appearance + IoU) ────────────────────────

    @staticmethod
    def _single_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _expand_box(box, buffer):
        """Expand bounding box by buffer pixels in each direction (C-BIoU)."""
        return [box[0] - buffer, box[1] - buffer,
                box[2] + buffer, box[3] + buffer]


    def _fused_associate(self, det_boxes, det_embs, det_scores, tracks,
                         frame_h=None, frozen_lids=None, cooldown_lids=None,
                         exit_trajectories=None, collision_partners=None, frame_count=None):
        """
        Hungarian matching using fused appearance + IoU cost matrix.

        PHASE 2 OPTIMIZATION: Full NumPy vectorization.
        All O(N×M) Python loops replaced with broadcasting + masking.
        Every cognitive constraint preserved exactly:
          - Direction-aware motion penalty (OC-SORT inspired)
          - Pseudo-depth quantization (PD-SORT inspired)
          - Adaptive appearance weight during partial occlusion
          - AMI (Ambiguous Match Improvement)
          - Frozen track gating
          - Cooldown trajectory commitment
          - Cross-partner isolation
        """
        import math
        frozen_lids = frozen_lids or set()
        cooldown_lids = cooldown_lids or set()
        exit_trajectories = exit_trajectories or {}
        collision_partners = collision_partners or {}
        n_det = len(det_boxes)
        n_trk = len(tracks)

        if n_det == 0 or n_trk == 0:
            return [], [], list(range(n_trk)), list(range(n_det)), np.empty((0, 0))

        GATE_VALUE = 1e5

        # ── Precompute track arrays (one pass, O(M)) ──────────────────
        # Build per-track scalar arrays instead of repeating lookups inside loops
        pred_boxes_list = [t.predicted_box.tolist() for t in tracks]
        pred_boxes = np.array(pred_boxes_list, dtype=np.float32)           # (M,4)
        det_boxes_arr = np.array(det_boxes, dtype=np.float32)              # (N,4)
        det_scores_arr = np.array(det_scores, dtype=np.float32)            # (N,)

        # Track centers (M,2) and detection centers (N,2)
        trk_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5              # (M,)
        trk_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5              # (M,)
        det_cx = (det_boxes_arr[:, 0] + det_boxes_arr[:, 2]) * 0.5        # (N,)
        det_cy = (det_boxes_arr[:, 1] + det_boxes_arr[:, 3]) * 0.5        # (N,)

        # Velocity arrays
        trk_vx = np.array([t.vel[0] for t in tracks], dtype=np.float32)   # (M,)
        trk_vy = np.array([t.vel[1] for t in tracks], dtype=np.float32)   # (M,)

        # Boolean masks for cognitive states
        is_frozen_arr   = np.array([t.local_id in frozen_lids   for t in tracks], dtype=bool)  # (M,)
        is_cooldown_arr = np.array([t.local_id in cooldown_lids for t in tracks], dtype=bool)  # (M,)
        track_has_emb   = np.array([t.get_averaged_embedding() is not None for t in tracks], dtype=bool)  # (M,)

        # Embeddings matrix (M,512) — use zeros where no embedding
        track_embs = np.zeros((n_trk, 512), dtype=np.float32)
        for j, t in enumerate(tracks):
            emb = t.get_averaged_embedding()
            if emb is not None:
                track_embs[j] = emb

        # ── IoU distance (N,M) ────────────────────────────────────────
        iou = iou_matrix(det_boxes, pred_boxes_list)                        # (N,M)
        iou_dist = 1.0 - iou                                               # (N,M)

        # ── Appearance distance (N,M) ─────────────────────────────────
        # det_embs: (N,512), track_embs: (M,512) — both L2-normalized
        # Cosine distance = 1 - (det @ trk.T)
        if det_embs.shape[0] > 0 and np.any(track_has_emb):
            sim = det_embs @ track_embs.T                                   # (N,M)
            app_dist = np.ones((n_det, n_trk), dtype=np.float32)
            app_dist[:, track_has_emb] = (1.0 - sim[:, track_has_emb])    # (N,M)
        else:
            app_dist = np.ones((n_det, n_trk), dtype=np.float32)

        # ── Compute trajectory-predicted centers for cooldown tracks ───
        # For each cooldown track with an exit trajectory, override its center
        # with the trajectory-projected position.
        traj_cx = trk_cx.copy()                                            # (M,)
        traj_cy = trk_cy.copy()                                            # (M,)
        traj_vx = trk_vx.copy()                                            # (M,)
        traj_vy = trk_vy.copy()                                            # (M,)
        has_traj = np.zeros(n_trk, dtype=bool)                             # (M,)

        for j, t in enumerate(tracks):
            if is_cooldown_arr[j] or is_frozen_arr[j]:
                traj = exit_trajectories.get(t.local_id)
                if traj is not None and frame_count is not None:
                    vx_t, vy_t = traj["velocity"]
                    frames_elapsed = max(1, frame_count - traj["freeze_frame"])
                    traj_cx[j] = traj["center"][0] + vx_t * frames_elapsed
                    traj_cy[j] = traj["center"][1] + vy_t * frames_elapsed
                    traj_vx[j] = vx_t
                    traj_vy[j] = vy_t
                    has_traj[j] = True

        # ── Center distance matrix (N,M) ─────────────────────────────
        # Vectorized: (det_cx[:,None] - traj_cx[None,:]) → (N,M)
        diff_cx = det_cx[:, None] - traj_cx[None, :]                       # (N,M)
        diff_cy = det_cy[:, None] - traj_cy[None, :]                       # (N,M)
        center_dist = np.sqrt(diff_cx**2 + diff_cy**2)                     # (N,M)

        # ── Adaptive appearance weight (N,M) ──────────────────────────
        # Start from base weight, then modify per cognitive state
        eff_app_w = np.where(
            is_frozen_arr[None, :],
            0.0,                                                            # frozen → no appearance
            np.where(
                is_cooldown_arr[None, :],
                np.minimum(self.appearance_weight * 1.5, 0.95),             # cooldown → boosted
                np.where(
                    (iou > 0.15) & (iou < 0.5),
                    self.appearance_weight * 0.5,                           # partial occlusion → reduced
                    self.appearance_weight                                   # normal
                )
            )
        ).astype(np.float32)                                               # (N,M)

        # ── Base fused cost (N,M) ─────────────────────────────────────
        # For cooldown tracks with trajectory: use trajectory distance cost
        # For all others: use fused appearance + IoU
        cost_eff_app_w_cd = 0.6  # cooldown trajectory commitment weight
        dist_cost = np.minimum(1.0, center_dist / 80.0)                    # (N,M) — normalized dist

        # cooldown+trajectory mask
        cd_traj_mask = is_cooldown_arr[None, :] & has_traj[None, :]       # (N,M) broadcast

        cost = np.where(
            cd_traj_mask,
            cost_eff_app_w_cd * app_dist + (1 - cost_eff_app_w_cd) * dist_cost,
            eff_app_w * app_dist + (1 - eff_app_w) * iou_dist
        ).astype(np.float32)                                               # (N,M)

        # ── Frozen track spatial gating (vectorized) ──────────────────
        # Gate out frozen track cells where detection is too far
        # (iou < 0.20 AND center_dist > 45)
        frozen_gate_mask = (
            is_frozen_arr[None, :]                                          # track is frozen
            & (iou < 0.20)                                                  # low IoU
            & (center_dist > 45.0)                                          # far away
        )                                                                   # (N,M)
        cost = np.where(frozen_gate_mask, GATE_VALUE, cost)

        # ── Cooldown trajectory corridor gating (vectorized) ──────────
        # Gate out cooldown cells where detection is outside the trajectory corridor
        has_strong_visual = (
            track_has_emb[None, :] & (app_dist < 0.22)
        )                                                                   # (N,M)
        max_corridor = np.where(has_strong_visual, 120.0, 50.0)            # (N,M)

        cooldown_corridor_gate = (
            is_cooldown_arr[None, :]
            & has_traj[None, :]
            & (center_dist > max_corridor)
        )
        cost = np.where(cooldown_corridor_gate, GATE_VALUE, cost)

        # ── Cooldown velocity direction gating (vectorized) ───────────
        # Require motion direction consistency for cooldown tracks
        vel_len = np.sqrt(traj_vx**2 + traj_vy**2)                        # (M,)
        disp_len = center_dist                                              # (N,M) reuse

        # Normalized dot product of detection displacement with track velocity
        # Only compute where vel_len > 0.5 and disp_len > 10
        valid_vel = (vel_len > 0.5)[None, :] & (disp_len > 10.0)          # (N,M)
        if np.any(valid_vel):
            # Vectorized dot: (diff_cx * traj_vx + diff_cy * traj_vy) / (disp_len * vel_len)
            dot_num = diff_cx * traj_vx[None, :] + diff_cy * traj_vy[None, :]  # (N,M)
            dot_denom = np.maximum(disp_len, 1e-6) * np.maximum(vel_len[None, :], 1e-6)
            dot_prod = dot_num / dot_denom                                  # (N,M)

            min_dot = np.where(has_strong_visual, -0.2, 0.6)               # (N,M)
            cooldown_vel_gate = (
                is_cooldown_arr[None, :]
                & valid_vel
                & (dot_prod < min_dot)
            )
            cost = np.where(cooldown_vel_gate, GATE_VALUE, cost)

        # ── Cooldown appearance gate (vectorized) ─────────────────────
        cooldown_app_gate = (
            is_cooldown_arr[None, :]
            & track_has_emb[None, :]
            & (app_dist > 0.28)
        )
        cost = np.where(cooldown_app_gate, GATE_VALUE, cost)

        # ── Cross-partner isolation (per-pair, sparse Python loop) ────
        # This is inherently sparse (only frozen/cooldown tracks with partners)
        # so it doesn't scale badly — most tracks have no partners.
        for j, t in enumerate(tracks):
            if not (is_frozen_arr[j] or is_cooldown_arr[j]):
                continue
            partners = collision_partners.get(t.local_id, set())
            if not partners:
                continue
            for k, t_k in enumerate(tracks):
                if k == j or t_k.local_id not in partners:
                    continue
                # Partner track k: compute distances from each det to k's position
                cx_k = traj_cx[k]
                cy_k = traj_cy[k]
                dist_to_j = center_dist[:, j]                              # (N,)
                dist_to_k = np.sqrt(
                    (det_cx - cx_k)**2 + (det_cy - cy_k)**2)              # (N,)

                # Gate cells where partner k is closer to the detection
                crossing_mask = dist_to_k < dist_to_j - 20.0              # (N,) bool
                cost[crossing_mask, j] = GATE_VALUE

                # Additional cooldown appearance check
                if is_cooldown_arr[j] and track_has_emb[j] and track_has_emb[k]:
                    app_prefer_k = app_dist[:, k] < app_dist[:, j] - 0.05  # (N,)
                    cost[app_prefer_k, j] = GATE_VALUE

        # ── Pseudo-Depth Quantization (vectorized) ────────────────────
        # Partition by bottom-edge y-position as depth proxy.
        if frame_h is not None and frame_h > 0:
            det_depth = det_boxes_arr[:, 3] / frame_h                      # (N,) 0=far 1=near
            trk_depth = pred_boxes[:, 3] / frame_h                        # (M,)
            depth_diff = np.abs(det_depth[:, None] - trk_depth[None, :])  # (N,M)
            cost = np.where(depth_diff > 0.25, cost + 0.5, cost)
            cost = np.where((depth_diff > 0.15) & (depth_diff <= 0.25), cost + 0.15, cost)

        # ── Appearance + Spatial Gating (vectorized) ──────────────────
        no_iou = iou < 0.001                                               # (N,M)
        vel_mag = np.sqrt(trk_vx**2 + trk_vy**2)                          # (M,)

        # Allow zero-IoU if: strong appearance + actually moving + reasonable distance
        good_app_fast = (
            track_has_emb[None, :]
            & (app_dist <= self.max_cosine_dist)
            & (vel_mag[None, :] > 2.0)
        )                                                                   # (N,M)
        max_cdist_allowed = np.minimum(120, np.maximum(40, vel_mag * 8.0)) # (M,)
        too_far = center_dist > max_cdist_allowed[None, :]                 # (N,M)

        # Gate zero-IoU cells: allow if good appearance+speed+distance, else gate
        cost = np.where(
            no_iou & ~(good_app_fast & ~too_far),
            GATE_VALUE, cost
        )

        # Low overlap + bad appearance gate
        bad_app_gate = (
            track_has_emb[None, :]
            & (app_dist > self.max_cosine_dist)
            & (iou < 0.5)
        )
        cost = np.where(bad_app_gate, GATE_VALUE, cost)

        # Crowd disambiguation: nearby but different looking
        crowd_gate = (
            track_has_emb[None, :]
            & (iou > 0.1) & (iou < 0.45)
            & (app_dist > 0.28)
        )
        cost = np.where(crowd_gate, GATE_VALUE, cost)

        # ── Hungarian matching ────────────────────────────────────────
        if cost.size > 0:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)

        matched_t, matched_d = [], []
        unmatched_d = set(range(n_det))
        unmatched_t = set(range(n_trk))

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < GATE_VALUE:
                # ── AMI: Ambiguous Match Improvement ─────────────────
                reject = False
                if n_trk > 1:
                    row_costs = cost[r, :]
                    sorted_costs = np.sort(row_costs)
                    valid = sorted_costs[sorted_costs < GATE_VALUE]
                    if len(valid) >= 2:
                        gap = valid[1] - valid[0]
                        if gap < self.ambiguity_margin:
                            reject = True
                if n_det > 1 and not reject:
                    col_costs = cost[:, c]
                    sorted_costs = np.sort(col_costs)
                    valid = sorted_costs[sorted_costs < GATE_VALUE]
                    if len(valid) >= 2:
                        gap = valid[1] - valid[0]
                        if gap < self.ambiguity_margin:
                            reject = True

                if not reject:
                    matched_d.append(r)
                    matched_t.append(c)
                    unmatched_d.discard(r)
                    unmatched_t.discard(c)

        return matched_t, matched_d, list(unmatched_t), list(unmatched_d), cost


        if n_det == 0 or n_trk == 0:
            return [], [], list(range(n_trk)), list(range(n_det)), np.empty((0, 0))

        # ── IoU distance ─────────────────────────────────────────────
        pred_boxes = [t.predicted_box.tolist() for t in tracks]
        iou = iou_matrix(det_boxes, pred_boxes)
        iou_dist = 1.0 - iou

        # ── Appearance distance (cosine) ─────────────────────────────
        app_dist = np.ones((n_det, n_trk), dtype=np.float32)

        track_embs = []
        track_has_emb = []
        for t in tracks:
            emb = t.get_averaged_embedding()
            if emb is not None:
                track_embs.append(emb)
                track_has_emb.append(True)
            else:
                track_embs.append(np.zeros(512, dtype=np.float32))
                track_has_emb.append(False)

        if det_embs.shape[0] > 0 and any(track_has_emb):
            track_emb_matrix = np.array(track_embs, dtype=np.float32)
            sim = det_embs @ track_emb_matrix.T
            cos_dist = 1.0 - sim
            for j in range(n_trk):
                if track_has_emb[j]:
                    app_dist[:, j] = cos_dist[:, j]

        # ── Precompute closest track index for each detection ────────
        closest_track_for_det = []
        import math
        for i in range(n_det):
            det_cx = (det_boxes[i][0] + det_boxes[i][2]) / 2.0
            det_cy = (det_boxes[i][1] + det_boxes[i][3]) / 2.0
            min_dist = float('inf')
            best_k = -1
            for k in range(n_trk):
                traj_k = exit_trajectories.get(tracks[k].local_id)
                if traj_k is not None and frame_count is not None:
                    vx_k, vy_k = traj_k["velocity"]
                    freeze_frame_k = traj_k["freeze_frame"]
                    frames_elapsed_k = max(1, frame_count - freeze_frame_k)
                    pred_cx_k = traj_k["center"][0] + vx_k * frames_elapsed_k
                    pred_cy_k = traj_k["center"][1] + vy_k * frames_elapsed_k
                else:
                    pred_cx_k = (pred_boxes[k][0] + pred_boxes[k][2]) / 2.0
                    pred_cy_k = (pred_boxes[k][1] + pred_boxes[k][3]) / 2.0
                
                dist_k = math.hypot(det_cx - pred_cx_k, det_cy - pred_cy_k)
                if dist_k < min_dist:
                    min_dist = dist_k
                    best_k = k
            closest_track_for_det.append(best_k)

        # ── Fused cost matrix (with per-pair adaptive app weight) ────
        cost = np.empty((n_det, n_trk), dtype=np.float32)
        GATE_VALUE = 1e5
        
        for i in range(n_det):
            for j in range(n_trk):
                is_frozen = tracks[j].local_id in frozen_lids
                is_cooldown = tracks[j].local_id in cooldown_lids
                
                # Occlusion-Aware Embedding Protection:
                # During partial occlusion (IoU 0.15-0.5), appearance is unreliable.
                # Reduce appearance weight to trust motion/position more.
                if is_frozen:
                    # HARD TRACK LOCKING: Disable appearance similarity alone for frozen participants
                    eff_app_w = 0.0
                elif is_cooldown:
                    # RECOVERY LOCK: Heavily bias previous appearance (keep eff_app_w high but restrict mismatch)
                    eff_app_w = self.appearance_weight * 1.5
                    if eff_app_w > 0.95:
                        eff_app_w = 0.95
                elif 0.15 < iou[i, j] < 0.5:
                    eff_app_w = self.appearance_weight * 0.5
                else:
                    eff_app_w = self.appearance_weight
                    
                det_cx = (det_boxes[i][0] + det_boxes[i][2]) / 2.0
                det_cy = (det_boxes[i][1] + det_boxes[i][3]) / 2.0
                
                dist_to_pred = 0.0
                traj = exit_trajectories.get(tracks[j].local_id)
                if is_cooldown and traj is not None and frame_count is not None:
                    # Predict future position along pre-collision exit trajectory corridor
                    vx, vy = traj["velocity"]
                    freeze_frame = traj["freeze_frame"]
                    frames_elapsed = max(1, frame_count - freeze_frame)
                    pred_cx = traj["center"][0] + vx * frames_elapsed
                    pred_cy = traj["center"][1] + vy * frames_elapsed
                    
                    dist_to_pred = math.hypot(det_cx - pred_cx, det_cy - pred_cy)
                    trk_cx, trk_cy = pred_cx, pred_cy
                else:
                    vx, vy = tracks[j].vel
                    trk_cx = (pred_boxes[j][0] + pred_boxes[j][2]) / 2.0
                    trk_cy = (pred_boxes[j][1] + pred_boxes[j][3]) / 2.0
                    dist_to_pred = math.hypot(det_cx - trk_cx, det_cy - trk_cy)

                if is_cooldown and traj is not None:
                    # Trajectory Commitment Cost: Trust predicted trajectory center distance over raw IoU
                    dist_cost = min(1.0, dist_to_pred / 80.0)
                    cost_eff_app_w = 0.6
                    cost[i, j] = (cost_eff_app_w * app_dist[i, j] +
                                  (1 - cost_eff_app_w) * dist_cost)
                else:
                    cost[i, j] = (eff_app_w * app_dist[i, j] +
                                  (1 - eff_app_w) * iou_dist[i, j])

                if is_frozen:
                    # Restrict association to very tight trajectory corridor
                    if iou[i, j] < 0.20 and dist_to_pred > 45.0:
                        cost[i, j] = GATE_VALUE
                        continue

                    # Collision Participant Isolation during Freeze:
                    partners = collision_partners.get(tracks[j].local_id, set())
                    for k in range(n_trk):
                        if k != j and tracks[k].local_id in partners:
                            traj_k = exit_trajectories.get(tracks[k].local_id)
                            if traj_k is not None and frame_count is not None:
                                vx_k, vy_k = traj_k["velocity"]
                                freeze_frame_k = traj_k["freeze_frame"]
                                frames_elapsed_k = max(1, frame_count - freeze_frame_k)
                                pred_cx_k = traj_k["center"][0] + vx_k * frames_elapsed_k
                                pred_cy_k = traj_k["center"][1] + vy_k * frames_elapsed_k
                                trk_cx_k, trk_cy_k = pred_cx_k, pred_cy_k
                            else:
                                trk_cx_k = (pred_boxes[k][0] + pred_boxes[k][2]) / 2.0
                                trk_cy_k = (pred_boxes[k][1] + pred_boxes[k][3]) / 2.0
                                
                            dist_k = math.hypot(det_cx - trk_cx_k, det_cy - trk_cy_k)
                            dist_j = math.hypot(det_cx - trk_cx, det_cy - trk_cy)
                            
                            # Forbid crossing identity exchange/stealing with a 20px margin
                            if dist_k < dist_j - 20.0:
                                cost[i, j] = GATE_VALUE
                                break

                elif is_cooldown:
                    # Determine if this detection is an extremely strong visual match
                    has_strong_visual_match = track_has_emb[j] and app_dist[i, j] < 0.22

                    # RECOVERY LOCK constraints
                    # 1. Flexible Trajectory Commitment (loosen to 120px if visual match is strong)
                    max_corridor = 120.0 if has_strong_visual_match else 50.0
                    if dist_to_pred > max_corridor:
                        cost[i, j] = GATE_VALUE
                        continue
                        
                    # 2. Strongly prefer historical trajectory / motion direction (loosen if appearance is extremely strong)
                    vel_len = math.hypot(vx, vy)
                    disp_x = det_cx - trk_cx
                    disp_y = det_cy - trk_cy
                    disp_len = math.hypot(disp_x, disp_y)
                    
                    if vel_len > 0.5 and disp_len > 10.0:
                        dot_prod = (disp_x * vx + disp_y * vy) / (disp_len * vel_len)
                        min_dot = -0.2 if has_strong_visual_match else 0.6
                        if dot_prod < min_dot:  # Allow unexpected turns if appearance matches perfectly
                            cost[i, j] = GATE_VALUE
                            continue
                            
                    # 3. Heavily bias previous appearance history (strictly reject mismatch)
                    if track_has_emb[j] and app_dist[i, j] > 0.28:
                        cost[i, j] = GATE_VALUE
                        continue
                        
                    # 4. Crossing-Trajectory Rejection (reject stealing/exchanging identities)
                    partners = collision_partners.get(tracks[j].local_id, set())
                    for k in range(n_trk):
                        if k != j and tracks[k].local_id in partners:
                            traj_k = exit_trajectories.get(tracks[k].local_id)
                            if traj_k is not None and frame_count is not None:
                                vx_k, vy_k = traj_k["velocity"]
                                freeze_frame_k = traj_k["freeze_frame"]
                                frames_elapsed_k = max(1, frame_count - freeze_frame_k)
                                pred_cx_k = traj_k["center"][0] + vx_k * frames_elapsed_k
                                pred_cy_k = traj_k["center"][1] + vy_k * frames_elapsed_k
                                trk_cx_k, trk_cy_k = pred_cx_k, pred_cy_k
                            else:
                                trk_cx_k = (pred_boxes[k][0] + pred_boxes[k][2]) / 2.0
                                trk_cy_k = (pred_boxes[k][1] + pred_boxes[k][3]) / 2.0
                                
                            dist_k = math.hypot(det_cx - trk_cx_k, det_cy - trk_cy_k)
                            dist_j = math.hypot(det_cx - trk_cx, det_cy - trk_cy)
                            
                            # Forbid crossing identity exchange/stealing with a 20px margin
                            if dist_k < dist_j - 20.0:
                                cost[i, j] = GATE_VALUE
                                break
                                
                            if track_has_emb[k] and track_has_emb[j]:
                                if app_dist[i, k] < app_dist[i, j] - 0.05:
                                    cost[i, j] = GATE_VALUE
                                    break
        # NOTE: Hard distance/direction gating removed from main fused
        # association. The fused IoU+appearance cost already handles match
        # quality well. Hard gates here caused more fragmentation than
        # they prevented wrong matches. Direction/distance checks remain
        # active in the C-BIoU fallback stage (see above).


        # ── Pseudo-Depth Quantization (PD-SORT inspired) ─────────────
        # Partition by bottom-edge y-position as depth proxy.
        # Objects at the bottom of the frame are closer to the camera.
        # Cross-depth matches get penalized to prevent FG/BG ID swaps.
        if frame_h is not None and frame_h > 0:
            for i in range(n_det):
                det_depth = det_boxes[i][3] / frame_h  # 0=top(far), 1=bottom(near)
                for j in range(n_trk):
                    trk_depth = pred_boxes[j][3] / frame_h
                    depth_diff = abs(det_depth - trk_depth)
                    if depth_diff > 0.25:
                        cost[i, j] += 0.5  # Strong cross-depth penalty
                    elif depth_diff > 0.15:
                        cost[i, j] += 0.15  # Mild cross-depth penalty

        # ── Appearance + Spatial Gating ─────────────────────────────────
        for i in range(n_det):
            for j in range(n_trk):
                if cost[i, j] >= GATE_VALUE:
                    continue  # Already gated by direction/distance

                # No spatial overlap: allow if appearance is strong enough
                if iou[i, j] < 0.001:
                    vel_mag_j = float(np.linalg.norm(tracks[j].vel))
                    if track_has_emb[j] and app_dist[i, j] <= self.max_cosine_dist and vel_mag_j > 2.0:
                        det_cx_g = (det_boxes[i][0] + det_boxes[i][2]) / 2.0
                        det_cy_g = (det_boxes[i][1] + det_boxes[i][3]) / 2.0
                        pred_cx_g = (pred_boxes[j][0] + pred_boxes[j][2]) / 2.0
                        pred_cy_g = (pred_boxes[j][1] + pred_boxes[j][3]) / 2.0
                        import math
                        cdist_g = math.hypot(det_cx_g - pred_cx_g, det_cy_g - pred_cy_g)
                        max_cdist = min(120, max(40, vel_mag_j * 8.0))
                        if cdist_g > max_cdist:
                            print(f"[TRACE] REJECT (cdist > max_cdist): local={tracks[j].local_id} iou={iou[i,j]:.3f} app={app_dist[i,j]:.3f}")
                            cost[i, j] = GATE_VALUE  # Too far even with good appearance
                    else:
                        print(f"[TRACE] REJECT (zero IoU, bad app/slow): local={tracks[j].local_id} iou={iou[i,j]:.3f} app={app_dist[i,j]:.3f} vel={vel_mag_j:.1f}")
                        cost[i, j] = GATE_VALUE  # No appearance, bad appearance, or stationary

                # Low overlap + bad appearance = wrong match
                if track_has_emb[j] and app_dist[i, j] > self.max_cosine_dist:
                    if iou[i, j] < 0.5:
                        print(f"[TRACE] REJECT (low overlap + bad app): local={tracks[j].local_id} iou={iou[i,j]:.3f} app={app_dist[i,j]:.3f}")
                        cost[i, j] = GATE_VALUE

                # Crowd disambiguation: nearby but different looking
                if track_has_emb[j] and 0.1 < iou[i, j] < 0.45:
                    if app_dist[i, j] > 0.28:
                        print(f"[TRACE] REJECT (crowd bad app): local={tracks[j].local_id} iou={iou[i,j]:.3f} app={app_dist[i,j]:.3f}")
                        cost[i, j] = GATE_VALUE

        # ── Hungarian matching ───────────────────────────────────────
        if cost.size > 0:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)

        matched_t, matched_d = [], []
        unmatched_d = set(range(n_det))
        unmatched_t = set(range(n_trk))

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < GATE_VALUE:
                # ── AMI: Ambiguous Match Improvement ─────────────────
                # Check if this match is sufficiently unambiguous.
                # If the gap between this cost and the second-best
                # alternative is too small, reject both to avoid
                # forcing a bad ID decision.
                reject = False
                if n_trk > 1:
                    # Second-best track for this detection
                    row_costs = cost[r, :]
                    sorted_costs = np.sort(row_costs)
                    valid = sorted_costs[sorted_costs < GATE_VALUE]
                    if len(valid) >= 2:
                        gap = valid[1] - valid[0]
                        if gap < self.ambiguity_margin:
                            reject = True
                if n_det > 1 and not reject:
                    # Second-best detection for this track
                    col_costs = cost[:, c]
                    sorted_costs = np.sort(col_costs)
                    valid = sorted_costs[sorted_costs < GATE_VALUE]
                    if len(valid) >= 2:
                        gap = valid[1] - valid[0]
                        if gap < self.ambiguity_margin:
                            reject = True

                if reject:
                    print(f"[TRACE] REJECT (AMI ambiguous match): local={tracks[c].local_id}")

                if not reject:
                    matched_d.append(r)
                    matched_t.append(c)
                    unmatched_d.discard(r)
                    unmatched_t.discard(c)

        return matched_t, matched_d, list(unmatched_t), list(unmatched_d), cost

    # ─── IoU-only Association (Stage 2) ──────────────────────────────


    @staticmethod
    def _iou_associate(det_boxes, tracks, iou_threshold, frozen_lids=None, cooldown_lids=None, collision_partners=None):
        """Pure IoU Hungarian matching for low-confidence recovery."""
        frozen_lids = frozen_lids or set()
        cooldown_lids = cooldown_lids or set()
        collision_partners = collision_partners or {}
        n_det = len(det_boxes)
        n_trk = len(tracks)

        if n_det == 0 or n_trk == 0:
            return [], [], list(range(n_trk)), list(range(n_det)), np.empty((0, 0))

        pred_boxes = [t.predicted_box.tolist() for t in tracks]
        iou = iou_matrix(det_boxes, pred_boxes)
        cost = 1.0 - iou

        if cost.size > 0:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)

        matched_t, matched_d = [], []
        unmatched_d = set(range(n_det))
        unmatched_t = set(range(n_trk))

        for r, c in zip(row_idx, col_idx):
            # Tighten IoU requirement for frozen/cooldown tracks to prevent theft
            is_frozen = tracks[c].local_id in frozen_lids
            is_cooldown = tracks[c].local_id in cooldown_lids
            req_iou = 0.6 if (is_frozen or is_cooldown) else iou_threshold

            # If in cooldown, check collision partners to prevent track swapping
            if is_cooldown:
                partners = collision_partners.get(tracks[c].local_id, set())
                should_skip = False
                for k in range(n_trk):
                    if k != c and tracks[k].local_id in partners:
                        # Compare IoU overlap with partner track k
                        if iou[r, k] > iou[r, c] + 0.05:
                            should_skip = True
                            break
                if should_skip:
                    continue

            if iou[r, c] >= req_iou:
                matched_d.append(r)
                matched_t.append(c)
                unmatched_d.discard(r)
                unmatched_t.discard(c)


        return matched_t, matched_d, list(unmatched_t), list(unmatched_d), cost

    # ─── IDSR: ID Switch Rectification ───────────────────────────────

    def _idsr_rectify(self, new_tracks):
        """
        Check if newly created tracks match recently removed tracks.
        If a new track's position aligns with the trajectory projection
        of a recently removed track, transfer the old local_id to prevent
        fragmentation.
        """
        if not self._recently_removed:
            return

        used_removed = set()
        for new_trk in new_tracks:
            new_cx = (new_trk.box[0] + new_trk.box[2]) / 2
            new_cy = (new_trk.box[1] + new_trk.box[3]) / 2
            new_aspect = STrack._box_aspect(new_trk.box)

            best_match = None
            best_dist = float('inf')

            for idx, (rem_box, rem_vel, rem_aspect, rem_lid, rem_age) \
                    in enumerate(self._recently_removed):
                if idx in used_removed:
                    continue

                # Project removed track's position forward
                proj_cx = (rem_box[0] + rem_box[2]) / 2 + rem_vel[0] * rem_age
                proj_cy = (rem_box[1] + rem_box[3]) / 2 + rem_vel[1] * rem_age

                dist = ((new_cx - proj_cx) ** 2 + (new_cy - proj_cy) ** 2) ** 0.5

                # Aspect ratio check (reject if body shape changed too much)
                aspect_diff = abs(new_aspect - rem_aspect) / max(rem_aspect, 0.1)
                if aspect_diff > 0.4:
                    continue

                if dist < 80 and dist < best_dist:  # 80px threshold
                    best_dist = dist
                    best_match = idx

            if best_match is not None:
                # Recover old local_id to maintain identity continuity
                _, _, _, old_lid, _ = self._recently_removed[best_match]
                new_trk.local_id = old_lid
                used_removed.add(best_match)

        # Remove used entries
        self._recently_removed = [
            r for i, r in enumerate(self._recently_removed)
            if i not in used_removed]

    # ─── Track Management ────────────────────────────────────────────

    def _cleanup(self):
        """Remove dead tracks. Save to IDSR history before removal."""
        survivors = []
        for t in self.tracks:
            if t.should_remove(self.max_lost):
                print(f"  [TRACKER DEBUG] TRACK DIED: local={t.local_id} hits={t.total_hits} age={t.age}")
                # Save to IDSR history for potential rectification
                if t.total_hits >= 3:  # Only save meaningful tracks
                    self._recently_removed.append((
                        t._last_observed_box.tolist(),
                        t._last_observed_vel.tolist(),
                        STrack._box_aspect(t._last_observed_box),
                        t.local_id,
                        t._frames_since_observation,
                    ))
            else:
                survivors.append(t)
        self.tracks = survivors

        # Age out IDSR history
        aged = []
        for entry in self._recently_removed:
            rem_box, rem_vel, rem_aspect, rem_lid, rem_age = entry
            aged.append((rem_box, rem_vel, rem_aspect, rem_lid, rem_age + 1))
        self._recently_removed = [
            e for e in aged if e[4] <= self._max_removed_history]

    def _output(self):
        """Return confirmed tracks using local_id as key."""
        result = {}
        for t in self.tracks:
            if t.is_confirmed and t.time_since_update <= 15:
                if t.time_since_update > 0:
                    result[t.local_id] = t.predicted_box.tolist()
                else:
                    # Use raw detection box — smooth_box lags during fast motion,
                    # causing the drawn box to trail behind the person visually
                    result[t.local_id] = t.box.tolist()
        return result

    # ─── Info ────────────────────────────────────────────────────────

    @property
    def track_count(self):
        return len(self.tracks)

    @property
    def confirmed_count(self):
        return sum(1 for t in self.tracks if t.is_confirmed)

    @property
    def lost_count(self):
        return sum(1 for t in self.tracks if t.is_lost)
