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
        self.max_lost = max_lost
        self.confirm_threshold = confirm_threshold
        self.embedding_history = embedding_history
        self.min_quality_score = min_quality_score
        self.biou_buffer = biou_buffer
        self.ambiguity_margin = ambiguity_margin

        self.tracks = []   # All active tracks (confirmed + tentative + lost)

        # IDSR: recently removed tracks for post-hoc rectification
        self._recently_removed = []  # list of (box, vel, aspect, local_id, age)
        self._max_removed_history = 30  # max items to keep

    # ─── Public API ──────────────────────────────────────────────────

    def update(self, boxes, scores, embeddings, crops=None, frame_shape=None,
               frame_delta=1, suppress_regions=None):
        """
        Run one frame of StrongSORT+ tracking.

        PURE TRACKER: no brain, no phantom, no identity mutation.
        Returns dict {local_id: [x1, y1, x2, y2]} for confirmed, active tracks.
        """
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
                frame_h=frame_h
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
                    frame_h=frame_h
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
                self._iou_associate(low_boxes, remaining_tracks, self.iou_thresh)

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
            # Dynamic C-BIoU: scale buffer by velocity magnitude, min 10, max 40
            biou_trk_boxes = []
            buffers_used = []
            for t in still_unmatched:
                vel_mag = np.linalg.norm(t.vel)
                dyn_buffer = max(10, min(40, int(vel_mag * 2.5)))
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

                if suppressed:
                    print(f"  [TRACKER DEBUG] BIRTH SUPPRESSED: overlap with region. conf={scores[orig_i]:.2f}")
                    continue

                new_track = STrack(
                    det_box, scores[orig_i], embeddings[orig_i],
                    label="person",
                    confirm_threshold=self.confirm_threshold,
                    embedding_history_size=self.embedding_history,
                )
                new_tracks.append(new_track)
                print(f"  [TRACKER DEBUG] NEW LOCAL TRACK SPAWNED: local={new_track.local_id} conf={scores[orig_i]:.2f}")

        # ─────────────────────────────────────────────────────────────
        # STAGE 4: IDSR — ID Switch Rectification
        #          Recently removed track data is kept in history for
        #          the cognitive memory layer to use. Direct local_id
        #          reassignment is NOT done here — GlobalIdentityManager
        #          handles identity recovery at a higher layer.
        # ─────────────────────────────────────────────────────────────

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
                         frame_h=None):
        """
        Hungarian matching using fused appearance + IoU cost matrix.

        Research-driven enhancements:
          - Direction-aware motion penalty (OC-SORT inspired)
          - Pseudo-depth quantization (PD-SORT inspired)
          - Adaptive appearance weight during partial occlusion
          - AMI (Ambiguous Match Improvement)
        """
        n_det = len(det_boxes)
        n_trk = len(tracks)

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

        # ── Fused cost matrix (with per-pair adaptive app weight) ────
        cost = np.empty((n_det, n_trk), dtype=np.float32)
        for i in range(n_det):
            for j in range(n_trk):
                # Occlusion-Aware Embedding Protection:
                # During partial occlusion (IoU 0.15-0.5), appearance is unreliable.
                # Reduce appearance weight to trust motion/position more.
                if 0.15 < iou[i, j] < 0.5:
                    eff_app_w = self.appearance_weight * 0.5
                else:
                    eff_app_w = self.appearance_weight
                cost[i, j] = (eff_app_w * app_dist[i, j] +
                              (1 - eff_app_w) * iou_dist[i, j])

        # ── Direction-Aware Motion Penalty (OC-SORT inspired) ────────
        # Penalize matches where detection velocity and track velocity
        # point in opposite directions — physically impossible for the
        # same person.
        for i in range(n_det):
            det_cx = (det_boxes[i][0] + det_boxes[i][2]) / 2
            det_cy = (det_boxes[i][1] + det_boxes[i][3]) / 2
            for j in range(n_trk):
                tv = tracks[j].vel
                tv_speed = float(np.linalg.norm(tv))
                if tv_speed < 0.5:
                    continue  # Track is nearly stationary, skip direction check

                # Compute implied detection velocity from predicted position
                pred = pred_boxes[j]
                pred_cx = (pred[0] + pred[2]) / 2
                pred_cy = (pred[1] + pred[3]) / 2
                dv = np.array([det_cx - pred_cx, det_cy - pred_cy],
                              dtype=np.float32)
                dv_speed = float(np.linalg.norm(dv))
                if dv_speed < 0.5:
                    continue  # Detection is very close to prediction, fine

                # Normalized dot product: 1=same direction, -1=opposite
                dot = float(np.dot(tv / tv_speed, dv / dv_speed))

                if dot < -0.3:
                    # Opposite direction: heavy penalty
                    cost[i, j] += 0.4
                elif dot < 0.1:
                    # Weak/perpendicular: mild penalty
                    cost[i, j] += 0.15

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

        # ── Gating ───────────────────────────────────────────────────
        GATE_VALUE = 1e5
        for i in range(n_det):
            for j in range(n_trk):
                # Gate on IoU: block when truly zero overlap
                if iou[i, j] < 0.001:
                    if tracks[j].is_lost and track_has_emb[j]:
                        if app_dist[i, j] > self.max_cosine_dist:
                            cost[i, j] = GATE_VALUE
                    else:
                        cost[i, j] = GATE_VALUE

                # Gate on appearance: too different
                if track_has_emb[j] and app_dist[i, j] > self.max_cosine_dist:
                    if iou[i, j] < 0.5:
                        cost[i, j] = GATE_VALUE

                # Crowd disambiguation
                if track_has_emb[j] and 0.1 < iou[i, j] < 0.45:
                    if app_dist[i, j] > 0.28:
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

                if not reject:
                    matched_d.append(r)
                    matched_t.append(c)
                    unmatched_d.discard(r)
                    unmatched_t.discard(c)

        return matched_t, matched_d, list(unmatched_t), list(unmatched_d), cost

    # ─── IoU-only Association (Stage 2) ──────────────────────────────

    @staticmethod
    def _iou_associate(det_boxes, tracks, iou_threshold):
        """Pure IoU Hungarian matching for low-confidence recovery."""
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
            if iou[r, c] >= iou_threshold:
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
            if t.is_confirmed and t.time_since_update <= 5:
                if t.time_since_update > 0:
                    result[t.local_id] = t.predicted_box.tolist()
                else:
                    result[t.local_id] = t.smooth_box.tolist()
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
