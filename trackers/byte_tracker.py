import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.iou import iou_matrix
from .strack import STrack

class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    Two-stage association:
      Stage 1: High-confidence detections ↔ all tracks (IoU-based Hungarian)
      Stage 2: Low-confidence detections  ↔ remaining unmatched tracks (IoU)

    Integrated with ReID IdentityMemory for persistent re-identification
    across track losses.

    Args:
        reid: ReIDExtractor instance for appearance embeddings.
        memory: IdentityMemory instance for long-term re-ID.
        high_thresh: Confidence threshold for first association (default 0.5).
        low_thresh: Confidence threshold for second association (default 0.1).
        iou_thresh: Minimum IoU to accept a match (default 0.3).
        max_lost: Max frames a track stays alive without match (default 30).
        min_hits: Min consecutive hits before a track is confirmed (default 3).
        new_id_cooldown: Frames to wait before assigning new ID (default 5).
    """

    def __init__(self, reid, memory,
                 high_thresh=0.5, low_thresh=0.1, iou_thresh=0.3,
                 max_lost=30, min_hits=3, new_id_cooldown=5):
        self.reid = reid
        self.memory = memory
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.new_id_cooldown = new_id_cooldown

        self.tracked = []    # active tracks
        self.lost = []       # recently lost tracks
        self._pending = []   # cooldown buffer for new IDs
        self.frame_id = 0

    def update(self, boxes, scores, embeddings, labels):
        """
        Run one frame of ByteTrack.

        Args:
            boxes: list of [x1,y1,x2,y2]
            scores: list of confidence floats
            embeddings: list of ReID embeddings (or None)
            labels: list of class label strings

        Returns:
            dict {track_id: [x1,y1,x2,y2]} for confirmed active tracks.
        """
        self.frame_id += 1
        # Predict all existing tracks
        for t in self.tracked + self.lost:
            t.predict()

        if len(boxes) == 0:
            # Age all tracks
            for t in self.tracked:
                t.mark_lost()
            self.lost.extend([t for t in self.tracked if t.time_since_update > 0])
            self.tracked = [t for t in self.tracked if t.time_since_update == 0]
            self._remove_dead()
            return self._output()

        boxes = [list(b) for b in boxes]
        scores = list(scores)

        # ────────────────────────────────────────────────────────────────
        # Split detections into HIGH and LOW confidence
        # ────────────────────────────────────────────────────────────────
        high_idx = [i for i, s in enumerate(scores) if s >= self.high_thresh]
        low_idx = [i for i, s in enumerate(scores)
                   if self.low_thresh <= s < self.high_thresh]

        high_boxes = [boxes[i] for i in high_idx]
        low_boxes = [boxes[i] for i in low_idx]

        # ────────────────────────────────────────────────────────────────
        # STAGE 1: High-conf detections ↔ ALL tracked + lost tracks
        # ────────────────────────────────────────────────────────────────
        all_tracks = self.tracked + self.lost
        matched_t, matched_d, unmatched_tracks, unmatched_dets = \
            self._associate(high_boxes, all_tracks, self.iou_thresh)

        # Update matched tracks
        for t_idx, d_idx in zip(matched_t, matched_d):
            trk = all_tracks[t_idx]
            orig_i = high_idx[d_idx]
            trk.update(boxes[orig_i], scores[orig_i], embeddings[orig_i])
            trk.is_activated = True
            # Move from lost back to tracked if re-found
            if trk in self.lost:
                self.lost.remove(trk)
                if trk not in self.tracked:
                    self.tracked.append(trk)

        # ────────────────────────────────────────────────────────────────
        # STAGE 2: Low-conf detections ↔ REMAINING unmatched tracks
        # ────────────────────────────────────────────────────────────────
        remaining_tracks = [all_tracks[i] for i in unmatched_tracks]

        if low_boxes and remaining_tracks:
            matched_t2, matched_d2, unmatched_tracks2, _ = \
                self._associate(low_boxes, remaining_tracks, self.iou_thresh)

            for t_idx, d_idx in zip(matched_t2, matched_d2):
                trk = remaining_tracks[t_idx]
                orig_i = low_idx[d_idx]
                trk.update(boxes[orig_i], scores[orig_i], embeddings[orig_i])
                trk.is_activated = True
                if trk in self.lost:
                    self.lost.remove(trk)
                    if trk not in self.tracked:
                        self.tracked.append(trk)
            # Update remaining to those still unmatched after stage 2
            still_unmatched = [remaining_tracks[i] for i in unmatched_tracks2]
        else:
            still_unmatched = remaining_tracks

        # ────────────────────────────────────────────────────────────────
        # Mark still-unmatched tracks as lost
        # ────────────────────────────────────────────────────────────────
        for trk in still_unmatched:
            trk.mark_lost()
            if trk in self.tracked:
                self.tracked.remove(trk)
                self.lost.append(trk)

        # ────────────────────────────────────────────────────────────────
        # Initialize new tracks from unmatched HIGH-conf detections
        # ────────────────────────────────────────────────────────────────
        for d_idx in unmatched_dets:
            orig_i = high_idx[d_idx]
            self._init_new_track(
                boxes[orig_i], scores[orig_i],
                embeddings[orig_i], labels[orig_i])

        # Cleanup dead tracks + archive to memory
        self._remove_dead()
        return self._output()

    def _associate(self, det_boxes, tracks, iou_threshold):
        """Hungarian matching based on IoU with predicted track positions."""
        if not det_boxes or not tracks:
            return [], [], list(range(len(tracks))), list(range(len(det_boxes)))

        pred_boxes = [t.predicted_box.tolist() for t in tracks]
        iou = iou_matrix(det_boxes, pred_boxes)
        cost = 1.0 - iou  # minimize cost = maximize IoU

        if cost.size > 0:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        matched_t, matched_d = [], []
        unmatched_d = set(range(len(det_boxes)))
        unmatched_t = set(range(len(tracks)))

        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= iou_threshold:
                matched_d.append(r)
                matched_t.append(c)
                unmatched_d.discard(r)
                unmatched_t.discard(c)

        return matched_t, matched_d, list(unmatched_t), list(unmatched_d)

    def _init_new_track(self, box, score, embedding, label):
        """Create new track or re-identify via ReID memory (with cooldown)."""
        # Try ReID match from memory
        matched_gid, sim = self.memory.query(embedding, candidate_key=None)
        if matched_gid is not None:
            # Re-identified — reuse the old ID
            trk = STrack(box, score, embedding, label)
            trk.id = matched_gid
            trk.is_activated = True
            self.memory.store(matched_gid, embedding, label)
            self.tracked.append(trk)
            return

        # Cooldown: wait N frames before confirming new identity
        best_iou = 0
        best_idx = -1
        for i, p in enumerate(self._pending):
            iou = iou_matrix([box], [p['box']])[0, 0]
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou > self.iou_thresh:
            p = self._pending[best_idx]
            p['count'] += 1
            p['box'], p['score'], p['emb'] = box, score, embedding
            p['last_frame'] = self.frame_id

            if p['count'] >= self.new_id_cooldown:
                trk = STrack(box, score, embedding, label)
                trk.is_activated = True
                self.memory.store(trk.id, embedding, label)
                self.tracked.append(trk)
                self._pending.pop(best_idx)
        else:
            self._pending.append({
                'box': box, 'score': score, 'emb': embedding, 
                'label': label, 'count': 1, 'last_frame': self.frame_id
            })

    def _remove_dead(self):
        """Archive dead tracks to memory, remove from lost list."""
        alive = []
        for trk in self.lost:
            if trk.time_since_update <= self.max_lost:
                alive.append(trk)
            else:
                # Archive embedding to memory before deletion
                if trk.embedding is not None:
                    self.memory.store(trk.id, trk.embedding,
                                      self.memory.get_label(trk.id))
        self.lost = alive

        # Also clean stale pending entries
        self._pending = [p for p in self._pending if self.frame_id - p['last_frame'] <= self.max_lost]

    def _output(self):
        """Return confirmed, active tracks as {id: [x1,y1,x2,y2]}."""
        result = {}
        for trk in self.tracked:
            if trk.is_activated and trk.time_since_update == 0:
                result[trk.id] = trk.smooth_box.tolist()
        return result
