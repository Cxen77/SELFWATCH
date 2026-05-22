import time
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def _cosine_dist(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 1.0
    return float(1.0 - (dot / norm))

class DeferredIdentityManager:
    """
    Implements Deferred Multi-Hypothesis Tracking.
    When tracks intersect, hard ID assignment is suspended.
    The system accumulates evidence (embeddings) over N frames,
    then resolves the identities retroactively using bipartite matching
    once the tracks separate.
    """
    def __init__(self, iou_thresh=0.4, min_frames=3, max_frames=15):
        self.iou_thresh = iou_thresh
        self.min_frames = min_frames
        self.max_frames = max_frames
        
        # cluster_id -> {
        #   'tracks_involved': set(track_ids),
        #   'pre_embeddings': {track_id: embedding},
        #   'post_candidates': {track_id: [embeddings]},
        #   'frames_elapsed': int
        # }
        self.active_clusters = {}
        self.next_cluster_id = 0
        
        # track_id -> cluster_id
        self.track_to_cluster = {}

        # Mappings overriding the tracker's IDs
        # original_id -> resolved_id
        self.resolved_aliases = {}

    def tick(self, tracks, brain):
        """
        Process the current frame's tracks.
        Returns a dictionary of overriding IDs for rendering.
        """
        active_track_ids = [t.id for t in tracks if t.is_confirmed]
        
        # 1. Clean up old aliases
        for tid in list(self.resolved_aliases.keys()):
            if tid not in active_track_ids:
                del self.resolved_aliases[tid]

        # 2. Detect Intersections and Create/Expand Clusters
        for i in range(len(tracks)):
            if not tracks[i].is_confirmed or tracks[i].embedding is None:
                continue
            for j in range(i + 1, len(tracks)):
                if not tracks[j].is_confirmed or tracks[j].embedding is None:
                    continue
                    
                tA = tracks[i]
                tB = tracks[j]
                
                iou = _iou(tA.smooth_box.tolist(), tB.smooth_box.tolist())
                if iou > self.iou_thresh:
                    self._handle_intersection(tA, tB, brain)

        # 3. Process Active Clusters (Accumulate evidence & Resolve)
        overrides = {}
        clusters_to_resolve = []
        
        for cid, cluster in list(self.active_clusters.items()):
            cluster['frames_elapsed'] += 1
            
            # Find all current tracks that belong to this cluster
            current_cluster_tracks = []
            for t in tracks:
                if t.id in self.track_to_cluster and self.track_to_cluster[t.id] == cid:
                    current_cluster_tracks.append(t)
            
            # Check if they have separated
            max_iou = 0.0
            for i in range(len(current_cluster_tracks)):
                for j in range(i + 1, len(current_cluster_tracks)):
                    iou = _iou(current_cluster_tracks[i].smooth_box.tolist(), 
                               current_cluster_tracks[j].smooth_box.tolist())
                    max_iou = max(max_iou, iou)

            separated = max_iou < (self.iou_thresh * 0.5)  # Hysteresis
            timeout = cluster['frames_elapsed'] >= self.max_frames
            ready = cluster['frames_elapsed'] >= self.min_frames
            
            if (separated and ready) or timeout:
                clusters_to_resolve.append(cid)
            else:
                # Still overlapping, accumulate evidence
                for t in current_cluster_tracks:
                    if t.embedding is not None:
                        if t.id not in cluster['post_candidates']:
                            cluster['post_candidates'][t.id] = []
                        cluster['post_candidates'][t.id].append(t.embedding.copy())
                    overrides[t.id] = "THINKING"

        # 4. Resolve Clusters
        for cid in clusters_to_resolve:
            new_aliases = self._resolve_cluster(cid)
            self.resolved_aliases.update(new_aliases)

        # 5. Apply resolved aliases to output
        final_overrides = overrides.copy()
        for t in tracks:
            if t.id in self.resolved_aliases:
                # Rewrite the tracker's ID internally so it persists
                old_id = t.id
                t.id = self.resolved_aliases[old_id]
                final_overrides[old_id] = t.id
                
                # We also need to update the tracker's internal dictionaries if needed,
                # but simply modifying t.id is usually enough for the next frame's rendering
                # and memory updates.
                
        return final_overrides

    def _handle_intersection(self, tA, tB, brain):
        """Put two tracks into a deferred decision cluster."""
        cid_A = self.track_to_cluster.get(tA.id)
        cid_B = self.track_to_cluster.get(tB.id)

        if cid_A is None and cid_B is None:
            # Create new cluster
            cid = self.next_cluster_id
            self.next_cluster_id += 1
            
            pre_emb_A = self._get_stable_embedding(tA.id, brain, tA.embedding)
            pre_emb_B = self._get_stable_embedding(tB.id, brain, tB.embedding)
            
            self.active_clusters[cid] = {
                'tracks_involved': {tA.id, tB.id},
                'pre_embeddings': {tA.id: pre_emb_A, tB.id: pre_emb_B},
                'post_candidates': {},
                'frames_elapsed': 0
            }
            self.track_to_cluster[tA.id] = cid
            self.track_to_cluster[tB.id] = cid
            
        elif cid_A is not None and cid_B is None:
            self._add_to_cluster(cid_A, tB, brain)
        elif cid_B is not None and cid_A is None:
            self._add_to_cluster(cid_B, tA, brain)
        elif cid_A != cid_B:
            # Merge clusters (complex multi-person overlap)
            self._merge_clusters(cid_A, cid_B)

    def _add_to_cluster(self, cid, track, brain):
        cluster = self.active_clusters[cid]
        if track.id not in cluster['tracks_involved']:
            cluster['tracks_involved'].add(track.id)
            pre_emb = self._get_stable_embedding(track.id, brain, track.embedding)
            cluster['pre_embeddings'][track.id] = pre_emb
            self.track_to_cluster[track.id] = cid

    def _merge_clusters(self, cid_A, cid_B):
        cluster_A = self.active_clusters[cid_A]
        cluster_B = self.active_clusters[cid_B]
        
        cluster_A['tracks_involved'].update(cluster_B['tracks_involved'])
        cluster_A['pre_embeddings'].update(cluster_B['pre_embeddings'])
        
        for tid, embs in cluster_B['post_candidates'].items():
            if tid not in cluster_A['post_candidates']:
                cluster_A['post_candidates'][tid] = []
            cluster_A['post_candidates'][tid].extend(embs)
            
        for tid in cluster_B['tracks_involved']:
            self.track_to_cluster[tid] = cid_A
            
        del self.active_clusters[cid_B]

    def _get_stable_embedding(self, track_id, brain, fallback_emb):
        """Try to get the stable pre-intersection embedding from Cognitive Memory."""
        identity = brain._active_identities.get(track_id)
        if identity and "stable_embedding" in identity:
            return identity["stable_embedding"].copy()
        return fallback_emb.copy()

    def _resolve_cluster(self, cid):
        """
        Solves the bipartite graph between pre-intersection identities
        and post-intersection candidate tracks.
        """
        cluster = self.active_clusters.pop(cid)
        
        # Clean up track mappings
        for tid in cluster['tracks_involved']:
            if tid in self.track_to_cluster:
                del self.track_to_cluster[tid]
                
        pre_ids = list(cluster['pre_embeddings'].keys())
        post_ids = list(cluster['post_candidates'].keys())
        
        if not pre_ids or not post_ids:
            return {}
            
        # Build cost matrix
        cost_matrix = np.zeros((len(pre_ids), len(post_ids)), dtype=np.float32)
        
        for i, pre_id in enumerate(pre_ids):
            pre_emb = cluster['pre_embeddings'][pre_id]
            for j, post_id in enumerate(post_ids):
                # Average the accumulated evidence embeddings
                post_embs = cluster['post_candidates'][post_id]
                if not post_embs:
                    cost_matrix[i, j] = 1.0
                    continue
                    
                avg_post_emb = np.mean(np.stack(post_embs), axis=0)
                norm = np.linalg.norm(avg_post_emb)
                if norm > 0:
                    avg_post_emb /= norm
                    
                cost_matrix[i, j] = _cosine_dist(pre_emb, avg_post_emb)
                
        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        aliases = {}
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost < 0.45: # Max distance to accept assignment
                old_id = pre_ids[r]
                current_id = post_ids[c]
                if old_id != current_id:
                    print(f"[DEFERRED-DECISION] Resolved Intersection! {current_id} is actually {old_id} (cost={cost:.2f})")
                    aliases[current_id] = old_id
                    
        return aliases

    def is_thinking(self, track_id):
        return track_id in self.track_to_cluster
