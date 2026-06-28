import numpy as np

def extract_forensics(tracker, unmatched_tracks, unmatched_dets, boxes, embeddings):
    for u_trk in unmatched_tracks:
        if u_trk.local_id == 2:
            print("\n==== FORENSIC EXTRACTION: local=2 vs unmatched detections ====")
            pred_cx = (u_trk.predicted_box[0] + u_trk.predicted_box[2]) / 2.0
            pred_cy = (u_trk.predicted_box[1] + u_trk.predicted_box[3]) / 2.0
            print(f"Predicted Center for local=2: ({pred_cx:.1f}, {pred_cy:.1f})")
            print(f"Velocity: ({u_trk.vel[0]:.1f}, {u_trk.vel[1]:.1f})")
            
            for d_idx in unmatched_dets:
                d_box = boxes[d_idx]
                d_emb = embeddings[d_idx]
                det_cx = (d_box[0] + d_box[2]) / 2.0
                det_cy = (d_box[1] + d_box[3]) / 2.0
                
                dist = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
                
                avg_sim = 0.0
                if u_trk.get_averaged_embedding() is not None:
                    avg_sim = np.dot(u_trk.get_averaged_embedding(), d_emb)
                    
                anchor_sim = 0.0
                if getattr(u_trk, '_anchor_embedding', None) is not None:
                    anchor_sim = np.dot(u_trk._anchor_embedding, d_emb)
                    
                print(f"Detection {d_idx} Center: ({det_cx:.1f}, {det_cy:.1f})")
                print(f"  pred_err: {dist:.1f}px")
                print(f"  avg_emb_sim: {avg_sim:.3f}")
                print(f"  anchor_sim: {anchor_sim:.3f}")
            print("==============================================================\n")
