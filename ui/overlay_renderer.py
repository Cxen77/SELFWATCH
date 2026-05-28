import cv2
import numpy as np

# Copied from pipeline constants
STATE_ACTIVE = 0
STATE_THINKING = 1

# Color map from UI
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128)
]

def id_color(idx):
    return COLORS[idx % len(COLORS)]

class OverlayRenderer:
    """
    Phase 3: Dedicated overlay renderer.
    Runs on the UI thread (or right before display update) to decouple 
    OpenCV drawing from the inference hot-path.
    """
    
    def __init__(self):
        self._cached_cone = None
        self._cached_cone_pts = None

    def render(self, frame: np.ndarray, stats: dict, cam_label: str = "", cam_id: int = 0) -> np.ndarray:
        """
        Takes a raw numpy frame and a complete stats dictionary,
        and renders all diagnostic, tracking, and forensic overlays.
        Returns the annotated frame.
        """
        if frame is None or stats is None:
            return frame

        # Optional: copy frame if we don't want to modify original, but since it's only for display, we can modify in place
        # Actually we must modify in-place or a copy, but let's do a fast copy to avoid polluting the async cache's raw frame if needed.
        # Wait, the AsyncStateCache saves the frame we give it. If we draw ON it now, it's fine because we do this just before display,
        # BUT scrubbing will re-draw over an already drawn frame if we do in-place!
        # So we MUST copy the frame before drawing.
        disp = frame.copy()
        
        layer_flags = stats.get("layer_flags", {})
        
        # 1. Suppress regions
        if layer_flags.get("forensic") and layer_flags.get("for_suppress"):
            for rbox in stats.get("suppress_regions", []):
                rx1, ry1, rx2, ry2 = map(int, rbox)
                cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (0, 0, 100), 2)
                cv2.putText(disp, "SUPPRESS", (rx1, ry1-5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,100), 1)

        # 2. Phantoms
        if layer_flags.get("motion") and layer_flags.get("motion_prediction"):
            for p in stats.get("phantoms", []):
                px1, py1, px2, py2 = map(int, p["position"])
                cv2.rectangle(disp, (px1, py1), (px2, py2), (150, 150, 150), 1, lineType=cv2.LINE_AA)
                cv2.putText(disp, f"P:{p['track_id']} ({p['confidence']}%)", (px1, py1 - 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (150, 150, 150), 1)
                
                if p.get("cone"):
                    tip, left, right = p["cone"]
                    pts_arr = np.array([
                        [int(tip[0]), int(tip[1])],
                        [int(left[0]), int(left[1])],
                        [int(right[0]), int(right[1])],
                    ], dtype=np.int32)
                    
                    # Add transparent cone (optimized)
                    sub_img = disp[0:disp.shape[0], 0:disp.shape[1]]
                    cv2.fillPoly(sub_img, [pts_arr], (0, 180, 0))
                    cv2.addWeighted(sub_img, 0.15, disp, 0.85, 0, disp)
                    cv2.polylines(disp, [pts_arr], True, (0, 200, 0), 1, cv2.LINE_AA)
                    
                    tcx, tcy = int(tip[0]), int(tip[1])
                    vx, vy = p.get("velocity", [0, 0])
                    dir_end = (int(tcx + vx * 15), int(tcy + vy * 15))
                    cv2.arrowedLine(disp, (tcx, tcy), dir_end, (0, 255, 0), 2, tipLength=0.4)

        # 3. Main Tracks
        frozen_gids = stats.get("frozen_gids", set())
        display_items = stats.get("display_items", {})
        color_map = stats.get("color_map", {})
        
        # Optimize polylines by batching if possible, but here they have different colors
        for gid, item in display_items.items():
            tbox = item["box"]
            state = item["state"]
            lid = item["lid"]
            vel = item["vel"]
            age = item["age"]
            assoc_data = item["assoc"]
            
            x1, y1, x2, y2 = map(int, tbox)
            color_id = color_map.get(gid, gid)
            color = id_color(color_id)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            frozen_flag = " [FROZEN]" if (gid in frozen_gids and layer_flags.get("cognitive") and layer_flags.get("cog_frozen")) else ""
            
            if state == STATE_THINKING and not (layer_flags.get("cognitive") and layer_flags.get("cog_thinking")):
                continue
                
            # Render Box and Label
            lbl_parts = []
            if layer_flags.get("tracking"):
                if layer_flags.get("tracking_ids"): lbl_parts.append(f"G:{gid}|L:{lid}")
                lbl_parts.append(f"(T){frozen_flag}" if state == STATE_THINKING else frozen_flag)
                if layer_flags.get("tracking_age"): lbl_parts.append(f"A:{age}")
            else:
                lbl_parts.append(f"(T){frozen_flag}" if state == STATE_THINKING else frozen_flag)
                
            lbl = " ".join(lbl_parts).strip()
            thickness = 1 if state == STATE_THINKING else 2
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
            
            # Label background
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(disp, (x1, y1 - 22), (x1 + tw + 6, y1), color, -1)
            cv2.putText(disp, lbl, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            
            # Forensic details
            if assoc_data and stats.get("debug_enabled") and layer_flags.get("association"):
                parts = []
                if layer_flags.get("assoc_method"): parts.append(assoc_data.get("method", "NEW"))
                if layer_flags.get("assoc_cost"): parts.append(f"Cost: {assoc_data.get('cost', 0.0):.2f}")
                if layer_flags.get("assoc_cbiou") and assoc_data.get("cbiou", 0) > 0:
                    parts.append(f"C-BIoU: {assoc_data['cbiou']}px")
                if parts:
                    cv2.putText(disp, " | ".join(parts), (x1, y2 + 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 0), 1)

            # Velocity vector — guard against numpy truth-value ambiguity
            if vel is not None and hasattr(vel, "__len__") and len(vel) == 2 and layer_flags.get("motion") and layer_flags.get("motion_velocity"):
                vx, vy = float(vel[0]), float(vel[1])
                end_pt = (int(cx + vx * 10), int(cy + vy * 10))
                cv2.arrowedLine(disp, (cx, cy), end_pt, color, 2, tipLength=0.3)
                
            # Camera level Global ID annotation (Moved from camera_stream.py)
            gid_label = f"CAM{cam_id} | LID{lid} | GID{gid}"
            cv2.putText(disp, gid_label, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

        # 4. Global HUD & Errors
        # ID switches
        id_switches = stats.get("id_switches", 0)
        if id_switches > 0 and layer_flags.get("forensic") and layer_flags.get("for_failure"):
            warn_msg = f"FAILURE EVENT: {id_switches} ID SWITCH(ES)"
            (ww, wh), _ = cv2.getTextSize(warn_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            wx = (disp.shape[1] - ww) // 2
            wy = 50
            cv2.rectangle(disp, (wx - 10, wy - wh - 10), (wx + ww + 10, wy + 10), (0, 0, 200), -1)
            cv2.putText(disp, warn_msg, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Rejects
        if layer_flags.get("forensic") and layer_flags.get("for_failure"):
            rejects = stats.get("recent_rejects", [])
            ry = 80
            for msg in rejects:
                cv2.putText(disp, msg, (10, ry), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 100, 255), 1)
                ry += 15

        # HUD top left
        hud = stats.get("hud", {})
        cv2.putText(disp, f"Tracked: {hud.get('tracked', 0)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, 
                    f"Active:{hud.get('active', 0)}  Think:{hud.get('thinking', 0)}  "
                    f"Ph:{hud.get('phantom', 0)}  Occ:{hud.get('occlusion', 0)}  "
                    f"Wm:{hud.get('warm', 0)}", 
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(disp, f"[{hud.get('reid_tag', 'cached')}] F#{hud.get('frame_count', 0)}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Camera Header
        if cam_label:
            cv2.putText(disp, cam_label, (disp.shape[1] - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return disp
