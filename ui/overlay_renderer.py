import cv2
import numpy as np

# Copied from pipeline constants
STATE_ACTIVE = 0
STATE_THINKING = 1

# Modern Lean Palette (BGR for OpenCV)
COLORS = [
    (235, 99, 37),   # Blue
    (199, 132, 2),   # Sky
    (136, 148, 13),  # Teal
    (229, 70, 79),   # Indigo
    (105, 150, 5),   # Emerald
    (6, 119, 217),   # Amber
]

# RF-DETR 0-indexed class names (raw_id - 1).
# Sourced from rfdetr/assets/coco_classes.py.
COCO_NAMES = {
    0:  "Person",
    26: "Backpack",
    27: "Umbrella",
    30: "Handbag",
    31: "Tie",
    32: "Suitcase",
    36: "Sports Ball",
    43: "Bottle",
    45: "Wine Glass",
    46: "Cup",
    72: "Laptop",
    75: "Keyboard",
    76: "Cell Phone",
    83: "Book",
    -1: "Unknown",
}

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
            track_class_id = item.get("class_id", 0)
            if track_class_id == 0:
                class_name = f"Person #{gid}"
            else:
                class_name = f"Item #{gid}"
            
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
                lbl_parts.append(class_name)
                lbl_parts.append(f"(T){frozen_flag}" if state == STATE_THINKING else frozen_flag)
                if layer_flags.get("tracking_age"): lbl_parts.append(f"A:{age}")
            else:
                lbl_parts.append(class_name)
                lbl_parts.append(f"(T){frozen_flag}" if state == STATE_THINKING else frozen_flag)
                
            lbl = " ".join(lbl_parts).strip()
            thickness = 1 if state == STATE_THINKING else 2
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
            
            # Modern dark tag label
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(disp, (x1, y1 - 20), (x1 + tw + 8, y1), (15, 15, 18), -1)
            cv2.putText(disp, lbl, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
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

        # ── Retail Theft Detection Overlay ──────────────────────────────
        retail = stats.get("retail")
        if retail:
            self._render_retail(disp, retail, display_items)
            self._render_interactions(disp, retail)

        # ── Shelf Change Zone Overlay ──────────────────────────────────
        shelf_change = stats.get("shelf_change")
        if shelf_change:
            self._render_shelf_zones(disp, shelf_change)

        return disp

    def _render_retail(self, disp, retail, display_items):
        ownership_map = retail.get("ownership_map", {})
        alerts = retail.get("alerts", [])
        zone_summary = retail.get("zone_summary", {})

        # Determine per-person operational state
        person_theft = {a.get("person_gid") for a in alerts if a.get("person_gid")}
        person_with_items = {
            info.get("owner_gid")
            for info in ownership_map.values()
            if info.get("owner_gid") is not None
        }

        # Show simple NORMAL / OBJECT REMOVED / THEFT DETECTED state per person
        for gid, item in display_items.items():
            if item.get("class_id", 0) != 0:
                continue
            box = item["box"]
            x1, y1 = int(box[0]), int(box[1])
            if gid in person_theft:
                cv2.putText(disp, "OBJECT REMOVED", (x1, y1 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(disp, "THEFT DETECTED", (x1, y1 - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif gid in person_with_items:
                cv2.putText(disp, "OBJECT REMOVED", (x1, y1 - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(disp, "NORMAL", (x1, y1 - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

        zy = disp.shape[0] - 20
        for zname, zdata in zone_summary.items():
            cv2.putText(disp, f"{zname}: {zdata.get('total', 0)} items",
                        (10, zy), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 100), 1)
            zy -= 15

    def _render_shelf_zones(self, disp, shelf_change):
        zones = shelf_change.get("zones", [])
        zone_status = shelf_change.get("zone_status", {})
        for zone in zones:
            zname, x1, y1, x2, y2 = zone
            has_ref = zone_status.get(zname, False)
            color = (0, 180, 0) if has_ref else (0, 0, 180)
            for i in range(0, x2 - x1, 12):
                sx = x1 + i
                cv2.line(disp, (sx, y1), (min(sx + 6, x2), y1), color, 1)
                cv2.line(disp, (sx, y2), (min(sx + 6, x2), y2), color, 1)
            for i in range(0, y2 - y1, 12):
                sy = y1 + i
                cv2.line(disp, (x1, sy), (x1, min(sy + 6, y2)), color, 1)
                cv2.line(disp, (x2, sy), (x2, min(sy + 6, y2)), color, 1)
            tag = f"{zname} [REF]" if has_ref else f"{zname} [NO REF]"
            cv2.putText(disp, tag, (x1 + 4, y2 - 6),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

    def _render_interactions(self, disp, retail):
        """Render interaction-driven theft pipeline overlays."""
        # 1. Active reach zone interactions (cyan dashed box)
        interactions = retail.get("active_interactions", [])
        for inter in interactions:
            rz = inter.get("reach_box")
            if rz:
                rx1, ry1, rx2, ry2 = map(int, rz)
                # Dashed rectangle for reach zone
                for i in range(rx1, rx2, 8):
                    cv2.line(disp, (i, ry1), (min(i + 4, rx2), ry1), (255, 255, 0), 1)
                    cv2.line(disp, (i, ry2), (min(i + 4, rx2), ry2), (255, 255, 0), 1)
                for i in range(ry1, ry2, 8):
                    cv2.line(disp, (rx1, i), (rx1, min(i + 4, ry2)), (255, 255, 0), 1)
                    cv2.line(disp, (rx2, i), (rx2, min(i + 4, ry2)), (255, 255, 0), 1)
                cv2.putText(disp, f"REACH P:{inter.get('person_gid', '?')}",
                            (rx1, ry1 - 4), cv2.FONT_HERSHEY_PLAIN, 0.7,
                            (255, 255, 0), 1)

        # 2. Candidate items (yellow boxes)
        candidates = retail.get("candidate_items", [])
        for cand in candidates:
            if cand.get("promoted"):
                continue  # Promoted items are rendered by persistence below
            box = cand.get("box")
            if box is not None and len(box) > 0:
                cx1, cy1, cx2, cy2 = map(int, box)
                cv2.rectangle(disp, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1)
                label = f"Cand#{cand['candidate_id']} A:{cand['age']}"
                cv2.putText(disp, label, (cx1, cy1 - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)

        # 3. Persistent items (state-colored boxes)
        persist_items = retail.get("persistent_items", {})
        for item_gid, item_data in persist_items.items():
            box = item_data.get("box")
            state = item_data.get("state", "")
            parent_gid = item_data.get("parent_gid")
            concealed = item_data.get("concealed", False)

            if box is not None and len(box) > 0:
                ix1, iy1, ix2, iy2 = map(int, box)

                if concealed:
                    color = (0, 0, 255)  # Red for concealed
                    # Dashed box locked to parent position
                    for i in range(ix1, ix2, 8):
                        cv2.line(disp, (i, iy1), (min(i + 4, ix2), iy1), color, 2)
                        cv2.line(disp, (i, iy2), (min(i + 4, ix2), iy2), color, 2)
                    for i in range(iy1, iy2, 8):
                        cv2.line(disp, (ix1, i), (ix1, min(i + 4, iy2)), color, 2)
                        cv2.line(disp, (ix2, i), (ix2, min(i + 4, iy2)), color, 2)
                elif state == "CARRIED":
                    color = (0, 165, 255)  # Orange for carried
                    cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), color, 2)
                elif state == "PICKUP":
                    color = (0, 255, 255)  # Yellow for pickup
                    cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), color, 1)
                else:
                    color = (200, 200, 200)
                    cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), color, 1)

                # Label
                label = f"Item#{item_gid} [{state}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
                cv2.rectangle(disp, (ix1, iy1 - 14), (ix1 + tw + 4, iy1), color, -1)
                cv2.putText(disp, label, (ix1 + 2, iy1 - 3),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)

                # Parent-child link line
                if parent_gid is not None:
                    # Draw a line from item center to parent label position
                    item_cx = (ix1 + ix2) // 2
                    item_cy = (iy1 + iy2) // 2
                    cv2.putText(disp, f"OWNER:P{parent_gid}",
                                (ix1, iy2 + 12), cv2.FONT_HERSHEY_PLAIN, 0.7,
                                color, 1)

        # 4. Theft alert banner
        alerts = retail.get("alerts", [])
        theft_alerts = [a for a in alerts if a.get("alert") == "THEFT" or a.get("reason")]
        if theft_alerts:
            alert = theft_alerts[-1]
            p_gid = alert.get("person_gid", "?")
            item_gid = alert.get("item_gid")
            if item_gid is None:
                obj_gids = alert.get("object_gids", [])
                item_gid = obj_gids[0] if obj_gids else None
            item_str = f"  Item #{item_gid}" if item_gid is not None else ""
            banner = f"!!! THEFT DETECTED   Person #{p_gid}{item_str} !!!"
            h, w = disp.shape[:2]
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
            bx = (w - tw) // 2
            cv2.rectangle(disp, (bx - 15, 5), (bx + tw + 15, 48), (0, 0, 200), -1)
            cv2.putText(disp, banner, (bx, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

