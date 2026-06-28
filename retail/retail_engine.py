"""
Retail Theft Engine — orchestrates behavioral theft detection.

Pipeline per frame:
  1. InteractionEngine:     person-shelf reach zone interaction detection
  2. ShelfChangeDetector:   (optional) category-agnostic shelf void extraction
  3. CandidateManager:      temporary unknown item tracking & promotion
  4. PersistenceEngine:     parent-child binding, concealment, exit alerting
  5. OwnershipEngine:       person ↔ known object proximity association
  6. ShelfZoneManager:      zone occupancy, ITEM_REMOVED / ITEM_RETURNED
  7. ConcealmentDetector:   ENTERED_BAG / BODY_CONCEALMENT (known objects)
  8. SuspicionEngine:       score accumulation per person
  9. AlertSystem:            THEFT_ALERT + ESP32 trigger
  10. EvidenceTimeline:      append-only forensic log
"""

import config
from .rfdetr_taxonomy import RETAIL_TARGET_CLASSES, CLASS_NAMES
from .ownership_engine import OwnershipEngine
from .shelf_zone_manager import ShelfZoneManager
from .concealment_detector import ConcealmentDetector
from .suspicion_engine import SuspicionEngine
from .alert_system import AlertSystem
from .evidence_timeline import EvidenceTimeline
from .interaction_engine import InteractionEngine
from .candidate_manager import CandidateManager
from .persistence_engine import PersistenceEngine


class RetailTheftEngine:

    def __init__(self):
        zones = getattr(config, "RETAIL_ZONES", None)
        if zones:
            self.shelf_manager = ShelfZoneManager(zones)
        else:
            self.shelf_manager = ShelfZoneManager()

        self.ownership = OwnershipEngine()
        self.concealment = ConcealmentDetector()
        self.suspicion = SuspicionEngine()
        self.alert_system = AlertSystem(
            esp32_url=getattr(config, "RETAIL_ESP32_URL", "http://192.168.1.100/alert"),
            cooldown=getattr(config, "RETAIL_ALERT_COOLDOWN", 30.0),
        )
        self.timeline = EvidenceTimeline()
        self._frame_count = 0

        # ── New: Category-Agnostic Interaction Pipeline ──
        self.interaction = InteractionEngine()
        self.candidates = CandidateManager()
        self.persistence = PersistenceEngine(
            frame_h=getattr(config, "RETAIL_FRAME_H", 540))

        # Cache shelf zones for interaction engine
        self._shelf_zones = zones if zones else [("ZONE_A", 0, 0, 960, 540)]

    def process_frame(self, display_items, id_states, frame_count,
                      shelf_change_dets=None):
        """
        Main frame processing.

        Args:
            display_items: dict of gid -> {"box", "vel", "class_id", ...}
            id_states: dict of gid -> {"state", ...}
            frame_count: current frame number
            shelf_change_dets: optional list of shelf change detections
                               from ShelfChangeDetector
        """
        self._frame_count = frame_count

        from .rfdetr_taxonomy import PERSON, ALL_TRACKED_CLASSES
        persons = [g for g, i in display_items.items()
                   if i.get("class_id", 0) == PERSON]
        objects = [g for g, i in display_items.items()
                   if i.get("class_id", 0) in ALL_TRACKED_CLASSES]

        if frame_count % 30 == 0:
            n_cand = len(self.candidates.get_active_candidates())
            n_persist = len(self.persistence.get_all_items())
            print(f"[RETAIL_TRACE] frame={frame_count} persons={persons} "
                  f"objects={objects} candidates={n_cand} "
                  f"persistent_items={n_persist} "
                  f"id_states_count={len(id_states)}")

        # ── 1. Interaction Detection ──
        active_interactions = self.interaction.update(
            display_items, self._shelf_zones, frame_count)

        # ── 2. Candidate Management ──
        # Build proposals from shelf change detections (if available)
        # AND from low-confidence RF-DETR detections already in display_items
        proposals = []
        if shelf_change_dets:
            for det in shelf_change_dets:
                proposals.append(det)

        # Also add any unknown-class detections from display_items
        for gid, item in display_items.items():
            cid = item.get("class_id", 0)
            if cid == -1:  # UNKNOWN_OBJECT
                proposals.append({
                    "box": item["box"],
                    "confidence": 0.5,
                    "zone": "auto",
                    "class_id": -1,
                })

        active_candidates = self.candidates.process_proposals(
            proposals, active_interactions, display_items, frame_count)

        # ── 3. Promote candidates → PersistenceEngine ──
        for promoted in self.candidates.get_promoted():
            item_gid = self.persistence.register_promoted(
                promoted, frame_count)
            # Add suspicion for the pickup
            self.suspicion.process_events([{
                "event": "ITEM_REMOVED",
                "person_gid": promoted.person_gid,
                "object_gid": item_gid,
                "zone": promoted.origin_zone,
                "frame": frame_count,
            }], {})
            self.timeline.record(
                frame=frame_count,
                event_type="UNKNOWN_ITEM_PICKUP",
                person_gid=promoted.person_gid,
                item_gid=item_gid,
                zone=promoted.origin_zone,
            )

        # ── 4. Persistence Engine (parent-child, concealment, exit) ──
        persist_result = self.persistence.update(
            display_items, self._shelf_zones,
            self.candidates.get_all(), frame_count)

        # Process persistence theft alerts
        persist_alerts = persist_result.get("theft_alerts", [])
        for alert in persist_alerts:
            self.suspicion.process_events([{
                "event": "EXIT_WITH_CONCEALED",
                "person_gid": alert["person_gid"],
                "object_gid": alert["item_gid"],
                "frame": frame_count,
            }], {})
            self.timeline.record(
                frame=frame_count,
                event_type="UNKNOWN_ITEM_THEFT",
                person_gid=alert["person_gid"],
                item_gid=alert["item_gid"],
                origin_zone=alert["origin_zone"],
                reason=alert["reason"],
            )

        # ── 5-8. Existing known-object pipeline ──
        ownership_map = self.ownership.update(display_items, id_states)

        shelf_events = self.shelf_manager.update(
            display_items, ownership_map, id_states, frame_count)

        conceal_events = self.concealment.update(
            display_items, id_states, self.shelf_manager,
            ownership_map, frame_count)

        all_events = shelf_events + conceal_events

        for ev in all_events:
            self.timeline.record(
                frame=frame_count,
                event_type=ev["event"],
                **{k: v for k, v in ev.items() if k not in ("event", "frame")},
            )

        self.suspicion.process_events(all_events, ownership_map)
        self.suspicion.tick()

        # Merge persistence items into ownership_map for alerts
        persist_items = persist_result.get("items", {})
        for item_gid, p_data in persist_items.items():
            ownership_map[item_gid] = {
                "owner_gid": p_data["parent_gid"],
                "zone": p_data.get("origin_zone", "unknown"),
                "class_id": -1,
                "label": "unknown",
            }

        fired_alerts = self.alert_system.check_and_alert(
            self.suspicion, ownership_map, self.timeline, frame_count)

        for alert in fired_alerts:
            self.timeline.record(
                frame=frame_count,
                event_type="THEFT_ALERT",
                person_gid=alert["person_gid"],
                confidence=alert["confidence"],
                object_gids=alert.get("object_gids", []),
            )

        # ── Build return dict ──
        # Merge persistence items into the rendering output
        persist_items = persist_result.get("items", {})

        return {
            "ownership_map": ownership_map,
            "shelf_events": shelf_events,
            "conceal_events": conceal_events,
            "suspicion_scores": self.suspicion.get_all_scores(),
            "alerts": fired_alerts + persist_alerts,
            "zone_summary": self.shelf_manager.get_zone_summary(),
            # New: interaction-driven data for overlay rendering
            "active_interactions": [{
                "person_gid": ev.person_gid,
                "zone_name": ev.zone_name,
                "reach_box": ev.reach_box,
            } for ev in active_interactions],
            "candidate_items": [{
                "candidate_id": c.candidate_id,
                "box": c.box,
                "age": c.age,
                "promoted": c.promoted,
                "person_gid": c.person_gid,
                "origin_zone": c.origin_zone,
            } for c in active_candidates],
            "persistent_items": persist_items,
        }

    def close(self):
        self.timeline.export_json()
        self.persistence.close()

        scores = self.suspicion.get_all_scores()
        n_alerts = len(self.alert_system.alerts)
        n_events = len(self.timeline.events)
        print(f"\n{'='*60}")
        print(f"  RETAIL THEFT ENGINE — Final Report")
        print(f"{'='*60}")
        print(f"  Total events recorded: {n_events}")
        print(f"  Total alerts fired:    {n_alerts}")
        print(f"  Final suspicion scores: {scores}")
        zone_summary = self.shelf_manager.get_zone_summary()
        for zname, zdata in zone_summary.items():
            print(f"  Zone {zname}: {zdata['total_on_shelf']} on shelf, "
                  f"{zdata['total_removed']} removed")
        print(f"{'='*60}")
