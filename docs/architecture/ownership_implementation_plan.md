# Phase 1: Ownership Engine Implementation Plan

## 1. Overview
The **Ownership Engine** sits strictly above the `StrongSORT` tracker and `IdentityMemory`, analyzing their outputs (bounding boxes, velocity, and state) to form graph-based ownership relationships between `PersonNode` and `ObjectNode`. This phase solely implements **Ownership Establishment**, avoiding any alerts or theft detection logic. 

The relationship is determined by fusing:
- **Spatial Proximity** (Distance from person's base/feet to object center)
- **Motion Similarity** (Cosine similarity of velocity vectors)
- **Temporal Persistence** (Time spent in proximity)

## 2. Exact Files to Modify
- **Create**: `memory/ownership_manager.py` (Completed - Code provided in Section 3)
- **Modify**: `engine/pipeline.py` (Integration point)
- **Modify**: `config.py` (Optional, add default threshold values)

## 3. Complete Code: `memory/ownership_manager.py`

This module is designed strictly with standard Python dataclasses, no external graph libraries, and consumes only standard tracking metrics:

```python
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import logging

logger = logging.getLogger("OwnershipEngine")

@dataclass
class OwnershipEdge:
    person_id: int
    object_id: int
    proximity_score: float = 0.0
    motion_similarity: float = 0.0
    temporal_score: float = 0.0
    frames_together: int = 0
    
    @property
    def total_score(self) -> float:
        return (0.4 * self.proximity_score + 
                0.4 * self.motion_similarity + 
                0.2 * self.temporal_score)

@dataclass
class PersonNode:
    track_id: int
    bbox: List[float]
    velocity: List[float]
    state: int

@dataclass
class ObjectNode:
    track_id: int
    bbox: List[float]
    velocity: List[float]
    state: int
    owner_id: Optional[int] = None
    owner_score: float = 0.0
    ownership_state: str = "UNOWNED"

class OwnershipManager:
    \"\"\"
    Phase 1 Ownership Engine implementation.
    Determines ownership of objects by persons based on proximity, motion, and time.
    \"\"\"
    def __init__(self, max_reach_dist: float = 150.0):
        self.max_reach_dist = max_reach_dist
        self.frames_to_tentative = 5
        self.frames_to_owned = 15
        self.score_tentative_threshold = 0.5
        self.score_owned_threshold = 0.8
        
        self.edges: Dict[Tuple[int, int], OwnershipEdge] = {}
        self.objects: Dict[int, ObjectNode] = {}
        self.persons: Dict[int, PersonNode] = {}
        
    def _calculate_proximity(self, p_box: List[float], o_box: List[float]) -> float:
        # Distance from bottom-center of person to center of object
        p_cx = (p_box[0] + p_box[2]) / 2.0
        p_by = p_box[3]
        o_cx = (o_box[0] + o_box[2]) / 2.0
        o_cy = (o_box[1] + o_box[3]) / 2.0
        dist = math.sqrt((p_cx - o_cx)**2 + (p_by - o_cy)**2)
        return max(0.0, 1.0 - (dist / self.max_reach_dist))
        
    def _calculate_motion_similarity(self, p_vel: List[float], o_vel: List[float]) -> float:
        p_speed = math.sqrt(p_vel[0]**2 + p_vel[1]**2)
        o_speed = math.sqrt(o_vel[0]**2 + o_vel[1]**2)
        
        if p_speed < 1.0 and o_speed < 1.0:
            return 1.0
        if p_speed < 1e-3 or o_speed < 1e-3:
            return 0.0
            
        cos_sim = (p_vel[0] * o_vel[0] + p_vel[1] * o_vel[1]) / (p_speed * o_speed)
        return max(0.0, (cos_sim + 1.0) / 2.0)
        
    def update(self, frame_count: int, persons: List[PersonNode], objects: List[ObjectNode]) -> List[ObjectNode]:
        self.persons = {p.track_id: p for p in persons}
        
        new_objects = {}
        for obj in objects:
            if obj.track_id in self.objects:
                prev = self.objects[obj.track_id]
                obj.owner_id = prev.owner_id
                obj.owner_score = prev.owner_score
                obj.ownership_state = prev.ownership_state
            new_objects[obj.track_id] = obj
        self.objects = new_objects
        
        active_edges: Set[Tuple[int, int]] = set()
        
        # Calculate pair-wise metrics
        for p_id, p in self.persons.items():
            for o_id, o in self.objects.items():
                edge_key = (p_id, o_id)
                active_edges.add(edge_key)
                if edge_key not in self.edges:
                    self.edges[edge_key] = OwnershipEdge(person_id=p_id, object_id=o_id)
                edge = self.edges[edge_key]
                
                edge.proximity_score = self._calculate_proximity(p.bbox, o.bbox)
                edge.motion_similarity = self._calculate_motion_similarity(p.velocity, o.velocity)
                
                if edge.proximity_score > 0:
                    edge.frames_together += 1
                else:
                    edge.frames_together = max(0, edge.frames_together - 1)
                edge.temporal_score = min(1.0, edge.frames_together / self.frames_to_owned)
                
        # Clean up old edges
        dead_edges = []
        for edge_key, edge in self.edges.items():
            if edge_key not in active_edges:
                edge.frames_together = max(0, edge.frames_together - 2)
                edge.temporal_score = min(1.0, edge.frames_together / self.frames_to_owned)
                if edge.frames_together == 0:
                    dead_edges.append(edge_key)
        for key in dead_edges:
            del self.edges[key]

        # Resolve ownership
        for o_id, obj in self.objects.items():
            best_person = None
            best_score = 0.0
            
            for p_id in self.persons.keys():
                edge = self.edges.get((p_id, o_id))
                if edge and edge.total_score > best_score:
                    best_score = edge.total_score
                    best_person = p_id
            
            prev_state = obj.ownership_state
            prev_owner = obj.owner_id
            
            # Hysteresis & Transitions
            if best_score > self.score_owned_threshold:
                obj.ownership_state = "OWNED"
                obj.owner_id = best_person
                obj.owner_score = best_score
                if prev_state != "OWNED" or prev_owner != best_person:
                    logger.info(f"[EVENT] ownership_confirmed: obj={o_id} owner={best_person} score={best_score:.2f}")
                    
            elif best_score > self.score_tentative_threshold:
                if prev_state == "OWNED" and prev_owner == best_person and best_score > (self.score_owned_threshold - 0.1):
                    obj.ownership_state = "OWNED"
                    obj.owner_score = best_score
                else:
                    obj.ownership_state = "TENTATIVE"
                    obj.owner_id = best_person
                    obj.owner_score = best_score
                    if prev_state == "UNOWNED":
                        logger.info(f"[EVENT] ownership_created: obj={o_id} tentative_owner={best_person} score={best_score:.2f}")
                    
            else:
                if prev_state == "OWNED" and prev_owner == best_person and best_score > (self.score_tentative_threshold - 0.2):
                    obj.ownership_state = "TENTATIVE"
                    obj.owner_score = best_score
                else:
                    obj.ownership_state = "UNOWNED"
                    obj.owner_id = None
                    obj.owner_score = best_score
                    if prev_state != "UNOWNED":
                        logger.info(f"[EVENT] ownership_lost: obj={o_id} previous_owner={prev_owner}")

        return list(self.objects.values())
```

## 4. Integration Points in Existing Pipeline

In `engine/pipeline.py`:

**Initialization:**
```python
from memory.ownership_manager import OwnershipManager, PersonNode, ObjectNode

class SelfWatchPipeline:
    def __init__(self, ...):
        # ... existing initializations
        self.ownership_manager = OwnershipManager(max_reach_dist=150.0)
```

**Inside `process_frame` after the tracker/ownership_arbitration runs:**
```python
# Convert tracker outputs to nodes for ownership engine
persons = []
objects = []

for track in active:
    gid = track.global_id
    vel = list(track.vel)
    bbox = track.smooth_box.tolist()
    state = track.state
    
    # We rely on track.class_id if exposed. Otherwise, assume PERSON_CLASS = 0
    if getattr(track, 'class_id', 0) == PERSON_CLASS:
        persons.append(PersonNode(track_id=gid, bbox=bbox, velocity=vel, state=state))
    else:
        objects.append(ObjectNode(track_id=gid, bbox=bbox, velocity=vel, state=state))

# Process Ownership relationships
updated_objects = self.ownership_manager.update(self.frame_count, persons, objects)

# Expose metadata to _id_states for drawing
for obj in updated_objects:
    if obj.track_id in self._id_states:
        self._id_states[obj.track_id]["owner_id"] = obj.owner_id
        self._id_states[obj.track_id]["owner_score"] = obj.owner_score
        self._id_states[obj.track_id]["ownership_state"] = obj.ownership_state
```

## 5. Required Config Values

If you wish to manage these in `config.py` rather than hardcoding in the manager:
```python
OWNERSHIP_MAX_REACH_DIST = 150.0       # Max distance (px) to consider for proximity
OWNERSHIP_FRAMES_TENTATIVE = 5         # Frames before marking TENTATIVE
OWNERSHIP_FRAMES_OWNED = 15            # Frames before marking OWNED
OWNERSHIP_TENTATIVE_THRESH = 0.5       # Score threshold for TENTATIVE
OWNERSHIP_OWNED_THRESH = 0.8           # Score threshold for OWNED
```

## 6. Unit-Test Strategy

1. **Test Spatial Proximity**: 
   - Pass a `PersonNode` and `ObjectNode` directly on top of each other. Verify `proximity_score == 1.0`. 
   - Pass them >150px apart. Verify `proximity_score == 0.0`.
2. **Test Motion Similarity**:
   - Both stationary `[0,0]`: verify `1.0`.
   - One stationary, one moving: verify `0.0`.
   - Both moving opposite directions `[10,0]` and `[-10,0]`: verify `0.0`.
   - Both moving same direction `[10,0]` and `[5,0]`: verify `1.0`.
3. **Test Temporal Accumulation (Hysteresis)**:
   - Feed overlapping nodes for 15 frames. Verify `[EVENT] ownership_created` fires at Frame 5, and `[EVENT] ownership_confirmed` fires at Frame 15.
   - Separate them for 1 frame. Verify state drops to `TENTATIVE` (not `UNOWNED`) due to the hysteresis buffer.
4. **Test Identity Stability**:
   - Introduce a new `PersonNode` close to the object. Ensure `owner_id` doesn't flicker instantly to the new person unless their temporal/motion score massively eclipses the current owner.

## 7. Example Ownership Lifecycle

```text
Person 10 approaches Backpack 5
↓
[Frame 1-4] UNOWNED (Score rising: Proximity increases, motion syncs up)
↓
[Frame 5] TENTATIVE (Score > 0.5)
Event Log: [EVENT] ownership_created: obj=5 tentative_owner=10 score=0.62
↓
[Frame 6-14] TENTATIVE (Score rising: Temporal score increasing)
↓
[Frame 15] OWNED (Score > 0.8)
Event Log: [EVENT] ownership_confirmed: obj=5 owner=10 score=0.85
↓
[Frame 16+] OWNED (Ownership Maintained)
```
