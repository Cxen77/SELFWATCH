import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import logging

# Use the existing logger format, or standard logging
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
    """
    Phase 1 Ownership Engine implementation.
    Determines ownership of objects by persons based on proximity, motion, and time.
    """
    def __init__(self, max_reach_dist: float = 150.0):
        # Configuration
        self.max_reach_dist = max_reach_dist
        self.frames_to_tentative = 5
        self.frames_to_owned = 15
        self.score_tentative_threshold = 0.5
        self.score_owned_threshold = 0.8
        
        # Graph State
        self.edges: Dict[Tuple[int, int], OwnershipEdge] = {}
        self.objects: Dict[int, ObjectNode] = {}
        self.persons: Dict[int, PersonNode] = {}
        
    def _calculate_proximity(self, p_box: List[float], o_box: List[float]) -> float:
        """Calculate spatial proximity score between person and object."""
        # Use person bottom center (feet/ground contact) and object center
        p_cx = (p_box[0] + p_box[2]) / 2.0
        p_by = p_box[3]
        
        o_cx = (o_box[0] + o_box[2]) / 2.0
        o_cy = (o_box[1] + o_box[3]) / 2.0
        
        dist = math.sqrt((p_cx - o_cx)**2 + (p_by - o_cy)**2)
        
        # Score decreases linearly from 1.0 (dist=0) to 0.0 (dist>=max_reach_dist)
        score = max(0.0, 1.0 - (dist / self.max_reach_dist))
        return score
        
    def _calculate_motion_similarity(self, p_vel: List[float], o_vel: List[float]) -> float:
        """Calculate motion similarity using cosine similarity of velocities."""
        p_speed = math.sqrt(p_vel[0]**2 + p_vel[1]**2)
        o_speed = math.sqrt(o_vel[0]**2 + o_vel[1]**2)
        
        # If both are effectively stationary, they have similar motion (or lack thereof)
        if p_speed < 1.0 and o_speed < 1.0:
            return 1.0
            
        # If one is moving and the other is stationary, motion is entirely dissimilar
        if p_speed < 1e-3 or o_speed < 1e-3:
            return 0.0
            
        cos_sim = (p_vel[0] * o_vel[0] + p_vel[1] * o_vel[1]) / (p_speed * o_speed)
        # Map cosine similarity [-1, 1] to [0, 1] range
        return max(0.0, (cos_sim + 1.0) / 2.0)
        
    def update(self, frame_count: int, persons: List[PersonNode], objects: List[ObjectNode]) -> List[ObjectNode]:
        """
        Main update step for the Ownership Engine.
        To be called after tracker and Identity Memory have updated.
        """
        # Register current persons
        self.persons = {p.track_id: p for p in persons}
        
        # Sync objects, retaining ownership state across frames for continuity
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
        
        # Calculate pair-wise metrics for all combinations of persons and objects
        for p_id, p in self.persons.items():
            for o_id, o in self.objects.items():
                edge_key = (p_id, o_id)
                active_edges.add(edge_key)
                
                if edge_key not in self.edges:
                    self.edges[edge_key] = OwnershipEdge(person_id=p_id, object_id=o_id)
                    
                edge = self.edges[edge_key]
                
                # Update metrics
                edge.proximity_score = self._calculate_proximity(p.bbox, o.bbox)
                edge.motion_similarity = self._calculate_motion_similarity(p.velocity, o.velocity)
                
                # Temporal logic: increment if in proximity, else decay
                if edge.proximity_score > 0:
                    edge.frames_together += 1
                else:
                    edge.frames_together = max(0, edge.frames_together - 1)
                    
                edge.temporal_score = min(1.0, edge.frames_together / self.frames_to_owned)
                
        # Clean up dead or untracked edges using exponential-like decay
        dead_edges = []
        for edge_key, edge in self.edges.items():
            if edge_key not in active_edges:
                edge.frames_together = max(0, edge.frames_together - 2)
                edge.temporal_score = min(1.0, edge.frames_together / self.frames_to_owned)
                if edge.frames_together == 0:
                    dead_edges.append(edge_key)
        for key in dead_edges:
            del self.edges[key]

        # Resolve ownership using hysteresis and state machine logic
        for o_id, obj in self.objects.items():
            best_person = None
            best_score = 0.0
            
            # Find the strongest edge (most likely owner)
            for p_id in self.persons.keys():
                edge = self.edges.get((p_id, o_id))
                if edge and edge.total_score > best_score:
                    best_score = edge.total_score
                    best_person = p_id
            
            prev_state = obj.ownership_state
            prev_owner = obj.owner_id
            
            # State transitions
            if best_score > self.score_owned_threshold:
                obj.ownership_state = "OWNED"
                obj.owner_id = best_person
                obj.owner_score = best_score
                if prev_state != "OWNED" or prev_owner != best_person:
                    print(f"[EVENT] ownership_confirmed: obj={o_id} owner={best_person} score={best_score:.2f}")
                    
            elif best_score > self.score_tentative_threshold:
                # Add hysteresis: if it was already OWNED by the same person, allow a slight dip
                if prev_state == "OWNED" and prev_owner == best_person and best_score > (self.score_owned_threshold - 0.1):
                    obj.ownership_state = "OWNED"
                    obj.owner_score = best_score
                else:
                    obj.ownership_state = "TENTATIVE"
                    obj.owner_id = best_person
                    obj.owner_score = best_score
                    if prev_state == "UNOWNED":
                        print(f"[EVENT] ownership_created: obj={o_id} tentative_owner={best_person} score={best_score:.2f}")
                    
            else:
                # Hysteresis: if previously OWNED, don't drop immediately unless score falls way down
                if prev_state == "OWNED" and prev_owner == best_person and best_score > (self.score_tentative_threshold - 0.2):
                    obj.ownership_state = "TENTATIVE"
                    obj.owner_score = best_score
                else:
                    obj.ownership_state = "UNOWNED"
                    obj.owner_id = None
                    obj.owner_score = best_score
                    if prev_state != "UNOWNED":
                        print(f"[EVENT] ownership_lost: obj={o_id} previous_owner={prev_owner}")

        return list(self.objects.values())
