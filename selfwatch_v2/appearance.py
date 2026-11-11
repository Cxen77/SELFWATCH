"""Reuse the original OSNet and attention policy behind explicit v2 contracts."""
from hashlib import sha256
from pathlib import Path
import time

from reid import EmbeddingExtractor
from memory.attention import CognitiveAttention
from .tracking import iou


class AppearanceEncoder:
    def __init__(self, weights, device="cuda:0", refresh_s=.5, batch_buckets=False):
        self.extractor = EmbeddingExtractor(weights, device=device, half=True,
            strict_weights=True, input_size=(256,128), max_batch=16)
        self.provenance = dict(checkpoint=str(Path(weights).resolve()),
            sha256=sha256(Path(weights).read_bytes()).hexdigest(), input_height_width=[256,128],
            load_report=self.extractor.load_report, device=device, dim=512,
            reused_component="reid.embedding_extractor.EmbeddingExtractor",
            batch_buckets=[1,2,4,8,16] if batch_buckets else None)
        self.batch_buckets=batch_buckets
        self.refresh_s = refresh_s
        self.states, self.attention = {}, {}

    def extract_all(self, observation, image):
        """Fixed-input experiments: all eligible crops, without track scheduling."""
        tracks=[dict(instance_index=i,local_id=i,hits=1) for i,p in enumerate(observation.instances)
                if p.label.lower()=='person']
        return self.extract(observation,image,tracks,scheduled=False)

    def extract(self, observation, image, tracks, scheduled=True):
        import cv2
        now = observation.frame.captured_at_s
        camera, epoch = observation.frame.camera_id, observation.frame.stream_epoch
        if scheduled and camera not in self.attention:
            self.attention[camera]=CognitiveAttention()
        attention=self.attention.get(camera)
        crops, indices, diagnostics = [], [], []
        for track in tracks:
            index = track["instance_index"]
            person = observation.instances[index]
            key = (camera, epoch, track["local_id"]) if scheduled else None
            if scheduled:
                state = self.states.setdefault(key, dict(first=now, extracted=-1e9, last_seen=now))
                state["last_seen"] = now
                ticks = int(round(now*15))
                attention.update_tier((epoch,track["local_id"]), int((now-state["first"])*15),
                                     track["hits"]>=2, frame_count=ticks)
                tier = attention.get_tier((epoch,track["local_id"]))
            else:
                tier = 'ALL_ELIGIBLE'
            interval = .1 if tier == "HIGH" else self.refresh_s if tier == "NORMAL" else 2.
            reason = None
            if scheduled and now-state["extracted"] < interval-1e-6:
                reason = "scheduled_skip"
            elif person.score < .4:
                reason = "low_detection_confidence"
            elif any(i != index and other.label.lower()=="person" and iou(person.box_xyxy,other.box_xyxy)>.3
                     for i,other in enumerate(observation.instances)):
                reason = "overlapping_person_crops"
            x1,y1,x2,y2 = map(int,person.box_xyxy)
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(image.shape[1],x2),min(image.shape[0],y2)
            if min(x2-x1,y2-y1) < 16:
                reason = "small_or_invalid_crop"
            if reason:
                diagnostics.append(dict(instance_index=index, reason=reason, tier=tier))
                continue
            crops.append(cv2.resize(image[y1:y2,x1:x2],(128,256)))
            indices.append((index,key))
            diagnostics.append(dict(instance_index=index, reason="fresh_embedding", tier=tier))
        started = time.perf_counter()
        vectors = []
        for start in range(0,len(crops),16):
            chunk=crops[start:start+16];count=len(chunk)
            # Eval-mode OSNet has no cross-example attention. Padding repeats a
            # crop solely to keep CUDA on five warmed batch shapes; padded rows
            # are discarded and never become observations or gallery evidence.
            if self.batch_buckets:
                capacity=next(size for size in (1,2,4,8,16) if size>=count)
                chunk=chunk+[chunk[-1]]*(capacity-count)
            vectors.extend(self.extractor.extract_batch(chunk)[:count])
        embeddings = {index:vector for (index,key),vector in zip(indices,vectors)}
        for _,key in indices:
            if key is not None:
                self.states[key]["extracted"] = now
        # Bound scheduler metadata to recently observed local tracklets.
        for key,state in list(self.states.items()):
            if now-state["last_seen"]>30:
                self.attention[key[0]].forget((key[1],key[2]))
                del self.states[key]
        return embeddings, diagnostics, (time.perf_counter()-started)*1000
