"""
SELFWATCH — Dataset Manager

Supports loading and iterating over standard MOT benchmarks
(DanceTrack, MOT20) and custom retail CCTV video directories.

Dataset structure expected:
  datasets/
    DanceTrack/
      val/
        dancetrack0001/
          img1/
            000001.jpg
            ...
          gt/gt.txt        (optional ground truth)
          seqinfo.ini
    MOT20/
      train/
        MOT20-01/
          img1/
            000001.jpg
          gt/gt.txt
          seqinfo.ini
    custom/
      video1.mp4
      video2.avi
"""

import os
import glob
import configparser
import csv


# ── Scenario Tags ────────────────────────────────────────────────────

SCENARIOS = {
    "all":        "All sequences",
    "crowd":      "Dense crowd scenes",
    "crossing":   "Path crossing / overlap",
    "occlusion":  "Long occlusion events",
    "re-entry":   "Exit and re-entry patterns",
}

# Manual tagging for well-known sequences
_SEQUENCE_TAGS = {
    # DanceTrack — most involve crossing & occlusion
    "dancetrack":   ["crossing", "occlusion"],
    # MOT20
    "MOT20-01":     ["crowd"],
    "MOT20-02":     ["crowd"],
    "MOT20-03":     ["crowd", "crossing"],
    "MOT20-05":     ["crowd", "crossing", "occlusion"],
}


class SequenceInfo:
    """Metadata for one benchmark sequence (or a single video)."""
    __slots__ = (
        "name", "dataset", "path", "frame_dir", "gt_path",
        "frame_rate", "seq_length", "im_width", "im_height",
        "is_video", "video_path", "tags",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return f"<Seq '{self.name}' ({self.dataset}) frames={self.seq_length}>"


class DatasetManager:
    """
    Discovers and manages benchmark datasets and custom videos.

    Usage:
        dm = DatasetManager(root="datasets")
        sequences = dm.list_sequences("DanceTrack")
        seq = sequences[0]
        frames = dm.iter_frames(seq)   # yields (frame_idx, frame_bgr)
    """

    SUPPORTED_DATASETS = ["DanceTrack", "MOT20", "Custom Videos"]

    def __init__(self, root="datasets"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ── Discovery ────────────────────────────────────────────────────

    def get_available_datasets(self):
        """Return list of dataset names that actually exist on disk."""
        available = []
        for ds in ["DanceTrack", "MOT20"]:
            ds_path = os.path.join(self.root, ds)
            if os.path.isdir(ds_path):
                available.append(ds)
        # Custom videos always available (user picks file)
        available.append("Custom Videos")
        return available

    def list_sequences(self, dataset_name, scenario_filter="all"):
        """List all sequences for a given dataset, optionally filtered."""
        if dataset_name == "Custom Videos":
            return self._list_custom_videos()
        elif dataset_name == "DanceTrack":
            return self._list_mot_sequences("DanceTrack", scenario_filter)
        elif dataset_name == "MOT20":
            return self._list_mot_sequences("MOT20", scenario_filter)
        return []

    def _list_mot_sequences(self, dataset, scenario_filter="all"):
        """Parse MOT-format dataset directories."""
        sequences = []
        ds_root = os.path.join(self.root, dataset)

        # Check standard splits: train, val, test
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(ds_root, split)
            if not os.path.isdir(split_dir):
                continue

            for seq_dir in sorted(os.listdir(split_dir)):
                seq_path = os.path.join(split_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                info = self._parse_seqinfo(seq_path, dataset)
                if info is None:
                    continue

                # Apply scenario filter
                if scenario_filter != "all":
                    if scenario_filter not in (info.tags or []):
                        continue

                sequences.append(info)

        return sequences

    def _parse_seqinfo(self, seq_path, dataset):
        """Read seqinfo.ini for a MOT sequence directory."""
        ini_path = os.path.join(seq_path, "seqinfo.ini")
        frame_dir = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")

        name = os.path.basename(seq_path)
        frame_rate = 30
        seq_length = 0
        im_width = 1920
        im_height = 1080

        if os.path.isfile(ini_path):
            cfg = configparser.ConfigParser()
            cfg.read(ini_path)
            sec = cfg["Sequence"] if "Sequence" in cfg else {}
            name = sec.get("name", name)
            frame_rate = int(sec.get("frameRate", 30))
            seq_length = int(sec.get("seqLength", 0))
            im_width = int(sec.get("imWidth", 1920))
            im_height = int(sec.get("imHeight", 1080))
        elif os.path.isdir(frame_dir):
            seq_length = len(glob.glob(os.path.join(frame_dir, "*.jpg")))

        if seq_length == 0 and not os.path.isdir(frame_dir):
            return None

        # Determine tags
        tags = []
        for key, tag_list in _SEQUENCE_TAGS.items():
            if key.lower() in name.lower():
                tags.extend(tag_list)
        tags = list(set(tags)) or ["all"]

        return SequenceInfo(
            name=name, dataset=dataset, path=seq_path,
            frame_dir=frame_dir,
            gt_path=gt_path if os.path.isfile(gt_path) else None,
            frame_rate=frame_rate, seq_length=seq_length,
            im_width=im_width, im_height=im_height,
            is_video=False, video_path=None, tags=tags,
        )

    def _list_custom_videos(self):
        """List video files from the custom directory."""
        custom_dir = os.path.join(self.root, "custom")
        os.makedirs(custom_dir, exist_ok=True)
        sequences = []

        exts = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm")
        for ext in exts:
            for vp in sorted(glob.glob(os.path.join(custom_dir, ext))):
                name = os.path.splitext(os.path.basename(vp))[0]
                sequences.append(SequenceInfo(
                    name=name, dataset="Custom Videos", path=vp,
                    frame_dir=None, gt_path=None,
                    frame_rate=30, seq_length=0,
                    im_width=0, im_height=0,
                    is_video=True, video_path=vp,
                    tags=["custom"],
                ))

        return sequences

    # ── Ground Truth Loader ──────────────────────────────────────────

    def load_ground_truth(self, seq_info):
        """
        Load MOT-format ground truth from gt.txt.
        Returns dict: frame_id -> list of (track_id, x1, y1, w, h)
        """
        if seq_info.gt_path is None or not os.path.isfile(seq_info.gt_path):
            return None

        gt = {}
        with open(seq_info.gt_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                frame_id = int(row[0])
                track_id = int(row[1])
                x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                # Some formats have confidence and class columns
                conf = float(row[6]) if len(row) > 6 else 1.0
                if conf <= 0:
                    continue  # ignored region or zero-conf
                if frame_id not in gt:
                    gt[frame_id] = []
                gt[frame_id].append((track_id, x, y, w, h))

        return gt
