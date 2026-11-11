import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from selfwatch_v2.contracts import FrameRef, Observation, Instance
from selfwatch_v2.replay import ordered_merge, load_manifest, Source
from selfwatch_v2.store import ObservationStore
from selfwatch_v2.yolo import convert_result


def frame(camera="A", epoch="recording1", index=0, timestamp=0):
    return FrameRef(camera, epoch, index, timestamp, 1000, 640, 480, "fixture")


class ReplayTests(unittest.TestCase):
    def test_capture_order_is_independent_of_receipt_and_camera_speed(self):
        streams = [[(frame("A", index=i, timestamp=t), None) for i, t in enumerate([0, 2, 4])],
                   [(replace(frame("B", index=i, timestamp=t), received_at_s=1), None)
                    for i, t in enumerate([1, 2, 3])]]
        merged = [f for f, _ in ordered_merge(streams)]
        self.assertEqual([f.captured_at_s for f in merged], [0, 1, 2, 2, 3, 4])
        self.assertEqual([f.camera_id for f in merged], ["A", "B", "A", "B", "B", "A"])

    def test_buffers_are_bounded_and_empty_camera_does_not_block(self):
        consumed = []
        def generate():
            for i in range(100):
                consumed.append(i)
                yield frame(index=i, timestamp=i), None
        merged = ordered_merge([generate(), iter(())])
        next(merged)
        self.assertEqual(consumed, [0])
        merged.close()

    def test_nonmonotonic_source_rejected(self):
        with self.assertRaises(ValueError):
            list(ordered_merge([iter([(frame(timestamp=2), None), (frame(timestamp=1), None)])]))

    def test_epoch_and_camera_scope_identity(self):
        self.assertNotEqual(frame().observation_id, frame(epoch="reconnected").observation_id)
        self.assertNotEqual(frame().observation_id, frame(camera="B").observation_id)
        self.assertNotEqual(frame(camera="a/b", epoch="c").observation_id,
                            frame(camera="a", epoch="b/c").observation_id)

    def test_invalid_frame_rejected(self):
        with self.assertRaises(ValueError):
            frame(timestamp=float("nan"))

    def test_duplicate_cameras_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "video").touch()
            source = dict(camera_id="A", path="video", stream_epoch="r", fps=30)
            path = root / "manifest.json"
            path.write_text(json.dumps(dict(schema_version=1, sources=[source, source])))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_nonuniform_timestamps_override_encoded_video_fps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video"
            path.touch()
            source = Source("A", path, "clip1", 30, .1, (0, .3, 1.7))
            self.assertAlmostEqual(source.timestamp(2), 1.8)
            with self.assertRaises(ValueError):
                Source("A", path, "clip1", 30, 0, (0, .3, .2))


class StoreTests(unittest.TestCase):
    def test_retries_conflicts_and_camera_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "observations.sqlite"
            item = Observation(frame(), (Instance((1, 2, 3, 4), .9, 0, "head"),), "custom", "detect", 1)
            with ObservationStore(path) as store:
                self.assertTrue(store.append(item))
                self.assertFalse(store.append(item))
                with self.assertRaises(ValueError):
                    store.append(replace(item, model="other"))
                self.assertTrue(store.append(replace(item, frame=frame(epoch="new"))))
            with ObservationStore(path) as store:
                records = list(store.records())
                self.assertEqual(len(records), 2)
                # Custom class zero remains "head", never implicitly "person".
                self.assertEqual(records[0]["instances"][0]["label"], "head")
                self.assertIsNone(records[0]["instances"][0]["keypoints_xyc"])


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class AdapterTests(unittest.TestCase):
    def result(self):
        return SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[1, 2, 30, 40]]),
            conf=Tensor([.8]), cls=Tensor([0])), names={0: "hands"}, masks=None, keypoints=None)

    def test_custom_taxonomy_and_absent_pose(self):
        record = convert_result(self.result(), frame(), "ppe", 2, "detect")
        self.assertEqual(record.instances[0].label, "hands")
        self.assertEqual(record.instances[0].box_xyxy, (1, 2, 30, 40))
        self.assertIsNone(record.instances[0].keypoints_xyc)

    def test_fast_boxes_do_not_materialize_expensive_mask_polygons(self):
        class Masks:
            @property
            def xy(self):raise AssertionError('Mask conversion must be skipped')
        result=self.result();result.masks=Masks()
        record=convert_result(result,frame(),'ppe',2,'detect',include_masks=False)
        self.assertEqual(record.instances[0].label,'hands')
        self.assertIsNone(record.instances[0].mask_polygon_xy)

    def test_pose_topology_mismatch_is_not_silently_accepted(self):
        result = self.result()
        result.keypoints = SimpleNamespace(data=Tensor([[[1, 2, .9]]]))
        with self.assertRaises(ValueError):
            convert_result(result, frame(), "pose", 2, "pose", "coco17")

    def test_instance_polygons_preserved_in_image_pixels(self):
        result = self.result()
        result.masks = SimpleNamespace(xy=[[(1, 2), (30, 2), (30, 40)]])
        record = convert_result(result, frame(), "seg", 2, "segment")
        self.assertEqual(record.instances[0].mask_polygon_xy[2], (30, 40))


if __name__ == "__main__":
    unittest.main()
