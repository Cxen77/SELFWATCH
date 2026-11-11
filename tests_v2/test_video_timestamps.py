"""Real video decoding checks; optional OpenCV dependency."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from selfwatch_v2.replay import Source, decode


@unittest.skipUnless(importlib.util.find_spec("cv2"), "Requires OpenCV runtime")
class TimestampDecodeTests(unittest.TestCase):
    def video(self, path):
        import cv2
        import numpy as np
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(2):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        finally:
            writer.release()

    def test_decode_uses_metadata_not_encoded_fps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.avi"
            self.video(path)
            frames = list(decode(Source("A", path, "r", 30, .2, (0, 1.7))))
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[1][0].captured_at_s, 1.9)

    def test_missing_or_extra_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.avi"
            self.video(path)
            for times in ((0,), (0, 1, 2)):
                with self.assertRaises(ValueError):
                    list(decode(Source("A", path, "r", 30, timestamps_s=times)))
