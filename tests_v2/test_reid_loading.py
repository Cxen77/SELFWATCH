from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from reid.embedding_extractor import EmbeddingExtractor


class WeightValidationTests(unittest.TestCase):
    def test_strict_loading_rejects_missing_feature_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'partial.pth'
            torch.save({'0.weight':torch.ones(2,3)},path)
            with patch('reid.embedding_extractor._build_osnet_x1_0',lambda:torch.nn.Sequential(torch.nn.Linear(3,2))):
                with self.assertRaisesRegex(ValueError,'Incomplete OSNet'):
                    EmbeddingExtractor(path,device='cpu',strict_weights=True)

    def test_batch_bounds_checked_before_device_work(self):
        encoder=EmbeddingExtractor.__new__(EmbeddingExtractor)
        encoder._MAX_BATCH=1
        encoder.input_height,encoder.input_width=256,128
        with self.assertRaisesRegex(ValueError,'capacity'):
            encoder.extract_batch([np.zeros((256,128,3),dtype=np.uint8)]*2)
        with self.assertRaisesRegex(ValueError,'configured input size'):
            encoder.extract_batch([np.zeros((128,128,3),dtype=np.uint8)])
