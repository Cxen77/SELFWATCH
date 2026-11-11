from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from selfwatch_v2.prepare_overhead_training import prepare,export_sources


class TrainingPartitionTests(unittest.TestCase):
    def fixture(self,folder):
        parent=Path(folder)/'parent';extension=Path(folder)/'extension'
        roles={'dev':'development','val':'validation','test':'sealed_test'}
        original=dict(source='official-source',items=[dict(image_id=g,source_group=g,
            image_sha256='source-checksum') for g in roles])
        expanded=dict(source='official-source',items=[dict(original['items'][0])])
        for root,selection in ((parent,original),(extension,expanded)):
            root.mkdir();(root/'NOTICE.txt').write_text('Research dataset notice')
            (root/'selection.json').write_text(json.dumps(selection))
        expanded['parent_selection_sha256']=sha256((parent/'selection.json').read_bytes()).hexdigest()
        (extension/'selection.json').write_text(json.dumps(expanded))
        def load(root):return (expanded if Path(root)==extension else original),roles
        return parent,extension,original,expanded,load

    def test_combined_export_decodes_only_development_extension_and_parent_validation(self):
        with TemporaryDirectory() as folder:
            parent,extension,original,expanded,load=self.fixture(folder)
            decoded=[]
            def pixels(root,item):
                decoded.append((root,item['image_id']))
                return np.zeros((20,30,3),np.uint8),None,[dict(box_xyxy=(1,2,10,15))],{}
            with patch('selfwatch_v2.prepare_overhead_training.load_selection',side_effect=load),patch(
                'selfwatch_v2.prepare_overhead_training.load_image',side_effect=pixels):
                manifest=prepare(extension,Path(folder)/'export',parent)
            self.assertEqual(decoded,[(extension,'dev'),(parent,'val')])
            self.assertEqual(manifest['counts']['validation_images'],1)
            self.assertEqual(manifest['partition_sources']['validation']['root'],str(parent.resolve()))
            self.assertFalse((Path(folder)/'export/images/sealed_test').exists())

    def test_reject_changed_parent_before_creating_output(self):
        with TemporaryDirectory() as folder:
            parent,extension,original,expanded,load=self.fixture(folder)
            with (parent/'selection.json').open('a') as file:file.write('\n')
            with patch('selfwatch_v2.prepare_overhead_training.load_selection',side_effect=load):
                with self.assertRaisesRegex(ValueError,'frozen validation parent'):
                    prepare(extension,Path(folder)/'export',parent)
            self.assertFalse((Path(folder)/'export').exists())

    def test_reject_validation_or_test_in_development_extension(self):
        with TemporaryDirectory() as folder:
            parent,extension,original,expanded,load=self.fixture(folder)
            for forbidden in original['items'][1:]:
                expanded['items']=[forbidden]
                with patch('selfwatch_v2.prepare_overhead_training.load_selection',side_effect=load):
                    with self.assertRaisesRegex(ValueError,'non-development'):
                        export_sources(extension,parent)

    def test_development_only_without_parent_is_rejected(self):
        with TemporaryDirectory() as folder:
            parent,extension,original,expanded,load=self.fixture(folder)
            with patch('selfwatch_v2.prepare_overhead_training.load_selection',side_effect=load):
                with self.assertRaisesRegex(ValueError,'No validation'):
                    export_sources(extension)

    def test_original_export_never_decodes_final_test(self):
        with TemporaryDirectory() as folder:
            parent,extension,original,expanded,load=self.fixture(folder)
            with patch('selfwatch_v2.prepare_overhead_training.load_selection',side_effect=load):
                _,_,_,plan=export_sources(parent)
            self.assertEqual([item['image_id'] for _,_,item in plan],['dev','val'])


if __name__=='__main__':unittest.main()
