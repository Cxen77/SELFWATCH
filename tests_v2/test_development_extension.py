import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.download_wepdtof_development import make_extension


class DevelopmentExtensionTests(unittest.TestCase):
    def fixture(self,root):
        groups=[f'scene{i}' for i in range(14)];items=[]
        for n in range(14):
            sequence=groups[n];path=root/f'{sequence}.json'
            images=[dict(id=f'{sequence}_{i}',file_name=f'{i:06d}.jpg') for i in range(40)]
            path.write_text(json.dumps(dict(images=images,annotations=[])),encoding='utf-8')
            for image in images[:8]:items.append(dict(sequence=sequence,source_group=sequence,image_id=image['id'],
                annotation_path=path.name,roi_path=f'{sequence}.png'))
        (root/'selection.json').write_text(json.dumps(dict(status='complete',group_order=groups,items=items)))
        (root/'download_manifest.json').write_text(json.dumps(dict(status='complete',files=[])))
        return groups

    def test_expansion_keeps_original_development_samples_and_excludes_all_other_groups(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);groups=self.fixture(root);result=make_extension(root,16)
            self.assertEqual({i['source_group'] for i in result['items']},set(groups[:7]))
            identifiers={i['image_id'] for i in result['items']}
            self.assertTrue(all(f'{group}_{index}' in identifiers for group in groups[:7] for index in range(8)))
            self.assertEqual(len(identifiers),len(result['items']))
            self.assertTrue(all(i['split']=='development' for i in result['items']))

    def test_parent_checksum_changes_if_frozen_pilot_changes(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);first=make_extension(root,16)['parent_selection_sha256']
            with (root/'selection.json').open('a') as file:file.write('\n')
            self.assertNotEqual(first,make_extension(root,16)['parent_selection_sha256'])


if __name__=='__main__':unittest.main()
