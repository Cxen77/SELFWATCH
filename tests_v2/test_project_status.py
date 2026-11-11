from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from selfwatch_v2.project_status import read_status


class ProjectStatusTests(unittest.TestCase):
    def test_site_reads_the_actual_master_log(self):
        status=read_status(Path(__file__).resolve().parents[1]/'docs/PROJECT_LOG.md')
        self.assertEqual([s['number'] for s in status['stages']],list(range(12)))
        self.assertTrue(0<=status['current_stage']<12)
        self.assertEqual(len(status['gates']),6)
        self.assertEqual(status['master_log'],'/project-log.md')

    def test_pilot_training_does_not_silently_complete_stages_or_gates(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'log.md'
            rows=['**Updated: today**','**Current stage: 2**','**Status summary:** pilot trained']
            for number in range(12):
                state='In progress' if number==2 else 'Pilot trained' if number==8 else 'Not started'
                rows.append(f'| {number} | Stage {number} | {state} | pending independent evaluation |')
            rows.extend(f'| {gate} | Gate {gate} | Not passed | unmeasured |' for gate in 'ABCDEF')
            path.write_text('\n'.join(rows))
            status=read_status(path)
            self.assertEqual(status['stages'][8]['status'],'Pilot trained')
            self.assertEqual(status['stages'][2]['status'],'In progress')
            self.assertTrue(all(gate['status']=='Not passed' for gate in status['gates']))

    def test_incomplete_master_cannot_silently_publish_a_partial_phase_table(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'log.md'
            path.write_text('**Updated: today**\n**Current stage: 2**\n**Status summary:** unfinished\n| 0 | Inventory | Done | preserved |\n')
            with self.assertRaisesRegex(ValueError,'stages'):read_status(path)


if __name__=='__main__':unittest.main()
