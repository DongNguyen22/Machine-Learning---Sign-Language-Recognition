import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_webcam_collection import LABELS, auto_assign_participants, validate


class SplitTests(unittest.TestCase):
    def test_complete_and_missing(self):
        assignment = {'p1': 'train', 'p2': 'validation', 'p3': 'test'}
        rows = [
            dict(
                folder_name=p + c,
                participant_id=p,
                session_id='s1',
                gesture_instance_id=c,
                handedness='right',
                background='plain',
                lighting='daylight',
                true_label=c,
                label_confirmed='True',
                image_sha256=p + c,
            )
            for p in assignment
            for c in LABELS
        ]
        self.assertEqual(validate(rows, assignment)[0], [])
        errors = validate(rows[:-1], assignment)[0]
        self.assertIn('test: missing class Z', errors)
        rows[0]['participant_id'] = ''
        self.assertTrue(any('missing participant_id' in e for e in validate(rows, assignment)[0]))

    def test_auto_assign_requires_three_participants(self):
        rows = [dict(participant_id='p1'), dict(participant_id='p2')]
        with self.assertRaises(ValueError):
            auto_assign_participants(rows)

        rows3 = [dict(participant_id=f'p{i}') for i in range(1, 6)]
        assigned = auto_assign_participants(rows3, seed=42)
        self.assertEqual(set(assigned.keys()), {'p1', 'p2', 'p3', 'p4', 'p5'})
        self.assertEqual(set(assigned.values()), {'train', 'validation', 'test'})


if __name__ == '__main__':
    unittest.main()
