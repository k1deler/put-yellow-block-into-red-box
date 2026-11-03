"""No GPU, checkpoint, private data or robot imports are needed for these tests."""
import copy
import unittest

from selection_quality import OBJECTIVES, select_candidate, evaluation_reference


def candidate(name, arm, grip, first_arm=None, first_grip=None, epoch=0):
    return dict(checkpoint=name, epoch=epoch, metrics=dict(zip(OBJECTIVES, (
        arm, grip, arm if first_arm is None else first_arm,
        grip if first_grip is None else first_grip))))


class SelectionQualityTests(unittest.TestCase):
    def test_full_data_diagnostics_require_acknowledgement(self):
        ck=dict(train_episodes=['a','b'],val_episodes=[],diagnostic_reference_episodes=['b'])
        with self.assertRaises(ValueError):evaluation_reference(ck)
        names,scope=evaluation_reference(ck,True)
        self.assertEqual(names,{'b'})
        self.assertFalse(scope['reference_is_held_out'])
        self.assertEqual(scope['reference_training_overlap'],['b'])

    def test_held_out_scope_is_preserved(self):
        names,scope=evaluation_reference(dict(train_episodes=['a'],val_episodes=['b']))
        self.assertEqual(names,{'b'})
        self.assertTrue(scope['reference_is_held_out'])

    def test_small_arm_gain_cannot_hide_large_gripper_regression(self):
        # The former arm-first tuple chooses A, despite its 3x gripper error.
        rows = [candidate('A', .100, .30), candidate('B', .105, .10)]
        self.assertEqual(min(rows, key=lambda r: (r['metrics']['right_arm_rad'],
                                               r['metrics']['right_grip']))['checkpoint'], 'A')
        selected, evidence = select_candidate(rows)
        self.assertEqual(selected['checkpoint'], 'B')
        self.assertFalse(evidence['deployment_authorized'])

    def test_dominated_checkpoint_is_excluded(self):
        _, evidence = select_candidate([candidate('good', .1, .1), candidate('bad', .2, .2)])
        self.assertEqual([r['checkpoint'] for r in evidence['ranking'] if r['pareto']], ['good'])

    def test_first_step_errors_are_included(self):
        rows = [candidate('late', .1, .1, .4, .4), candidate('balanced', .11, .11, .1, .1)]
        self.assertEqual(select_candidate(rows)[0]['checkpoint'], 'balanced')

    def test_units_and_input_order_do_not_change_selection(self):
        rows = [candidate('A', .1, .3), candidate('B', .105, .1)]
        scaled = copy.deepcopy(rows)
        for row in scaled:
            row['metrics']['right_arm_rad'] *= 1000
            row['metrics']['first_step_right_arm_rad'] *= 1000
        self.assertEqual(select_candidate(rows)[0]['checkpoint'], select_candidate(scaled[::-1])[0]['checkpoint'])

    def test_invalid_results_are_rejected(self):
        for value in (float('nan'), float('inf'), -.1, True, '0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_candidate([candidate('bad', value, .1)])
        with self.assertRaises(ValueError):
            select_candidate([])
        with self.assertRaises(KeyError):
            select_candidate([dict(checkpoint='missing', epoch=0, metrics={})])

    def test_zero_errors_and_ties_are_deterministic(self):
        rows = [candidate('later', 0., 0., epoch=10), candidate('earlier', 0., 0., epoch=5)]
        self.assertEqual(select_candidate(rows)[0]['checkpoint'], 'earlier')

    def test_input_records_are_not_modified(self):
        rows = [candidate('A', .1, .3), candidate('B', .105, .1)]
        original = copy.deepcopy(rows)
        select_candidate(rows)
        self.assertEqual(rows, original)


if __name__ == '__main__':
    unittest.main()
