import copy
import unittest

from fine_tune import validate_parent


class ParentContractTests(unittest.TestCase):
    def setUp(self):
        self.parent = dict(format='yam_train/act_ref-v1', task='put_block_in_bowl',
            action_mode='delta', action_shift=0, chunk_size=100, fps=30,
            image_size=[480, 640], camera_names=['right_wrist', 'left_wrist'],
            train_episodes=['put_block_in_bowl/episode_0000'],
            val_episodes=['put_block_in_bowl/episode_0001'])

    def test_valid_contract_preserved(self):
        original = copy.deepcopy(self.parent)
        validate_parent(self.parent)
        self.assertEqual(original, self.parent)

    def test_incompatible_action_and_image_contracts_rejected(self):
        for key, value in [('action_mode', 'abs'), ('action_shift', 1),
                           ('image_size', [240, 320]), ('fps', 15),
                           ('camera_names', ['left_wrist', 'right_wrist']),
                           ('task', 'other_task')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_parent({**self.parent, key: value})

    def test_invalid_split_rejected(self):
        for values in [[], self.parent['train_episodes']]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_parent({**self.parent, 'val_episodes': values})


if __name__ == '__main__':
    unittest.main()
