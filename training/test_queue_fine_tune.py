import unittest
from unittest.mock import patch

from queue_fine_tune import idle_gpus


class IdleResourceTests(unittest.TestCase):
    def test_allocated_idle_gpu_is_not_free(self):
        with patch('queue_fine_tune.subprocess.check_output', return_value='0, 29617, 0\n1, 5, 0\n'):
            self.assertEqual(idle_gpus(), [1])

    def test_busy_or_nearly_threshold_gpu_is_not_free(self):
        with patch('queue_fine_tune.subprocess.check_output',
                   return_value='0, 999, 5\n1, 1000, 0\n2, 999, 4\n'):
            self.assertEqual(idle_gpus(), [2])

    def test_resource_query_failure_does_not_claim_a_free_gpu(self):
        with patch('queue_fine_tune.subprocess.check_output', side_effect=RuntimeError('query failed')):
            with self.assertRaises(RuntimeError):
                idle_gpus()


if __name__ == '__main__':
    unittest.main()
