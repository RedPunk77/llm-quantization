"""Проверяем выбор модели и передачу его дочерним процессам без GPU."""
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.model_config import artifact_path, MODELS
from scripts.run_smoke_matrix import main


class ModelSelectionTests(unittest.TestCase):
    def test_artifacts_are_isolated(self):
        self.assertEqual(MODELS['qwen2.5-0.5b'], 'Qwen/Qwen2.5-0.5B')
        for fmt in ('fp16', 'int8', 'int4'):
            self.assertNotEqual(artifact_path('qwen2.5-0.5b', fmt), artifact_path('smollm2-135m', fmt))
        self.assertEqual(artifact_path('qwen2.5-0.5b', 'int4').name, 'qwen2.5-0.5b-int4-g64')

    def test_matrix_propagates_model_to_every_child(self):
        with patch('sys.argv', ['run_smoke_matrix.py', '--model', 'qwen2.5-0.5b']), patch('scripts.run_smoke_matrix.subprocess.run') as run:
            main()
        self.assertEqual(run.call_count, 5)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('--model') + 1], 'qwen2.5-0.5b')
            self.assertTrue(call.kwargs['check'])
        self.assertEqual(Path(run.call_args_list[0].args[0][1]).name, 'smoke_mlx.py')
