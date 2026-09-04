"""Проверки сохранения выбранной ревизии без GPU и сети"""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.smoke_mlx import prepare_artifact


class ArtifactTests(unittest.TestCase):
    def test_converter_receives_local_snapshot_and_reuses_completed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'snapshots' / ('a' * 40)
            snapshot.mkdir(parents=True)
            artifact = root / 'output'
            def download(**kwargs):
                self.assertEqual(kwargs['revision'], 'a' * 40)
                return str(snapshot)
            def convert(path, **kwargs):
                self.assertEqual(Path(path), snapshot)
                self.assertNotIn('revision', kwargs)
                Path(kwargs['mlx_path']).mkdir()
            source = prepare_artifact(artifact, 'test/repo', 'a' * 40, convert, download)
            self.assertEqual(source['revision'], 'a' * 40)
            self.assertEqual(json.loads((artifact / 'source_revision.json').read_text()), source)
            def unexpected(*args, **kwargs):
                self.fail('Completed artifact must be reused')
            self.assertEqual(prepare_artifact(artifact, 'test/repo', None, unexpected, unexpected), source)

    def test_failed_conversion_is_not_published_and_old_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / 'output'
            artifact.mkdir()
            (artifact / 'partial').write_text('keep me')
            def fail(path, **kwargs):
                Path(kwargs['mlx_path']).mkdir()
                raise RuntimeError('save failed')
            with self.assertRaisesRegex(RuntimeError, 'save failed'):
                prepare_artifact(artifact, 'test/repo', None, fail, lambda **kw: str(root / ('b' * 40)))
            self.assertFalse(artifact.exists())
            backup, = root.glob('output.incomplete-*')
            self.assertEqual((backup / 'partial').read_text(), 'keep me')
            self.assertEqual(list(root.glob('.conversion-*')), [])
