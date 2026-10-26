import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from utils import relocate_vscode_storage as migration


class RelocationTests(unittest.TestCase):
    def make_source(self, root):
        source = root / "source"
        source.mkdir()
        (source / "extensions").mkdir()
        (source / "extensions" / "example").write_bytes(b"preserve extension data\x00")
        (source / "backup").symlink_to("missing-target")
        return source

    def test_copy_link_and_idempotent_preserve(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = self.make_source(root)
            expected = migration.inventory(src)
            dst = root / "data"
            migration.migrate(src, dst, check_processes=False, require_other_fs=False)
            self.assertTrue(src.is_symlink())
            self.assertEqual(migration.inventory(src), expected)
            self.assertEqual(migration.inventory(dst), expected)
            self.assertEqual(list(root.glob("*.verified_root_*")), [])
            migration.migrate(src, dst, check_processes=False, require_other_fs=False)

    def test_source_change_keeps_original(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = self.make_source(root)
            real_copy = migration.shutil.copytree
            def changing_copy(source, destination, *args, **kwargs):
                result = real_copy(source, destination, *args, **kwargs)
                if Path(source) == src:
                    (src / "new-log").write_bytes(b"writer restarted")
                return result
            with patch.object(migration.shutil, "copytree", side_effect=changing_copy):
                with self.assertRaises(RuntimeError):
                    migration.migrate(src, root / "data", check_processes=False, require_other_fs=False)
            self.assertFalse(src.is_symlink())
            self.assertTrue((src / "new-log").exists())

    def test_link_error_restores_original(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = self.make_source(root)
            expected = migration.inventory(src)
            with patch.object(Path, "symlink_to", side_effect=OSError("injected ENOSPC")):
                with self.assertRaises(OSError):
                    migration.migrate(src, root / "data", check_processes=False, require_other_fs=False)
            self.assertFalse(src.is_symlink())
            self.assertEqual(migration.inventory(src), expected)

    def test_existing_destination_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = self.make_source(root)
            dst = root / "data"
            dst.mkdir()
            (dst / "existing").write_bytes(b"keep")
            with self.assertRaises(RuntimeError):
                migration.migrate(src, dst, check_processes=False, require_other_fs=False)
            self.assertEqual((dst / "existing").read_bytes(), b"keep")
            self.assertFalse(src.is_symlink())


if __name__ == "__main__":
    unittest.main()
