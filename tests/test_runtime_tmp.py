"""Inherited NAS scratch, cached tempfile choice and real spawn cleanup."""
import multiprocessing
import multiprocessing.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.runtime_tmp import configure_local_temporary_directory


def _report_worker_directory(queue):
    from multiprocessing.util import get_temp_dir
    queue.put((get_temp_dir(), tempfile.gettempdir(), os.environ['TMPDIR']))


class RuntimeTemporaryDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.saved_cache = tempfile.tempdir
        self.environment = patch.dict(os.environ)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        tempfile.tempdir = self.saved_cache

    def test_replaces_inherited_nas_and_cached_directory_with_explicit_local_scratch(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as root:
            target = Path(root)/'ipc'
            os.environ['HEARTFFD_TMPDIR'] = str(target)
            for name in ('TMPDIR','TEMP','TMP'):
                os.environ[name] = '/nonexistent/nfs/runtime'
            tempfile.tempdir = '/nonexistent/nfs/runtime'
            self.assertEqual(configure_local_temporary_directory(), target)
            self.assertEqual(tempfile.gettempdir(), str(target))
            for name in ('TMPDIR','TEMP','TMP'):
                self.assertEqual(os.environ[name], str(target))
            with tempfile.TemporaryFile() as stream:
                stream.write(b'working')

    def test_default_ignores_inherited_tmpdir(self):
        os.environ.pop('HEARTFFD_TMPDIR', None)
        os.environ['TMPDIR'] = '/nonexistent/nfs/runtime'
        target = configure_local_temporary_directory()
        self.assertEqual(target, Path(f'/tmp/heartssm-{os.getuid()}'))
        self.assertEqual(tempfile.gettempdir(), str(target))

    def test_spawned_workers_use_local_pymp_directories_and_remove_them_on_exit(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as root:
            os.environ['HEARTFFD_TMPDIR'] = root
            configure_local_temporary_directory()
            context = multiprocessing.get_context('spawn')
            # Repeat worker teardown, as happens at successive epoch boundaries.
            for _ in range(2):
                queue = context.Queue()
                process = context.Process(target=_report_worker_directory, args=(queue,))
                process.start()
                directory, selected, inherited = queue.get(timeout=15)
                process.join(timeout=15)
                if process.is_alive():
                    process.terminate(); process.join()
                    self.fail('Worker did not exit')
                queue.close(); queue.join_thread()
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(Path(directory).parent, Path(root))
                self.assertEqual(selected, root)
                self.assertEqual(inherited, root)
                self.assertFalse(Path(directory).exists())


if __name__ == '__main__':
    unittest.main()
