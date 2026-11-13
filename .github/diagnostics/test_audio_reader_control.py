"""Non-audio process/receipt regressions. No Swift, model, or GPU execution."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('control', ROOT / 'audio-reader-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class ProcessTests(unittest.TestCase):
    def test_timeout_stops_child_before_returning_final_logs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            child = folder / 'child.py'
            child.write_text('''import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    print('child still running', flush=True)
    print('child stderr', file=sys.stderr, flush=True)
    time.sleep(0.02)
''')
            parent = folder / 'parent.py'
            parent.write_text('''import subprocess, sys, time
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
time.sleep(30)
''')
            pid_file = folder / 'child.pid'
            result = control.run_command([sys.executable, str(parent), str(child), str(pid_file)],
                                         folder, timeout=0.5, grace=0.3)
            self.assertTrue(pid_file.exists(), 'fixture child never started')
            pid = int(pid_file.read_text())
            try:
                self.assertFalse(alive(pid), 'timed-out descendant survives phase return')
                before = [(folder / stream).read_bytes() for stream in ('stdout.log', 'stderr.log')]
                time.sleep(0.08)
                self.assertEqual(before, [(folder / stream).read_bytes() for stream in ('stdout.log', 'stderr.log')])
                self.assertEqual(result['exitCode'], 124)
                self.assertTrue(result['timedOut'])
                self.assertTrue(result['cleanupComplete'])
                self.assertIn('SIGKILL', result['cleanup']['signals'])
            finally:
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)  # Only this fixture's recorded child.

    def test_clean_assertion_failure_still_collects_matrix(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            command = self.make_command(folder, "sys.exit(2)")
            verdicts = control.run_phases(self.phases(), folder, executable=str(command))
            self.assertEqual(verdicts['original']['classification'], 'process_failure')
            self.assertFalse(verdicts['original']['timedOut'])
            self.assertEqual(verdicts['original']['exitCode'], 2)
            self.assertEqual(verdicts['matrix']['classification'], 'passed')

    def test_timeout_is_distinct_and_next_phase_requires_drained_group(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            command = self.make_command(folder, 'time.sleep(30)')
            verdicts = control.run_phases(self.phases(), folder, timeout=0.3, grace=0.2, executable=str(command))
            self.assertEqual(verdicts['original']['classification'], 'timeout')
            self.assertEqual(verdicts['original']['exitCode'], 124)
            self.assertTrue(verdicts['original']['cleanupComplete'])
            self.assertEqual(verdicts['matrix']['classification'], 'passed')

    def test_cleanup_failure_refuses_matrix_and_log_interpretation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            command = self.make_command(folder, 'time.sleep(30)')
            real_killpg = os.killpg

            def denied(group, sig):
                if sig != 0:
                    raise PermissionError('fixture-only signal denial')
                return real_killpg(group, sig)

            try:
                with patch.object(control.os, 'killpg', side_effect=denied):
                    verdicts = control.run_phases(self.phases(), folder, timeout=0.3, grace=0.2, executable=str(command))
                self.assertEqual(list(verdicts), ['original'])
                original = verdicts['original']
                self.assertEqual(original['classification'], 'cleanup_failure')
                self.assertTrue(original['timedOut'])
                self.assertFalse(original['logsFinal'])
                self.assertNotIn('passed', original)
                self.assertFalse((folder / 'matrix').exists())
                self.assertTrue(alive(original['processGroup']))
            finally:
                pid_file = folder / 'original.pid'
                if pid_file.exists():
                    pid = int(pid_file.read_text())
                    real_killpg(pid, signal.SIGKILL)  # Exactly this fixture-owned group.
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass

    def test_missing_duplicate_and_skip_cannot_pass_with_zero_exit(self):
        cases = {'missing': "pass", 'duplicate': "print(\"Test Case '-[Tests original]' passed.\\n\" * 2)",
                 'skip': "print(\"Test Case '-[Tests original]' passed.\\nTest skipped\")"}
        for name, body in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                folder = Path(directory)
                command = self.make_command(folder, body)
                verdicts = control.run_phases(self.phases(), folder, executable=str(command))
                self.assertEqual(verdicts['original']['classification'], 'test_inventory_failure')
                self.assertEqual(verdicts['matrix']['classification'], 'passed')

    @staticmethod
    def phases():
        return {'original': ['Tests/original'], 'matrix': ['Tests/matrix']}

    @staticmethod
    def make_command(folder, original):
        command = folder / 'swift-fixture'
        # The matrix stub refuses to claim a pass while the previous process group exists.
        command.write_text(f'''#!{sys.executable}
import os, sys, time
from pathlib import Path
folder = Path({str(folder)!r})
if 'Tests/original' in sys.argv[sys.argv.index('--filter') + 1]:
    (folder / 'original.pid').write_text(str(os.getpid()))
    {original}
else:
    group = int((folder / 'original.pid').read_text())
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        print("Test Case '-[Tests matrix]' passed.")
    else:
        sys.exit('prior group is still alive')
''')
        command.chmod(0o700)
        return command


class IdentityTests(unittest.TestCase):
    def test_no_test_bundle_is_not_an_identity_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            source = folder / 'mlx.metallib'
            source.write_bytes(b'fixture metal')
            with self.assertRaises(ValueError):
                control.binary_identity(folder, source)

    def test_complete_nonempty_proof_contains_expected_binary_and_matching_metal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            source, binary, staged = self.fixture(folder)
            proof = control.binary_identity(folder, source)
            self.assertEqual(len(proof['bundles']), 1)
            self.assertEqual(proof['bundles'][0]['executable']['path'], str(binary))
            self.assertGreater(proof['bundles'][0]['executable']['bytes'], 0)
            self.assertEqual(proof['sourceMetallib']['sha256'], proof['bundles'][0]['metallib']['sha256'])
            self.assertEqual(proof['bundles'][0]['metallib']['path'], str(staged))

    def test_empty_missing_unexecutable_or_mismatched_files_are_rejected(self):
        for defect in ('empty-source', 'empty-binary', 'missing-binary', 'no-exec', 'empty-staged', 'missing-staged', 'different-staged'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                folder = Path(directory)
                source, binary, staged = self.fixture(folder)
                if defect == 'empty-source':
                    source.write_bytes(b'')
                elif defect == 'empty-binary':
                    binary.write_bytes(b'')
                elif defect == 'missing-binary':
                    binary.unlink()
                elif defect == 'no-exec':
                    binary.chmod(0o600)
                elif defect == 'empty-staged':
                    staged.write_bytes(b'')
                elif defect == 'missing-staged':
                    staged.unlink()
                else:
                    staged.write_bytes(b'wrong metallib')
                with self.assertRaises(ValueError):
                    control.binary_identity(folder, source)

    @staticmethod
    def fixture(folder):
        source = folder / 'mlx.metallib'
        source.write_bytes(b'fixture metal')
        macos = folder / 'mlx-swift-lmPackageTests.xctest/Contents/MacOS'
        macos.mkdir(parents=True)
        binary = macos / 'mlx-swift-lmPackageTests'
        binary.write_bytes(b'fixture executable')
        binary.chmod(0o700)
        staged = macos / 'mlx.metallib'
        staged.write_bytes(source.read_bytes())
        return source, binary, staged


if __name__ == '__main__':
    unittest.main()
