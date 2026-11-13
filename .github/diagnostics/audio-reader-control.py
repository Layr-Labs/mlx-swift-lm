"""Bounded audio diagnostic control with exact source and runner identity."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SOURCE = '76ee2ff71649d8d0982546a88b9ccf8aa04013fc'
RECEIPT = None
CONTROL = Path('../control/.github/diagnostics')
FILTER = r'^MLXLMTests\.(AudioReaderRetryTests|MiMoV26EncodedAACAudioTests)/'

def output(*command):
    return subprocess.check_output(command, text=True).strip()

def save(name, value):
    (RECEIPT / name).write_text(json.dumps(value, indent=2) + '\n')

def binary_identity(binary_dir, source_metallib):
    def proof(file, executable=False):
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError('missing/empty identity file: ' + str(file))
        if executable and not os.access(file, os.X_OK):
            raise ValueError('test binary is not executable: ' + str(file))
        return {'path': str(file), 'bytes': file.stat().st_size,
                'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}

    source = proof(source_metallib)
    bundles = sorted(binary_dir.glob('*.xctest'))
    if not bundles or not any(bundle.name == 'mlx-swift-lmPackageTests.xctest' for bundle in bundles):
        raise ValueError('expected mlx-swift-lmPackageTests.xctest is absent')
    verified = []
    for bundle in bundles:
        executable = proof(bundle / 'Contents/MacOS' / bundle.stem, executable=True)
        staged = proof(bundle / 'Contents/MacOS/mlx.metallib')
        if staged['sha256'] != source['sha256'] or staged['bytes'] != source['bytes']:
            raise ValueError('staged metallib differs from built source: ' + staged['path'])
        verified.append({'executable': executable, 'metallib': staged})
    return {'sourceMetallib': source, 'bundles': verified}

def group_present(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can report EPERM briefly while a terminated leader is reaped.
        # Permission denial is never absence proof; keep polling within the bound.
        return True


def await_group_exit(process, deadline):
    while True:
        process.poll()  # Reap the direct child, including after group termination.
        if not group_present(process.pid) and process.returncode is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def drain_group(process, grace):
    """Only target the session created by this exact Popen invocation."""
    cleanup = {'complete': False, 'signals': [], 'errors': []}
    if process.pid == os.getpgrp():
        cleanup['errors'].append('refusing to signal the runner process group')
        return cleanup
    try:
        if await_group_exit(process, time.monotonic()):
            cleanup['complete'] = True
            return cleanup
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
                cleanup['signals'].append(sig.name)
            except ProcessLookupError:
                pass
            if await_group_exit(process, time.monotonic() + grace):
                cleanup['complete'] = True
                return cleanup
    except OSError as error:
        cleanup['errors'].append(str(error))
    return cleanup


def run_command(command, folder, timeout=600, grace=5):
    """Return only after group drainage, or an explicit failed isolation receipt.

    Raw streams are regular files (no pipe-reader deadlock). Their final bytes are
    read by the caller only when the process group has no remaining writers.
    SwiftPM/test children inherit this session; deliberately detached daemons
    are outside this control's execution contract.
    """
    with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup = drain_group(process, grace)
        return {'exitCode': 124 if timed_out else process.returncode,
                'processReturnCode': process.returncode, 'timedOut': timed_out,
                'processGroup': process.pid, 'cleanupComplete': cleanup['complete'],
                'cleanup': cleanup,
                'descendantsOutlivedLeader': not timed_out and bool(cleanup['signals'])}


def run_phases(phases, receipt, timeout=600, grace=5, executable='swift'):
    verdicts = {}
    for phase, selected in phases.items():
        folder = receipt / phase
        folder.mkdir()
        selection = '^(?:' + '|'.join(re.escape(test) for test in selected) + ')$'
        command = [executable, 'test', '--build-system', 'native', '--skip-build', '--no-parallel',
                   '--filter', selection, '--xunit-output', str(folder / 'xunit.xml'),
                   '--event-stream-output-path', str(folder / 'events.jsonl'), '--event-stream-version', '0']
        execution = run_command(command, folder, timeout, grace)
        verdict = {**execution, 'expected': sorted(selected),
                   'logsFinal': execution['cleanupComplete']}
        if not execution['cleanupComplete']:
            # Inherited descriptors may still have writers. Do not interpret mutable logs.
            verdict['classification'] = 'cleanup_failure'
        else:
            combined = (folder / 'stdout.log').read_text() + '\n' + (folder / 'stderr.log').read_text()
            passed = re.findall(r"Test Case '-\[([^ ]+) ([^]]+)\]' passed", combined)
            verdict['passed'] = sorted(class_name + '/' + method for class_name, method in passed)
            verdict['hasSkip'] = bool(re.search(r'Test skipped|with [1-9][0-9]* tests? skipped', combined))
            if execution['timedOut']:
                verdict['classification'] = 'timeout'
            elif execution['descendantsOutlivedLeader']:
                verdict['classification'] = 'descendant_leak'
            elif execution['exitCode'] != 0:
                verdict['classification'] = 'process_failure'
            elif verdict['passed'] != sorted(selected) or verdict['hasSkip']:
                verdict['classification'] = 'test_inventory_failure'
            else:
                verdict['classification'] = 'passed'
        (folder / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
        verdicts[phase] = verdict
        if not execution['cleanupComplete']:
            break  # A following phase would violate isolation.
    return verdicts


def main():
    global RECEIPT
    RECEIPT = Path(os.environ["RUNNER_TEMP"]) / "audio-receipt"
    RECEIPT.mkdir(exist_ok=True)
    if sys.argv[1] == 'identity':
        identity = {
            'source': output('git', 'rev-parse', 'HEAD'),
            'tree': output('git', 'rev-parse', 'HEAD^{tree}'),
            'cpu': output('sysctl', '-n', 'machdep.cpu.brand_string'),
            'cpus': output('sysctl', '-n', 'hw.ncpu'),
            'memory': output('sysctl', '-n', 'hw.memsize'),
            'os': output('sw_vers'), 'xcode': output('xcodebuild', '-version'),
            'developerDir': output('xcode-select', '-p'),
            'runnerName': os.environ.get('RUNNER_NAME'),
        }
        save('identity.json', identity)
        if identity['source'] != SOURCE or identity['tree'] != '7bae22208eb081ee1f16e189d224014f1a1df661':
            raise SystemExit('NOT RUN: source identity mismatch')
        if (identity['cpu'], identity['cpus'], identity['memory']) != ('Apple M5 Max (Virtual)', '12', '34359738368'):
            raise SystemExit('NOT RUN: allocated hardware differs from observed failing class')
        if 'Xcode 27.2' not in identity['xcode'] or '27B52' not in identity['xcode']:
            raise SystemExit('NOT RUN: compiler differs from control')
    elif sys.argv[1] == 'pins':
        expected = json.loads((CONTROL / 'audio-reader-control.resolved.json').read_text())
        actual = json.loads(Path('Package.resolved').read_text())
        normalize = lambda data: sorted(data['pins'], key=lambda pin: pin['identity'])
        (RECEIPT / 'Package.resolved').write_bytes(Path('Package.resolved').read_bytes())
        save('pins.json', {'match': normalize(expected) == normalize(actual), 'pins': normalize(actual)})
        if normalize(expected) != normalize(actual):
            raise SystemExit('NOT RUN: dependency graph changed')
    elif sys.argv[1] == 'tests':
        expected = (CONTROL / 'audio-reader-control.tests.txt').read_text().splitlines()
        actual = [line for line in (RECEIPT / 'tests.txt').read_text().splitlines() if re.match(FILTER, line)]
        if sorted(actual) != sorted(expected) or len(expected) != 9:
            raise SystemExit('NOT RUN: audio test inventory changed')
        binary_dir = Path(output('swift', 'build', '--build-system', 'native', '--show-bin-path'))
        try:
            save('binary-hashes.json', binary_identity(binary_dir, RECEIPT / 'mlx.metallib'))
        except ValueError as error:
            save('identity-error.json', {'classification': 'not_run_identity', 'error': str(error)})
            raise SystemExit('NOT RUN: ' + str(error))
        phases = {
            'original': [test for test in expected if 'testAudioReaderDiagnosticMatrix' not in test],
            'matrix': [test for test in expected if 'testAudioReaderDiagnosticMatrix' in test],
        }
        verdicts = run_phases(phases, RECEIPT)
        save('phases.json', verdicts)
        failures = [phase for phase, verdict in verdicts.items() if verdict['classification'] != 'passed']
        if failures:
            raise SystemExit('FAILED: ' + ', '.join(phase + '=' + verdicts[phase]['classification'] for phase in failures))

    else:
        raise SystemExit('identity | pins | tests')


if __name__ == "__main__":
    main()
