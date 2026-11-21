#!/usr/bin/env python3
"""Offline checks of wait-for-products.py: it returns only after both jobs succeeded, and fails loudly otherwise.

Run: python3 .github/scripts/test-wait-for-products.py
"""

import contextlib
import email.message
import importlib.util
import io
from pathlib import Path
import re
import time
import unittest
import urllib.error


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/wait-for-products.py"
SPEC = importlib.util.spec_from_file_location("wait_for_products", SCRIPT)
waiter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(waiter)


def job(name, status="in_progress", conclusion=None):
    return {"name": name, "status": status, "conclusion": conclusion}


def done(name, conclusion="success"):
    return job(name, "completed", conclusion)


def http_error(code, body=b"", **headers):
    message = email.message.Message()
    for key, value in headers.items():
        message[key.replace("_", "-")] = value
    return urllib.error.HTTPError("https://api.github.com/x", code, "refused", message, io.BytesIO(body))


class Clock:
    """A fake clock that sleep advances; fetch answers one poll after another."""

    def __init__(self, answers):
        self.now = 0.0
        self.answers = list(answers)
        self.sleeps = []

    def fetch(self):
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def wait(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return waiter.wait(fetch=self.fetch, clock=lambda: self.now, sleep=self.sleep, **kwargs)


class WaitForProducts(unittest.TestCase):
    def test_returns_once_both_jobs_succeeded(self):
        clock = Clock([
            [job("Build", "queued"), job("Metal library")],
            [job("Build"), done("Metal library")],
            [done("Build"), done("Metal library"), job("Test shard 1")],
        ])
        self.assertEqual(clock.wait(), 30)
        self.assertEqual(clock.sleeps, [15, 15])

    def test_a_finished_build_does_not_end_the_wait_for_the_metal_library(self):
        clock = Clock([[done("Build"), job("Metal library")], [done("Build"), done("Metal library")]])
        self.assertEqual(clock.wait(), 15)

    def test_fails_when_either_job_did_not_succeed(self):
        for failed in ("Build", "Metal library"):
            for conclusion in ("failure", "cancelled", "skipped", "timed_out"):
                with self.subTest(failed=failed, conclusion=conclusion):
                    other = "Metal library" if failed == "Build" else "Build"
                    clock = Clock([[job(other), done(failed, conclusion)]])
                    with self.assertRaisesRegex(waiter.WaitError, f"{failed} ended {conclusion}"):
                        clock.wait()

    def test_fails_when_a_job_is_never_listed(self):
        clock = Clock([[done("Build")]])
        with self.assertRaisesRegex(waiter.WaitError, "lists no job named Metal library"):
            clock.wait()
        self.assertGreaterEqual(clock.now, waiter.MISSING_SECONDS)

    def test_fails_at_the_limit(self):
        clock = Clock([[job("Build"), job("Metal library")]])
        with self.assertRaisesRegex(waiter.WaitError, f"not ready after {waiter.LIMIT_SECONDS} s"):
            clock.wait()
        self.assertEqual(clock.now, waiter.LIMIT_SECONDS)

    def test_the_wait_ends_before_the_test_job_times_out(self):
        # The wait counts against the job's own timeout; it must end with its
        # message first and leave the shard its time.
        workflow = (ROOT / "workflows/fork-tests.yml").read_text()
        for name in ("test", "test-12x32"):
            job = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  \S)", workflow).group(1)
            minutes = int(re.search(r"(?m)^    timeout-minutes: (\d+)$", job).group(1))
            self.assertEqual(minutes, 45, name)
            self.assertLessEqual(waiter.LIMIT_SECONDS, (minutes - 15) * 60, name)

    def test_a_failed_request_is_tried_again_at_the_next_poll(self):
        clock = Clock([OSError("reset"), [done("Build"), done("Metal library")]])
        self.assertEqual(clock.wait(), 15)

    def test_a_refused_token_fails_at_once(self):
        for code in (401, 403, 404):
            with self.subTest(code=code):
                clock = Clock([http_error(code)])
                with self.assertRaisesRegex(waiter.WaitError, "actions: read"):
                    clock.wait()
                self.assertEqual(clock.sleeps, [])

    def test_a_rate_limit_waits_as_long_as_github_asks(self):
        clock = Clock([http_error(403, Retry_After="40"), [done("Build"), done("Metal library")]])
        self.assertEqual(clock.wait(), 40)
        clock = Clock([http_error(429), [done("Build"), done("Metal library")]])
        self.assertEqual(clock.wait(), waiter.RATE_LIMIT_PAUSE_SECONDS)
        # A secondary limit may name itself only in the body.
        secondary = http_error(403, b'{"message": "You have exceeded a secondary rate limit."}')
        clock = Clock([secondary, [done("Build"), done("Metal library")]])
        self.assertEqual(clock.wait(), waiter.RATE_LIMIT_PAUSE_SECONDS)
        # No request left: until the budget resets, and never past the limit.
        reset = str(int(time.time()) + 3 * waiter.LIMIT_SECONDS)
        clock = Clock([http_error(403, X_RateLimit_Remaining="0", X_RateLimit_Reset=reset)])
        with self.assertRaisesRegex(waiter.WaitError, "not ready after"):
            clock.wait()
        self.assertEqual(clock.sleeps, [waiter.LIMIT_SECONDS])

    def test_the_workflow_job_names_are_the_ones_it_waits_for(self):
        workflow = (ROOT / "workflows/fork-tests.yml").read_text()
        for name in waiter.JOBS:
            self.assertEqual(len(re.findall(rf"(?m)^    name: {re.escape(name)}$", workflow)), 1, name)


if __name__ == "__main__":
    unittest.main()
