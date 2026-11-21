#!/usr/bin/env python3
"""Wait until this run's Build and Metal library jobs succeeded.

The Fork Tests test jobs do not need Build and Metal library: they start
beside them, so their Mac, checkout and Xcode are ready when the products
are, as in Layr-Labs/d-inference (scripts/ci-test-products.py wait). This
script holds a test job until both jobs of this run's latest attempt have
ended. It fails when one of them did not succeed (there is nothing to test),
when the run lists none of them after MISSING_SECONDS, or after
LIMIT_SECONDS. A re-run of failed test jobs reads the earlier attempt's
products: their jobs are the latest that ran.

It reads the run's jobs with the job's token, which needs `actions: read`.
A request that fails is tried again at the next poll; a refused token fails.

Run: python3 .github/scripts/wait-for-products.py
"""

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

JOBS = ("Build", "Metal library")
POLL_SECONDS = 15
# The build takes 5 to 10 minutes. The limit ends the wait with a message
# before the job's own timeout.
LIMIT_SECONDS = 45 * 60
# GitHub lists every job of a run when it starts; a job still missing after
# this long will not appear.
MISSING_SECONDS = 120
RATE_LIMIT_PAUSE_SECONDS = 60


class WaitError(Exception):
    pass


def run_jobs():
    """The latest execution of each job of this run (one page: the run has fewer than 100)."""
    try:
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        repository, run = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
        token = os.environ["GITHUB_TOKEN"]
    except KeyError as missing:
        raise WaitError(f"{missing.args[0]} is not set; the script runs only in a workflow job")
    request = urllib.request.Request(
        f"{api}/repos/{repository}/actions/runs/{run}/jobs?filter=latest&per_page=100",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "fork-tests-wait-for-products",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response).get("jobs", [])


def pause_after(failure, poll):
    """Seconds to wait after a failed request; raise when the token is refused.

    GitHub answers a rate-limited request with 429, or with 403 and
    Retry-After or no request left. Any other 401, 403 or 404 means that the
    token cannot read the run's jobs.
    """
    if not isinstance(failure, urllib.error.HTTPError) or failure.code not in (401, 403, 404, 429):
        return poll
    headers = failure.headers
    retry_after = headers.get("Retry-After")
    exhausted = headers.get("X-RateLimit-Remaining") == "0"
    if failure.code != 429 and not retry_after and not exhausted:
        raise WaitError(f"The run's jobs cannot be read ({failure}); the job needs the token permission actions: read")
    try:
        if retry_after:
            return max(poll, float(retry_after))
        if exhausted:
            return max(poll, float(headers.get("X-RateLimit-Reset")) - time.time())
    except (TypeError, ValueError):
        pass
    return max(poll, RATE_LIMIT_PAUSE_SECONDS)


def wait(fetch=run_jobs, names=JOBS, poll=POLL_SECONDS, limit=LIMIT_SECONDS, missing=MISSING_SECONDS,
         clock=time.monotonic, sleep=time.sleep):
    """Return once every named job ended successfully; raise when one ended otherwise."""
    started = clock()
    last = None
    while True:
        waited = clock() - started
        try:
            jobs, error, pause = {job.get("name"): job for job in fetch()}, None, poll
        # urllib's HTTPError and URLError are OSErrors; a response cut short
        # raises one of http.client's own errors.
        except (OSError, http.client.HTTPException, ValueError) as failure:
            jobs, error, pause = {}, failure, pause_after(failure, poll)
        found = [jobs[name] for name in names if name in jobs]
        for job in found:
            if job.get("status") == "completed" and job.get("conclusion") != "success":
                raise WaitError(f"{job.get('name')} ended {job.get('conclusion')} after this job had waited "
                                f"{waited:.0f} s; there are no products to test")
        if not error and len(found) == len(names) and all(job.get("status") == "completed" for job in found):
            print(f"{' and '.join(names)} succeeded; waited {waited:.0f} s for the products")
            return waited
        if error:
            state = f"request failed: {error}"
        else:
            state = ", ".join(f"{name} {jobs[name].get('status') if name in jobs else 'not listed'}" for name in names)
        if state != last:
            print(f"At {waited:.0f} s: {state}", flush=True)
            last = state
        if not error and len(found) < len(names) and waited >= missing:
            raise WaitError(f"This run lists no job named {', '.join(n for n in names if n not in jobs)} after {waited:.0f} s")
        if waited >= limit:
            raise WaitError(f"The products were not ready after {waited:.0f} s ({state})")
        sleep(min(pause, limit - waited))


def main():
    try:
        wait()
    except WaitError as error:
        print(f"::error title=Test products::{error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
