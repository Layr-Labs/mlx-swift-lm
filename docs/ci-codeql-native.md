# Native CodeQL activation

`codeql-native.yml` is inactive unless the repository variable `CODEQL_NATIVE`
is exactly `on`. Merging the workflow does not switch GitHub's default setup or
change its security coverage. The operator must review that switch separately.

The scan job has `contents: read` only. It runs the pinned CodeQL2.27.4 ARM64
bundle against the Actions event commit, without building repository code. It
uses the same default query suites for Actions, C/C++ and Python. The Darkbloom
route has one `codeql` Linux ARM64 lane; hosted fallback is also ARM64.

The separate GitHub-hosted upload job has only `contents: read`, `actions: read`
and `security-events: write`. It does not check out or run repository code. Its
only inputs are the same run's three SARIF files and a bounded manifest, checked
against the event repository, commit, ref, run and attempt. Unexpected files,
symlinks, changed hashes, different tools and missing fingerprints are refused.
The official pinned uploader preserves these categories:

- `/language:actions`
- `/language:c-cpp`
- `/language:python`

Native CodeQL already records the primary-line fingerprints. The uploader gets
an empty directory as `checkout_path`; it preserves the supplied fingerprints
and uses the Actions event SHA/ref. No source files are transferred to this job.
GitHub permits code-scanning uploads for `pull_request` runs, including runs with
read-only token scopes. The uploader ignores ref/SHA overrides for fork PRs.

## Operator gates

1. Keep `CODEQL_NATIVE` unset/off while reviewing and merging this source. Verify
   default setup remains enabled and its three analyses are current. Record its
   settings and latest successful analysis IDs for rollback.
2. Approve the single `codeql` lane for the reviewed Linux image and its existing
   network template. Confirm the image/worker and capacity before switching.
3. Switch GitHub from default to advanced setup with admin authorization. Default
   setup blocks CodeQL API uploads, so it must not remain the active uploader.
   Ensure the new workflow is enabled, then set `CODEQL_NATIVE=on` and dispatch
   `main`. The weekly schedule and subsequent main pushes/PRs use this workflow.
4. Require successful native compute, all three processed GitHub analyses on the
   expected commit/categories, and physical runner retirement. Also qualify a
   fork PR upload using its merge ref and unprivileged PR token. An artifact-only
   scan does not prove the upload/alert integration.
5. After the new analyses are healthy, review any stale default-setup analysis
   configuration before removing it; category names alone do not merge distinct
   workflow configurations. Never delete the last working coverage first.

Rollback: turn `CODEQL_NATIVE` off, restore the recorded default-setup settings,
and require its three initial analyses to succeed. Do not leave both modes active.

References:

- [Default versus advanced setup](https://docs.github.com/en/code-security/reference/code-scanning/troubleshoot-analysis-errors/results-different-than-expected)
- [PR uploads with read-only scopes](https://docs.github.com/en/enterprise-cloud@latest/code-security/reference/code-scanning/troubleshoot-analysis-errors/resource-not-accessible)
- [Pinned uploader inputs](https://github.com/github/codeql-action/blob/33d7b85da1a04171e211befe2c739303b9a4c066/upload-sarif/action.yml)
