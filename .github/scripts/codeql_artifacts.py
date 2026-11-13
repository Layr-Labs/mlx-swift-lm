#!/usr/bin/env python3
"""Bind read-only CodeQL SARIF to the exact Actions run before upload."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path, PurePosixPath

# BEGIN DATA VALIDATOR
LANGUAGES = {
    "actions": "/language:actions",
    "cpp": "/language:c-cpp",
    "python": "/language:python",
}
CODEQL_VERSION = "2.27.4"
MAX_FILE = 16 * 1024 * 1024
MAX_BUNDLE = 64 * 1024 * 1024


def context():
    return {
        key: os.environ[env]
        for key, env in {
            "repository": "GITHUB_REPOSITORY",
            "sha": "GITHUB_SHA",
            "ref": "GITHUB_REF",
            "run_id": "GITHUB_RUN_ID",
            "attempt": "GITHUB_RUN_ATTEMPT",
        }.items()
    }


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("invalid bundle path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("..", ".git") for part in path.parts):
        raise ValueError("bundle path escapes its data directory")
    if path.as_posix() != value:
        raise ValueError("bundle path is not canonical")
    return path


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_sarif(sarif):
    if sarif.get("version") != "2.1.0" or not sarif.get("runs"):
        raise ValueError("invalid SARIF")
    if not any(
        run.get("tool", {}).get("driver", {}).get("rules") for run in sarif["runs"]
    ):
        raise ValueError("analysis has no rules")
    for run in sarif["runs"]:
        driver = run.get("tool", {}).get("driver", {})
        if (
            driver.get("name") != "CodeQL"
            or driver.get("semanticVersion") != CODEQL_VERSION
        ):
            raise ValueError("SARIF was not produced by the qualified CodeQL version")
        artifacts = run.get("artifacts", [])
        for result in run.get("results", []):
            locations = result.get("locations", [])
            if not locations:
                continue
            primary = locations[0].get("physicalLocation", {})
            location = primary.get("artifactLocation", {})
            if "uri" not in location and "index" in location:
                index = location["index"]
                if type(index) is not int or not 0 <= index < len(artifacts):
                    raise ValueError("invalid artifact index")
                location = artifacts[index]["location"]
            uri = urllib.parse.urlsplit(location.get("uri", ""))
            if uri.scheme or uri.netloc:
                raise ValueError("source location is not repository relative")
            relative_path(urllib.parse.unquote(uri.path))
            if primary.get("region", {}).get("startLine") is not None:
                fingerprint = result.get("partialFingerprints", {}).get(
                    "primaryLocationLineHash", ""
                )
                if not re.fullmatch(r"[0-9a-f]+:[1-9][0-9]*", fingerprint):
                    raise ValueError("source-line finding lacks a stable fingerprint")


def verify_bundle(root, expected):
    root = root.resolve(strict=True)
    manifest_file = root / "manifest.json"
    if manifest_file.is_symlink() or manifest_file.stat().st_size > 256 * 1024:
        raise ValueError("invalid manifest")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("context") != expected or manifest.get("version") != 1:
        raise ValueError(
            "scan belongs to another run, attempt, repository, ref or commit"
        )
    if (
        manifest.get("codeql_version") != CODEQL_VERSION
        or manifest.get("categories") != LANGUAGES
    ):
        raise ValueError("unexpected CodeQL recipe")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 3:
        raise ValueError("invalid file inventory")
    required = {f"sarif/{language}.sarif" for language in LANGUAGES}
    if not required.issubset(files):
        raise ValueError("missing language")
    actual = set()
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks are not scan data")
        if not path.is_file():
            if path.is_dir():
                continue
            raise ValueError("only regular files are scan data")
        rel = path.relative_to(root).as_posix()
        if rel == "manifest.json":
            continue
        relative_path(rel)
        if not path.resolve().is_relative_to(root):
            raise ValueError("file escapes bundle")
        spec = files.get(rel)
        size = path.stat().st_size
        if not isinstance(spec, dict) or not 0 <= size <= MAX_FILE:
            raise ValueError("unexpected or oversized file")
        if spec.get("size") != size or spec.get("sha256") != digest(path):
            raise ValueError("scan data changed")
        if rel not in required:
            raise ValueError("only the three SARIF files may be supplied")
        actual.add(rel)
        total += size
    if actual != set(files) or total > MAX_BUNDLE:
        raise ValueError("file inventory differs or bundle exceeds limit")
    for language in LANGUAGES:
        sarif = json.loads((root / f"sarif/{language}.sarif").read_text())
        validate_sarif(sarif)
    # The official uploader falls back to event SHA/ref without a checkout.
    # Native CodeQL already supplies every primary-line fingerprint.
    empty_checkout = root / "empty-checkout"
    empty_checkout.mkdir(exist_ok=True)
    return empty_checkout


# END DATA VALIDATOR


def pack_bundle(checkout, results, output):
    checkout = checkout.resolve(strict=True)
    expected = context()
    actual_sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != expected["sha"]:
        raise ValueError("checkout is not the event commit")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sarif").mkdir()
    used = 0
    for language in LANGUAGES:
        source = results / f"{language}.sarif"
        if source.is_symlink() or source.stat().st_size > MAX_FILE:
            raise ValueError("invalid SARIF input")
        data = json.loads(source.read_text())
        validate_sarif(data)
        used += source.stat().st_size
        if used > MAX_BUNDLE:
            raise ValueError("bundle exceeds limit")
        destination = output / "sarif" / source.name
        shutil.copyfile(source, destination)
        destination.chmod(0o644)
    files = {
        path.relative_to(output).as_posix(): {
            "sha256": digest(path),
            "size": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "version": 1,
        "context": expected,
        "codeql_version": CODEQL_VERSION,
        "categories": LANGUAGES,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_bundle(output, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack_bundle(args.checkout, args.results, args.output)
