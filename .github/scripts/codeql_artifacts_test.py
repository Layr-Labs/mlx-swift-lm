import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "bundle", Path(__file__).with_name("codeql_artifacts.py")
)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.expected = {
            "repository": "Layr-Labs/mlx",
            "sha": "a" * 40,
            "ref": "refs/pull/42/merge",
            "run_id": "7",
            "attempt": "1",
        }
        (self.root / "sarif").mkdir()
        for language in bundle.LANGUAGES:
            data = {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": "CodeQL",
                                "semanticVersion": "2.27.4",
                                "rules": [{"id": "example"}],
                            }
                        },
                        "results": [],
                    }
                ],
            }
            (self.root / f"sarif/{language}.sarif").write_text(json.dumps(data))
        self.manifest = {
            "version": 1,
            "context": self.expected,
            "codeql_version": "2.27.4",
            "categories": bundle.LANGUAGES,
        }
        self.seal()

    def seal(self):
        self.manifest["files"] = {
            p.relative_to(self.root).as_posix(): {
                "sha256": bundle.digest(p),
                "size": p.stat().st_size,
            }
            for p in self.root.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_bundle_and_fork_merge_attribution(self):
        self.assertEqual(
            bundle.verify_bundle(self.root, self.expected),
            self.root.resolve() / "empty-checkout",
        )

    def test_every_context_dimension_is_bound(self):
        for key in self.expected:
            changed = copy.deepcopy(self.expected)
            changed[key] += "-different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                bundle.verify_bundle(self.root, changed)

    def test_content_changes_are_rejected(self):
        (self.root / "sarif/actions.sarif").write_text("altered content\n")
        with self.assertRaises(ValueError):
            bundle.verify_bundle(self.root, self.expected)

    def test_missing_language_is_rejected(self):
        (self.root / "sarif/python.sarif").unlink()
        self.seal()
        with self.assertRaises(ValueError):
            bundle.verify_bundle(self.root, self.expected)

    def test_extra_files_and_incorrect_inventory_are_rejected(self):
        (self.root / "unexpected.sh").write_text("exit 0\n")
        with self.assertRaises(ValueError):
            bundle.verify_bundle(self.root, self.expected)
        self.seal()
        with self.assertRaises(ValueError):
            bundle.verify_bundle(self.root, self.expected)

    def test_source_symlinks_are_rejected(self):
        (self.root / "link").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "symlink"):
            bundle.verify_bundle(self.root, self.expected)

    def test_ruleless_or_malformed_sarif_is_rejected(self):
        p = self.root / "sarif/actions.sarif"
        p.write_text(
            json.dumps(
                {"version": "2.1.0", "runs": [{"tool": {"driver": {"rules": []}}}]}
            )
        )
        self.seal()
        with self.assertRaisesRegex(ValueError, "no rules"):
            bundle.verify_bundle(self.root, self.expected)

    def test_bounds_are_enforced(self):
        with patch.object(bundle, "MAX_BUNDLE", 1), self.assertRaises(ValueError):
            bundle.verify_bundle(self.root, self.expected)
        with patch.object(bundle, "MAX_FILE", 1), self.assertRaises(ValueError):
            bundle.verify_bundle(self.root, self.expected)

    def test_traversal_git_and_ambiguous_paths_are_rejected(self):
        for path in (
            "../private",
            "/tmp/private",
            "source/../private",
            "source/.git/config",
            "a\\b",
            "a//b",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bundle.relative_path(path)

    def test_primary_locations_and_fingerprints_are_required(self):
        primary = {
            "artifactLocation": {"uri": "mlx/ops.cpp"},
            "region": {"startLine": 12},
        }
        result = {
            "locations": [{"physicalLocation": primary}],
            "partialFingerprints": {"primaryLocationLineHash": "ab12:1"},
        }
        data = {
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "CodeQL",
                            "semanticVersion": "2.27.4",
                            "rules": [{"id": "rule"}],
                        }
                    },
                    "results": [result],
                }
            ],
        }
        bundle.validate_sarif(data)
        del result["partialFingerprints"]
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            bundle.validate_sarif(data)
        primary["artifactLocation"]["uri"] = "../private"
        with self.assertRaises(ValueError):
            bundle.validate_sarif(data)

    def test_different_tool_version_is_rejected(self):
        p = self.root / "sarif/cpp.sarif"
        data = json.loads(p.read_text())
        data["runs"][0]["tool"]["driver"]["semanticVersion"] = "different"
        p.write_text(json.dumps(data))
        self.seal()
        with self.assertRaisesRegex(ValueError, "CodeQL version"):
            bundle.verify_bundle(self.root, self.expected)

    def test_publisher_embeds_the_exact_tested_validator(self):
        source = (
            Path(bundle.__file__)
            .read_text()
            .split("# BEGIN DATA VALIDATOR\n", 1)[1]
            .split("# END DATA VALIDATOR", 1)[0]
        )
        workflow = (
            Path(__file__)
            .parents[1]
            .joinpath("workflows/codeql-native.yml")
            .read_text()
        )
        # YAML removes this common indentation before the shell sees the program.
        embedded = "\n".join(
            line[10:] if line.startswith("          ") else line
            for line in workflow.splitlines()
        )
        self.assertIn(source.strip(), embedded)


if __name__ == "__main__":
    unittest.main()
