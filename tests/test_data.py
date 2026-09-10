import csv
import json
from pathlib import Path
import tempfile
import unittest

from genpads.data import DOMAINS, LABELS, dataset_path, sha256, verify_dataset


class DatasetTests(unittest.TestCase):
    def fixture(self, root):
        manifest = {"files": {}}
        for domain in DOMAINS:
            common = {"conversation_id": domain + "-dialogue", "split": "train"}
            datasets = {
                "padd": [{**common, "utterance_id": domain + "-turn", "speaker": "USER", "text": "please book", "label": "2"}],
                "cleaned": [{**common, "utterance_id": domain + "-turn", "turn_id": 0, "speaker": "USER",
                             "text": "please book", "segments": '[{"text": "book"}]'}],
                "dg": [{**common, "example_id": "pair1", "input_text": "request", "target_text": "please book"}],
                "g": [{**common, "example_id": "g" + str(i), "source_id": "pair1",
                       "input_text": "please book", "target_text": "could you book"} for i in range(2)],
            }
            for task, rows in datasets.items():
                path = dataset_path(root, task, domain)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                entry = {"sha256": sha256(path), "rows": len(rows), "splits": {"train": len(rows)}}
                if task == "padd":
                    entry["labels"] = {"2": 1}
                manifest["files"][path.relative_to(root).as_posix()] = entry
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "labels.json").write_text(json.dumps(dict(enumerate(LABELS))))
        return manifest

    def test_complete_fixture_verifies_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            self.assertEqual(verify_dataset(root)["files"], 28)
            path = dataset_path(root, "padd", "flights")
            path.write_text(path.read_text().replace("please book", "edited text"))
            with self.assertRaisesRegex(ValueError, "Checksum differs"):
                verify_dataset(root)

    def test_wrong_generation_lineage_rejected_even_with_fresh_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            path = dataset_path(root, "g", "flights")
            path.write_text(path.read_text().replace("pair1", "missing-pair"))
            manifest["files"][path.relative_to(root).as_posix()]["sha256"] = sha256(path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "G source differs"):
                verify_dataset(root)

    def test_cross_task_dialogue_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            path = dataset_path(root, "dg", "flights")
            path.write_text(path.read_text().replace("train", "test"))
            spec = manifest["files"][path.relative_to(root).as_posix()]
            spec.update(sha256=sha256(path), splits={"test": 1})
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Dialogue crosses splits"):
                verify_dataset(root)

    def test_incorrect_label_names_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "labels.json").write_text('{"0": "polite"}')
            with self.assertRaisesRegex(ValueError, "canonical class map"):
                verify_dataset(root)

    def test_manifest_summary_counts_are_checked_against_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            manifest["politeness_counts"] = {"2": 8}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Global label counts differ"):
                verify_dataset(root)


if __name__ == "__main__":
    unittest.main()
