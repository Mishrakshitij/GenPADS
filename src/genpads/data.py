"""CSV access and integrity checks; no model downloads are needed."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

DOMAINS = ("flights", "food-ordering", "hotels", "movies", "music",
           "restaurant-search", "sports")
LABELS = ("impolite", "somewhat_impolite", "somewhat_polite", "polite")
SPLITS = ("train", "validation", "test")


def load_rows(path, split=None):
    """Read UTF-8 records, optionally selecting a complete-dialogue partition."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if split is None or row["split"] == split]


def dataset_path(root, task, domain):
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    if task not in ("padd", "cleaned", "dg", "g"):
        raise ValueError(f"Unknown dataset: {task}")
    return Path(root) / ("padd" if task == "padd" else f"gendd/{task}") / f"{domain}.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(root="datasets"):
    """Validate checksums, schemas, split membership, labels and pair grouping."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    label_map = json.loads((root / "labels.json").read_text())
    if label_map != {str(i): name for i, name in enumerate(LABELS)}:
        raise ValueError("Dataset verification failed: labels.json differs from the canonical class map")
    errors, counts = [], {}
    global_conversations, utterance_keys = {}, set()
    global_labels = Counter()
    padd_ids = defaultdict(set)
    padd_utterances = {}
    padd_text_labels = {}
    cleaned_ids = set()
    dg_ids, g_counts, response_splits = {}, Counter(), {}
    for task in ("padd", "cleaned", "dg", "g"):
        for domain in DOMAINS:
            path = dataset_path(root, task, domain)
            relative = path.relative_to(root).as_posix()
            spec = manifest["files"].get(relative)
            if not path.is_file() or spec is None:
                errors.append(f"Missing dataset or manifest entry: {relative}")
                continue
            if sha256(path) != spec["sha256"]:
                errors.append(f"Checksum differs: {relative}")
            rows = load_rows(path)
            required = {"conversation_id", "split"}
            required.update({
                "padd": {"utterance_id", "speaker", "text", "label"},
                "cleaned": {"utterance_id", "turn_id", "speaker", "text", "segments"},
                "dg": {"example_id", "input_text", "target_text"},
                "g": {"example_id", "source_id", "input_text", "target_text"},
            }[task])
            if not rows or not required.issubset(rows[0]):
                errors.append(f"Missing required columns or empty file: {relative}")
                continue
            if len(rows) != spec["rows"]:
                errors.append(f"Row count differs: {relative}")
            if "columns" in spec and list(rows[0]) != spec["columns"]:
                errors.append(f"Column order differs: {relative}")
            if "conversations" in spec and len({r.get("conversation_id") for r in rows}) != spec["conversations"]:
                errors.append(f"Conversation count differs: {relative}")
            counts[relative] = len(rows)
            split_counts, label_counts = Counter(), Counter()
            seen_examples = set()
            for index, row in enumerate(rows, 2):
                at = f"{relative}:{index}"
                if None in row or any(value is None for value in row.values()):
                    errors.append(f"Malformed CSV row: {at}")
                    continue
                conversation = row.get("conversation_id", "")
                split = row.get("split", "")
                if not conversation or split not in SPLITS:
                    errors.append(f"Invalid dialogue or split: {at}")
                if global_conversations.setdefault(conversation, split) != split:
                    errors.append(f"Dialogue crosses splits: {at}")
                split_counts[split] += 1
                texts = ("text",) if task in ("padd", "cleaned") else ("input_text", "target_text")
                if any(not row.get(key, "").strip() for key in texts):
                    errors.append(f"Empty text: {at}")
                if task in ("padd", "cleaned") and row.get("speaker") not in {"USER", "ASSISTANT"}:
                    errors.append(f"Invalid speaker: {at}")
                if task == "padd":
                    if row.get("label") not in {"0", "1", "2", "3"}:
                        errors.append(f"Invalid label: {at}")
                    label_counts[row.get("label")] += 1
                    global_labels[row.get("label")] += 1
                    text_key = (domain, row["text"])
                    if padd_text_labels.setdefault(text_key, row["label"]) != row["label"]:
                        errors.append(f"Repeated text has conflicting labels: {at}")
                    key = (domain, row.get("utterance_id"))
                    if key in utterance_keys or not key[1]:
                        errors.append(f"Repeated or missing utterance ID: {at}")
                    utterance_keys.add(key)
                    padd_ids[domain].add(conversation)
                    padd_utterances[key] = (conversation, row["speaker"], row["text"], split)
                elif conversation not in padd_ids[domain]:
                    errors.append(f"Dialogue absent from PADD: {at}")
                if task == "cleaned":
                    key = (domain, row["utterance_id"])
                    if key in cleaned_ids:
                        errors.append(f"Duplicate cleaned utterance: {at}")
                    cleaned_ids.add(key)
                    if padd_utterances.get(key) != (conversation, row["speaker"], row["text"], split):
                        errors.append(f"Cleaned turn differs from PADD: {at}")
                    try:
                        segments = json.loads(row["segments"])
                        if not isinstance(segments, list) or not segments:
                            raise ValueError("empty slot list")
                        int(row["turn_id"])
                    except (ValueError, TypeError):
                        errors.append(f"Invalid slot data or turn index: {at}")
                if task == "dg":
                    example = row.get("example_id")
                    if not example or example in seen_examples:
                        errors.append(f"Duplicate DG example ID: {at}")
                    seen_examples.add(example)
                    dg_ids[(domain, example)] = row
                    response_key = (domain, " ".join(row.get("target_text", "").casefold().split()))
                    if response_splits.setdefault(response_key, split) != split:
                        errors.append(f"Generation source text crosses splits: {at}")
                if task == "g":
                    source = dg_ids.get((domain, row.get("source_id")))
                    if (source is None or row["input_text"] != source["target_text"]
                            or conversation != source["conversation_id"]):
                        errors.append(f"G source differs from DG target: {at}")
                    key = row.get("example_id")
                    if not key or key in seen_examples:
                        errors.append(f"Duplicate G example ID: {at}")
                    seen_examples.add(key)
                    g_counts[(domain, row.get("source_id"))] += 1
            if dict(split_counts) != spec["splits"]:
                errors.append(f"Split counts differ: {relative}")
            if task == "padd" and dict(label_counts) != spec["labels"]:
                errors.append(f"Label counts differ: {relative}")
            if task == "g":
                statistics = {
                    "source_examples": len({r.get("source_id") for r in rows}),
                    "unchanged_target_rows": sum(r.get("input_text") == r.get("target_text") for r in rows),
                    "distinct_pairs": len({(r.get("input_text"), r.get("target_text")) for r in rows}),
                    "distinct_changed_pairs": len({(r.get("input_text"), r.get("target_text")) for r in rows
                                                   if r.get("input_text") != r.get("target_text")}),
                }
                for key, value in statistics.items():
                    if key in spec and value != spec[key]:
                        errors.append(f"Generator statistic {key} differs: {relative}")
    if any(g_counts[key] != 2 for key in dg_ids):
        errors.append("Every DG target must have two G records")
    if "conversation_ids" in manifest and len(global_conversations) != manifest["conversation_ids"]:
        errors.append("Global conversation count differs")
    if "politeness_counts" in manifest and dict(global_labels) != manifest["politeness_counts"]:
        errors.append("Global label counts differ")
    if errors:
        raise ValueError("Dataset verification failed:\n" + "\n".join(errors[:30]))
    return {"verified": True, "files": len(counts), "rows": counts,
            "conversations": len(global_conversations)}
