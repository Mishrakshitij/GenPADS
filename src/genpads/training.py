"""Supervised PC, DG and G training with explicit PyTorch optimization."""

import json
import math
from pathlib import Path
import random

from .data import LABELS, dataset_path, load_rows
from .models import resolve_device


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_rows(rows, tokenizer, task, max_length=128):
    import torch
    inputs = [row["text" if task == "pc" else "input_text"] for row in rows]
    batch = tokenizer(inputs, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    if task == "pc":
        batch["labels"] = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    else:
        targets = tokenizer(text_target=[row["target_text"] for row in rows], padding=True,
                            truncation=True, max_length=max_length, return_tensors="pt")
        labels = targets["input_ids"]
        labels[targets["attention_mask"] == 0] = -100
        batch["labels"] = labels
    return batch


def forward_loss(model, batch, task):
    """Return summed cross entropy, target count and logits."""
    import torch.nn.functional as F
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    if task != "pc":
        inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=labels)
    logits = model(**inputs).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return loss, int((labels != -100).sum().item()), logits


def classification_metrics(confusion):
    total = sum(map(sum, confusion))
    accuracy = sum(confusion[i][i] for i in range(4)) / total if total else 0.0
    f1s = []
    for i in range(4):
        denominator = sum(confusion[i]) + sum(row[i] for row in confusion)
        f1s.append(2 * confusion[i][i] / denominator if denominator else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / 4,
            "per_class_f1": dict(zip(LABELS, f1s)), "confusion_matrix": confusion}


def evaluate_loader(model, loader, task, device):
    import torch
    was_training = model.training
    model.eval()
    loss_sum, target_count, examples = 0.0, 0, 0
    confusion = [[0] * 4 for _ in range(4)]
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, count, logits = forward_loss(model, batch, task)
            loss_sum += loss.item()
            target_count += count
            examples += len(batch["labels"])
            if task == "pc":
                for target, predicted in zip(batch["labels"].tolist(), logits.argmax(-1).tolist()):
                    confusion[target][predicted] += 1
    model.train(was_training)
    if not target_count:
        raise ValueError("Evaluation partition has no targets")
    nll = loss_sum / target_count
    metrics = {"examples": examples, "targets": target_count, "loss": nll}
    if task == "pc":
        metrics.update(classification_metrics(confusion))
    else:
        metrics["perplexity"] = math.exp(nll) if nll < 700 else float("inf")
    return metrics


def make_loader(rows, tokenizer, task, batch_size, max_length, shuffle=False, seed=42):
    from functools import partial
    import torch
    from torch.utils.data import DataLoader
    if not rows:
        raise ValueError("Selected dataset partition is empty")
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate_rows, tokenizer=tokenizer,
                                         task=task, max_length=max_length))


def train_supervised(task, domain, output, *, data_root="datasets", model_name=None,
                     epochs=None, batch_size=None, learning_rate=4e-5,
                     max_length=128, accumulation_steps=1, max_steps=None,
                     device="auto", seed=42):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
    if task not in {"pc", "dg", "g"}:
        raise ValueError("task must be pc, dg or g")
    epochs = epochs if epochs is not None else (2 if task == "pc" else 6)
    batch_size = batch_size if batch_size is not None else (8 if task == "pc" else 4)
    if min(epochs, batch_size, accumulation_steps, max_length) < 1 or learning_rate <= 0:
        raise ValueError("Training sizes and learning rate must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory is not empty: {output}")
    seed_everything(seed)
    device = resolve_device(device)
    model_name = model_name or ("distilbert/distilbert-base-uncased" if task == "pc" else "facebook/bart-large")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task == "pc":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4, id2label=dict(enumerate(LABELS)),
            label2id={name: i for i, name in enumerate(LABELS)})
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    path = dataset_path(data_root, "padd" if task == "pc" else task, domain)
    train_rows, validation_rows = load_rows(path, "train"), load_rows(path, "validation")
    loader = make_loader(train_rows, tokenizer, task, batch_size, max_length, True, seed)
    validation = make_loader(validation_rows, tokenizer, task, batch_size, max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    output.mkdir(parents=True, exist_ok=True)
    configuration = dict(task=task, domain=domain, model_name=str(model_name),
                         epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                         max_length=max_length, accumulation_steps=accumulation_steps,
                         max_steps=max_steps, seed=seed)
    best, steps, history = float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_targets = 0
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, count, _ = forward_loss(model, batch, task)
            loss.backward()
            accumulated_targets += count
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_targets)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_targets = 0
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
        metrics = evaluate_loader(model, validation, task, device)
        history.append({"epoch": epoch + 1, "optimizer_steps": steps, "validation": metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "genpads.json").write_text(json.dumps({**configuration, "best_validation": metrics}, indent=2) + "\n")
        if max_steps is not None and steps >= max_steps:
            break
    result = {"task": task, "domain": domain, "optimizer_steps": steps,
              "best_validation_loss": best, "history": history}
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate_checkpoint(checkpoint, *, data_root="datasets", split="test", device="auto",
                        batch_size=8, text_metrics=False):
    from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
    configuration = json.loads((Path(checkpoint) / "genpads.json").read_text())
    task, domain = configuration["task"], configuration["domain"]
    rows = load_rows(dataset_path(data_root, "padd" if task == "pc" else task, domain), split)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model_class = AutoModelForSequenceClassification if task == "pc" else AutoModelForSeq2SeqLM
    device = resolve_device(device)
    model = model_class.from_pretrained(checkpoint).to(device)
    loader = make_loader(rows, tokenizer, task, batch_size, configuration["max_length"])
    result = evaluate_loader(model, loader, task, device)
    if text_metrics:
        if task == "pc":
            raise ValueError("Text metrics apply to DG and G checkpoints")
        from .metrics import generation_metrics
        from collections import OrderedDict
        import torch
        model.eval()
        groups = OrderedDict()
        for row in rows:
            groups.setdefault(row["input_text"], set()).add(row["target_text"])
        generation_rows = [{"input_text": source, "target_text": sorted(targets)[0]}
                           for source, targets in groups.items()]
        generation_loader = make_loader(generation_rows, tokenizer, task, batch_size,
                                        configuration["max_length"])
        predictions = []
        with torch.inference_mode():
            for batch in generation_loader:
                encoded = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                generated = model.generate(**encoded, max_new_tokens=configuration["max_length"],
                                           num_beams=4, length_penalty=2.0, early_stopping=True)
                predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        result.update(generation_metrics([sorted(targets) for targets in groups.values()], predictions))
        result["generated_inputs"] = len(groups)
    return {"task": task, "domain": domain, "split": split, **result}
