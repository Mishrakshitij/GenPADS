"""Offline model checks with tiny randomly initialized local checkpoints."""

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (BartConfig, BartForConditionalGeneration, DistilBertConfig,
                          DistilBertForSequenceClassification, PreTrainedTokenizerFast)

from genpads.models import PolitenessClassifier, ResponseGenerator
from genpads.cli import main
from genpads.training import (classification_metrics, collate_rows, evaluate_checkpoint,
                              forward_loss, train_supervised)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tokenizer(self, directory):
        vocabulary = {word: i for i, word in enumerate(
            ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "hello", "please", "book", "a", "flight", "thanks"])}
        core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        core.pre_tokenizer = Whitespace()
        core.post_processor = TemplateProcessing(single="[BOS] $A [EOS]",
                                                   special_tokens=[("[BOS]", 2), ("[EOS]", 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]",
                                            unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
                                            model_input_names=["input_ids", "attention_mask"])
        tokenizer.save_pretrained(directory)
        return tokenizer

    def fixture(self, root, task):
        checkpoint = root / "initial"
        checkpoint.mkdir()
        tokenizer = self.tokenizer(checkpoint)
        if task == "pc":
            model = DistilBertForSequenceClassification(DistilBertConfig(
                vocab_size=10, dim=16, hidden_dim=24, n_layers=1, n_heads=2,
                max_position_embeddings=32, num_labels=4))
        else:
            model = BartForConditionalGeneration(BartConfig(
                vocab_size=10, d_model=16, encoder_layers=1, decoder_layers=1,
                encoder_attention_heads=2, decoder_attention_heads=2,
                encoder_ffn_dim=24, decoder_ffn_dim=24, max_position_embeddings=32,
                pad_token_id=0, bos_token_id=2, eos_token_id=3, decoder_start_token_id=2,
                forced_eos_token_id=3))
        model.save_pretrained(checkpoint)
        data = root / "datasets" / ("padd" if task == "pc" else "gendd/g")
        data.mkdir(parents=True)
        fields = ["text", "label", "split"] if task == "pc" else ["input_text", "target_text", "split"]
        with (data / "flights.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for split in ("train", "validation", "test"):
                for i in range(4):
                    row = {"text": "hello " + "please " * i, "label": i} if task == "pc" else {
                        "input_text": "book a flight", "target_text": "please " * i + "book a flight"}
                    writer.writerow({**row, "split": split})
        return checkpoint, tokenizer, model

    def test_padding_is_ignored_in_generation_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, tokenizer, model = self.fixture(Path(temporary), "g")
            rows = [{"input_text": "hello", "target_text": "thanks"},
                    {"input_text": "hello", "target_text": "please book a flight"}]
            batch = collate_rows(rows, tokenizer, "g", 16)
            self.assertTrue((batch["labels"] == -100).any())
            model.eval()
            actual, count, _ = forward_loss(model, batch, "g")
            expected = model(**batch).loss
            self.assertTrue(torch.allclose(actual / count, expected, atol=1e-6))
            self.assertEqual(count, sum(len(tokenizer(row["target_text"])["input_ids"]) for row in rows))

    def test_classifier_metrics_include_absent_classes(self):
        metrics = classification_metrics([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 0.25)

    def test_sampling_is_varied_repeatable_and_isolates_global_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _, _ = self.fixture(Path(temporary), "g")
            generator = ResponseGenerator(checkpoint, device="cpu", max_length=8,
                                          num_beams=1, do_sample=True, seed=19)
            global_rng = torch.get_rng_state().clone()
            initial = generator.get_rng_state()
            first = [generator("book a flight") for _ in range(8)]
            self.assertGreater(len(set(first)), 1)
            self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
            generator.set_rng_state(initial)
            self.assertEqual(first, [generator("book a flight") for _ in range(8)])
            generator.reseed(19)
            self.assertEqual(first, [generator("book a flight") for _ in range(8)])

    def test_train_reload_evaluate_both_model_types(self):
        for task in ("pc", "g"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint, _, original = self.fixture(root, task)
                initial = {name: value.detach().clone() for name, value in original.named_parameters()}
                result = train_supervised(task, "flights", root / "trained", data_root=root / "datasets",
                                          model_name=checkpoint, epochs=1, batch_size=1,
                                          accumulation_steps=3, max_steps=2, max_length=16, device="cpu")
                self.assertEqual(result["optimizer_steps"], 2)
                metrics = evaluate_checkpoint(root / "trained", data_root=root / "datasets", device="cpu")
                self.assertEqual(metrics["examples"], 4)
                self.assertGreater(metrics["loss"], 0)
                adapter = (PolitenessClassifier if task == "pc" else ResponseGenerator)(root / "trained", device="cpu")
                self.assertEqual(adapter.max_length, 16)
                self.assertTrue(any(not torch.equal(initial[name], value) for name, value in adapter.model.named_parameters()))
                # Longer than the tiny models' 32 positions: checkpoint-based
                # truncation must work without an explicit adapter override.
                prediction = adapter("please book a flight " * 20)
                self.assertIn(prediction, range(4)) if task == "pc" else self.assertIsInstance(prediction, str)
                cli_output = io.StringIO()
                with redirect_stdout(cli_output):
                    main(["classify" if task == "pc" else "generate",
                          "--checkpoint", str(root / "trained"), "--device", "cpu",
                          "--text", "please book a flight " * 20])
                self.assertIn("label" if task == "pc" else "text", json.loads(cli_output.getvalue()))
                config = json.loads((root / "trained/genpads.json").read_text())
                self.assertEqual(config["task"], task)


if __name__ == "__main__":
    unittest.main()
