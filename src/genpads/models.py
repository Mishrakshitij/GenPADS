"""DistilBERT politeness classification and BART response generation."""

import json
from contextlib import contextmanager
from functools import lru_cache
from numbers import Integral
from pathlib import Path

from .data import LABELS


def resolve_device(device="auto"):
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def checkpoint_max_length(checkpoint, override=None):
    """Keep inference token limits consistent with a trained checkpoint."""
    if override is None:
        metadata = Path(checkpoint) / "genpads.json"
        override = json.loads(metadata.read_text()).get("max_length", 128) if metadata.is_file() else 128
    if isinstance(override, bool) or not isinstance(override, Integral) or override < 1:
        raise ValueError("max_length must be a positive integer")
    return int(override)


class PolitenessClassifier:
    """Callable PC adapter returning the canonical four-class label."""

    def __init__(self, checkpoint, device="auto", max_length=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        if self.model.config.num_labels != 4:
            raise ValueError("A four-class GenPADS classifier is required")
        mapping = self.model.config.id2label
        if tuple(mapping.get(i) for i in range(4)) != LABELS:
            raise ValueError("Classifier id2label must match datasets/labels.json")
        self.max_length = checkpoint_max_length(checkpoint, max_length)

    def probabilities(self, texts):
        import torch
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self.model(**encoded).logits.softmax(-1).cpu().tolist()

    @lru_cache(maxsize=4096)
    def __call__(self, text):
        probabilities = self.probabilities([text])[0]
        return max(range(4), key=probabilities.__getitem__)


class ResponseGenerator:
    """Callable seq2seq G or DG adapter using a local trained checkpoint."""

    def __init__(self, checkpoint, device="auto", max_length=None, num_beams=4,
                 do_sample=False, temperature=0.9, top_p=0.9, seed=42):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device).eval()
        self.max_length = checkpoint_max_length(checkpoint, max_length)
        self.num_beams = num_beams
        if self.max_length < 2 or num_beams < 1:
            raise ValueError("max_length must be >= 2 and num_beams >= 1")
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Generation with isolated random state supports CPU and CUDA")
        self.do_sample = bool(do_sample)
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.reseed(seed)

    def reseed(self, seed):
        """Start a private sampling stream without changing PyTorch's global RNG."""
        import torch
        self._cpu_rng_state = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self._cuda_rng_state = (torch.Generator(device=self.device).manual_seed(seed).get_state()
                                if self.device.type == "cuda" else None)

    def get_rng_state(self):
        return {"device": str(self.device), "cpu": self._cpu_rng_state.clone(),
                "cuda": self._cuda_rng_state.clone() if self._cuda_rng_state is not None else None}

    def set_rng_state(self, state):
        if state["device"] != str(self.device):
            raise ValueError("Generator RNG state belongs to a different device")
        self._cpu_rng_state = state["cpu"].clone()
        self._cuda_rng_state = state["cuda"].clone() if state["cuda"] is not None else None

    @contextmanager
    def _sampling_rng(self):
        import torch
        cuda_index = (self.device.index if self.device.index is not None
                      else torch.cuda.current_device()) if self.device.type == "cuda" else None
        devices = [] if cuda_index is None else [cuda_index]
        # generate() samples using the device's default RNG. Swap in our state
        # for this call only, then restore the process state even on errors.
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._cpu_rng_state)
            if cuda_index is not None:
                torch.cuda.set_rng_state(self._cuda_rng_state, cuda_index)
            try:
                yield
            finally:
                self._cpu_rng_state = torch.get_rng_state()
                if cuda_index is not None:
                    self._cuda_rng_state = torch.cuda.get_rng_state(cuda_index)

    def generate(self, texts):
        import torch
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
        sampling_options = ({"temperature": self.temperature, "top_p": self.top_p}
                            if self.do_sample else {})
        beam_options = ({"length_penalty": 2.0, "early_stopping": True}
                        if self.num_beams > 1 else {})
        with torch.inference_mode(), self._sampling_rng():
            tokens = self.model.generate(**encoded, max_new_tokens=self.max_length,
                                         num_beams=self.num_beams, do_sample=self.do_sample,
                                         **beam_options, **sampling_options)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def __call__(self, text):
        return self.generate([text])[0]


def checkpoint_task(path):
    metadata = Path(path) / "genpads.json"
    if not metadata.exists():
        raise ValueError(f"Checkpoint has no GenPADS task metadata: {metadata}")
    return json.loads(metadata.read_text())["task"]
