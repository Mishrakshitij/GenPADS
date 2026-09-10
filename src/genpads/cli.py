"""Command line entry points for data, supervised models and dialogue policies."""

import argparse
import json
from pathlib import Path

from .data import DOMAINS, SPLITS, verify_dataset


def parser():
    root = argparse.ArgumentParser(prog="genpads", description="GenPADS training and evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-data", help="Verify all released CSV files")
    verify.add_argument("--data-root", default="datasets")
    train = commands.add_parser("train", help="Train a domain-specific PC, DG or G model")
    train.add_argument("--task", choices=("pc", "dg", "g"), required=True)
    train.add_argument("--domain", choices=DOMAINS, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--data-root", default="datasets")
    train.add_argument("--model-name")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float, default=4e-5)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--accumulation-steps", type=int, default=1)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="Evaluate a supervised checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-root", default="datasets")
    evaluate.add_argument("--split", choices=SPLITS, default="test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--text-metrics", action="store_true")
    generate = commands.add_parser("generate", help="Generate with a trained G or DG checkpoint")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--max-length", type=int, help="Override the checkpoint token limit")
    generate.add_argument("--num-beams", type=int, default=4)
    generate.add_argument("--sample", action="store_true", help="Sample a response instead of beam decoding")
    generate.add_argument("--seed", type=int, default=42)
    classify = commands.add_parser("classify", help="Predict a politeness label")
    classify.add_argument("--checkpoint", required=True)
    classify.add_argument("--text", required=True)
    classify.add_argument("--device", default="auto")
    classify.add_argument("--max-length", type=int, help="Override the checkpoint token limit")
    for name, help_text in (("simulate", "Run or evaluate a slot-filling dialogue policy"),
                            ("train-policy", "Train LSTM REINFORCE with dialogue rewards")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--domain", choices=DOMAINS, required=True)
        command.add_argument("--episodes", type=int, default=100 if name == "simulate" else 8000)
        command.add_argument("--reward-mode", choices=("baseline", "prrp"), default="prrp")
        command.add_argument("--classifier", help="Trained domain PC checkpoint")
        command.add_argument("--generator", help="Trained domain G checkpoint")
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--max-turns", type=int)
        command.add_argument("--threads", type=int, default=1)
        if name == "simulate":
            command.add_argument("--policy", help="Saved policy.pt or checkpoint directory")
            command.add_argument("--greedy", action="store_true")
            command.add_argument("--record-dialogues", action="store_true")
        else:
            command.add_argument("--output", required=True)
            command.add_argument("--evaluate-every", type=int, default=400)
            command.add_argument("--evaluation-episodes", type=int, default=500)
            command.add_argument("--exploration", type=float, default=0.1)
            command.add_argument("--learning-rate", type=float, default=1.0)
            command.add_argument("--discount", type=float, default=0.9)
    return root


def main(argv=None):
    root = parser()
    args = vars(root.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "verify-data":
            result = verify_dataset(args["data_root"])
        elif command == "train":
            from .training import train_supervised
            result = train_supervised(**args)
        elif command == "evaluate":
            from .training import evaluate_checkpoint
            result = evaluate_checkpoint(**args)
        elif command == "generate":
            from .models import ResponseGenerator, checkpoint_task
            text = args.pop("text")
            if checkpoint_task(args["checkpoint"]) not in ("dg", "g"):
                raise ValueError("Generation requires a DG or G checkpoint")
            args["do_sample"] = args.pop("sample")
            result = {"text": ResponseGenerator(**args)(text)}
        elif command == "classify":
            from .data import LABELS
            from .models import PolitenessClassifier
            text = args.pop("text")
            probabilities = PolitenessClassifier(**args).probabilities([text])[0]
            label = max(range(4), key=probabilities.__getitem__)
            result = {"label": label, "name": LABELS[label], "probabilities": probabilities}
        else:
            from .models import PolitenessClassifier, ResponseGenerator, resolve_device
            from .simulation import simulate, train_policy
            from .policy import ReinforcePolicy
            threads = args.pop("threads")
            if threads < 1:
                raise ValueError("threads must be positive")
            if command == "train-policy" or args.get("policy") or args["classifier"] or args["generator"]:
                import torch
                torch.set_num_threads(threads)
                args["device"] = str(resolve_device(args["device"]))
            if args["generator"] and not args["classifier"]:
                raise ValueError("Generated responses require --classifier for politeness feedback")
            for flag, expected_task in (("classifier", "pc"), ("generator", "g")):
                if args[flag]:
                    metadata = json.loads((Path(args[flag]) / "genpads.json").read_text())
                    if metadata["task"] != expected_task or metadata["domain"] != args["domain"]:
                        raise ValueError(f"--{flag} must be a {expected_task} checkpoint for {args['domain']}")
            args["classifier"] = (PolitenessClassifier(args["classifier"], device=args["device"])
                                  if args["classifier"] else None)
            args["generator"] = (ResponseGenerator(args["generator"], device=args["device"],
                                                    do_sample=True, seed=args["seed"])
                                 if args["generator"] else None)
            if command == "simulate":
                args["policy"] = ReinforcePolicy.load(args["policy"], device=args["device"]) if args["policy"] else None
                args.pop("device")
                result = simulate(**args)
            else:
                output = Path(args["output"])
                if output.exists() and (output.is_file() or any(output.iterdir())):
                    raise ValueError(f"Output path is not empty: {output}")
                result = train_policy(**args, progress=lambda record: print(json.dumps(record), flush=True))
                report = output.parent / (output.stem + "-training.json") if output.suffix == ".pt" else output / "training.json"
                report.write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, FileNotFoundError, ImportError) as exc:
        root.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
