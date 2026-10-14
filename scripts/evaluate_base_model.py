from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


NO_ISSUES = "No issues found."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the base ReviewPy model."
    )

    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Path to evaluation configuration.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Optional dataset override. If omitted, the dataset is selected "
            "from evaluation.split."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted evaluation from saved predictions.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard saved progress and start the evaluation again.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_output(text: str) -> str:
    return text.strip().strip('"').strip()


def is_no_issue(text: str) -> bool:
    return normalize_output(text) == NO_ISSUES


def generate_review(
    model,
    tokenizer,
    record: dict,
    generation_config: dict,
) -> tuple[str, float]:
    messages = [
        {
            "role": "system",
            "content": record["system"],
        },
        {
            "role": "user",
            "content": record["user"],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    start = time.perf_counter()

    generation_kwargs = {
        "max_new_tokens": generation_config["max_new_tokens"],
        "do_sample": generation_config["do_sample"],
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    elapsed = time.perf_counter() - start

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1] :,
    ]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return normalize_output(text), elapsed


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)

    exact_matches = 0
    positive_exact_matches = 0
    negative_exact_matches = 0

    positive_examples = 0
    negative_examples = 0

    predicted_comments = 0
    predicted_no_issue = 0

    correct_no_issue_decision = 0

    true_positive_no_issue = 0
    false_positive_no_issue = 0
    false_negative_no_issue = 0

    total_output_chars = 0
    total_latency = 0.0

    for result in results:
        prediction = result["prediction"]
        reference = result["reference"]

        prediction_no_issue = is_no_issue(prediction)
        reference_no_issue = is_no_issue(reference)

        if reference_no_issue:
            negative_examples += 1
        else:
            positive_examples += 1

        if prediction_no_issue:
            predicted_no_issue += 1
        else:
            predicted_comments += 1

        if prediction == reference:
            exact_matches += 1

            if reference_no_issue:
                negative_exact_matches += 1
            else:
                positive_exact_matches += 1

        if prediction_no_issue == reference_no_issue:
            correct_no_issue_decision += 1

        if prediction_no_issue and reference_no_issue:
            true_positive_no_issue += 1

        elif prediction_no_issue and not reference_no_issue:
            false_positive_no_issue += 1

        elif not prediction_no_issue and reference_no_issue:
            false_negative_no_issue += 1

        total_output_chars += len(prediction)
        total_latency += result["latency_seconds"]

    precision_denominator = (
        true_positive_no_issue + false_positive_no_issue
    )

    recall_denominator = (
        true_positive_no_issue + false_negative_no_issue
    )

    precision = (
        true_positive_no_issue / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        true_positive_no_issue / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "examples": total,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "predicted_comments": predicted_comments,
        "predicted_no_issue": predicted_no_issue,
        "exact_match_accuracy": (
            exact_matches / total if total else 0.0
        ),
        "positive_exact_match_accuracy": (
            positive_exact_matches / positive_examples
            if positive_examples
            else 0.0
        ),
        "negative_exact_match_accuracy": (
            negative_exact_matches / negative_examples
            if negative_examples
            else 0.0
        ),
        "no_issue_decision_accuracy": (
            correct_no_issue_decision / total
            if total
            else 0.0
        ),
        "no_issue_precision": precision,
        "no_issue_recall": recall,
        "no_issue_f1": f1,
        "generated_comment_rate": (
            predicted_comments / total
            if total
            else 0.0
        ),
        "average_output_characters": (
            total_output_chars / total
            if total
            else 0.0
        ),
        "average_latency_seconds": (
            total_latency / total
            if total
            else 0.0
        ),
    }


def get_hardware_info() -> dict:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = None

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
    }


def write_json_atomic(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    temporary_path.replace(path)


def build_run_signature(
    model_name: str,
    model_revision: str,
    split: str,
    max_examples: int | None,
    dataset_path: Path,
    dataset_sha256: str,
    generation_config: dict,
) -> dict:
    return {
        "model_name": model_name,
        "model_revision": model_revision,
        "split": split,
        "max_examples": max_examples,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "generation": {
            "max_new_tokens": generation_config["max_new_tokens"],
            "do_sample": generation_config["do_sample"],
        },
    }


def validate_saved_results(results: list[dict]) -> None:
    for expected_index, result in enumerate(results, start=1):
        if result.get("index") != expected_index:
            raise RuntimeError(
                "Saved prediction indices are not contiguous. "
                "Use --restart to start a clean evaluation."
            )


def main() -> None:
    args = parse_args()

    if args.resume and args.restart:
        raise ValueError(
            "Choose either --resume or --restart, not both."
        )

    config_path = Path(args.config)
    config = load_config(config_path)

    split = config["evaluation"]["split"]

    if args.dataset is not None:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = Path(
            f"data/processed/{split}.jsonl"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    dataset_sha256 = sha256_file(dataset_path)

    records = load_jsonl(dataset_path)

    max_examples = config["evaluation"].get(
        "max_examples"
    )

    if max_examples is not None:
        records = records[:max_examples]

    model_name = config["model"]["name"]
    model_revision = config["model"]["revision"]

    generation_config = config["generation"]

    seed = config["evaluation"].get("seed", 42)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_directory = Path(
        config["output"]["directory"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_directory
        / "base_predictions.jsonl"
    )

    metrics_path = (
        output_directory
        / "base_metrics.json"
    )

    progress_path = (
        output_directory
        / "base_progress.json"
    )

    run_signature = build_run_signature(
        model_name=model_name,
        model_revision=model_revision,
        split=split,
        max_examples=max_examples,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        generation_config=generation_config,
    )

    if args.restart:
        predictions_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    results: list[dict] = []

    if args.resume:
        if not predictions_path.exists():
            raise RuntimeError(
                "No saved predictions were found to resume."
            )

        if not progress_path.exists():
            raise RuntimeError(
                "No progress metadata was found to resume."
            )

        with progress_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            saved_progress = json.load(file)

        if saved_progress.get("run_signature") != run_signature:
            raise RuntimeError(
                "The saved run does not match the current model, "
                "dataset, split, or generation settings. "
                "Use --restart if you intentionally want a new run."
            )

        results = load_jsonl(predictions_path)
        validate_saved_results(results)

        print(
            f"Resuming from example {len(results) + 1:,} "
            f"of {len(records):,}."
        )

    else:
        if progress_path.exists():
            raise RuntimeError(
                "Saved evaluation progress already exists. "
                "Use --resume to continue it or --restart "
                "to discard it."
            )

        predictions_path.write_text(
            "",
            encoding="utf-8",
        )

    completed_examples = len(results)

    progress = {
        "run_signature": run_signature,
        "completed_examples": completed_examples,
        "total_examples": len(records),
        "complete": completed_examples == len(records),
    }

    write_json_atomic(
        progress_path,
        progress,
    )

    print(f"Model: {model_name}")
    print(f"Revision: {model_revision}")
    print(f"Split: {split}")
    print(f"Dataset: {dataset_path}")
    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"Examples: {len(records):,}")
    print(f"Already completed: {completed_examples:,}")

    if completed_examples < len(records):
        print("\nLoading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
        )

        print("Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            torch_dtype="auto",
            device_map="auto",
        )

        model.eval()

        try:
            with predictions_path.open(
                "a",
                encoding="utf-8",
            ) as prediction_file:
                for zero_based_index in range(
                    completed_examples,
                    len(records),
                ):
                    record = records[zero_based_index]
                    index = zero_based_index + 1

                    prediction, latency = generate_review(
                        model,
                        tokenizer,
                        record,
                        generation_config,
                    )

                    result = {
                        "index": index,
                        "prediction": prediction,
                        "reference": record["assistant"],
                        "is_negative": (
                            record["metadata"]["is_negative"]
                        ),
                        "repo_name": (
                            record["metadata"]["repo_name"]
                        ),
                        "file_path": (
                            record["metadata"]["file_path"]
                        ),
                        "latency_seconds": latency,
                    }

                    results.append(result)

                    prediction_file.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    prediction_file.flush()

                    progress["completed_examples"] = len(results)

                    write_json_atomic(
                        progress_path,
                        progress,
                    )

                    print(
                        f"[{index}/{len(records)}] "
                        f"{prediction[:100]}"
                    )

        except KeyboardInterrupt:
            print(
                "\nEvaluation interrupted safely."
            )
            print(
                f"Saved {len(results):,} / "
                f"{len(records):,} predictions."
            )
            print(
                "Resume later with:"
            )
            print(
                "python scripts/evaluate_base_model.py --resume"
            )
            return

    if len(results) != len(records):
        raise RuntimeError(
            "Evaluation ended before all examples were completed."
        )

    task_metrics = compute_metrics(results)

    metadata = {
        "model": {
            "name": model_name,
            "revision": model_revision,
        },
        "evaluation": {
            "split": split,
            "max_examples": max_examples,
            "seed": seed,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
        },
        "generation": {
            "max_new_tokens": (
                generation_config["max_new_tokens"]
            ),
            "do_sample": (
                generation_config["do_sample"]
            ),
        },
        "environment": get_hardware_info(),
    }

    report = {
        "metadata": metadata,
        "metrics": task_metrics,
    }

    write_json_atomic(
        metrics_path,
        report,
    )

    progress["completed_examples"] = len(results)
    progress["complete"] = True

    write_json_atomic(
        progress_path,
        progress,
    )

    print("\n=== BASE MODEL METRICS ===")

    for key, value in task_metrics.items():
        if isinstance(value, float):
            print(
                f"{key:35} "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key:35} "
                f"{value}"
            )

    print("\n=== REPRODUCIBILITY INFO ===")
    print(f"Model revision:    {model_revision}")
    print(f"Dataset SHA-256:   {dataset_sha256}")
    print(
        f"GPU:               "
        f"{metadata['environment']['gpu_name']}"
    )
    print(
        f"PyTorch:           "
        f"{metadata['environment']['torch_version']}"
    )
    print(
        f"Transformers:      "
        f"{metadata['environment']['transformers_version']}"
    )

    print(
        f"\nPredictions: {predictions_path}"
    )

    print(
        f"Metrics:     {metrics_path}"
    )

    print(
        f"Progress:    {progress_path}"
    )


if __name__ == "__main__":
    main()