from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


NO_ISSUES = "No issues found."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the base ReviewPy model on the frozen test set."
    )

    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Path to evaluation configuration.",
    )

    parser.add_argument(
        "--dataset",
        default="data/processed/test.jsonl",
        help="Path to frozen test JSONL.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config["do_sample"],
            temperature=generation_config["temperature"],
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
    correct_no_issue_decision = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    generated_comments = 0
    total_output_chars = 0
    total_latency = 0.0

    for result in results:
        prediction = result["prediction"]
        reference = result["reference"]

        predicted_no_issue = is_no_issue(prediction)
        reference_no_issue = is_no_issue(reference)

        if prediction == reference:
            exact_matches += 1

        if predicted_no_issue == reference_no_issue:
            correct_no_issue_decision += 1

        if predicted_no_issue and reference_no_issue:
            true_positive += 1

        elif predicted_no_issue and not reference_no_issue:
            false_positive += 1

        elif not predicted_no_issue and reference_no_issue:
            false_negative += 1

        if not predicted_no_issue:
            generated_comments += 1

        total_output_chars += len(prediction)
        total_latency += result["latency_seconds"]

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
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
        "exact_match_accuracy": exact_matches / total if total else 0.0,
        "no_issue_decision_accuracy": (
            correct_no_issue_decision / total if total else 0.0
        ),
        "no_issue_precision": precision,
        "no_issue_recall": recall,
        "no_issue_f1": f1,
        "generated_comment_rate": (
            generated_comments / total if total else 0.0
        ),
        "average_output_characters": (
            total_output_chars / total if total else 0.0
        ),
        "average_latency_seconds": (
            total_latency / total if total else 0.0
        ),
    }


def main() -> None:
    args = parse_args()

    config = load_config(
        Path(args.config)
    )

    records = load_jsonl(
        Path(args.dataset)
    )

    max_examples = config["evaluation"].get(
        "max_examples"
    )

    if max_examples is not None:
        records = records[:max_examples]

    model_name = config["model"]["name"]

    print(f"Model: {model_name}")
    print(f"Examples: {len(records):,}")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    results = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        prediction, latency = generate_review(
            model,
            tokenizer,
            record,
            config["generation"],
        )

        result = {
            "index": index,
            "prediction": prediction,
            "reference": record["assistant"],
            "is_negative": record["metadata"]["is_negative"],
            "repo_name": record["metadata"]["repo_name"],
            "file_path": record["metadata"]["file_path"],
            "latency_seconds": latency,
        }

        results.append(result)

        print(
            f"[{index}/{len(records)}] "
            f"{prediction[:100]}"
        )

    metrics = compute_metrics(
        results
    )

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

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_path = (
        output_directory
        / "base_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\n=== BASE MODEL METRICS ===")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{key:30} "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key:30} "
                f"{value}"
            )

    print(
        f"\nPredictions: {predictions_path}"
    )

    print(
        f"Metrics:     {metrics_path}"
    )


if __name__ == "__main__":
    main()