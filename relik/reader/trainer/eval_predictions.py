import argparse
import json
import sys
from collections import Counter, defaultdict
from pprint import pprint

from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.utils.special_symbols import NME_SYMBOL
from relik.reader.utils.strong_matching_eval import StrongMatching


def load_relik_reader_samples(path: str) -> list[RelikReaderSample]:
    samples = []
    malformed_lines = 0
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                samples.append(RelikReaderSample(**json.loads(raw)))
            except json.JSONDecodeError as exc:
                malformed_lines += 1
                print(
                    f"Warning: skipping malformed JSON on line {line_number} of {path}: {exc}",
                    file=sys.stderr,
                )
    if malformed_lines:
        print(
            f"Warning: skipped {malformed_lines} malformed record(s) from {path}.",
            file=sys.stderr,
        )
    return samples


def sample_key(sample: RelikReaderSample) -> tuple[int, int]:
    return sample.doc_id, sample.window_id


def normalize_predictions(sample: RelikReaderSample) -> None:
    predicted_window_labels = sample.predicted_window_labels or []
    sample._d["predicted_window_labels_chars"] = {
        (start, end, label) for start, end, label in predicted_window_labels
    }


def load_aligned_predictions(
    gold_path: str, predicted_path: str
) -> list[RelikReaderSample]:
    gold_samples = list(load_relik_reader_samples(gold_path))
    predicted_samples = list(load_relik_reader_samples(predicted_path))

    gold_by_key = {sample_key(sample): sample for sample in gold_samples}
    pred_by_key = {sample_key(sample): sample for sample in predicted_samples}

    missing_predictions = sorted(set(gold_by_key) - set(pred_by_key))
    extra_predictions = sorted(set(pred_by_key) - set(gold_by_key))
    if missing_predictions:
        print(
            "Warning: missing predictions for "
            f"{len(missing_predictions)} gold sample(s); dropping them. "
            f"Examples: {missing_predictions[:5]}",
            file=sys.stderr,
        )
    if extra_predictions:
        print(
            "Warning: found "
            f"{len(extra_predictions)} predicted sample(s) without gold; dropping them. "
            f"Examples: {extra_predictions[:5]}",
            file=sys.stderr,
        )

    aligned_samples = []
    for key in sorted(set(gold_by_key) & set(pred_by_key)):
        gold_sample = gold_by_key[key]
        predicted_sample = pred_by_key[key]
        predicted_sample._d["window_labels"] = gold_sample.window_labels
        predicted_sample._d["window_labels_tokens"] = gold_sample.window_labels_tokens
        normalize_predictions(predicted_sample)
        aligned_samples.append(predicted_sample)

    if not aligned_samples:
        raise ValueError("No overlapping parsable samples found between gold and predicted files.")

    return aligned_samples


def aggregate_doc_level_samples(
    window_samples: list[RelikReaderSample],
) -> list[RelikReaderSample]:
    samples_by_doc = {}

    for sample in window_samples:
        doc_sample = samples_by_doc.setdefault(
            sample.doc_id,
            RelikReaderSample(
                doc_id=sample.doc_id,
                window_id=0,
                text="",
                offset=0,
                window_labels=[],
                window_labels_tokens=[],
                predicted_window_labels_chars=set(),
                span_candidates=[],
                probs_window_labels_chars={},
            ),
        )

        gold_annotations = {
            (start, end, label)
            for start, end, label in (sample.window_labels or [])
            if label != NME_SYMBOL
        }
        predicted_annotations = set(sample.predicted_window_labels_chars or set())

        existing_gold = set(doc_sample.window_labels)
        existing_gold.update(gold_annotations)
        doc_sample._d["window_labels"] = sorted(existing_gold)

        doc_sample.predicted_window_labels_chars.update(predicted_annotations)

        existing_candidates = set(doc_sample.span_candidates)
        existing_candidates.update(sample.span_candidates or [])
        doc_sample._d["span_candidates"] = sorted(existing_candidates)

    return list(samples_by_doc.values())


def annotation_text(sample: RelikReaderSample, start: int, end: int) -> str:
    text = sample.text or ""
    offset = sample.offset or 0
    local_start = start - offset
    local_end = end - offset
    if 0 <= local_start <= local_end <= len(text):
        return text[local_start:local_end].replace("\n", " ")
    return ""


def overlap_length(
    first_span: tuple[int, int], second_span: tuple[int, int]
) -> int:
    first_start, first_end = first_span
    second_start, second_end = second_span
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def any_overlap(
    span: tuple[int, int], other_spans: set[tuple[int, int]]
) -> bool:
    return any(overlap_length(span, other_span) > 0 for other_span in other_spans)


def best_overlapping_annotation(
    span: tuple[int, int, str], candidates: set[tuple[int, int, str]]
) -> tuple[int, int, str] | None:
    start, end, _ = span
    best_candidate = None
    best_overlap = 0
    for candidate in candidates:
        candidate_start, candidate_end, _ = candidate
        current_overlap = overlap_length((start, end), (candidate_start, candidate_end))
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_candidate = candidate
    return best_candidate


def count_weak_overlap_ratio(
    strong_matching: StrongMatching,
    predicted_annotation: tuple[int, int, str],
    gold_annotation: tuple[int, int, str],
) -> float:
    pred_start, pred_end, _ = predicted_annotation
    gold_start, gold_end, _ = gold_annotation
    return strong_matching._span_overlap_ratio(
        (pred_start, pred_end), (gold_start, gold_end)
    )


def match_weak_annotations(
    predicted_annotations: set[tuple[int, int, str]],
    gold_annotations: set[tuple[int, int, str]],
    strong_matching: StrongMatching,
) -> tuple[
    list[tuple[tuple[int, int, str], tuple[int, int, str], float]],
    set[tuple[int, int, str]],
    set[tuple[int, int, str]],
]:
    available_gold = set(gold_annotations)
    matches = []
    unmatched_predictions = set()

    for predicted_annotation in predicted_annotations:
        pred_start, pred_end, pred_label = predicted_annotation
        best_gold_match = None
        best_overlap = 0.0

        for gold_annotation in available_gold:
            gold_start, gold_end, gold_label = gold_annotation
            if pred_label != gold_label:
                continue
            overlap_ratio = strong_matching._span_overlap_ratio(
                (pred_start, pred_end), (gold_start, gold_end)
            )
            if (
                overlap_ratio >= strong_matching.weak_match_threshold
                and overlap_ratio > best_overlap
            ):
                best_overlap = overlap_ratio
                best_gold_match = gold_annotation

        if best_gold_match is None:
            unmatched_predictions.add(predicted_annotation)
            continue

        available_gold.remove(best_gold_match)
        matches.append((predicted_annotation, best_gold_match, best_overlap))

    return matches, unmatched_predictions, available_gold


def compute_debug_stats(
    samples: list[RelikReaderSample],
    strong_matching: StrongMatching,
) -> tuple[dict, dict[str, list[dict]]]:
    false_positive_labels = Counter()
    false_negative_labels = Counter()
    exact_label_confusions = Counter()
    false_positive_texts = Counter()
    false_negative_texts = Counter()
    span_boundary_errors = Counter()
    weak_false_positive_labels = Counter()
    weak_false_negative_labels = Counter()
    weak_false_positive_texts = Counter()
    weak_false_negative_texts = Counter()
    weak_near_miss_errors = Counter()
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    examples = {
        "label_confusions": [],
        "boundary_errors": [],
        "false_positives": [],
        "false_negatives": [],
        "weak_false_positives": [],
        "weak_false_negatives": [],
        "weak_near_misses": [],
    }

    for sample in samples:
        predicted_annotations = set(sample.predicted_window_labels_chars or set())
        gold_annotations = {
            (start, end, label)
            for start, end, label in (sample.window_labels or [])
            if label != NME_SYMBOL
        }
        exact_matches = predicted_annotations & gold_annotations
        predicted_spans = {(start, end) for start, end, _ in predicted_annotations}
        gold_spans = {(start, end) for start, end, _ in gold_annotations}

        for _, _, label in exact_matches:
            per_label[label]["tp"] += 1

        false_positives = predicted_annotations - gold_annotations
        false_negatives = gold_annotations - predicted_annotations
        _, weak_false_positives, weak_false_negatives = match_weak_annotations(
            predicted_annotations, gold_annotations, strong_matching
        )

        for start, end, label in false_positives:
            false_positive_labels[label] += 1
            false_positive_texts[(annotation_text(sample, start, end), label)] += 1
            per_label[label]["fp"] += 1

            overlapping_gold = best_overlapping_annotation(
                (start, end, label), false_negatives
            )
            if overlapping_gold is not None and (start, end) == overlapping_gold[:2]:
                exact_label_confusions[(overlapping_gold[2], label)] += 1
                if len(examples["label_confusions"]) < 10:
                    examples["label_confusions"].append(
                        {
                            "doc_id": sample.doc_id,
                            "window_id": sample.window_id,
                            "text": annotation_text(sample, start, end),
                            "gold_label": overlapping_gold[2],
                            "predicted_label": label,
                        }
                    )
            elif any_overlap((start, end), gold_spans):
                overlapping_gold = best_overlapping_annotation(
                    (start, end, label), gold_annotations
                )
                if overlapping_gold is not None:
                    gold_start, gold_end, gold_label = overlapping_gold
                    span_boundary_errors[
                        (
                            annotation_text(sample, gold_start, gold_end),
                            gold_label,
                            annotation_text(sample, start, end),
                            label,
                        )
                    ] += 1
                    if len(examples["boundary_errors"]) < 10:
                        examples["boundary_errors"].append(
                            {
                                "doc_id": sample.doc_id,
                                "window_id": sample.window_id,
                                "gold_text": annotation_text(
                                    sample, gold_start, gold_end
                                ),
                                "gold_label": gold_label,
                                "predicted_text": annotation_text(sample, start, end),
                                "predicted_label": label,
                            }
                        )
            elif len(examples["false_positives"]) < 10:
                examples["false_positives"].append(
                    {
                        "doc_id": sample.doc_id,
                        "window_id": sample.window_id,
                        "text": annotation_text(sample, start, end),
                        "label": label,
                    }
                )

        for start, end, label in false_negatives:
            false_negative_labels[label] += 1
            false_negative_texts[(annotation_text(sample, start, end), label)] += 1
            per_label[label]["fn"] += 1

            if not any_overlap((start, end), predicted_spans) and len(
                examples["false_negatives"]
            ) < 10:
                examples["false_negatives"].append(
                    {
                        "doc_id": sample.doc_id,
                        "window_id": sample.window_id,
                        "text": annotation_text(sample, start, end),
                        "label": label,
                    }
                )

        for start, end, label in weak_false_positives:
            weak_false_positive_labels[label] += 1
            weak_false_positive_texts[(annotation_text(sample, start, end), label)] += 1

            same_label_gold = [
                gold_annotation
                for gold_annotation in weak_false_negatives
                if gold_annotation[2] == label
            ]
            if same_label_gold:
                overlapping_gold = max(
                    same_label_gold,
                    key=lambda gold_annotation: count_weak_overlap_ratio(
                        strong_matching, (start, end, label), gold_annotation
                    ),
                )
                overlap_ratio = count_weak_overlap_ratio(
                    strong_matching, (start, end, label), overlapping_gold
                )
                if 0 < overlap_ratio < strong_matching.weak_match_threshold:
                    gold_start, gold_end, gold_label = overlapping_gold
                    weak_near_miss_errors[
                        (
                            annotation_text(sample, gold_start, gold_end),
                            gold_label,
                            annotation_text(sample, start, end),
                            label,
                            round(overlap_ratio, 4),
                        )
                    ] += 1
                    if len(examples["weak_near_misses"]) < 10:
                        examples["weak_near_misses"].append(
                            {
                                "doc_id": sample.doc_id,
                                "window_id": sample.window_id,
                                "gold_text": annotation_text(
                                    sample, gold_start, gold_end
                                ),
                                "gold_label": gold_label,
                                "predicted_text": annotation_text(sample, start, end),
                                "predicted_label": label,
                                "overlap_ratio": round(overlap_ratio, 4),
                            }
                        )
            elif len(examples["weak_false_positives"]) < 10:
                examples["weak_false_positives"].append(
                    {
                        "doc_id": sample.doc_id,
                        "window_id": sample.window_id,
                        "text": annotation_text(sample, start, end),
                        "label": label,
                    }
                )

        for start, end, label in weak_false_negatives:
            weak_false_negative_labels[label] += 1
            weak_false_negative_texts[(annotation_text(sample, start, end), label)] += 1

            if len(examples["weak_false_negatives"]) < 10:
                examples["weak_false_negatives"].append(
                    {
                        "doc_id": sample.doc_id,
                        "window_id": sample.window_id,
                        "text": annotation_text(sample, start, end),
                        "label": label,
                    }
                )

    summary = {
        "false_positive_labels": false_positive_labels,
        "false_negative_labels": false_negative_labels,
        "exact_label_confusions": exact_label_confusions,
        "false_positive_texts": false_positive_texts,
        "false_negative_texts": false_negative_texts,
        "span_boundary_errors": span_boundary_errors,
        "weak_false_positive_labels": weak_false_positive_labels,
        "weak_false_negative_labels": weak_false_negative_labels,
        "weak_false_positive_texts": weak_false_positive_texts,
        "weak_false_negative_texts": weak_false_negative_texts,
        "weak_near_miss_errors": weak_near_miss_errors,
        "per_label": per_label,
    }
    return summary, examples


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def top_counter_entries(counter: Counter, limit: int) -> list[tuple]:
    return counter.most_common(limit)


def print_per_label_stats(per_label: dict, top_k: int) -> None:
    ranked = []
    for label, counts in per_label.items():
        precision = ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = ratio(counts["tp"], counts["tp"] + counts["fn"])
        f1 = ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        support = counts["tp"] + counts["fn"]
        ranked.append((f1, support, label, counts, precision, recall))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    print("\nLowest per-label F1")
    for f1, support, label, counts, precision, recall in ranked[:top_k]:
        print(
            f"- {label}: f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"tp={counts['tp']} fp={counts['fp']} fn={counts['fn']} support={support}"
        )


def print_most_frequent_label_stats(per_label: dict, top_k: int) -> None:
    ranked = []
    for label, counts in per_label.items():
        precision = ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = ratio(counts["tp"], counts["tp"] + counts["fn"])
        f1 = ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        support = counts["tp"] + counts["fn"]
        ranked.append((support, label, counts, precision, recall, f1))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    print("\nMost frequent labels")
    for support, label, counts, precision, recall, f1 in ranked[:top_k]:
        print(
            f"- {label}: support={support} f1={f1:.4f} precision={precision:.4f} "
            f"recall={recall:.4f} tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}"
        )


def print_counter_section(title: str, entries: list[tuple]) -> None:
    print(f"\n{title}")
    for item, count in entries:
        print(f"- {item}: {count}")


def print_examples(title: str, entries: list[dict]) -> None:
    print(f"\n{title}")
    for entry in entries:
        pprint(entry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-path", required=True)
    parser.add_argument("--predicted-path", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--weak-match-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted_samples = load_aligned_predictions(args.gold_path, args.predicted_path)
    doc_level_samples = aggregate_doc_level_samples(predicted_samples)
    strong_matching = StrongMatching(weak_match_threshold=args.weak_match_threshold)

    print("Window-level StrongMatching metrics")
    pprint(strong_matching(predicted_samples))

    print("\nDoc-level StrongMatching metrics")
    pprint(strong_matching(doc_level_samples))

    debug_stats, examples = compute_debug_stats(predicted_samples, strong_matching)

    print_counter_section(
        "Most common false positive labels",
        top_counter_entries(debug_stats["false_positive_labels"], args.top_k),
    )
    print_counter_section(
        "Most common false negative labels",
        top_counter_entries(debug_stats["false_negative_labels"], args.top_k),
    )
    print_counter_section(
        "Most common exact-span label confusions (gold_label, predicted_label)",
        top_counter_entries(debug_stats["exact_label_confusions"], args.top_k),
    )
    print_counter_section(
        "Most common false positive spans ((text, label))",
        top_counter_entries(debug_stats["false_positive_texts"], args.top_k),
    )
    print_counter_section(
        "Most common false negative spans ((text, label))",
        top_counter_entries(debug_stats["false_negative_texts"], args.top_k),
    )
    print_counter_section(
        "Most common boundary errors ((gold_text, gold_label, predicted_text, predicted_label))",
        top_counter_entries(debug_stats["span_boundary_errors"], args.top_k),
    )
    print_counter_section(
        "Most common weak false positive labels",
        top_counter_entries(debug_stats["weak_false_positive_labels"], args.top_k),
    )
    print_counter_section(
        "Most common weak false negative labels",
        top_counter_entries(debug_stats["weak_false_negative_labels"], args.top_k),
    )
    print_counter_section(
        "Most common weak false positive spans ((text, label))",
        top_counter_entries(debug_stats["weak_false_positive_texts"], args.top_k),
    )
    print_counter_section(
        "Most common weak false negative spans ((text, label))",
        top_counter_entries(debug_stats["weak_false_negative_texts"], args.top_k),
    )
    print_counter_section(
        "Most common weak near misses ((gold_text, gold_label, predicted_text, predicted_label, overlap_ratio))",
        top_counter_entries(debug_stats["weak_near_miss_errors"], args.top_k),
    )
    print_most_frequent_label_stats(debug_stats["per_label"], args.top_k)
    print_per_label_stats(debug_stats["per_label"], args.top_k)

    print_examples("Label confusion examples", examples["label_confusions"])
    print_examples("Boundary error examples", examples["boundary_errors"])
    print_examples("False positive examples", examples["false_positives"])
    print_examples("False negative examples", examples["false_negatives"])
    print_examples("Weak false positive examples", examples["weak_false_positives"])
    print_examples("Weak false negative examples", examples["weak_false_negatives"])
    print_examples("Weak near miss examples", examples["weak_near_misses"])


if __name__ == "__main__":
    main()
