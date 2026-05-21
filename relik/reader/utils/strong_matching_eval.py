from typing import Dict, List, Tuple

from lightning.pytorch.callbacks import Callback

from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.utils.relik_reader_predictor import RelikReaderPredictor
from relik.reader.utils.metrics import f1_measure, safe_divide
from relik.reader.utils.special_symbols import NME_SYMBOL


class StrongMatching:
    def __init__(self, weak_match_threshold: float = 0.5) -> None:
        self.weak_match_threshold = weak_match_threshold

    @staticmethod
    def _span_overlap_ratio(
        predicted_span: Tuple[int, int], gold_span: Tuple[int, int]
    ) -> float:
        pred_start, pred_end = predicted_span
        gold_start, gold_end = gold_span
        overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
        gold_length = max(0, gold_end - gold_start)
        if gold_length == 0:
            return 0.0
        return overlap / gold_length

    def _count_weak_matches(
        self,
        predicted_annotations: set[Tuple[int, int, str]],
        gold_annotations: set[Tuple[int, int, str]],
    ) -> int:
        available_gold = set(gold_annotations)
        matched_predictions = 0

        for pred_start, pred_end, pred_entity in predicted_annotations:
            best_gold_match = None
            best_overlap = 0.0

            for gold_annotation in available_gold:
                gold_start, gold_end, gold_entity = gold_annotation
                if pred_entity != gold_entity:
                    continue

                overlap_ratio = self._span_overlap_ratio(
                    (pred_start, pred_end), (gold_start, gold_end)
                )
                if (
                    overlap_ratio >= self.weak_match_threshold
                    and overlap_ratio > best_overlap
                ):
                    best_overlap = overlap_ratio
                    best_gold_match = gold_annotation

            if best_gold_match is not None:
                available_gold.remove(best_gold_match)
                matched_predictions += 1

        return matched_predictions

    def __call__(self, predicted_samples: List[RelikReaderSample]) -> Dict:
        # accumulators
        correct_predictions = 0
        weak_correct_predictions = 0
        correct_predictions_at_k = 0
        total_predictions = 0
        total_weak_predictions = 0
        total_gold = 0
        correct_span_predictions = 0
        correct_labels_on_matched_spans = 0
        total_matched_predicted_spans = 0
        total_matched_gold_spans = 0
        miss_due_to_candidates = 0

        # prediction index stats
        avg_correct_predicted_index = []
        avg_wrong_predicted_index = []
        less_index_predictions = []

        # collect data from samples
        for sample in predicted_samples:
            predicted_annotations = sample.predicted_window_labels_chars
            predicted_annotations_probabilities = (
                sample.probs_window_labels_chars or {}
            )
            gold_annotations = {
                (ss, se, entity)
                for ss, se, entity in sample.window_labels
                if entity != NME_SYMBOL
            }
            total_predictions += len(predicted_annotations)
            total_weak_predictions += len(predicted_annotations)
            total_gold += len(gold_annotations)

            # correct named entity detection
            predicted_spans = {(s, e) for s, e, _ in predicted_annotations}
            gold_spans = {(s, e) for s, e, _ in gold_annotations}
            matched_spans = predicted_spans.intersection(gold_spans)
            correct_span_predictions += len(matched_spans)
            total_matched_predicted_spans += len(matched_spans)
            total_matched_gold_spans += len(matched_spans)

            # correct entity linking
            correct_predictions += len(
                predicted_annotations.intersection(gold_annotations)
            )

            weak_correct_predictions += self._count_weak_matches(
                predicted_annotations, gold_annotations
            )

            for ss, se, ge in gold_annotations.difference(predicted_annotations):
                if ge not in sample.span_candidates:
                    miss_due_to_candidates += 1
                predicted_candidates = set(
                    predicted_annotations_probabilities.get((ss, se), set())
                )
                if ge in predicted_candidates:
                    correct_predictions_at_k += 1

            # indices metrics
            predicted_spans_index = {
                (ss, se): ent for ss, se, ent in predicted_annotations
            }
            gold_spans_index = {(ss, se): ent for ss, se, ent in gold_annotations}

            for matched_span in matched_spans:
                if predicted_spans_index[matched_span] == gold_spans_index[matched_span]:
                    correct_labels_on_matched_spans += 1

            for pred_span, pred_ent in predicted_spans_index.items():
                gold_ent = gold_spans_index.get(pred_span)

                if pred_span not in gold_spans_index:
                    continue

                # missing candidate
                if gold_ent not in sample.span_candidates:
                    continue

                gold_idx = sample.span_candidates.index(gold_ent)
                if gold_idx is None:
                    continue
                pred_idx = sample.span_candidates.index(pred_ent)

                if gold_ent != pred_ent:
                    avg_wrong_predicted_index.append(pred_idx)

                    if gold_idx is not None:
                        if pred_idx > gold_idx:
                            less_index_predictions.append(0)
                        else:
                            less_index_predictions.append(1)

                else:
                    avg_correct_predicted_index.append(pred_idx)

        # compute NED metrics
        span_precision = safe_divide(correct_span_predictions, total_predictions)
        span_recall = safe_divide(correct_span_predictions, total_gold)
        span_f1 = f1_measure(span_precision, span_recall)

        # compute EL metrics
        precision = safe_divide(correct_predictions, total_predictions)
        recall = safe_divide(correct_predictions, total_gold)
        weak_precision = safe_divide(weak_correct_predictions, total_weak_predictions)
        weak_recall = safe_divide(weak_correct_predictions, total_gold)
        matched_span_label_precision = safe_divide(
            correct_labels_on_matched_spans, total_matched_predicted_spans
        )
        matched_span_label_recall = safe_divide(
            correct_labels_on_matched_spans, total_matched_gold_spans
        )
        recall_at_k = safe_divide(
            (correct_predictions + correct_predictions_at_k), total_gold
        )

        f1 = f1_measure(precision, recall)
        weak_f1 = f1_measure(weak_precision, weak_recall)
        matched_span_label_f1 = f1_measure(
            matched_span_label_precision, matched_span_label_recall
        )

        wrong_for_candidates = safe_divide(miss_due_to_candidates, total_gold)

        out_dict = {
            "span_precision": span_precision,
            "span_recall": span_recall,
            "span_f1": span_f1,
            "core_precision": precision,
            "core_recall": recall,
            "weak_core_precision": weak_precision,
            "weak_core_recall": weak_recall,
            "matched_span_label_precision": matched_span_label_precision,
            "matched_span_label_recall": matched_span_label_recall,
            "core_recall-at-k": recall_at_k,
            "core_f1": round(f1, 4),
            "weak_core_f1": round(weak_f1, 4),
            "matched_span_label_f1": round(matched_span_label_f1, 4),
            "wrong-for-candidates": wrong_for_candidates,
            "index_errors_avg-index": safe_divide(
                sum(avg_wrong_predicted_index), len(avg_wrong_predicted_index)
            ),
            "index_correct_avg-index": safe_divide(
                sum(avg_correct_predicted_index), len(avg_correct_predicted_index)
            ),
            "index_avg-index": safe_divide(
                sum(avg_correct_predicted_index + avg_wrong_predicted_index),
                len(avg_correct_predicted_index + avg_wrong_predicted_index),
            ),
            "index_percentage-favoured-smaller-idx": safe_divide(
                sum(less_index_predictions), len(less_index_predictions)
            ),
        }

        return {k: round(v, 5) for k, v in out_dict.items()}


class ELStrongMatchingCallback(Callback):
    def __init__(
        self,
        dataset_path: str,
        dataset_conf,
        weak_match_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_conf = dataset_conf
        self.strong_matching_metric = StrongMatching(
            weak_match_threshold=weak_match_threshold
        )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        relik_reader_predictor = RelikReaderPredictor(pl_module.relik_reader_core_model)
        predicted_samples = relik_reader_predictor.predict(
            self.dataset_path,
            samples=None,
            dataset_conf=self.dataset_conf,
        )
        predicted_samples = list(predicted_samples)
        for k, v in self.strong_matching_metric(predicted_samples).items():
            pl_module.log(f"val_{k}", v)
