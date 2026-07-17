import os
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

def load_json(path: str) -> Optional[Any]:
    """Return parsed JSON or None if file is missing or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

_ALNUM = re.compile(r"[^a-zA-Z0-9]")

def _norm(s: Any) -> str:
    """Normalize to lowercase alphanumeric; None -> ''."""
    return _ALNUM.sub("", str(s or "")).lower()


def _yield_text_items(x: Any) -> Iterable[str]:
    if x is None:
        return
    if isinstance(x, dict):
        if "content" in x:
            yield str(x.get("content", ""))
        else:
            for v in x.values():
                yield from _yield_text_items(v)
    elif isinstance(x, list):
        for v in x:
            yield from _yield_text_items(v)
    else:
        yield str(x)


def unique_norm_set(seq: Any) -> Set[str]:
    out: Set[str] = set()
    for t in _yield_text_items(seq):
        n = _norm(t)
        if n:
            out.add(n)
    return out

def prf1(pred_set: Set[str], gold_set: Set[str]) -> Dict[str, float]:
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def overlap_metrics(pred_set: Set[str], gold_set: Set[str]) -> Tuple[int, int, float]:
    in_hist = len(pred_set & gold_set)
    total = len(pred_set)
    denom = len(gold_set)
    ratio = in_hist / denom if denom > 0 else 0.0
    return in_hist, total, ratio

def evaluate_all(
    indices = range(100),
    path_google: str = "Generated_report/2_step_gemma-3-27b-it",
    path_gpt4o: str = "Generated_report/2_step_gpt-4o",
    hist_search_path: str = "historical_search_results.json",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    f1_per_id: Dict[str, Dict[str, float]] = {}

    hist_search = load_json(hist_search_path) or {}
    if not isinstance(hist_search, dict):
        hist_search = {}

    for i in indices:
        print(i)
        report_id = str(i)

        react_search_path = os.path.join(path_google, report_id, "search.json")
        react_store_path  = os.path.join(path_google, report_id, "store.json")
        new_search_path   = os.path.join(path_gpt4o, report_id, "search.json")
        new_store_path    = os.path.join(path_gpt4o, report_id, "store.json")

        react_search = load_json(react_search_path)
        # If there is no search.json for this id in the baseline path, skip (matches original behavior)
        if react_search is None:
            continue

        react_store  = load_json(react_store_path)
        new_search   = load_json(new_search_path)
        new_store    = load_json(new_store_path)

        gold_entry = hist_search.get(report_id, {})
        gold_set   = unique_norm_set(gold_entry)

        # Unique normalized sets for predictions
        rs_set      = unique_norm_set(react_search)
        rstore_set  = unique_norm_set(react_store)
        nrs_set     = unique_norm_set(new_search)
        nrstore_set = unique_norm_set(new_store)

        print(rs_set)
        print(rstore_set)

        # Overlap (legacy-style) metrics
        rs_in, rs_tot, rs_ratio                 = overlap_metrics(rs_set, gold_set)
        rstore_in, rstore_tot, rstore_ratio     = overlap_metrics(rstore_set, gold_set)
        nrs_in, nrs_tot, nrs_ratio              = overlap_metrics(nrs_set, gold_set)
        nrstore_in, nrstore_tot, nrstore_ratio  = overlap_metrics(nrstore_set, gold_set)

        # Precision/Recall/F1
        rs_prf1      = prf1(rs_set, gold_set)
        rstore_prf1  = prf1(rstore_set, gold_set)
        nrs_prf1     = prf1(nrs_set, gold_set)
        nrstore_prf1 = prf1(nrstore_set, gold_set)

        # Collect metrics
        result = {
            # Overlap counts/ratios (unique-only)
            "react_search_in_hist_count": rs_in,
            "react_search_in_hist_total": rs_tot,
            "react_search_in_hist_ratio": rs_ratio,

            "react_store_in_hist_count": rstore_in,
            "react_store_in_hist_total": rstore_tot,
            "react_store_in_hist_ratio": rstore_ratio,

            "new_react_search_in_hist_count": nrs_in,
            "new_react_search_in_hist_total": nrs_tot,
            "new_react_search_in_hist_ratio": nrs_ratio,

            "new_react_store_in_hist_count": nrstore_in,
            "new_react_store_in_hist_total": nrstore_tot,
            "new_react_store_in_hist_ratio": nrstore_ratio,

            # Precision / Recall / F1
            "react_search_precision": rs_prf1["precision"],
            "react_search_recall":    rs_prf1["recall"],
            "react_search_f1":        rs_prf1["f1"],

            "react_store_precision":  rstore_prf1["precision"],
            "react_store_recall":     rstore_prf1["recall"],
            "react_store_f1":         rstore_prf1["f1"],

            "new_react_search_precision": nrs_prf1["precision"],
            "new_react_search_recall":    nrs_prf1["recall"],
            "new_react_search_f1":        nrs_prf1["f1"],

            "new_react_store_precision":  nrstore_prf1["precision"],
            "new_react_store_recall":     nrstore_prf1["recall"],
            "new_react_store_f1":         nrstore_prf1["f1"],
        }

        results[report_id] = result
        f1_per_id[report_id] = {
            "react_search_f1": rs_prf1["f1"],
            "react_store_f1": rstore_prf1["f1"],
            "new_react_search_f1": nrs_prf1["f1"],
            "new_react_store_f1": nrstore_prf1["f1"],
        }

    # Averages across all collected results
    avg_scores: Dict[str, float] = {}
    if results:
        example = next(iter(results.values()))
        numeric_keys = [k for k, v in example.items() if isinstance(v, (int, float))]

        def is_rate_metric(key: str) -> bool:
            return key.endswith(("_ratio", "_precision", "_recall", "_f1"))

        for k in numeric_keys:
            vals_all = [v[k] for v in results.values() if isinstance(v.get(k), (int, float))]
            # print(k,' ',len([x for x in vals_all if x == 0])/len(vals_all))
            if not vals_all:
                continue

            if k.endswith("_ratio"):
                # Skip zeros for ratio; use complement-mean on the nonzero subset
                vals = [x for x in vals_all if x != 0]
                if vals:
                    n = len(vals)
                    avg_scores[k] = 1.0 - (sum(1.0 - x for x in vals) / n)
                else:
                    avg_scores[k] = 0.0
            elif is_rate_metric(k):
                # For precision/recall/F1: average over non-zero values only
                vals = [x for x in vals_all if x != 0]
                avg_scores[k] = (sum(vals) / len(vals)) if vals else 0.0
            else:
                # Counts/totals and other integers: regular mean (zeros included)
                avg_scores[k] = sum(vals_all) / len(vals_all)

    # Save outputs
    write_json("evaluation/2_step_eval/per_id_scores.json", results)
    write_json("evaluation/2_step_eval/average_scores.json", avg_scores)
    write_json("evaluation/2_step_eval/f1_scores.json", f1_per_id)

    return results, avg_scores


if __name__ == "__main__":
    evaluate_all()
