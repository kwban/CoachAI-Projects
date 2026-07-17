import os, json, re, argparse, collections
import pandas as pd

DIMENSIONS = [
    "Factual Consistency",
    "Logical Consistency",
    "Importance",
    "Readability",
    "Objectivity",
    "Journalistic",
    "Information Density",
    "Overall",
]

FENCE_JSON_RE = re.compile(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", re.S)

def _find_first_json_object(text: str):
    start = text.find('{')
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        start = text.find('{', start + 1)
    return None

def _extract_json_text(content: str):
    m = FENCE_JSON_RE.search(content)
    if m:
        return m.group(1).strip()
    if content.strip().startswith('{'):
        obj = _find_first_json_object(content)
        if obj:
            return obj.strip()
    obj = _find_first_json_object(content)
    return obj.strip() if obj else None

def _safe_load_result(meta: dict):
    # Newer format:
    if "parsed" in meta:
        if meta.get("parsed") and isinstance(meta.get("result"), dict):
            return {"ok": True, "data": meta["result"], "raw_text": meta.get("raw_text")}
        # try to parse from raw_text if present
        raw = meta.get("raw_text")
        if raw:
            jt = _extract_json_text(raw)
            if jt:
                try:
                    return {"ok": True, "data": json.loads(jt), "raw_text": raw}
                except Exception:
                    pass
        return {"ok": False, "data": None, "raw_text": meta.get("raw_text")}
    # Older format (result likely the raw dict already)
    res = meta.get("result")
    if isinstance(res, dict):
        return {"ok": True, "data": res, "raw_text": None}
    # Sometimes only raw_text exists
    raw = meta.get("raw_text")
    if raw:
        jt = _extract_json_text(raw)
        if jt:
            try:
                return {"ok": True, "data": json.loads(jt), "raw_text": raw}
            except Exception:
                pass
    return {"ok": False, "data": None, "raw_text": meta.get("raw_text")}

def analyze_raw(outdir: str = "evaluation/LLM_eval"):
    raw_root = os.path.join(outdir, "raw_judge_json")
    if not os.path.isdir(raw_root):
        raise SystemExit(f"raw folder not found: {raw_root}")

    counters = {dim: collections.defaultdict(lambda: {"win":0,"lose":0,"tie":0,"matches":0})
                for dim in DIMENSIONS}

    ties_rows = []  # rows for ties.csv

    total_files = 0
    parsed_files = 0

    for report_dir in sorted(os.listdir(raw_root)):
        rpath = os.path.join(raw_root, report_dir)
        if not os.path.isdir(rpath):
            continue
        # Expect files like: <i>_vs_<j>.json
        for fname in sorted(os.listdir(rpath)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(rpath, fname)
            total_files += 1
            try:
                meta = json.load(open(fpath, "r", encoding="utf-8"))
            except Exception:
                continue

            label_i = meta.get("label_i", f"idx_{meta.get('i')}")
            label_j = meta.get("label_j", f"idx_{meta.get('j')}")
            flip = bool(meta.get("flipped_presentation", True))  # True means (first,second)==(i,j)
            report_id = meta.get("report_id")

            load = _safe_load_result(meta)
            if not load["ok"] or not isinstance(load["data"], dict):
                # can't parse; skip
                continue
            parsed_files += 1
            data = load["data"]

            for dim in DIMENSIONS:
                entry = data.get(dim)
                if not isinstance(entry, dict):
                    continue
                winner = entry.get("winner")
                reasoning = entry.get("reasoning")

                # Only count valid tokens; unknowns are ignored
                if winner not in ("first","second","tie"):
                    continue

                if winner == "tie":
                    counters[dim][label_i]["tie"] += 1
                    counters[dim][label_j]["tie"] += 1
                    counters[dim][label_i]["matches"] += 1
                    counters[dim][label_j]["matches"] += 1
                    ties_rows.append({
                        "report_id": report_id,
                        "dimension": dim,
                        "model_left": label_i,
                        "model_right": label_j,
                        "file": fpath,
                        "reasoning": reasoning,
                    })
                elif winner == "first":
                    if flip:
                        wi, wj = label_i, label_j
                    else:
                        wi, wj = label_j, label_i
                    counters[dim][wi]["win"] += 1
                    counters[dim][wj]["lose"] += 1
                    counters[dim][wi]["matches"] += 1
                    counters[dim][wj]["matches"] += 1
                elif winner == "second":
                    if flip:
                        wi, wj = label_j, label_i
                    else:
                        wi, wj = label_i, label_j
                    counters[dim][wi]["win"] += 1
                    counters[dim][wj]["lose"] += 1
                    counters[dim][wi]["matches"] += 1
                    counters[dim][wj]["matches"] += 1

    # Output directory
    out_analysis = os.path.join(outdir, "analysis")
    os.makedirs(out_analysis, exist_ok=True)

    # Save per-dimension W/L/T
    for dim in DIMENSIONS:
        rows = []
        for model, c in counters[dim].items():
            rows.append({
                "model": model,
                "wins": c["win"],
                "losses": c["lose"],
                "ties": c["tie"],
                "matches": c["matches"],
                "win_rate_%": round(100.0 * c["win"] / c["matches"], 2) if c["matches"] else 0.0,
                "tie_rate_%": round(100.0 * c["tie"] / c["matches"], 2) if c["matches"] else 0.0,
            })
        df = pd.DataFrame(rows).sort_values(["win_rate_%","tie_rate_%"], ascending=[False, True])
        safe = dim.lower().replace(" ", "_")
        df.to_csv(os.path.join(out_analysis, f"wlt_counts_{safe}.csv"), index=False)

    # Save all tie records
    ties_df = pd.DataFrame(ties_rows)
    ties_df.to_csv(os.path.join(out_analysis, "ties.csv"), index=False)

    # Quick console summary
    print(f"Scanned files: {total_files}, parsed: {parsed_files}")
    for dim in DIMENSIONS:
        total_matches = sum(counters[dim][m]["matches"] for m in counters[dim])
        total_ties = sum(counters[dim][m]["tie"] for m in counters[dim]) // 2  # each tie counted for both models
        print(f"{dim}: matches={total_matches//2}, ties={total_ties}")

def main():
    ap = argparse.ArgumentParser(description="Analyze raw_judge_json to get per-model W/L/T and tie tracebacks.")
    ap.add_argument("--outdir", default="evaluation/LLM_eval", help="Evaluator OUTDIR (default: evaluation/LLM_eval)")
    args = ap.parse_args()
    analyze_raw(args.outdir)

if __name__ == "__main__":
    main()
