from __future__ import annotations

import os
import json
import re
import ast
import asyncio
import random
from itertools import combinations
from typing import Optional, Tuple, List, Dict, Iterable, Set, Any
import pandas as pd

# --- OpenAI async client (OpenAI or OpenAI-compatible like DeepInfra) -----
from openai import AsyncOpenAI

# ---------- Configure here ----------
BASE_DIR = "Crawl_data/june_july_news"

MODEL_DIRS = [
    "Generated_report/1_step_gpt-4o",
    "Generated_report/1_step_gemma-3-27b-it",
    "Generated_report/2_step_gpt-4o",
    "Generated_report/2_step_gpt-4o-mini",
    "Generated_report/2_step_gemma-3-27b-it",
    "Generated_report/2_step_Qwen3-32B",
    "Generated_report/2_step_Llama-4-Scout-17B-16E-Instruct",
    "Generated_report/rulebase",
    "Generated_report/1_step_gpt-4o-mini",
]  # 9 folders; the 10th participant (reference) is added automatically

DISPLAY_NAME_MAP: Dict[str, str] = {
    "Generated_report/1_step_gpt-4o": "gpt-4o/1step",
    "Generated_report/1_step_gemma-3-27b-it": "gemma-3-27b-it/1step",
    "Generated_report/2_step_gpt-4o": "gpt-4o/2step",
    "Generated_report/2_step_gpt-4o-mini": "gpt-4o-mini/2step",
    "Generated_report/2_step_gemma-3-27b-it": "gemma-3-27b-it/2step",
    "Generated_report/2_step_Qwen3-32B": "Qwen3-32B/2step",
    "Generated_report/2_step_Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B-16E-Instruct/2step",
    "Generated_report/1_step_gpt-4o-mini": "gpt-4o-mini/1step",
    "Generated_report/rulebase": "rule_base",
    "__reference__": "reference",
}

INDICES = range(100)                      # report_ids to evaluate
OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # e.g., "gpt-4o"
SEED = 42
OUTDIR = "evaluation/LLM_eval"
STATE_FILENAME = "state.json"

# Concurrency of judge API requests (tune based on your rate limits)
_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "3"))
# Optional: support OpenAI-compatible endpoints (e.g., DeepInfra)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# -------------------------------------

# Async client singleton
_client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    _client_kwargs["base_url"] = OPENAI_BASE_URL
client = AsyncOpenAI(**_client_kwargs)

# global semaphore
_sem = asyncio.Semaphore(_CONCURRENCY)

PARA_RE = re.compile(r'paragraphs=\((.*?)\)\)\s*(?:,|\])', re.S)

# 8 evaluation dimensions (keep order stable)
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

# ----------------- I/O helpers -----------------

def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def pull_news(text: str) -> str:
    """Optional parser (unchanged)"""
    lead = ''
    if not text.lstrip().startswith('[ArticleSection'):
        lead, _, text = text.partition('[')
        lead = lead.strip()
    paragraphs: List[str] = []
    for chunk in PARA_RE.findall(text):
        paragraphs.extend(ast.literal_eval('(' + chunk + ')'))
    # return lead + "\n".join(paragraphs)
    return "".join(paragraphs)

def _labels_for(model_dirs_with_ref: List[str]) -> List[str]:
    labels = []
    for p in model_dirs_with_ref:
        base = os.path.basename(p.rstrip("/")) or p
        labels.append(DISPLAY_NAME_MAP.get(base, base))
    return labels

def read_ref(report_id: int) -> str:
    """
    Reference is treated as a 10th 'path' (candidate).
    We keep reading from BASE_DIR/<report_id>/article.txt via this helper.
    """
    ref_path = os.path.join(BASE_DIR, f"{report_id}", "article.txt")
    return pull_news(read_text(ref_path))

# --- Robust JSON extraction helpers ---
FENCE_JSON_RE = re.compile(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", re.S)

def _find_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced top-level {...} JSON object in text, or None."""
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

def _extract_json_text(content: str) -> Optional[str]:
    """Prefer fenced block; else first top-level object; else None."""
    m = FENCE_JSON_RE.search(content)
    if m:
        return m.group(1).strip()
    # If content already starts with { ... }, keep only the first full object
    if content.strip().startswith('{'):
        obj = _find_first_json_object(content)
        if obj:
            return obj.strip()
    # Fallback: search anywhere
    obj = _find_first_json_object(content)
    return obj.strip() if obj else None


# -------------- table + state building/saving --------------

def _build_pairwise_df(labels: List[str], wins: List[List[int]], counts: List[List[int]]) -> pd.DataFrame:
    M = len(labels)
    pairwise = []
    for i in range(M):
        row = []
        for j in range(M):
            if i == j:
                row.append(None)
            else:
                c = counts[i][j]
                row.append(round(100.0 * wins[i][j] / c, 2) if c > 0 else None)
        pairwise.append(row)
    return pd.DataFrame(pairwise, index=labels, columns=labels)

def _build_summary_df(labels: List[str], wins: List[List[int]], counts: List[List[int]]) -> pd.DataFrame:
    summary_rows = []
    M = len(labels)
    for i in range(M):
        total_wins = sum(wins[i][j] for j in range(M) if j != i)
        total_matches = sum(counts[i][j] for j in range(M) if j != i)
        win_rate = (100.0 * total_wins / total_matches) if total_matches > 0 else 0.0
        summary_rows.append({
            "model": labels[i],
            "wins": total_wins,
            "matches": total_matches,
            "win_rate_%": round(win_rate, 2)
        })
    return pd.DataFrame(summary_rows).sort_values(by="win_rate_%", ascending=False).reset_index(drop=True)

def _save_all_tables(
    outdir: str,
    labels: List[str],
    wins_overall: List[List[int]],
    counts_overall: List[List[int]],
    wins_by_dim: Dict[str, List[List[int]]],
    counts_by_dim: Dict[str, List[List[int]]],
    checkpoint_tag: Optional[str] = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Save overall matrices
    pairwise_overall = _build_pairwise_df(labels, wins_overall, counts_overall)
    summary_overall = _build_summary_df(labels, wins_overall, counts_overall)
    pairwise_overall.to_csv(os.path.join(outdir, "pairwise_win_rates_overall.csv"))
    summary_overall.to_csv(os.path.join(outdir, "overall_win_rates_overall.csv"), index=False)

    # Save per-dimension (excluding Overall which already has its own files above)
    for dim in DIMENSIONS[:-1]:
        pairwise_dim = _build_pairwise_df(labels, wins_by_dim[dim], counts_by_dim[dim])
        summary_dim = _build_summary_df(labels, wins_by_dim[dim], counts_by_dim[dim])
        safe = dim.lower().replace(" ", "_")
        pairwise_dim.to_csv(os.path.join(outdir, f"pairwise_win_rates_{safe}.csv"))
        summary_dim.to_csv(os.path.join(outdir, f"overall_win_rates_{safe}.csv"), index=False)

    # Optional checkpoints per circle
    if checkpoint_tag is not None:
        ckpt_dir = os.path.join(outdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        pairwise_overall.to_csv(os.path.join(ckpt_dir, f"pairwise_win_rates_overall_{checkpoint_tag}.csv"))
        summary_overall.to_csv(os.path.join(ckpt_dir, f"overall_win_rates_overall_{checkpoint_tag}.csv"), index=False)
        for dim in DIMENSIONS[:-1]:
            pairwise_dim = _build_pairwise_df(labels, wins_by_dim[dim], counts_by_dim[dim])
            summary_dim = _build_summary_df(labels, wins_by_dim[dim], counts_by_dim[dim])
            safe = dim.lower().replace(" ", "_")
            pairwise_dim.to_csv(os.path.join(ckpt_dir, f"pairwise_win_rates_{safe}_{checkpoint_tag}.csv"))
            summary_dim.to_csv(os.path.join(ckpt_dir, f"overall_win_rates_{safe}_{checkpoint_tag}.csv"), index=False)

def _state_path(outdir: str) -> str:
    return os.path.join(outdir, STATE_FILENAME)

def save_state(
    outdir: str,
    wins_overall: List[List[int]],
    counts_overall: List[List[int]],
    wins_by_dim: Dict[str, List[List[int]]],
    counts_by_dim: Dict[str, List[List[int]]],
    completed_ids: Iterable[int],
    model_dirs: List[str],
    seed: int,
) -> None:
    state = {
        "wins_overall": wins_overall,
        "counts_overall": counts_overall,
        "wins_by_dim": wins_by_dim,
        "counts_by_dim": counts_by_dim,
        "completed_ids": list(sorted(set(completed_ids))),
        "model_dirs": model_dirs,
        "display_map": DISPLAY_NAME_MAP,
        "seed": seed,
    }
    with open(_state_path(outdir), "w", encoding="utf-8") as f:
        json.dump(state, f)

def load_state(outdir: str) -> Optional[dict]:
    path = _state_path(outdir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------- Async judge -----------------

def _judge_prompt(first: str, second: str) -> list[dict]:
    user = f"""You are an expert evaluator of news articles.

Evaluate the two candidate articles on **8 dimensions** below. Decide the winner per dimension, then pick an **Overall** winner. Briefly explain each choice.
**Return a single JSON object. Do NOT use Markdown code fences or backticks.**

Dimensions:
1. Factual Consistency — factually sound and correct.
2. Logical Consistency — coherent and self-consistent.
3. Importance — conveys more important information.
4. Readability — fluent and easy to read.
5. Objectivity — neutral, minimal opinion.
6. Journalistic — adheres to journalistic style.
7. Information Density — more useful information per length.
8. Overall — best considering all dimensions above. **No tie allowed.**

First Article:
{first}

Second Article:
{second}

Return **only** JSON with this schema (no extra text):

{{
  "Factual Consistency": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Logical Consistency": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Importance": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Readability": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Objectivity": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Journalistic": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Information Density": {{"winner": "first"|"second"|"tie", "reasoning": "brief"}},
  "Overall": {{"winner": "first"|"second", "reasoning": "brief"}}
}}
"""
    return [
        {"role": "system", "content": "You are an expert evaluator of news articles. Respond with valid JSON only."},
        {"role": "user", "content": user},
    ]

def _normalize_winner(w: Optional[str], flipped: bool) -> Optional[str]:
    if w not in ("first", "second", "tie"):
        return None
    if w == "tie":
        return "tie"
    if flipped:
        # (first,second) == (a,b)
        return w
    else:
        # we showed (b,a); swap back to (a,b)
        return "first" if w == "second" else "second"

async def _judge_pair_async(
    text_a: str,
    text_b: str,
    flip: bool,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Returns structured result with per-dimension winners and raw JSON, or None on failure."""
    first, second = (text_a, text_b) if flip else (text_b, text_a)
    messages = _judge_prompt(first, second)

    async with _sem:
        try:
            rsp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=700,
            )
            content = rsp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[judge] API error: {e}")
            return None
    raw_text = content
    json_text = _extract_json_text(content)
    try:
        # print(content)
        data = json.loads(json_text)
    except Exception:
        print("[judge] could not parse JSON")
        return None

    # Normalize winners to (a,b) coordinates
    winners: Dict[str, Optional[str]] = {}
    for dim in DIMENSIONS:
        entry = data.get(dim, {})
        winners[dim] = _normalize_winner(entry.get("winner"), flip)

    return {
        "raw": data,            # original JSON
        "winners": winners,     # normalized per-dimension winners in (a,b)
    }


# ----------------- Main evaluation (async per circle) -----------------

def _zeros_matrix(n: int) -> List[List[int]]:
    return [[0 for _ in range(n)] for __ in range(n)]

async def evaluate_round_robin_async(
    model_dirs: List[str] = MODEL_DIRS,
    indices=INDICES,
    include_openai: bool = True,
    seed: int = SEED,
    outdir: str = OUTDIR,
    resume: bool = False,
    model_name: str = OPENAI_MODEL,
) -> None:
    rnd = random.Random(seed)

    # Add reference as a 10th participant
    model_dirs_with_ref = list(model_dirs) + ["__reference__"]
    labels = _labels_for(model_dirs_with_ref)
    M = len(model_dirs_with_ref)  # should be 10

    # Overall (uses "Overall" dimension)
    wins_overall = _zeros_matrix(M)
    counts_overall = _zeros_matrix(M)

    # Per-dimension
    wins_by_dim: Dict[str, List[List[int]]] = {dim: _zeros_matrix(M) for dim in DIMENSIONS[:-1]}
    counts_by_dim: Dict[str, List[List[int]]] = {dim: _zeros_matrix(M) for dim in DIMENSIONS[:-1]}

    completed_ids: Set[int] = set()

    if resume:
        prior = load_state(outdir)
        if prior and prior.get("model_dirs") == model_dirs:
            wins_overall = prior["wins_overall"]
            counts_overall = prior["counts_overall"]
            wins_by_dim = {k: v for k, v in prior["wins_by_dim"].items()}
            counts_by_dim = {k: v for k, v in prior["counts_by_dim"].items()}
            completed_ids = set(prior.get("completed_ids", []))
            print(f"Resuming from state with {len(completed_ids)} completed report_ids.")
        elif prior:
            print("Existing state.json uses different model_dirs; ignoring resume.")

    for report_id in indices:
        if report_id in completed_ids:
            continue
        # if report_id < 100:
        #     continue

        print(f"Processing report_id {report_id}...")

        # Load all candidate drafts for this report_id (9 model dirs)
        texts: List[str] = []
        skip = False
        for d in model_dirs:
            draft_path = os.path.join(d, f"{report_id}", "draft.txt")
            txt = read_text(draft_path)
            # if d == print(txt)
            if txt == "":
                skip = True
                # print(d)
                break
            texts.append(txt)

        # print(skip)

        if skip:
            completed_ids.add(report_id)
            save_state(outdir, wins_overall, counts_overall, wins_by_dim, counts_by_dim, completed_ids, model_dirs, seed)
            continue

        # Load the 10th participant (reference) as another "path"
        reference_text = read_ref(report_id)
        # print(reference_text)
        if reference_text.strip() == "":
            print(f"[warn] Missing reference for report_id {report_id}; skipping.")
            completed_ids.add(report_id)
            save_state(outdir, wins_overall, counts_overall, wins_by_dim, counts_by_dim, completed_ids, model_dirs, seed)
            continue
        texts.append(reference_text)  # now length 10, aligned with model_dirs_with_ref

        # Prepare all pairings for this circle with deterministic flips
        pairs = list(combinations(range(M), 2))  # 45 for M=10
        flips = [rnd.random() < 0.5 for _ in pairs]

        if include_openai and not OPENAI_API_KEY and not OPENAI_BASE_URL:
            print("Warning: OPENAI_API_KEY is empty. Skipping OpenAI evaluation for this circle.")
            completed_ids.add(report_id)
            save_state(outdir, wins_overall, counts_overall, wins_by_dim, counts_by_dim, completed_ids, model_dirs, seed)
            continue

        # Launch async judge calls for all pairs (bounded by _CONCURRENCY)
        tasks: List[asyncio.Task] = []
        for (pair_idx, (i, j)) in enumerate(pairs):
            a, b = texts[i], texts[j]
            flip = flips[pair_idx]
            tasks.append(asyncio.create_task(_judge_pair_async(a, b, flip, model_name)))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Directory for raw outputs
        raw_dir = os.path.join(outdir, "raw_judge_json", f"report_{report_id}")
        os.makedirs(raw_dir, exist_ok=True)

        # Aggregate
        for (res, (i, j), flip) in zip(results, pairs, flips):
            if isinstance(res, Exception) or res is None:
                continue

            # Save raw JSON with some metadata
            meta = {
                "i": i, "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "flipped_presentation": flip,
                "report_id": report_id,
                "result": res["raw"],
            }
            with open(os.path.join(raw_dir, f"{i}_vs_{j}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            winners = res["winners"]

            # Per-dimension aggregation (ties do not add to wins but still count as a comparison)
            for dim in DIMENSIONS[:-1]:
                w = winners.get(dim)
                if w is None:
                    continue
                counts_by_dim[dim][i][j] += 1
                counts_by_dim[dim][j][i] += 1
                if w == "first":
                    wins_by_dim[dim][i][j] += 1
                elif w == "second":
                    wins_by_dim[dim][j][i] += 1
                # ties: no wins increment

            # Overall aggregation (must be first/second only)
            w_overall = winners.get("Overall")
            if w_overall in ("first", "second"):
                counts_overall[i][j] += 1
                counts_overall[j][i] += 1
                if w_overall == "first":
                    wins_overall[i][j] += 1
                else:
                    wins_overall[j][i] += 1

        # Save after each circle
        _save_all_tables(
            outdir=outdir,
            labels=labels,
            wins_overall=wins_overall,
            counts_overall=counts_overall,
            wins_by_dim=wins_by_dim,
            counts_by_dim=counts_by_dim,
            checkpoint_tag=f"report_{report_id}",
        )
        completed_ids.add(report_id)
        save_state(outdir, wins_overall, counts_overall, wins_by_dim, counts_by_dim, completed_ids, model_dirs, seed)

    # Final printout (overall only for brevity)
    pairwise_df_overall = _build_pairwise_df(labels, wins_overall, counts_overall)
    summary_df_overall = _build_summary_df(labels, wins_overall, counts_overall)
    print("\nPairwise win-rate matrix (Overall, %):")
    print(pairwise_df_overall)
    print("\nOverall win rates (Overall dimension):")
    print(summary_df_overall)


# ----------------- Convenience resume wrapper -----------------

async def resume_round_robin_async(
    model_dirs: List[str] = MODEL_DIRS,
    indices=INDICES,
    include_openai: bool = True,
    outdir: str = OUTDIR,
    model_name: str = OPENAI_MODEL,
) -> None:
    await evaluate_round_robin_async(
        model_dirs=model_dirs,
        indices=indices,
        include_openai=include_openai,
        seed=SEED,
        outdir=outdir,
        resume=True,
        model_name=model_name,
    )


# ----------------- Script entry -----------------

if __name__ == "__main__":
    try:
        asyncio.run(evaluate_round_robin_async())
        # To resume later:
        # asyncio.run(resume_round_robin_async())
    except KeyboardInterrupt:
        print("Interrupted by user")
