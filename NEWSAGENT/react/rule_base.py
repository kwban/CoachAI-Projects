from __future__ import annotations

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List
import time

# Optional: stream the huge JSON (set to True to use ijson)
USE_STREAMING = True
try:
    import ijson
except Exception:
    USE_STREAMING = False

from json_keyword_extractor import semantic_search_in_json

with open("report_dataset.json", "r", encoding="utf-8") as f:
    data_solid = json.load(f)

def search_object(json_data: dict, query: str, year: str, month: str, day: str) -> List[Dict[str, Any]]:
    # Keep original behavior: searches the global data_solid
    return semantic_search_in_json(data_solid, query, 5, year, month, day)

def insert_object(report_store: List[Dict[str, Any]], obj: Dict[str, Any], section: str = "default") -> None:
    report_store.append({'section': section, 'content': obj, 'source': 'search'})

def modify_object(report_store: List[Dict[str, Any]], object_id: str, edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    modified = []
    for entry in report_store:
        obj = entry['content']
        if obj.get("Description") and obj["Description"].startswith(object_id):
            text = obj["Description"]
            for e in sorted(edits, key=lambda x: -x["start"]):
                text = text[:e["start"]] + e["replacement"] + text[e["end"]:]
            obj["Description"] = text
            modified.append(entry)
    return modified

def remove_object(report_store: List[Dict[str, Any]], object_id: str) -> None:
    report_store[:] = [e for e in report_store if not (
        e['content'].get('ID') == object_id or e['content'].get('Speaker') == object_id
    )]

def rephrase_content(report_entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in report_entries:
        obj = entry.get('content', {})
        text = obj.get("Text")
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return "\n".join(lines)

def build_report_from_firsthand_info(firsthand_info: Dict[str, Any]) -> str:
    parts: List[str] = []

    if isinstance(firsthand_info, dict):
        # Descriptions
        for desc in firsthand_info.get("Description", []) or []:
            d = desc.get("Description")
            if isinstance(d, str) and d.strip():
                parts.append(d.strip())

        # Speakers (only their 'Text')
        for sp in firsthand_info.get("Speaker", []) or []:
            t = sp.get("Text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())

    return "\n".join(parts)

_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
_sem = asyncio.Semaphore(_CONCURRENCY)

def _build_queries_from_inputs(title: str, firsthand_info: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    if isinstance(title, str) and title.strip():
        queries.append(title.strip())

    if isinstance(firsthand_info, dict):
        for sp in firsthand_info.get("Speaker", []) or []:
            txt = sp.get("Text")
            if isinstance(txt, str) and txt.strip():
                queries.append(txt.strip())
        for desc in firsthand_info.get("Description", []) or []:
            d = desc.get("Description")
            if isinstance(d, str) and d.strip():
                queries.append(d.strip())
        for img in firsthand_info.get("Image", []) or []:
            cap = img.get("Caption")
            if isinstance(cap, str) and cap.strip():
                queries.append(cap.strip())

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped

async def call_llm_with_react_async(
    slot: int,
    report_id: str,
    json_data: dict,
    title: str,
    firsthand_info: dict,
    year: str,
    month: str,
    day: str,
    output_dir: Path,
    query_count: int = 10
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    all_time_start = time.time()

    action_counts = {"search": 0, "insert": 0}
    insert_fail_count = 0

    all_search_results: List[Dict[str, Any]] = []
    report_store: List[Dict[str, Any]] = []
    used_queries: List[str] = []

    search_time = 0.0
    insert_time = 0.0

    top_k = 5
    score_threshold = 0.8  # << added threshold

    # Build queries
    queries = _build_queries_from_inputs(title, firsthand_info)
    if query_count and query_count > 0:
        queries = queries[:query_count]

    # Run searches
    for query in queries:
        search_time_start = time.time()
        action_counts["search"] += 1

        used_queries.append(query)
        results = search_object(json_data, query, year, month, day) or []
        all_search_results.extend(results)

        search_time += time.time() - search_time_start

    # Sort and keep top_k
    sorted_results = sorted(all_search_results, key=lambda x: x.get('score', 0.0), reverse=True)
    objs = sorted_results#[:top_k]

    # Insert only when score > 0.8
    for obj in objs:
        if float(obj.get('score', 0.0)) <= score_threshold:
            continue
        insert_time_start = time.time()
        action_counts["insert"] += 1
        insert_object(report_store, obj)
        insert_time += time.time() - insert_time_start

    # Save search/store once
    (output_dir / "store.json").write_text(json.dumps(report_store, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "search.json").write_text(json.dumps(all_search_results, ensure_ascii=False, indent=2), encoding="utf-8")

    write_time_start = time.time()

    search_text_block = rephrase_content(report_store)

    firsthand_text_block = build_report_from_firsthand_info(firsthand_info)

    blocks = [b for b in [firsthand_text_block, search_text_block] if b.strip()]
    final_report = "\n\n".join(blocks) if blocks else ""

    (output_dir / "draft.txt").write_text(final_report, encoding="utf-8")

    write_time_end = time.time()

    all_time_end = time.time()
    avg_search_time = (search_time / action_counts["search"]) if action_counts["search"] else 0.0
    denom = action_counts["insert"] + insert_fail_count
    avg_insert_time = (insert_time / denom) if denom else 0.0

    stats = {
        "action_counts": action_counts,
        "insert_fail_count": insert_fail_count,
        "all_time = ": all_time_end - all_time_start,  # keep original key shape
        "avg_search_time": avg_search_time,
        "avg_insert_time": avg_insert_time,
        "write_time": write_time_end - write_time_start,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logging.info("report %s finished", report_id)

async def main():
    output_root = Path("Generated_report/rule_base")
    output_root.mkdir(exist_ok=True)

    tasks: List[asyncio.Task[None]] = []

    if USE_STREAMING:
        with open("report_dataset.json", "rb") as f:
            for report_id, content in ijson.kvitems(f, ""):
                int_id = int(report_id)
                # keep your test condition
                if int_id > 100 or int_id < 0:
                    continue

                info = content["Report_info"][0]
                title = info['Title']

                firsthand_info = content.get("Firsthand_Information", {})
                for key in ["Speaker", "Description", "Image"]:
                    if key in firsthand_info and isinstance(firsthand_info[key], list):
                        for item in firsthand_info[key]:
                            if isinstance(item, dict) and "Encode" in item:
                                del item["Encode"]

                slot = len(tasks) % _CONCURRENCY
                out_dir = output_root / str(report_id)

                task = asyncio.create_task(
                    call_llm_with_react_async(
                        slot,
                        str(report_id),
                        content,
                        title,
                        firsthand_info,
                        info['Year'],
                        info['Month'],
                        info['Day'],
                        out_dir,
                        query_count=20,
                    )
                )
                tasks.append(task)

                if len(tasks) >= _CONCURRENCY:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

    if tasks:
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
