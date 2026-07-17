from __future__ import annotations

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import re
import copy

# Optional: stream the huge JSON (set to True to use ijson)
USE_STREAMING = True
try:
    import ijson  # pip install ijson
except Exception:
    USE_STREAMING = False

TYPE = os.getenv("LLM_PROVIDER", "GPT")  # "GPT" or "DEEPINFRA"
def _default_model(provider: str) -> str:
    if provider == "DEEPINFRA":
        return os.getenv("LLM_MODEL", "google/gemma-3-27b-it")
    return os.getenv("LLM_MODEL", "gpt-4o")

MODEL = _default_model(TYPE)

from openai import AsyncOpenAI  # DeepInfra also speaks OpenAI protocol

if TYPE == "GPT":
    OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for GPT provider")
    _client_singleton = AsyncOpenAI(api_key=OPENAI_API_KEY)

    def make_client(slot: int) -> AsyncOpenAI:  # shared client
        return _client_singleton

elif TYPE == "DEEPINFRA":
    import deepinfra
    from openai import AsyncOpenAI 

    DEEPINFRA_KEYS: List[str] = [
        os.getenv("DEEPINFRA_API_KEY_1"),
        os.getenv("DEEPINFRA_API_KEY_2"),
        os.getenv("DEEPINFRA_API_KEY_3"),
    ]

    # Pre‑create a client per key so sockets + TLS handshakes are reused.
    _deepinfra_clients: List[AsyncOpenAI] = []
    for k in DEEPINFRA_KEYS:
        _deepinfra_clients.append(
            AsyncOpenAI(
                api_key=k,
                base_url="https://api.deepinfra.com/v1/openai",
            )
        )

    def make_client(slot: int) -> AsyncOpenAI:
        return _deepinfra_clients[slot % len(_deepinfra_clients)]
else:
    raise ValueError(f"Unknown LLM_PROVIDER {TYPE!r}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
for _noisy in ("openai", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from json_keyword_extractor import semantic_search_in_json

with open("report_dataset.json", "r", encoding="utf-8") as f:
    data_solid = json.load(f)

def search_object(json_data: dict, query: str, year: str, month: str, day: str) -> List[Dict[str, Any]]:
    # Keep code1's behavior: searches the global data_solid
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

def rephrase_content(report_entries: List[Dict[str, Any]], target: str, style: str) -> str:
    segments: List[str] = []
    for entry in report_entries:
        obj = entry['content']
        if obj.get("Speaker") and obj.get("Text"):
            segments.append(f"[Speaker] {obj['Speaker']}: {obj['Text']}")
        elif obj.get("Caption"):
            segments.append(f"[Caption] {obj['Caption']}")
        elif obj.get("Conversation"):
            segments.append(f"[Conversation] {obj['Conversation']}")
        elif obj.get("Text") or obj.get("Description"):
            val = obj.get("Text", obj.get("Description"))
            segments.append(f"[Description] {val}")
        elif obj.get("text"):
            segments.append(f"[Description] {obj['text']}")
    return "\n".join(segments)

def extract_json_like_substring(text: str) -> str:
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        return text[start:end + 1]
    return ""

_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
_sem = asyncio.Semaphore(_CONCURRENCY)

async def call_chat(slot: int, messages: list[dict[str, str]]):
    async with _sem:
        client = make_client(slot)
        rsp = await client.chat.completions.create(model=MODEL, messages=messages)
    return rsp.choices[0].message, getattr(rsp, "usage", None)

async def call_llm_with_react_async(
    slot: int,
    report_id: str,
    json_data: dict,
    system_prompt: str,
    user_prompt: str,
    year: str,
    month: str,
    day: str,
    output_dir: Path,
    query_count: int = 10
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    all_time_start = time.time()

    total_prompt_tokens = 0
    total_completion_tokens = 0
    action_counts = {"search": 0, "insert": 0, "remove": 0}
    insert_fail_count = 0

    all_search_results: List[Dict[str, Any]] = []
    report_store: List[Dict[str, Any]] = []
    used_queries: List[str] = []

    search_time = 0.0
    insert_time = 0.0

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # ReAct loop
    for i in range(query_count):
        # Ask the agent for the next JSON action
        retries = 0
        while True:
            retries += 1
            if retries > 5:
                # give up on this turn
                break

            msg, usage = await call_chat(slot, messages)
            if usage:
                total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            raw = msg.content or ""
            try:
                # keep code1's parsing approach for parallelism in behaviour
                content = json.loads(extract_json_like_substring(raw).replace("\n", "").replace("\\", ""))
                action = content.get("Action")
                if action in {"search", "insert", "remove", "modify", "rephrase", "terminate"}:
                    break
                else:
                    # nudge the model
                    messages.append({"role": "assistant", "content": json.dumps({"Error": "Unknown Action"}, ensure_ascii=False)})
            except (json.JSONDecodeError, KeyError):
                messages.append({"role": "assistant", "content": json.dumps({"Error": "output format error, must fit json format."}, ensure_ascii=False)})

        # If we failed to get a valid action after retries, continue
        if retries > 5:
            continue

        # Echo the assistant JSON back (as in code1)
        messages.append({"role": "assistant", "content": json.dumps(extract_json_like_substring(raw).replace("\n", "").replace("\\", ""))})

        if action == "search":
            action_counts["search"] += 1
            search_time_start = time.time()

            obj = content.get("query", {})
            query = obj.get("text", "")
            if query:
                used_queries.append(query)
                results = search_object(json_data, query, year, month, day)
                all_search_results.extend(results)
                messages.append({"role": "assistant", "content": json.dumps({"Observation": results}, ensure_ascii=False)})
            else:
                messages.append({"role": "assistant", "content": json.dumps({"Observation": " Empty query"}, ensure_ascii=False)})

            search_time += time.time() - search_time_start

        elif action == "insert":
            insert_time_start = time.time()
            obj = content.get("query", {})

            # code1 compatibility: TEXT extraction
            if 'text' in obj:
                TEXT = obj['text']
            elif 'Text' in obj:
                TEXT = obj['Text']
            else:
                messages.append({"role": "assistant", "content": json.dumps({"Observation": " Insert failed: wrong output format"}, ensure_ascii=False)})
                TEXT = None

            if TEXT is not None and re.sub(r'[^a-zA-Z0-9]', '', TEXT) in re.sub(r'[^a-zA-Z0-9]', '', str(all_search_results)):
                insert_object(report_store, obj)
                messages.append({"role": "assistant", "content": json.dumps({"Observation": " Object inserted"}, ensure_ascii=False)})
                action_counts["insert"] += 1
            else:
                messages.append({"role": "assistant", "content": json.dumps({"Observation": " Insert failed: object not found in search results"}, ensure_ascii=False)})
                insert_fail_count += 1

            insert_time += time.time() - insert_time_start

        elif action == "remove":
            action_counts["remove"] += 1
            object_id = content.get("query", "")
            before = len(report_store)
            remove_object(report_store, object_id)
            after = len(report_store)
            messages.append({"role": "assistant", "content": json.dumps({"Observation": f" Removed {before - after} object(s)"}, ensure_ascii=False)})

        elif action in {"modify", "rephrase"}:
            # code1 simply broke; keep the no-op behaviour for parity
            pass

        elif action == "terminate":
            break

        # Persist step messages (like code1)
        with open(output_dir / f"messages{i}.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    # After loop: save stores like code1
    (output_dir / "store.json").write_text(json.dumps(report_store, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "search.json").write_text(json.dumps(all_search_results, ensure_ascii=False, indent=2), encoding="utf-8")

    write_time_start = time.time()
    with open("react/part2_prompt.txt", "r", encoding="utf-8") as file:
        final_sys_prompt = file.read()
    summary_messages = [
        {"role": "system", "content": final_sys_prompt},
        {"role": "user", "content": user_prompt + '\nHistirical data :' + str(report_store)}
    ]

    # retry until JSON parses (as code1)
    while True:
        msg, usage = await call_chat(slot, summary_messages)
        raw = msg.content or ""
        try:
            parsed = json.loads(extract_json_like_substring(raw).replace("\n", "").replace("\\", ""))
            break
        except json.JSONDecodeError:
            continue

    (output_dir / "draft.txt").write_text(parsed.get("Report", ""), encoding="utf-8")
    write_time_end = time.time()

    all_time_end = time.time()
    avg_search_time = (search_time / action_counts["search"]) if action_counts["search"] else 0.0
    denom = action_counts["insert"] + insert_fail_count
    avg_insert_time = (insert_time / denom) if denom else 0.0

    stats = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "action_counts": action_counts,
        "insert_fail_count": insert_fail_count,
        "all_time = ": all_time_end - all_time_start,  # keep original key shape
        "avg_search_time": avg_search_time,
        "avg_insert_time": avg_insert_time,
        "write_time": write_time_end - write_time_start,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logging.info("✅ report %s finished (tokens: %s)", report_id, stats["total_tokens"])

async def main():
    # Mirror code2's output directory naming
    output_root = Path(f"Generated_report/1_step_{MODEL.split('/')[0]}")
    output_root.mkdir(exist_ok=True)

    # Load the system prompt once
    with open("react/agent_prompt_ori.txt", "r", encoding="utf-8") as f:
        system_p = f.read()

    tasks: list[asyncio.Task[None]] = []

    if USE_STREAMING:
        # Stream top-level kv pairs
        with open("report_dataset.json", "rb") as f:
            for report_id, content in ijson.kvitems(f, ""):
                int_id = int(report_id)
                if int_id > 100 or int_id < 0 or int_id < 0:
                    continue

                info = content["Report_info"][0]
                title = info['Title']
                date = f"{info['Year']}-{info['Month']}-{info['Day']}"
                firsthand_info = content.get("Firsthand_Information", {})
                for key in ["Speaker", "Description", "Image"]:
                    if key in firsthand_info and isinstance(firsthand_info[key], list):
                        for item in firsthand_info[key]:
                            if isinstance(item, dict) and "Encode" in item:
                                del item["Encode"]

                user_p = f"Title: {title}\nDate: {date}\nFirsthand information: {firsthand_info}"
                slot = len(tasks) % _CONCURRENCY
                out_dir = output_root / str(report_id)

                task = asyncio.create_task(
                    call_llm_with_react_async(
                        slot,
                        str(report_id),
                        content,
                        system_p,
                        user_p,
                        info['Year'],
                        info['Month'],
                        info['Day'],
                        out_dir,
                        query_count=20,
                    )
                )
                tasks.append(task)

                # keep at most _CONCURRENCY in-flight tasks
                if len(tasks) >= _CONCURRENCY:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)
    else:
        # Fallback: load whole JSON then schedule tasks
        with open("report_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for report_id, content in data.items():
            int_id = int(report_id)
            if int_id > 100 or int_id < 0 or int_id < 0:
                continue

            info = content["Report_info"][0]
            title = info['Title']
            date = f"{info['Year']}-{info['Month']}-{info['Day']}"
            firsthand_info = content.get("Firsthand_Information", {})
            for key in ["Speaker", "Description", "Image"]:
                if key in firsthand_info and isinstance(firsthand_info[key], list):
                    for item in firsthand_info[key]:
                        if isinstance(item, dict) and "Encode" in item:
                            del item["Encode"]

            user_p = f"Title: {title}\nDate: {date}\nFirsthand information: {firsthand_info}"
            slot = len(tasks) % _CONCURRENCY
            out_dir = output_root / str(report_id)

            task = asyncio.create_task(
                call_llm_with_react_async(
                    slot,
                    str(report_id),
                    content,
                    system_p,
                    user_p,
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
