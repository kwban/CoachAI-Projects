from __future__ import annotations

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

import ijson  # pip install ijson
import time
import copy

TYPE = os.getenv("LLM_PROVIDER", "GPT")  # "GPT" or "DEEPINFRA" or "GEMINI"
def _default_model(provider: str) -> str:

    if provider == "DEEPINFRA":return "Qwen/Qwen3-32B"
    elif provider == "GPT":return "gpt-4o-mini"
    elif provider == "GEMINI":return "gemini-2.5-flash"

MODEL = os.getenv("LLM_MODEL", _default_model(TYPE))

if TYPE == "GPT":
    from openai import AsyncOpenAI  # async client

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    _client_singleton = AsyncOpenAI(api_key=OPENAI_API_KEY)

    def make_client(_) -> "AsyncOpenAI": 
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

    def make_client(slot: int) -> "AsyncOpenAI":
        return _deepinfra_clients[slot % len(_deepinfra_clients)]
    
elif TYPE == "GEMINI":
    from openai import AsyncOpenAI          # still use the openai package
    import google.genai
    # ideally load from ENV:  GEMINI_API_KEYS="key1,key2,key3"
    GEMINI_KEYS: List[str] = [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ]

    _gemini_clients: List[AsyncOpenAI] = []
    for k in GEMINI_KEYS:
        _gemini_clients.append(
            AsyncOpenAI(
                api_key=k,
                # <-- the official OpenAI-compat endpoint
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        )

    def make_client(slot: int) -> "AsyncOpenAI":
        return _gemini_clients[slot % len(_gemini_clients)]


else:  # defensive – shouldn't happen
    raise ValueError(f"Unknown TYPE {TYPE!r}. Use 'GPT' or 'DEEPINFRA' or 'GEMINI'.")

# ----------------------- LOGGING ------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Silence noisy per‑request logs from the OpenAI + httpx libraries
for _noisy in ("openai", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from json_keyword_extractor import semantic_search_in_json 
import re  

with open("report_dataset.json", "r", encoding="utf-8") as f:
    data_solid = json.load(f)

def search_object(json_data: dict, query: str, year: str, month: str, day: str):
    return semantic_search_in_json(data_solid, query, 5, year, month, day)


def insert_object(report_store: List[Dict[str, Any]], obj: Dict[str, Any], section: str = "default"):
    report_store.append({"section": section, "content": obj, "source": "search"})


def remove_object(report_store: List[Dict[str, Any]], object_id: str):
    report_store[:] = [
        e
        for e in report_store
        if not (e["content"].get("ID") == object_id or e["content"].get("Speaker") == object_id)
    ]


def extract_json_like_substring(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return ""

_CONCURRENCY = 1  # max simultaneous reports
_sem = asyncio.Semaphore(_CONCURRENCY)

async def call_chat(slot: int, messages: list[dict[str, str]]):
    """Thin async wrapper that respects the global parallelism cap."""

    async with _sem:
        client = make_client(slot)
        rsp = await client.chat.completions.create(model=MODEL, messages=messages)
    return rsp.choices[0].message.content, getattr(rsp, "usage", None)

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
    query_count: int = 10,
):

    all_time_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    total_prompt_tokens = total_completion_tokens = 0
    action_counts = {"search": 0, "insert": 0, "remove": 0}
    insert_fail_count = 0

    all_search_results: list[str] = []
    report_store: list[dict[str, Any]] = []
    used_queries: list[str] = []

    search_time = 0
    insert_time = 0

    # Load prompt templates once – *not* inside each loop.
    with open("react/agent_prompt_easy_search.txt", "r", encoding="utf-8") as f:
        system_p_search = f.read()
    with open("react/agent_prompt_easy_insert.txt", "r", encoding="utf-8") as f:
        system_p_insert = f.read()
    with open("react/agent_prompt_easy_remove.txt", "r", encoding="utf-8") as f:
        system_p_remove = f.read()

    for i in range(query_count):
        # break
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": f"searching record : {all_search_results}\ncurrent draft : {report_store}",
            },
        ]

        # Ask the agent what action to take.
        content, usage = await call_chat(slot, base_messages)

        if usage:
            total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
            total_completion_tokens += getattr(usage, "completion_tokens", 0)

        # Decide which branch to execute.
        
        print(content)
        action: str
        if "search" in content:
            action = "search"
            print("action: ",'search')
        elif "insert" in content:
            action = "insert"
            print("action: ",'insert')
        elif "remove" in content:
            action = "remove"
        elif "terminate" in content:
            break
        else:
            continue

        if action == "search":
            search_time_start = time.time()
            action_counts[action] += 1
            messages = [
                {"role": "system", "content": system_p_search},
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"searching record : {all_search_results}\ncurrent draft : {report_store}",
                },
            ]
            query, usage = await call_chat(slot, messages)
            if usage:
                total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                total_completion_tokens += getattr(usage, "completion_tokens", 0)

            logging.debug("[R%s] search‑query = %s", report_id, query)
            used_queries.append(query)
            query = re.sub(r'</?think>', '', query).strip()
            results = search_object({report_id: json_data}, query, year, month, day)
            if results:
                all_search_results.append(results[0]["text"])

            search_time_end = time.time()
            search_time += (search_time_end - search_time_start)

        elif action == "insert":
            insert_time_start = time.time()
            messages = [
                {"role": "system", "content": system_p_insert},
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"searching record : {all_search_results}\ncurrent draft : {report_store}",
                },
            ]
            query, usage = await call_chat(slot, messages)
            if usage:
                total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                total_completion_tokens += getattr(usage, "completion_tokens", 0)

            def extract_second_or_first(sql: str):
                # Regex to capture all quoted strings after VALUES
                tokens = re.findall(r'(["\'])(.*?)\1', sql, flags=re.DOTALL)
                if not tokens:
                    return None
                # tokens is a list of tuples like [('"', 'text'), ...]
                if len(tokens) > 1:
                    return tokens[1][1]
                return tokens[0][1]

            if 'INSERT INTO' in query:
                print('print(query)',query)
                query = extract_second_or_first(query)
                print('print(query result)',query)

            obj = re.sub(r'</?think>', '',query).strip()
            print("insert_obj :",obj)

            if re.sub(r"[^a-zA-Z0-9]", "", obj) in re.sub(
                r"[^a-zA-Z0-9]", "", str(all_search_results)
            ):
                insert_object(report_store, obj)
                action_counts[action] += 1
            else:
                insert_fail_count += 1

            insert_time_end = time.time()
            insert_time += (insert_time_end - insert_time_start)

        elif action == "remove":
            action_counts[action] += 1
            messages = [
                {"role": "system", "content": system_p_remove},
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"searching record : {all_search_results}\ncurrent draft : {report_store}",
                },
            ]
            object_id, usage = await call_chat(slot, messages)
            if usage:
                total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                total_completion_tokens += getattr(usage, "completion_tokens", 0)
            remove_object(report_store, object_id)

        # Persist intermediate state on disk (optional but kept from original).
        payload = copy.deepcopy(base_messages)
        extra = {"role": "system", "content": query}
        payload.append(extra)
        with open(output_dir / f"messages{i}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    write_time_start = time.time()
    with open("react/part2_prompt.txt", "r", encoding="utf-8") as file:
        final_sys_prompt = file.read()

    summary_messages = [
        {"role": "system", "content": final_sys_prompt},
        {
            "role": "user",
            "content": user_prompt + "\nHistirical data :" + str(report_store),
        },
    ]

    print(summary_messages)

    while True:
        content, _ = await call_chat(slot, summary_messages)
        try:
            parsed = json.loads(
                extract_json_like_substring(content).replace("\n", "").replace("\\", "")
            )
            break
        except json.JSONDecodeError:
            continue  # try again
    
    write_time_end = time.time()

    (output_dir / "draft.txt").write_text(parsed.get("Report", ""), encoding="utf-8")
    (output_dir / "store.json").write_text(json.dumps(report_store, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "search.json").write_text(json.dumps(all_search_results, ensure_ascii=False, indent=2), encoding="utf-8")

    all_time_end = time.time()

    if action_counts["search"] != 0:
        avg_search_time = search_time/action_counts["search"]
    else:
        avg_search_time = 0

    if action_counts["insert"]+insert_fail_count != 0:
        avg_insert_time = insert_time/(action_counts["insert"]+insert_fail_count)
    else:
        avg_insert_time = 0

    stats = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "action_counts": action_counts,
        "insert_fail_count" : insert_fail_count,
        "all_time = " : all_time_end - all_time_start,
        "avg_search_time" : avg_search_time,
        "avg_insert_time" : avg_insert_time,
        "write_time" : write_time_end - write_time_start,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logging.info("report %s finished (tokens: %s)", report_id, stats["total_tokens"])

async def main():
    output_root = Path(f"Generated_report/2_step_{MODEL.split('/')[0]}")
    output_root.mkdir(exist_ok=True)

    tasks: list[asyncio.Task[None]] = []

    # Stream top‑level key/value pairs from the huge JSON file.
    with open("report_dataset.json", "rb") as f:
        for report_id, content in ijson.kvitems(f, ""):
            int_id = int(report_id)
            if not (0 <= int_id <= 100):
                continue  # keep your filter

            info = content["Report_info"][0]
            title = info["Title"]
            date = f"{info['Year']}-{info['Month']}-{info['Day']}"
            firsthand = content.get("Firsthand_Information", {})
            for key in ("Speaker", "Description", "Image"):
                # strip heavy fields like "Encode"
                if key in firsthand and isinstance(firsthand[key], list):
                    for item in firsthand[key]:
                        if isinstance(item, dict) and "Encode" in item:
                            del item["Encode"]

            user_p = f"Title: {title}\nDate: {date}\nFirsthand information: {firsthand}"

            slot = len(tasks) % _CONCURRENCY  # 0,1,2 – good enough
            out_dir = output_root / report_id
            task = asyncio.create_task(
                call_llm_with_react_async(
                    slot,
                    report_id,
                    content,
                    system_prompt=open("react/agent_prompt_easy.txt", "r", encoding="utf-8").read(),
                    user_prompt=user_p,
                    year=info["Year"],
                    month=info["Month"],
                    day=info["Day"],
                    output_dir=out_dir,
                    query_count=20,
                )
            )
            tasks.append(task)

            # Limit queue length to keep only n = _CONCURRENCY live tasks.
            if len(tasks) >= _CONCURRENCY:
                # Wait for the *first* one to finish before spawning more.
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Re‑add any unfinished tasks.
                tasks = list(tasks)

    # Wait for the last few to finish.
    if tasks:
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
