import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.distance import cosine
# from react.json_keyword_extractor import semantic_search_in_json

INPUT_FILE = 'report_dataset.json'
OUTPUT_FILE = 'historical_search_results.json'
CHECKPOINT_FILE = 'historical_search_results_checkpoint.json'
TOP_K = 5  # You can adjust this as needed
MAX_REPORTS = 50  # Only process the first positive-keyed report for testing


def extract_historical_info_entries_with_encoding(historical_info):
    entries = []
    if not isinstance(historical_info, dict):
        return entries
    # Descriptions
    for desc in historical_info.get('Description', []):
        if isinstance(desc, dict) and 'Description' in desc:
            entries.append((desc['Description'], desc.get('Encode', None)))
        elif isinstance(desc, str):
            entries.append((desc, None))
    # Speakers
    for speaker in historical_info.get('Speaker', []):
        if isinstance(speaker, dict) and 'Text' in speaker:
            entries.append((speaker['Text'], speaker.get('Encode', None)))
        elif isinstance(speaker, str):
            entries.append((speaker, None))
    return entries


def extract_all_entries_with_encoding(report):
    results = []
    # Firsthand_Information
    fh = report.get('Firsthand_Information', {})
    if isinstance(fh, dict):
        for desc in fh.get('Description', []):
            if isinstance(desc, dict) and 'Description' in desc:
                results.append({'text': desc['Description'], 'encode': desc.get('Encode', None)})
        for speaker in fh.get('Speaker', []):
            if isinstance(speaker, dict) and 'Text' in speaker:
                results.append({'text': speaker['Text'], 'encode': speaker.get('Encode', None)})
        for image in fh.get('Image', []):
            if isinstance(image, dict) and 'Caption' in image:
                results.append({'text': image['Caption'], 'encode': image.get('Encode', None)})
    # Historical_Information
    hi = report.get('Historical_Information', {})
    if isinstance(hi, dict):
        for desc in hi.get('Description', []):
            if isinstance(desc, dict) and 'Description' in desc:
                results.append({'text': desc['Description'], 'encode': desc.get('Encode', None)})
        for speaker in hi.get('Speaker', []):
            if isinstance(speaker, dict) and 'Text' in speaker:
                results.append({'text': speaker['Text'], 'encode': speaker.get('Encode', None)})
        for image in hi.get('Image', []):
            if isinstance(image, dict) and 'Caption' in image:
                results.append({'text': image['Caption'], 'encode': image.get('Encode', None)})
    return results


def process_report(args):
    key, report, combined_data = args
    print(f"Processing report key: {key}")
    # Extract date info
    report_info = report.get('Report_info', [{}])[0]
    year = report_info.get('Year', '')
    month = report_info.get('Month', '')
    day = report_info.get('Day', '')
    # Extract historical information entries
    historical_info = report.get('Historical_Information', {})
    entries = extract_historical_info_entries_with_encoding(historical_info)
    entry_results = []
    for idx, (entry, entry_encode) in enumerate(entries):
        if not entry.strip() or entry_encode is None:
            entry_results.append([])
            continue
        print(f"  Semantic search for entry {idx+1}/{len(entries)} in report {key}...")
        results = semantic_search_in_json_with_encoding(combined_data, entry_encode, TOP_K, year, month, day)
        print(f"  Done semantic search for entry {idx+1}/{len(entries)} in report {key}.")
        entry_results.append(results)
    print(f"Finished processing report key: {key}")
    return key, entry_results


def semantic_search_in_json_with_encoding(json_data, query_encode, top_k, year, month, day):
    all_results = []
    threshold = 0.7
    query_vec = np.array(query_encode)
    for key, report in json_data.items():
        report_info = report.get("Report_info", [{}])[0]
        r_year = report_info.get("Year", "")
        r_month = report_info.get("Month", "")
        r_day = report_info.get("Day", "")
        # Only consider reports before the query date
        if (r_year < year) or (r_year == year and r_month < month) or (r_year == year and r_month == month and r_day < day):
            entries = extract_all_entries_with_encoding(report)
            for item in entries:
                text = item['text']
                encode = item['encode']
                if encode is None:
                    continue
                score = 1 - cosine(query_vec, np.array(encode))
                if score >= threshold:
                    all_results.append({'text': text, 'score': float(score)})   
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    return sorted_results[:top_k]


def main():
    print("Loading input file...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        combined = json.load(f)
    print(f"Loaded {len(combined)} reports.")
    # Only process positive integer keys (as strings)
    positive_keys = [k for k in combined.keys() if k.isdigit() and int(k) >= 0]
    positive_keys = sorted(positive_keys, key=int)[1990:]
    items = [(k, combined[k], combined) for k in positive_keys]
    print(f"Processing {len(items)} positive-keyed reports: {positive_keys}")
    # Try to load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            results_dict = json.load(f)
    else:
        results_dict = {}
    # Skip already processed keys
    items = [(k, v, c) for (k, v, c) in items if k not in results_dict]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_report, item): item[0] for item in items}
        for future in as_completed(futures):
            key, results = future.result()
            results_dict[key] = results
            print(f"Checkpointing after report {key}...")
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print("Saving final results...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved to {OUTPUT_FILE}")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == '__main__':
    main() 