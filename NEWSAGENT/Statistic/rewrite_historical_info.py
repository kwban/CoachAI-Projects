import json
import openai
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import datetime
from typing import Any

def paraphrase_with_gpt(text: str, max_retries=5) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Please rewrite the input sentence to have the same meaning but a different syntax."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"GPT paraphrasing failed: {e}")
            return text  # fallback: return original
    return text

def extract_text_with_path(data: Any, path: str = ''):
    results = []
    if isinstance(data, dict):
        if 'Description' in data and isinstance(data['Description'], str):
            results.append({"path": path, "text": data['Description']})
        else:
            for key, value in data.items():
                if key == "Image":
                    continue
                new_path = f"{path}->{key}" if path else key
                results += extract_text_with_path(value, new_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            results += extract_text_with_path(item, new_path)
    elif isinstance(data, str):
        results.append({"path": path, "text": data})
    return results

def process_report(args):
    key, report = args
    historical = report.get('Historical_Information', {})
    if not isinstance(historical, dict) or 'Description' not in historical:
        return None
    descriptions = historical.get('Description', [])
    rewritten = False
    new_descriptions = []
    for desc in descriptions:
        orig_text = desc.get('Description', '')
        if not orig_text:
            new_descriptions.append(desc)
            continue
        rewrite_text = paraphrase_with_gpt(orig_text)
        if rewrite_text and rewrite_text != orig_text:
            rewritten = True
            new_desc = desc.copy()
            new_desc['Description'] = rewrite_text
            new_descriptions.append(new_desc)
        else:
            new_descriptions.append(desc)
    if rewritten:
        new_historical = {'Description': new_descriptions}
        # Copy Speaker as-is if present
        if 'Speaker' in historical:
            new_historical['Speaker'] = historical['Speaker']
        # Rewrite Image captions
        if 'Image' in historical:
            new_images = []
            for image in historical['Image']:
                new_image = dict(image)
                if 'Caption' in new_image and new_image['Caption']:
                    new_image['Caption'] = paraphrase_with_gpt(new_image['Caption'])
                new_images.append(new_image)
            new_historical['Image'] = new_images
        new_report = {'Historical_Information': new_historical}
        # Copy Report_info and minus one day on date
        if 'Report_info' in report:
            orig_info = report['Report_info']
            new_info = []
            for info in orig_info:
                info_copy = info.copy()
                year = int(info_copy.get('Year', '1900'))
                month = int(info_copy.get('Month', '1'))
                day = int(info_copy.get('Day', '1'))
                try:
                    date_obj = datetime.date(year, month, day) - datetime.timedelta(days=1)
                    info_copy['Year'] = str(date_obj.year)
                    info_copy['Month'] = f"{date_obj.month:02d}"
                    info_copy['Day'] = f"{date_obj.day:02d}"
                except Exception as e:
                    print(f"Date error for key {key}: {e}")
                new_info.append(info_copy)
            new_report['Report_info'] = new_info
        return ('-' + str(key), new_report)
    return None

def rewrite_historical_info(data, output_path="rewritten_historical_info.json", max_reports=None, num_workers=10):
    output_data = {}
    processed = 0
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        items = list(data.items())
        if max_reports is not None:
            items = items[:max_reports]
        for args in items:
            futures.append(executor.submit(process_report, args))
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                output_key, new_report = result
                output_data[output_key] = new_report
                processed += 1
                if processed % 100 == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                    print(f"Progress saved after {processed} rewritten reports.")
    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nTotal rewritten historical information entries: {len(output_data)}")

if __name__ == "__main__":
    # Allow optional output filename as argument
    json_path = "english_articles_june_july.json"
    output_path = "rewritten_historical_info.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rewrite_historical_info(data, output_path, num_workers=10) 