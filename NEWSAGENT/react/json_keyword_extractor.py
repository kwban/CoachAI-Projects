import json
from typing import Any, List, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

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
                results.append({'text': f"{speaker['Speaker']}: {speaker['Text']}", 'encode': speaker.get('Encode', None)})
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
                results.append({'text': f"{speaker['Speaker']}: {speaker['Text']}", 'encode': speaker.get('Encode', None)})
        for image in hi.get('Image', []):
            if isinstance(image, dict) and 'Caption' in image:
                results.append({'text': image['Caption'], 'encode': image.get('Encode', None)})
    return results


def semantic_search_in_json(json_data, query_encode, top_k, year, month, day):
    all_results = []
    threshold = 0.5

    query_embedding = model.encode(query_encode, convert_to_tensor=True)
    query_embedding = query_embedding.cpu()

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
                # Ensure encode is a tensor and on the same device as query_embedding
                if encode is None:
                    continue
                if isinstance(encode, list):
                    encode = torch.tensor(encode)
                encode = encode.cpu()
                score = util.cos_sim(query_embedding, encode).item()
                if score >= threshold:
                    all_results.append({'text': text, 'score': float(score)})   
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    return sorted_results[:top_k]