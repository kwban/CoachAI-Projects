import json
from sentence_transformers import SentenceTransformer

INPUT_FILE_1 = 'english_articles_june_july.json'
INPUT_FILE_2 = 'rewritten_historical_info.json'
OUTPUT_FILE = 'report_dataset.json'

model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_text(text):
    if not isinstance(text, str):
        return []
    return model.encode(text).tolist()

def read_entire_json(filename):
    """Read the entire JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_nested_fields(section):
    if not isinstance(section, dict):
        return
    # Encode 'Text' in 'Speaker'
    if 'Speaker' in section and isinstance(section['Speaker'], list):
        for speaker in section['Speaker']:
            if isinstance(speaker, dict) and 'Text' in speaker:
                speaker['Encode'] = encode_text(speaker['Text'])
    # Encode 'Caption' in 'Image'
    if 'Image' in section and isinstance(section['Image'], list):
        for image in section['Image']:
            if isinstance(image, dict) and 'Caption' in image:
                image['Encode'] = encode_text(image['Caption'])
    # Encode 'Description' in 'Description'
    if 'Description' in section and isinstance(section['Description'], list):
        for desc in section['Description']:
            if isinstance(desc, dict) and 'Description' in desc:
                desc['Encode'] = encode_text(desc['Description'])

def encode_fields(entry):
    # Encode in 'Firsthand_Information' if present
    if 'Firsthand_Information' in entry:
        encode_nested_fields(entry['Firsthand_Information'])
    # Encode in 'Historical_Information' if present
    if 'Historical_Information' in entry:
        encode_nested_fields(entry['Historical_Information'])
    return entry

def main():
    # Read the entire files
    data1 = read_entire_json(INPUT_FILE_1)
    data2 = read_entire_json(INPUT_FILE_2)

    print(f"Loaded {INPUT_FILE_1}: type={type(data1)}, len={len(data1) if hasattr(data1, '__len__') else 'N/A'}")
    print(f"Loaded {INPUT_FILE_2}: type={type(data2)}, len={len(data2) if hasattr(data2, '__len__') else 'N/A'}")

    # Always combine as dicts, rewritten_thread.json (data2) takes precedence
    if isinstance(data1, dict) and isinstance(data2, dict):
        combined = data1.copy()
        combined.update(data2)
    else:
        raise ValueError('Input files must be dictionaries with report keys.')

    total = len(combined)
    batch_size = 100
    output_dict = {}

    for idx, (key, entry) in enumerate(combined.items(), 1):
        output_dict[key] = encode_fields(entry)
        if idx % batch_size == 0 or idx == total:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=2)
            print(f"Saved {idx} reports to {OUTPUT_FILE}")

if __name__ == '__main__':
    main() 