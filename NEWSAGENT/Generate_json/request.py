import json
import os
from openai import OpenAI
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

data_folder = 'Crawl_data/june_july_news' 
save_file = "_data_june_july.json"

def is_english_text(text):
    """
    Check if the given text is in English.
    Returns True if English, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False
    
    # Clean the text - remove special characters and extra whitespace
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # If text is too short, skip it
    if len(cleaned_text) < 10:
        return False
    
    try:
        # Detect language
        lang = detect(cleaned_text)
        return lang == 'en'
    except LangDetectException:
        # If detection fails, assume it's not English
        return False

def extract_text_from_article(article_path):
    """
    Read the article.txt and return its content for language detection.
    """
    try:
        with open(article_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def ask_gpt_and_get_message(model='gpt-4o', path='./', mode=None, id = 0):
    return_content = get_user_content(id)
    count = 0
    message = get_message(return_content, mode)
    clean_message = get_message(return_content, mode)
    while(1):
        count += 1
        if count > 5:
            return
        response_json = request_message(message, model, path)
        message = clean_message

        if response_json == 'error':
            return
        temp_origin = re.sub(r'[^a-zA-Z0-9]', '', return_content).lower()

        all_is_in = 1

        # Check for 'Speaker' in 'Firsthand_Information'
        for i in response_json["Firsthand_Information"]['Speaker']:
            temp = re.sub(r'[^a-zA-Z0-9]', '', i['Text']).lower()
            if temp not in temp_origin:
                message.append({"role": "assistant","content": "The text you found isn't exactly from the news article verbatim:" + i['Text']})
                all_is_in = 0


        # Check for 'Description' in 'Firsthand_Information'
        for i in response_json["Firsthand_Information"]['Description']:
            temp = re.sub(r'[^a-zA-Z0-9]', '', i['Description']).lower()
            if temp not in temp_origin:
                message.append({"role": "assistant","content": "The text you found isn't exactly from the news article verbatim:" + i['Description']})
                all_is_in = 0

        # Check for 'Image' in 'Firsthand_Information'
        for i in response_json["Firsthand_Information"]['Image']:
            temp = re.sub(r'[^a-zA-Z0-9]', '', i['Caption']).lower()
            if temp not in temp_origin:
                message.append({"role": "assistant","content": "The text you found isn't exactly from the news article verbatim:" + i['Caption']})
                all_is_in = 0

        # Repeat the checks for 'Historical_Information'
        # Check for 'Speaker' in 'Historical_Information'
        for i in response_json["Historical_Information"]['Speaker']:
            temp = re.sub(r'[^a-zA-Z0-9]', '', i['Text']).lower()
            if temp not in temp_origin:
                message.append({"role": "assistant","content": "The text you found isn't exactly from the news article verbatim:" + i['Text']})
                all_is_in = 0

        # Check for 'Description' in 'Historical_Information'
        for i in response_json["Historical_Information"]['Description']:
            temp = re.sub(r'[^a-zA-Z0-9]', '', i['Description']).lower()
            if temp not in temp_origin:
                message.append({"role": "assistant","content": "The text you found isn't exactly from the news article verbatim:" + i['Description']})
                all_is_in = 0

        # Check for 'Image' in 'Historical_Information'
        for i in response_json["Historical_Information"]['Image']:
            temp = re.sub(r'[^a-zA-Z0-9]', '', i['Caption']).lower()
            if temp not in temp_origin:
                message.append({"role": "assistant","content": "The text you found isn't exactly from the news article verbatim:" + i['Caption']})
                all_is_in = 0

        if all_is_in == 0:
            print('something goes wrong')
        if all_is_in == 1 and response_json != 'error':
            break

    first_data = {
        f"{id}":
        {
            "Report_info": response_json.get("Report_info", []),
            "Firsthand_Information": response_json.get("Firsthand_Information", []),
            "Historical_Information": response_json.get("Historical_Information", [])
        }
    }

    # Save the data into separate JSON files
    append_firsthand_data(first_data , os.path.join(path,f"{save_file}"))

    return response_json

def get_message(user_content=' ', mode=None):
    with open("Generate_json/Instruction.txt", "r", encoding="utf-8") as file:
        role_content = file.read()

    message = [
        {"role": "system", "content": role_content},
        {"role": "user", "content": user_content}
    ]

    return message

def get_user_content(id = 0):
    with open(f"{data_folder}/{id}/article.txt", "r", encoding="utf-8") as file:
        user_content = file.read()

    return user_content

def request_message(message, model_='gpt-4o-mini', path='./'):
    # api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    error_count = 0

    while(1):
        error_count += 1
        if error_count > 5:
            return 'error'
        completion = client.chat.completions.create(
            model=model_,
            messages=message
        )

        response_content = completion.choices[0].message.content
        # print(response_content)

        try:
            response_json = json.loads(response_content)  # Convert text to dictionary
            break
        except json.JSONDecodeError:
            print("Error: GPT response is not valid JSON. Saving as raw text.")
    return response_json

def append_firsthand_data(new_data, file_path):
    """Helper function to append historical data to an existing JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            existing_data = json.load(json_file)
        
        existing_data.update(new_data)
        
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4, ensure_ascii=False)
    else:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(new_data, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    for i in range(0,10):
        article_path = f"{data_folder}/{i}/article.txt"
        text = extract_text_from_article(article_path)
        if is_english_text(text):
            ask_gpt_and_get_message('gpt-4.1', './', None, i)
            print(f'{i} is finished!')
        else:
            print(f'{i} skipped (not English)')

