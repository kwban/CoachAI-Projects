import json
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

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

def extract_text_from_article(article):
    texts = []
    
    # Extract title from Report_info
    if 'Report_info' in article and article['Report_info']:
        for info in article['Report_info']:
            if 'Title' in info:
                texts.append(info['Title'])
    
    # Extract text from Firsthand_Information
    if 'Firsthand_Information' in article:
        firsthand = article['Firsthand_Information']
        
        # Extract speaker text
        if 'Speaker' in firsthand:
            for speaker in firsthand['Speaker']:
                if 'Text' in speaker:
                    texts.append(speaker['Text'])
        
        # Extract description text
        if 'Description' in firsthand:
            for desc in firsthand['Description']:
                if 'Description' in desc:
                    texts.append(desc['Description'])
    
    # Extract text from Historical_Information
    if 'Historical_Information' in article:
        historical = article['Historical_Information']
        
        # Extract speaker text
        if 'Speaker' in historical:
            for speaker in historical['Speaker']:
                if 'Text' in speaker:
                    texts.append(speaker['Text'])
        
        # Extract description text
        if 'Description' in historical:
            for desc in historical['Description']:
                if 'Description' in desc:
                    texts.append(desc['Description'])
    
    # return ' '.join(texts)
    return ' '.join(str(t) for t in texts if t is not None)

def filter_english_articles(input_file, output_file):
    """
    Filter articles to keep only those written in English.
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total articles: {len(data)}")
    
    english_articles = {}
    non_english_count = 0
    language_stats = {}
    
    for article_id, article in data.items():
        # Extract text content for language detection
        text_content = extract_text_from_article(article)
        
        if is_english_text(text_content):
            english_articles[article_id] = article
        else:
            non_english_count += 1
            # Track detected languages for debugging
            try:
                lang = detect(text_content)
                language_stats[lang] = language_stats.get(lang, 0) + 1
            except:
                language_stats['unknown'] = language_stats.get('unknown', 0) + 1
            
        # Progress indicator
        if int(article_id) % 1000 == 0:
            print(f"Processed {article_id} articles...")
    
    print(f"English articles: {len(english_articles)}")
    print(f"Non-English articles: {non_english_count}")
    print(f"Language distribution of non-English articles:")
    for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count}")
    
    # Save English articles to new file
    print(f"Saving English articles to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(english_articles, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "_data_june_july.json"
    output_file = "english_articles_june_july.json"
    
    filter_english_articles(input_file, output_file) 