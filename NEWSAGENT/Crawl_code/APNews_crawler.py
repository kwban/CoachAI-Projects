import datetime as dt
from fundus import Crawler                             
from fundus import PublisherCollection
APNews = PublisherCollection.us.APNews


import os
import requests

OUTPUT_DIR = "Crawl_data/APNews_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def only_2025(extracted: dict) -> bool:
    """
    Fundus filters are *negative* predicates:
    return True  -> article is discarded
    return False -> article is kept
    """
    pub_date = extracted.get("publishing_date")
    return (pub_date is None) or (pub_date.year != 2025)

crawler = Crawler(APNews)

print("APNews Articles from 2025:")
print("=" * 80)

for idx, art in enumerate(crawler.crawl(
        max_articles=1,         
        timeout=60,              
        only_complete=only_2025
)):
    print("Attributes of art object:")
    for attr in dir(art):
        if not attr.startswith('__'):
            print(attr, "=", getattr(art, attr))
    break

for idx, art in enumerate(crawler.crawl(
        max_articles=None,
        timeout=60,
        only_complete=only_2025
)):
    if 'article' not in art.html.requested_url:
        continue

    article_dir = os.path.join(OUTPUT_DIR, str(idx))
    images_dir = os.path.join(article_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    text_file = os.path.join(article_dir, "article.txt")

    lines = []
    lines.append(f"Title: {art.title}")
    lines.append(f"Date: {art.publishing_date:%Y-%m-%d}")
    
    urls_file = os.path.join(article_dir, "urls.txt")
    url_lines = []
    url_lines.append(f"Article URL: {art.html.requested_url}")

    # Download images and save URLs/captions
    if hasattr(art, 'images') and art.images:
        for img_idx, img_obj in enumerate(art.images, 1):
            # Extract URL from Image object - try different attributes
            if hasattr(img_obj, 'url'):
                img_url = img_obj.url
            elif hasattr(img_obj, 'src'):
                img_url = img_obj.src
            else:
                img_str = str(img_obj)
                if "URL:" in img_str:
                    url_start = img_str.find("URL:") + 4
                    url_end = img_str.find("\n", url_start)
                    if url_end == -1:
                        url_end = len(img_str)
                    img_url = img_str[url_start:url_end].strip().strip("'")
                else:
                    img_url = img_str
            img_caption = art.meta.get('og:image:alt') or art.meta.get('twitter:image:alt') or "No caption"
            url_lines.append(f"Image_{img_idx} URL: {img_url}")
            lines.append(f"[image_{img_idx}] Caption: {img_caption}")
            # Download image
            img_ext = os.path.splitext(img_url)[-1].split('?')[0]
            if not img_ext or len(img_ext) > 5:
                img_ext = ".jpg"
            img_file = os.path.join(images_dir, f"image_{img_idx}{img_ext}")
            try:
                img_data = requests.get(img_url, timeout=10).content
                with open(img_file, 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                lines.append(f"[image_{img_idx}] Failed to download: {img_url} ({e})")
            break

    # Save article body preview
    if art.body:
        body_text = ' '.join(str(paragraph) for paragraph in art.body)
        lines.append("")
        lines.append("Body:")
        lines.append(body_text)
    # Save article.txt (text/captions only)
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    # Save urls.txt (article and image URLs)
    with open(urls_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(url_lines))
    print(f"Saved article {idx} to {article_dir}")
