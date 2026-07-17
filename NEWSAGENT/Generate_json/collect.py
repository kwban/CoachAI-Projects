import os
import shutil
from datetime import datetime

def collect_june_july_articles(source_dirs, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    idx = 0
    for source in source_dirs:
        for subdir in sorted(os.listdir(source), key=lambda x: int(x)):
            article_path = os.path.join(source, subdir, "article.txt")
            if os.path.isfile(article_path):
                with open(article_path, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("Date:"):
                            date_str = line.strip().split("Date:")[1].strip()
                            try:
                                date = datetime.strptime(date_str, "%Y-%m-%d")
                                if date.month in (6, 7):
                                    new_subdir = os.path.join(dest_dir, str(idx))
                                    shutil.copytree(os.path.join(source, subdir), new_subdir)
                                    idx += 1
                            except Exception:
                                pass
                            break

if __name__ == "__main__":
    collect_june_july_articles(
        ["Crawl_data/BBC_output", "Crawl_data/APNews_output"],
        "Crawl_data/june_july_news_"
    )
    print("Done! All June and July articles are now in 'june_july_news' and renumbered.")
    #31097 bbc 15539 AP 15529