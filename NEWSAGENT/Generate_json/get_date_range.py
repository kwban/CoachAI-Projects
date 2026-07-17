import os
from datetime import datetime

def get_date_range(output_dir):
    dates = []
    june_july_count = 0
    for subdir in os.listdir(output_dir):
        article_path = os.path.join(output_dir, subdir, "article.txt")
        if os.path.isfile(article_path):
            with open(article_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Date:"):
                        date_str = line.strip().split("Date:")[1].strip()
                        try:
                            date = datetime.strptime(date_str, "%Y-%m-%d")
                            dates.append(date)
                            if date.month in (6, 7):
                                june_july_count += 1
                        except Exception as e:
                            print(f"Failed to parse date in {article_path}: {date_str}")
                        break
    if dates:
        print(f"{output_dir}: {min(dates).date()} to {max(dates).date()} ({len(dates)} articles)")
        print(f"  Articles in June and July: {june_july_count}")
    else:
        print(f"{output_dir}: No dates found.")

get_date_range("Crawl_data/BBC_output")
get_date_range("Crawl_data/APNews_output")