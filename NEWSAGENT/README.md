# NEWSAGENT

Official implementation and benchmark resources for:

> **Benchmarking Agentic Newswriting via Journalistic Workflows**  
> Findings of the Association for Computational Linguistics: ACL 2026

- Paper: https://aclanthology.org/2026.findings-acl.1816/
- Dataset: https://osf.io/nq83p/overview?view_only=c027289281964472aed2b8122ca46acc

## Overview

**NEWSAGENT** is a benchmark for evaluating whether large language model agents can carry out a realistic journalistic workflow rather than generate an article in a single pass.

Given a news title, release date, writing instruction, and partial firsthand materials, an agent must:

1. identify a narrative perspective;
2. search for relevant historical context available before the article's release date;
3. iteratively insert or remove evidence while revising a draft; and
4. rephrase the completed draft into a final news article.

The benchmark contains **6,237 human-verified examples** derived from real-world news. It is designed to evaluate both the agent's intermediate search and editing decisions and the quality of the final article.

## Benchmark Tasks

NEWSAGENT models newswriting as an iterative **search–edit–rephrase** workflow.

### Time-aware search

The agent issues keyword-based queries to retrieve relevant historical information. Retrieved material is constrained by the article's release date, preventing the use of future information.

### Iterative editing

The agent incrementally inserts or removes contextual evidence as the draft develops.

### Final rephrasing

After the search and editing process is complete, the agent rewrites the draft as a coherent news article.

## Evaluation

The benchmark supports two complementary levels of evaluation:

- **Function-wise evaluation:** measures how closely an agent's retrieval and editing decisions align with evidence selected in human-written reference articles.
- **End-to-end evaluation:** compares completed articles across six journalistic dimensions:
  - factuality;
  - logical consistency;
  - importance;
  - readability;
  - objectivity; and
  - journalistic style.

Function-wise scores diagnose evidence-selection behavior. They should not be interpreted as a complete measure of article quality.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Then activate the environment defined in `environment.yml`.

## Dataset

Download the dataset from OSF:

https://osf.io/nq83p/overview?view_only=c027289281964472aed2b8122ca46acc

To prepare it:

1. Download `report_dataset.zip`.
2. Extract `report_dataset.json`.
3. Place `report_dataset.json` in the repository root.

Expected layout:

```text
NEWSAGENT/
├── report_dataset.json
├── environment.yml
├── react/
├── evaluation/
└── ...
```

If you only want to run the agent workflows and evaluation scripts, download the prepared dataset and skip the data-collection pipeline.

## Running the Benchmark

### Agent workflows

```bash
# Run the one-step ReAct workflow.
# The current script processes articles 0–100 by default.
python react/react_1_step.py

# Run the two-step ReAct workflow.
python react/react_2_step.py

# Run the rule-based baseline.
python react/rule_base.py
```

### Evaluation

```bash
# Run pairwise LLM evaluation.
# Outputs are written under evaluation/LLM_eval.
python evaluation/LLM_evaluation.py

# Evaluate search and retrieval for the one-step workflow.
# Outputs are written under evaluation/1_step_eval.
python evaluation/1_step_evaluation.py

# Evaluate search and retrieval for the two-step workflow.
# Outputs are written under evaluation/2_step_eval.
python evaluation/2_step_evaluation.py

# Analyze raw LLM-evaluation outputs.
# Outputs are written under evaluation/LLM_eval/analysis.
python evaluation/anal_raw.py
```

## Rebuilding the Dataset

The following steps are only required when reconstructing the benchmark from the original news sources.

### 1. Crawl news articles

```bash
# Crawl AP News articles and create Crawl_data/APNews_output/.
python Crawl_code/APNews_crawler.py

# Crawl BBC News articles and create Crawl_data/BBC_output/.
python Crawl_code/bbc_crawler.py

# Collect June and July articles from both sources.
python Generate_json/collect.py

# Check publication-date ranges and create Crawl_data/june_july_news/.
python Generate_json/get_date_range.py
```

### 2. Extract structured information

```bash
# Extract report metadata, firsthand information, and historical information.
# The script creates _data_june_july.json.
python Generate_json/request.py
```

### 3. Filter, combine, and index the data

```bash
# Retain English-language articles.
python Statistic/filter_english_news.py

# Optionally rewrite historical information for parallel data.
python Statistic/rewrite_historical_info.py

# Combine and tokenize the data to create report_dataset.json.
python Statistic/tokenize_and_combine_json.py

# Generate historical_search_results.json for evaluation.
python Statistic/semantic_search_historical_parallel.py
```

## Dataset Format

Each example contains task metadata and structured firsthand information. A simplified example is shown below:

```json
{
'0':
{
    "Title": "Harvey Weinstein is back on trial in New York. Jury selection begins Tuesday.",
    "Firsthand_Information": (
        "Firsthand_Information": {
            "Speaker": [
                {
                    "Speaker": "Gloria Allred",
                    "Text": "It’s painful, to go through the process again about a traumatic event.",
                    "Citation": {
                        "Paragraph": 5,
                        "StartChar": 151,
                        "EndChar": 207
                    },
                    "Encdoe":[]
                },
                {
                    "Speaker": "Lindsay Goldbrum",
                    "Text": "She is one of the bravest, strongest women that I have ever had the pleasure of knowing.",
                    "Citation": {
                        "Paragraph": 9,
                        "StartChar": 40,
                        "EndChar": 98
                    },
                    "Encdoe":[]
                }
            ],
            "Description": [
                {
                    "Description": "Prosecutors have also added a new accuser in the retrial.",
                    "Citation": {
                        "Paragraph": 9,
                        "StartChar": 0,
                        "EndChar": 76
                    },
                    "Encdoe":[]
                }
            ],
            "Image": []
        },
    )
}
}
```

## Citation

```bibtex
@inproceedings{chien-etal-2026-benchmarking,
    title = "Benchmarking Agentic Newswriting via Journalistic Workflows",
    author = "Chien, Yen-Che  and
      Wang, Kuang-Da  and
      Wang, Wei-Yao  and
      Peng, Wen-Chih",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1816/",
    doi = "10.18653/v1/2026.findings-acl.1816",
    pages = "36450--36463",
    ISBN = "979-8-89176-395-1"
}
```
