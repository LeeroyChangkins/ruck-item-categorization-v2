# Ruck Item Categorization — v2 Pipeline

Matches production items to a taxonomy of leaf categories, generates attributes per category via LLM, and uploads results to the database.

---

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials (DB connection, OpenAI API key).

---

## Running the Pipeline

```bash
python3 run_v2_pipeline.py
```

You'll be asked two questions upfront:
1. **Prod or Dev?** — selects the item source file and suffixes all outputs (`-prod` / `-dev`)
2. **Speed run or Normal?** *(dev only)* — speed run auto-categorizes everything end-to-end with no prompts

Then choose which step to start from (defaults to 1.1 for a full run).

---

## Pipeline Steps

| Step | Script | What it does |
|------|--------|--------------|
| **1.1** | `step-1/1_1_build_similar_title_groups.py` | Groups items with similar titles together |
| **1.2** | `step-1/1_2_interactive_similar_title_match.py` | You manually assign each group to a taxonomy leaf |
| **2.1** | `step-2-bigram-keyword-matching/2_1_generate_keywords.py` + bigram scripts | Extracts keyword frequencies; builds bigram→category mappings |
| **2.2** | `step-2-bigram-keyword-matching/2_2_match_items_to_bigrams.py` | Auto-matches items to categories via bigram rules |
| **2.3** | `step-2-bigram-keyword-matching/2_3_interactive_keyword_match.py` | You manually handle leftover bigram groups |
| **3** | `step-3/3_llm_match_unmatched.py` | LLM matches everything still unmatched |
| **4** | `step-4-dedupe-and-merge-matched-items/4_dedupe_and_summaries.py` | Deduplicates all results across steps 1–3 |
| **5** | `step-5-attribute-generation-and-unit-value-assignment/5_generate_attributes.py` | LLM generates product attributes per leaf category |
| **6** | `step-6/6_upload_to_db.py` | Uploads categories, item links, and attributes to the DB |

Step 6 requires an active SSM tunnel (`ruck-db-staging` or `ruck-db-prod`) in a separate terminal.

---

## Source Files

| File | Purpose |
|------|---------|
| `source-files/categories_v1.json` | The taxonomy tree (source of truth for leaf categories) |
| `source-files/prod-items-with-stores.json` | Production item catalog |
| `source-files/dev-items-with-stores.json` | Smaller dev item catalog for testing |

---

## Outputs

All generated files land in timestamped folders inside each step's `outputs/` directory (e.g. `step-1/outputs/20260401_120000-prod/`). These are git-ignored. Final consolidated outputs are written to `final-output/<timestamp>/` after step 5.

---

## Dev Speed Run

When you select **dev → speed run**, the pipeline:
- Auto-randomly assigns all interactive categorization steps (1.2, 2.3)
- Runs all automated steps (2.1, 2.2, 3, 4, 5) with default settings
- Asks once at the end if you want to upload to the staging database

Useful for quickly testing the full pipeline end-to-end.
