# Categorization v2 Pipeline (Raw Categories + Items)

This doc explains the end-to-end workflow we implemented in `temp/categorization/v2`, starting from:

- a construction taxonomy (`categories_v1.json`)
- production items (`raw-prod-items-non-deleted.json`)
- derived keyword signals used to build high-precision bigram rules.

The goal is to support a “category navigator” where:

- top-level navigation is **format / product type first**
- finer details (species, material family, color, etc.) can be handled as attributes/filters later.

---

## 0) Files (current locations)

### Pipeline scripts and outputs (v2 layout)

| Folder | Scripts | Outputs |
|--------|---------|---------|
| `step-1/` | `1_1_build_similar_title_groups.py`, `1_2_interactive_similar_title_match.py` | `step-1/outputs/<run_id>/` (groups JSON, manuals, `unmatched_after_step1.json`) |
| `step-2/` | `2_1_generate_keywords.py`, `2_1_generate_bigrams_taxonomy.py`, `2_1_generate_bigrams_openai.py`, `2_2_match_items_to_bigrams.py`, `2_3_interactive_keyword_match.py` | `step-2/outputs/` (keyword + bigram + match + split + manual JSON), `step-2/checkpoints/` |
| `step-3/` | `3_llm_match_unmatched.py` | `step-3/outputs/`, `step-3/checkpoints/` |
| `step-4/` | `4_dedupe_and_summaries.py` | `step-4/outputs/<run_id>/` (deduped + summaries) |

Orchestrator: `run_v2_pipeline.py` (optional `--start-step`). Shared helpers: `pipeline_paths.py`, `taxonomy_cascade.py`.

### Inputs
- `source-files/categories_v1.json`
  - The category taxonomy (tree)
  - We treat these slugs as the source of truth for where things belong.

- `source-files/raw-prod-items-non-deleted.json`
  - Production items (title/subtitle, etc.)
  - For the bigram-to-category mapping step, we *do not* use item category fields.

- `1.0-title_subtitle_keyword_frequencies.json`
  - Curated “master list” of common words extracted from item `title` and `subtitle`.
  - This is the only thing used to generate the bigram candidates.

### Outputs
- `1.1-bigram_categories_mapping.json`
  - Bigram rules → “T1 parent category slug” + confidence.

- `1.2-bigram_sorted_items.json`
  - Each item is scanned for which bigram rules match its title/subtitle.
  - Output includes a ranked `categories: []` array per item and a separate `unmatched_items: []` section.

---

## Constants Used (Current Settings)

- Keyword extraction (`1.0-title_subtitle_keyword_frequencies.json`)
  - Tokenization: letters-only runs (A–Z), lowercase
  - Min token length: `> 3` (so 4+)
  - No numbers: tokens containing digits are excluded
  - Word frequency counting: `1` per item per word (dedup inside an item)
  - Min items per word: `20`

- Bigram mapping (`1.1-bigram_categories_mapping.json`)
  - Bigrams are generated only from the curated `title` word list (title bigrams) and only from the curated `subtitle` word list (subtitle bigrams)
  - Bigram match: order-independent (the two words can appear in any order)
  - Mapping uses taxonomy text from `categories_v1.json` only (slug + display_name tokens)
  - Word-to-parent association heuristic:
    - exact token match counts as a hit
    - otherwise, lexical similarity is computed and compared to `>= 0.58`
    - similarity comparisons are gated to taxonomy tokens with the same first 3 letters
  - Bigram → T1 selection rule:
    - the two words must share exactly `1` common T1 parent category to be considered a “best parent”
  - Confidence tiers:
    - both words map uniquely → `1.0`
    - one word maps uniquely → `0.85`
    - fallback shared mapping → `0.7`
  - Output filter for 1.1:
    - keep only entries with `confidence >= 0.85`

- Item scan + ranking (`1.2-bigram_sorted_items.json`)
  - Matching text: item `title` vs `title_bigrams` and item `subtitle` vs `subtitle_bigrams` (we do not mix title words into subtitle rules)
  - Tokenization for matching: letters-only runs; token comparisons are case-insensitive
  - Bigram trigger: item side matches if BOTH bigram words are present anywhere in that side (not necessarily adjacent)
  - Multiple bigrams per category:
    - an item can list the same category multiple times via different triggering bigrams
    - within a category, triggering bigrams are sorted by strongest `confidence` first
    - duplicate triggering bigrams are de-duped per category
  - Category ranking per item:
    - sort categories by `matched_bigram_count` (desc)
    - then by `max_confidence` (desc)
    - then by `category_slug` (asc)
  - Output ordering:
    - matched items sorted by `total_triggered_bigrams` (desc), then `max_confidence` (desc)
  - Unmatched items:
    - if an item triggers zero categories, it goes into `unmatched_items` (with `id`, `title`, `subtitle`)

## 1) Taxonomy cleanup (what we changed in `categories_v1.json`)

These edits were made to make the taxonomy consistent with “format/product-type first” logic and to remove inline note metadata.

### 1.1 Flatten some “species/stone identity” nodes
- **Natural stone pavers**
  - Instead of having subcategories split as `bluestone_flagstone`, `travertine_pavers`, `limestone_pavers`, etc., we keep:
    - `concrete_pavers` as its own leaf
    - `stone_pavers` as the natural-stone paver leaf
  - Stone type / cut becomes a filter-like concept (attributes later), not the navigation tree.

- **Structural timbers**
  - Instead of species subcategories under `heavy_timbers`, we keep `heavy_timbers` as the leaf and represent species as an attribute-level concept later.

- **Slabs & live edge**
  - Instead of species subcategories, we keep `slabs_and_live_edge` as a leaf and treat species as an attribute-level facet later.

### 1.2 Remove `"notes"` fields everywhere in the taxonomy
- The JSON tree was scrubbed so that no `"notes"` keys remain (including `notes: null`).

### 1.3 Ensure category slugs are unique across the entire tree
- We found duplicate slug strings used in multiple places (e.g. `finish_plaster`, `coarse_thread_screws`, `deck_gate_kits`, `square_tubing`, `rectangular_tubing`).
- We resolved collisions by renaming the conflicting nodes so every slug string is globally unique.

---

## 2) Keyword extraction (for `1.0-title_subtitle_keyword_frequencies.json`)

We produced a curated list of high-frequency words from production item text.

### 2.1 Rules used to build word lists
For each item, we tokenize:

- `title` (for the “title” list)
- `subtitle` (for the “subtitle” list)

Token rules:
- letters-only tokens (A–Z)
- lowercase
- token length must be **> 3** characters
- tokens must contain **no numbers**

Counting:
- each word counts **once per item** (so repeated mentions inside the same title don’t inflate counts)
- we keep words appearing in **at least 20 items**

### 2.2 Output format (structure)
`1.0-title_subtitle_keyword_frequencies.json` contains:

- `title`: `[{ "word": <string>, "item_count": <int> }, ...]`
- `subtitle`: `[{ "word": <string>, "item_count": <int> }, ...]`

---

## 3) Bigram-to-category mapping (creates `1.1-bigram_categories_mapping.json`)

This step **must not** use:
- `items[].category`
- `items[].subcategory`
- item title/subtitle text for deriving bigrams

It uses only:
- `1.0-title_subtitle_keyword_frequencies.json` word lists
- `source-files/categories_v1.json` taxonomy text

### 3.1 Define T0 and T1
- **T0** = top-level branches in the taxonomy:
  - `materials`, `tools_and_gear`, `services`
- **T1** = the direct children under each T0 node (their slugs).

Example:
- If you consider `materials/...` as T0, then `materials` children are T1.

### 3.2 Bigram candidate generation
We generate bigrams from the curated word list:

- Title bigrams:
  - formed from the `title` word list only
- Subtitle bigrams:
  - formed from the `subtitle` word list only

Each bigram is represented as:
```json
{ "bigram": ["wordA", "wordB"] }
```

Matching is **order-independent** (the words can appear in any order later).

### 3.3 Mapping a word to a T1 parent category (taxonomy-based “semantic-ish” logic)
For each T1 parent category:
1. We build a token set derived from that category’s:
   - `slug`
   - `display_name`
2. A word maps to a T1 parent if:
   - exact token match exists in that parent’s token set; OR
   - lexical similarity is strong enough against some taxonomy token.

Similarity:
- Implemented with `difflib.SequenceMatcher` (built-in Python)
- A fast gating heuristic is used:
  - compare only taxonomy tokens sharing the same first 3 letters

Threshold (implemented):
- similarity >= `0.58` adds the parent to the word’s candidate parent set.

### 3.4 Mapping a bigram to a T1 parent
For bigram `[A, B]`:
- compute `parents(A)` and `parents(B)`
- if `parents(A) ∩ parents(B)` contains exactly **one** T1 parent,
  - that parent is the “best” T1 category.

Confidence assignment:
- if both words map uniquely → `confidence = 1.0`
- if one word maps uniquely → `confidence = 0.85`
- otherwise → `confidence = 0.7`

Filtering for the 1.1 output:
- we keep only `confidence >= 0.85`

### 3.5 Output format
`1.1-bigram_categories_mapping.json` contains:
```json
{
  "version": "1.1",
  "min_confidence_kept": 0.85,
  "title_bigrams": [
    {
      "bigram": ["steel", "tube"],
      "suggested_parent_category_slug": "metals_and_metal_fabrication",
      "confidence": 1.0,
      "source": "taxonomy_based"
    }
  ],
  "subtitle_bigrams": [ ... ]
}
```

---

## 4) Scan items for matching bigrams (creates `1.2-bigram_sorted_items.json`)

This step reads:
- item list: `source-files/raw-prod-items-non-deleted.json`
- bigram rules: `1.1-bigram_categories_mapping.json`

### 4.1 IMPORTANT: We do not reference item category fields
The match uses only:
- `item.title` vs `title_bigrams`
- `item.subtitle` vs `subtitle_bigrams`

Item `.category` and `.subcategory` are ignored entirely.

### 4.2 Tokenization for matching
We tokenize each side text into:
- letters-only tokens (A–Z)
- preserves the original casing for output purposes

Matching is:
- case-insensitive for comparisons
- order-independent: bigram words can be anywhere in the text, not adjacent.

### 4.3 Triggering a category
For each item side (title/subtitle), for each bigram rule:
- if **both** bigram words are present in the side token set,
  - the category is triggered by that bigram.

Multiple bigrams can trigger the same category for one item.

### 4.4 Output ranking rules per item
For a given item:
- For each category_slug, we gather all triggering bigrams.
- Categories are sorted by:
  1. `matched_bigram_count` (descending)
  2. `max_confidence` among triggers (descending)
  3. category_slug (ascending tie-break)

Each category entry includes:
- `triggering_bigrams`: sorted strongest-to-weakest within that category

### 4.5 Unmatched items
Items that match **zero** categories are placed into:
- `unmatched_items: []`

Each entry includes:
- `id`, `title`, `subtitle`

---

## 5) JSON output schemas (quick reference)

### `1.1-bigram_categories_mapping.json`
- `title_bigrams`: array of `{ bigram: [w1,w2], suggested_parent_category_slug, confidence, source }`
- `subtitle_bigrams`: array of same structure

### `1.2-bigram_sorted_items.json`
Top level:
- `matched_items`: array of items with categories
- `unmatched_items`: array of items that triggered none (older outputs used `unmatched_words`)

Matched item fields:
- `id`
- `title`
- `subtitle`
- `total_triggered_bigrams`
- `max_confidence`
- `categories`: ordered array of
  - `category_slug`
  - `matched_bigram_count`
  - `triggering_bigrams`: list of triggering bigrams with casing + confidence + source
  - `max_confidence`

---

## 6) Notes for generating future code

If you ask an AI agent to regenerate the pipeline:

1. Treat the keyword extraction output (`1.0...json`) as the canonical word vocabulary.
2. Build bigrams from that vocabulary only (title and subtitle separately).
3. Build bigram→T1 mapping from taxonomy text only (slug + display_name).
4. Apply bigram rules to items by checking word presence in title/subtitle tokens (no adjacency required).
5. Rank categories by count of triggering bigrams and then confidence.

This ordering keeps the pipeline deterministic and avoids “sneaking in” item.category signals.

