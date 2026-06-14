# Architecture

This document walks through the stages of [`notebooks/snomed-ner-project.ipynb`](notebooks/snomed-ner-project.ipynb),
the notebook used to build the custom `acebirim/snomed-ner-model`, and how its output
connects to the deployed [`gradio_app/app.py`](gradio_app/app.py).

## Compute environment

The pipeline was run on a **Kaggle notebook with a T4 GPU**. Fine-tuning Bio_ClinicalBERT
on 137K training examples for 3 epochs is computationally expensive — Kaggle was chosen
because it provides free GPU access (T4, ~16GB VRAM) with no local hardware requirements,
which was sufficient to complete training within Kaggle's session time limits while also
hosting the SNOMED CT and MedMentions datasets as notebook inputs/outputs.

## Pipeline overview

```
┌────────────────────┐   ┌──────────────────────┐
│ SNOMED CT RF2       │   │ MedMentions corpus    │
│ (descriptions file) │   │ (PubMed abstracts +   │
│                     │   │  UMLS annotations)    │
└─────────┬───────────┘   └───────────┬───────────┘
          │                            │
          ▼                            ▼
┌────────────────────┐   ┌──────────────────────┐
│ term → SNOMED code  │   │ Parse documents &      │
│ lookup dictionary   │   │ entity annotations     │
│ (~1.24M terms)      │   └───────────┬───────────┘
└─────────┬───────────┘               │
          │                            │
          └────────────┬───────────────┘
                        ▼
          ┌──────────────────────────────┐
          │ Match entities → SNOMED codes  │
          │ (~61% match rate)               │
          │ + build ±100-char context        │
          │   windows with BIO labels         │
          └───────────────┬──────────────┘
                          ▼
          ┌──────────────────────────────┐
          │ 171K BIO-tagged examples        │
          │ split 80/10/10                  │
          │ (train/val/test)                │
          └───────────────┬──────────────┘
                          ▼
          ┌──────────────────────────────┐
          │ Tokenize (word-aligned, -100   │
          │ for subword continuations)     │
          └───────────────┬──────────────┘
                          ▼
          ┌──────────────────────────────┐
          │ Fine-tune Bio_ClinicalBERT     │
          │ (3 epochs, lr 5e-5,            │
          │  class weights [1, 5, 5])      │
          └───────────────┬──────────────┘
                          ▼
          ┌──────────────────────────────┐
          │ Evaluate on validation set      │
          │ (per-label precision/recall/F1) │
          └───────────────┬──────────────┘
                          ▼
          ┌──────────────────────────────┐
          │ Upload to HF Hub:                │
          │ acebirim/snomed-ner-model        │
          └───────────────┬──────────────┘
                          ▼
          ┌──────────────────────────────┐
          │ gradio_app/app.py (HF Space)    │
          │ — dropdown: production vs       │
          │   custom model                  │
          └──────────────────────────────┘
```

## Stage-by-stage

### 1. Setup (cells 0–6)
GPU availability is checked, the `emilyalsentzer/Bio_ClinicalBERT` tokenizer is loaded,
and the label scheme is fixed for the whole notebook:

```python
label_list = ["O", "B-DISEASE", "I-DISEASE"]
```

### 2. Data acquisition (cells 7–9, 13–14)
- The SNOMED CT International RF2 release (~554MB) is downloaded and extracted.
- The MedMentions corpus (~6.6MB, gzipped) is downloaded and extracted.

### 3. SNOMED lookup construction (cells 10–12)
The RF2 descriptions file is filtered to **active, English-language preferred terms and
synonyms** (`typeId` 900000000000013009 / 900000000000003001), then reduced to a
`term → SNOMED concept ID` dictionary of ~1.24M unique lowercase terms. This dictionary
is the entity-to-code lookup used later for matching and (in the root app) for
SNOMED-code annotation.

### 4. Exploratory data analysis (cells 15–25)
Distributions are examined for both datasets independently: SNOMED description-type and
term-length distributions, and MedMentions document counts, semantic-type distribution,
and entity-text-length distribution. This stage doesn't feed the model directly — it
informs decisions made in stage 5 (e.g. the ±100-character context window, and the
expectation of a partial SNOMED match rate).

### 5. Entity matching & training example construction (cell 23)
For every MedMentions annotation, `match_entity_to_snomed()` attempts three lookups in
order: exact lowercase match, punctuation-stripped match, then first-word match against
the SNOMED lookup. **~61% of the 352K annotations match.** For each match, a context
window of 100 characters before and after the entity span is extracted from the source
abstract, split into tokens, and BIO-labelled (`B-DISEASE` for the first token of the
span, `I-DISEASE` for the rest, `O` elsewhere) — producing **171K training examples**.

### 6. Train/val/test split (cell 26)
An 80/10/10 split of the 171K examples gives 137,093 train / 17,136 validation / 17,138
test examples.

### 7. Tokenization & dataset preparation (cells 27–29)
Each example is tokenized with `is_split_into_words=True`; subword continuation tokens
are assigned label `-100` so they're ignored by the loss function. A `torch.utils.data.Dataset`
wraps the tokenized tensors for batching.

### 8. Class weighting (cell 30)
The label distribution is computed (`O` ≈ 95.5%, `B-DISEASE` ≈ 3.3%, `I-DISEASE` ≈ 1.2%).
To counter this imbalance, cross-entropy loss is weighted `[1.0, 5.0, 5.0]` for
`[O, B-DISEASE, I-DISEASE]`. (A more aggressive `[1, 20, 20]` weighting was tried and
rejected — it traded precision for recall too heavily, per the notebook's commented-out code.)

### 9. Model training (cells 31–32)
`Bio_ClinicalBERT` is loaded as `AutoModelForTokenClassification` with 3 output labels,
and fine-tuned for **3 epochs**, batch size 16, learning rate 5e-5, using the weighted
cross-entropy loss from stage 8.

### 10. Model persistence (cells 33–35)
The fine-tuned model, tokenizer, and `label_mappings.json` (`id2label`/`label2id`/`label_list`)
are saved to `./snomed_ner_model_1`.

### 11. Inference & post-processing (cells 36–38)
`predict_snomed_entities()` runs the trained model over raw text and reconstructs entity
spans from BIO predictions. A `STOPWORD_BLOCKLIST` filters common false positives
(e.g. "patient", "with", "was") that the model occasionally tags as `DISEASE`. This
filter is reused in `gradio_app/app.py`.

### 12. Evaluation (cell 39)
Precision, recall, and F1 are computed per label on the validation set using
`sklearn.metrics.precision_recall_fscore_support`. Results: B-DISEASE F1 = 0.91,
I-DISEASE F1 = 0.82, overall weighted F1 = 0.99 (see `README.md` for the full table).

### 13. Model upload (cell 41)
The saved model directory is pushed to the Hugging Face Hub at `acebirim/snomed-ner-model`
via `HfApi().upload_folder()`.

> **Security note**: this cell currently has a Hugging Face access token hardcoded as a
> string literal. Before sharing or publishing this notebook, revoke that token at
> huggingface.co/settings/tokens and replace it with an environment variable or
> Kaggle Secret.

## From notebook output to deployed app

`gradio_app/app.py` does not repeat this training pipeline — it only consumes its
output: the `acebirim/snomed-ner-model` artifact from stage 13, alongside the
off-the-shelf `ugaray96/biobert_ncbi_disease_ner` model. A dropdown lets the user choose
which model runs the same BIO-tagging → entity-span → confidence-score logic from
stage 11 (including the shared `STOPWORD_BLOCKLIST`), so the live demo directly reflects
the trade-offs documented in the notebook's evaluation (stage 12) and the
README's "Model Development & Benchmarking" section.
