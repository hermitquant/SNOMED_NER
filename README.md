
# SNOMED NER — Clinical Disease Entity Extractor

A Named Entity Recognition model for identifying disease entities in clinical free-text,
deployed as part of an NHS/OneLondon Secure Data Environment capstone project.

## Models

### Production Model (Deployed)
- **Hub**: [`ugaray96/biobert_ncbi_disease_ner`](https://huggingface.co/ugaray96/biobert_ncbi_disease_ner)
- **Base model**: BioBERT
- **Task**: Token classification (NER) — `O`, `B-DISEASE`, `I-DISEASE`
- **Training data**: NCBI Disease dataset (793 PubMed abstracts, 6,892 disease mentions)

### Custom Trained Model (Benchmarked)
- **Hub**: [`acebirim/snomed-ner-model`](https://huggingface.co/acebirim/snomed-ner-model)
- **Base model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Task**: Token classification (NER) — `O`, `B-DISEASE`, `I-DISEASE`
- **Training data**: MedMentions (132K PubMed abstracts with UMLS annotations)
- **Training**: 3 epochs, learning rate 5e-5, class weights [1.0, 5.0, 5.0]

## How it works

Paste any clinical free-text into the form. The model identifies disease entity spans
using BIO tagging and highlights them inline. Results are also shown as a summary table.

## Model Development & Benchmarking

A custom NER model was trained from scratch on the MedMentions corpus using
Bio_ClinicalBERT as the base encoder. After training and evaluation, the custom model
was benchmarked against `ugaray96/biobert_ncbi_disease_ner`, a BioBERT model
fine-tuned on the NCBI Disease dataset.

The benchmarking revealed a significant performance gap in favour of the NCBI-trained
model, which was subsequently selected as the production deployment. This finding
illustrates a key principle in applied ML: **a smaller, cleaner, well-annotated dataset
consistently outperforms a larger but noisier one**. The NCBI Disease dataset contains
6,892 carefully curated disease mentions across 793 abstracts, while the custom model's
effective training set was reduced to ~61% of MedMentions due to SNOMED CT filtering,
and further degraded by short context windows and inconsistent span boundaries in the
source annotations.

## Known Limitations of the Custom Trained Model

### 1. Terminology Style Sensitivity
The model was trained on PubMed abstracts which use formal clinical language. It performs
poorly on lay terms — for example it recognises `hypertension` but not `high blood pressure`.

### 2. Multi-token Entity Truncation
The model frequently tags only the first token of a multi-word disease name rather than
the complete span. For example, `lupus erythematosus` is detected as `lupus` only.
This is attributable to inconsistent span boundaries in the MedMentions training annotations
and the short context windows used during training data construction.

### 3. Class Imbalance
Despite applying class weights of 5x for disease labels, the model remains conservative
and biases toward predicting `O`. High weights (20x) introduced false positives on
common tokens; lower weights (5x) improved precision but at the cost of recall.

### 4. Context Sensitivity
Model performance is sensitive to sentence structure. The same disease term may be
detected in one sentence but missed in another depending on the broader context window.

### 5. SNOMED Coverage Gap
Only 61% of MedMentions annotations could be mapped to a SNOMED CT term via the NHS
TRUD RF2 release file. The model was trained exclusively on this 61% subset, limiting
coverage of less common or recently introduced disease terms.

### 6. Single Entity Type
The model only distinguishes `DISEASE` vs non-entity. A production clinical NER system
would typically include `PROCEDURE`, `MEDICATION`, `ANATOMY`, and `FINDING` entity types.
The absence of these categories causes misclassification of procedural and contextual terms.

## Key Finding: Data Quality vs Data Quantity

> *The NCBI-trained model (6,892 entities, clean annotations) significantly outperforms
> the custom MedMentions-trained model (132K examples, filtered and noisy). This
> demonstrates that dataset quality and annotation consistency are more important than
> raw training set size for clinical NER tasks.*

## Future Work

- Retrain custom model on full MedMentions without SNOMED filtering and with full sentence context windows
- Expand label schema to include PROCEDURE, MEDICATION, ANATOMY entity types
- Integrate SNOMED CT terminology lookup to return concept codes alongside entity spans
- Evaluate on a held-out clinical corpus distinct from PubMed abstracts
- Implement model drift monitoring using Evidently AI to track entity distribution shifts over time

*Built as part of an NHS/OneLondon Secure Data Environment capstone project.*

## Confidence Scoring

Each detected entity is assigned a **confidence score** — the average softmax probability
across all tokens that make up the entity span. This is surfaced directly in the output
table alongside the entity text and type.

### Interpreting Confidence Scores

| Score | Interpretation |
|-------|----------------|
| ≥ 90% | High confidence — entity is well-represented in training data |
| 50–90% | Moderate confidence — entity recognised but with some uncertainty |
| < 50% | Low confidence — the model is not plurality-confident in its DISEASE prediction; term may be ambiguous, rare, or out-of-distribution |

### Why 50% as the Threshold?

In a 3-class classification problem (O, B-DISEASE, I-DISEASE), **50% is the natural
decision boundary** — it means the model assigns more probability mass to DISEASE than
to any other single class. Below 50%, another class has higher probability, meaning the
prediction is not even plurality-confident. This makes 50% statistically more defensible
than an arbitrary higher threshold.

In a production system this threshold would be further calibrated empirically — by plotting
a reliability diagram comparing predicted confidence against actual accuracy on a held-out
set, and applying temperature scaling to correct for the overconfidence typical of
transformer models.

### MLOps Relevance

Confidence scores are a lightweight but powerful production monitoring signal. In a
deployed clinical NLP pipeline, systematically low confidence scores across incoming
text are an early indicator of **model drift** — the real-world data distribution is
diverging from the training distribution. This could be caused by:

- New disease terminology entering clinical use (e.g. newly classified conditions)
- Changes in documentation style across clinical settings
- Seasonal variation in disease prevalence affecting the types of terms submitted

Rather than waiting for downstream data quality failures, confidence score monitoring
allows teams to proactively identify when a model requires retraining or recalibration.
This is the same principle underpinning tools like Evidently AI, implemented directly
at the inference layer.

## Model Drift Monitoring

A clinical NER model deployed in an NHS setting is particularly vulnerable to drift: SNOMED CT
releases new terminology twice a year, clinical documentation styles vary across settings (GP
letters vs. discharge summaries vs. PubMed abstracts), and disease prevalence shifts seasonally.
The confidence scores already returned by this app are a lightweight first signal for this — a
sustained drop in mean confidence on incoming text suggests the input distribution has shifted
away from the training data. A production deployment would log these scores over time and compare
them against the validation baseline using a tool such as [Evidently AI](https://www.evidentlyai.com/)
(see Future Work).

## Custom Model Evaluation Metrics

The custom trained model (`acebirim/snomed-ner-model`) was evaluated on a held-out
test set drawn from MedMentions (10% of 132K examples). Results are as follows:

### Per-Label Performance

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| O | 0.9984 | 0.9936 | 0.9960 | 498,064 |
| B-DISEASE | 0.8707 | 0.9508 | 0.9090 | 17,136 |
| I-DISEASE | 0.7730 | 0.8743 | 0.8205 | 6,260 |

### Summary Metrics

| Metric | Value |
|--------|-------|
| Overall accuracy | 99% |
| Weighted F1 | 0.9910 |
| Disease entities F1 (micro-avg) | 0.8850 |
| Disease entities Precision | 0.8439 |
| Disease entities Recall | 0.9303 |

### Interpretation

The model achieves **F1 of 0.91 for B-DISEASE** and **0.82 for I-DISEASE** on the
MedMentions test set, which are strong results for a clinical NER task on this corpus.

The high recall (0.95 for B-DISEASE) indicates the model is effective at finding disease
entity starts, while the lower precision for I-DISEASE (0.77) reflects the multi-token
span boundary challenge — the model sometimes extends entity spans too far or truncates them.

The gap between strong test set metrics and weaker real-world performance on the live
demo is explained by **distribution shift** — the model was evaluated on MedMentions
(PubMed abstracts) but the demo uses manually constructed clinical sentences with different
linguistic patterns. This is precisely the scenario that Evidently AI drift monitoring
is designed to detect.
