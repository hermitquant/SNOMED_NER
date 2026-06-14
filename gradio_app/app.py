import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
import warnings
warnings.filterwarnings("ignore")

MODEL_OPTIONS = {
    "Production — BioBERT (NCBI Disease)": "ugaray96/biobert_ncbi_disease_ner",
    "Custom — Bio_ClinicalBERT (MedMentions / SNOMED)": "acebirim/snomed-ner-model",
}
DEFAULT_MODEL_LABEL = "Production — BioBERT (NCBI Disease)"

LABEL_LIST = ["O", "B-DISEASE", "I-DISEASE"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loaded models are cached here so switching back to a model already
# used in this session doesn't require reloading it from the Hub.
loaded_models = {}


def get_model(model_label: str):
    model_id = MODEL_OPTIONS[model_label]
    if model_id not in loaded_models:
        print(f"Loading model from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        model.eval()
        model.to(device)
        loaded_models[model_id] = (tokenizer, model)
        print(f"Model loaded on {device}")
    return loaded_models[model_id]


# Pre-load the default model so the first request isn't slow
get_model(DEFAULT_MODEL_LABEL)

STOPWORD_BLOCKLIST = {
    'patient', 'subject', 'presents', 'with', 'of', 'the', 'a', 'an', 'and',
    'or', 'but', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'as', 'was',
    'were', 'is', 'are', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'diagnosed', 'biopsy', 'admitted', 'he', 'history', 'referred', 'presented',
    'she', 'staging', 'systemic', 'acute', 'chronic', 'severe', 'bilateral',
    'complications', 'complication', 'imaging', 'progressive'
}


def predict(text: str, model_label: str):
    if not text.strip():
        return [], "No text provided."

    tokenizer, model = get_model(model_label)

    tokens = text.split()
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in tokenized.items()})
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    word_ids = tokenized.word_ids()
    predicted_labels = []
    predicted_probs = []
    previous_word_idx = None
    for word_idx, pred_id, prob in zip(word_ids, predictions, probs):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            predicted_labels.append(ID2LABEL[pred_id.item()])
            predicted_probs.append(prob[pred_id.item()].item())
        previous_word_idx = word_idx

    entities = []
    current_entity_tokens = []
    current_label = None
    current_probs = []

    for token, label, prob in zip(tokens, predicted_labels, predicted_probs):
        if label.startswith("B-"):
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                avg_conf = sum(current_probs) / len(current_probs)
                if entity_text.lower() not in STOPWORD_BLOCKLIST:
                    entities.append({"text": entity_text, "label": current_label, "confidence": avg_conf})
            current_entity_tokens = [token]
            current_label = label.replace("B-", "")
            current_probs = [prob]
        elif label.startswith("I-") and current_entity_tokens:
            current_entity_tokens.append(token)
            current_probs.append(prob)
        else:
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                avg_conf = sum(current_probs) / len(current_probs)
                if entity_text.lower() not in STOPWORD_BLOCKLIST:
                    entities.append({"text": entity_text, "label": current_label, "confidence": avg_conf})
            current_entity_tokens = []
            current_label = None
            current_probs = []

    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        avg_conf = sum(current_probs) / len(current_probs)
        if entity_text.lower() not in STOPWORD_BLOCKLIST:
            entities.append({"text": entity_text, "label": current_label, "confidence": avg_conf})

    # Strip trailing punctuation
    entities = [{"text": e["text"].rstrip(".,;:!?"), "label": e["label"], "confidence": e["confidence"]} for e in entities]
    entities = [e for e in entities if e["text"].lower() not in STOPWORD_BLOCKLIST]

    # Highlighted text
    clean_tokens = [t.rstrip(".,;:!?") for t in tokens]
    entity_set = {e["text"]: e["label"] for e in entities}
    highlighted = []
    i = 0
    while i < len(clean_tokens):
        matched = False
        for length in range(min(10, len(clean_tokens) - i), 0, -1):
            span = " ".join(clean_tokens[i:i+length])
            if span in entity_set:
                highlighted.append((span, entity_set[span]))
                i += length
                matched = True
                break
        if not matched:
            highlighted.append((tokens[i] + " ", None))
            i += 1

    # Markdown table with confidence
    if entities:
        md = "| Entity | Type | Confidence |\n|--------|------|------------|\n"
        for e in entities:
            conf_pct = f"{e['confidence']*100:.1f}%"
            md += f"| {e['text']} | {e['label']} | {conf_pct} |\n"
    else:
        md = "No disease entities detected."

    return highlighted, md


EXAMPLES = [
    ["Patient presents with hypertension and type 2 diabetes mellitus."],
    ["History includes breast cancer treated with chemotherapy."],
    ["Subject presents with symptoms of asthma and chronic obstructive pulmonary disease."],
    ["The patient was diagnosed with pneumonia and required hospitalization."],
    ["Patient suffers from depression and anxiety disorder following myocardial infarction."],
]

with gr.Blocks(title="Clinical Disease Entity Extractor") as demo:

    gr.Markdown("""
    # 🏥 Clinical Disease Entity Extractor

    Paste any clinical free-text below and pick a model. The model will identify **disease entities** and return a confidence score for each.

    - **Production — BioBERT (NCBI Disease)**: [`ugaray96/biobert_ncbi_disease_ner`](https://huggingface.co/ugaray96/biobert_ncbi_disease_ner)
    - **Custom — Bio_ClinicalBERT (MedMentions / SNOMED)**: [`acebirim/snomed-ner-model`](https://huggingface.co/acebirim/snomed-ner-model)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL_LABEL,
                label="Model",
            )
            text_input = gr.Textbox(
                label="Clinical Text",
                placeholder="e.g. Patient presents with hypertension and type 2 diabetes mellitus.",
                lines=6,
            )
            run_btn = gr.Button("Extract Entities", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

        with gr.Column(scale=1):
            highlighted_output = gr.HighlightedText(
                label="Highlighted Entities",
                combine_adjacent=True,
                show_legend=True,
            )
            markdown_output = gr.Markdown()

    run_btn.click(fn=predict, inputs=[text_input, model_dropdown], outputs=[highlighted_output, markdown_output])
    text_input.submit(fn=predict, inputs=[text_input, model_dropdown], outputs=[highlighted_output, markdown_output])

    gr.Markdown("""
    ---
    **Confidence Score**: Average softmax probability across entity tokens. Scores below 50% indicate the model is not plurality-confident in its DISEASE prediction — a natural decision boundary for a 3-class problem — a useful signal for monitoring model reliability in production.

    *Built as a capstone project for the Advanced ML course from https://ml.electricsheep.africa/grade2/.*
    """)

demo.launch()
