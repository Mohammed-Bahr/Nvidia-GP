# ============================================================
# STAGE 2: Transcript Cleaning + Structured Notes Generation
# AI Pipeline: Speech-to-Text (Whisper) -> Text Organization (Flan-T5 / AraT5)
# ============================================================

# ---- Installation (Colab) ----
# Run this once in Google Colab before executing the script:
# !pip install -q transformers sentencepiece accelerate torch

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
TRANSCRIPT_FILE = "transcript.txt"
OUTPUT_FILE = "organized_notes.txt"

ENGLISH_MODEL = "google/flan-t5-base"   # instruction-tuned, best for English/code-switched text
ARABIC_MODEL = "UBC-NLP/AraT5-base"     # used when transcript is mostly Arabic

MAX_INPUT_TOKENS = 480      # stay under the model's ~512 token limit, leaving room for the prompt
CHUNK_OVERLAP_WORDS = 20    # small overlap so context isn't lost between chunks

# ------------------------------------------------------------
# 2. READ TRANSCRIPT
# ------------------------------------------------------------
def read_transcript(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ------------------------------------------------------------
# 3. CLEANING
# ------------------------------------------------------------
ENGLISH_FILLERS = [
    "um", "uh", "erm", "hmm", "like", "you know", "i mean",
    "actually", "basically", "literally", "sort of", "kind of", "okay so"
]

# Common Egyptian Arabic filler expressions
ARABIC_FILLERS = [
    "يعني", "امم", "اه", "آه", "طيب", "خلاص كده", "بصراحة",
    "يعني كده", "ماشي", "تمام كده"
]

FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in ENGLISH_FILLERS + ARABIC_FILLERS) + r")\b",
    flags=re.IGNORECASE
)

def remove_fillers(text):
    """Strip filler words/phrases (English + Egyptian Arabic) without touching meaning-bearing words."""
    return FILLER_PATTERN.sub("", text)

def remove_repeated_words(text):
    """Collapse immediate word repetitions caused by stutters/ASR artifacts, e.g. 'the the cat' -> 'the cat'."""
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def clean_transcript(text):
    text = remove_fillers(text)
    text = remove_repeated_words(text)
    text = normalize_spaces(text)
    return text

# ------------------------------------------------------------
# 4. LANGUAGE DETECTION (simple heuristic, no extra dependency)
# ------------------------------------------------------------
def detect_language_mix(text):
    """If Arabic script makes up more than 30% of characters, treat as Arabic-dominant."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(re.findall(r'[^\s]', text))
    if total_chars == 0:
        return "en"
    return "ar" if (arabic_chars / total_chars) > 0.3 else "en"

# ------------------------------------------------------------
# 5. MODEL LOADING
# ------------------------------------------------------------
def load_model(lang):
    model_name = ARABIC_MODEL if lang == "ar" else ENGLISH_MODEL
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return gen_pipeline, tokenizer, model_name

# ------------------------------------------------------------
# 6. CHUNKING LONG TRANSCRIPTS
# ------------------------------------------------------------
def chunk_text(text, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    """Split by word count while checking real token length, with small overlap between chunks."""
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(tokenizer.encode(" ".join(current))) >= max_tokens:
            current.pop()
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP_WORDS:] + [word]

    if current:
        chunks.append(" ".join(current))

    return chunks

# ------------------------------------------------------------
# 7. PROMPT ENGINEERING
# ------------------------------------------------------------
def build_prompt(chunk_text, lang):
    if lang == "ar":
        prompt = (
            "النص التالي هو نسخة من اجتماع أو محادثة، وقد يحتوي على مزيج من العربية المصرية والإنجليزية. "
            "لخص النص، واستخرج أهم النقاط، واستخرج أي مهام أو إجراءات مطلوبة (Action Items). "
            "حافظ على المعنى الأصلي حتى لو وردت بعض الكلمات بالإنجليزية.\n\n"
            f"النص:\n{chunk_text}\n\n"
            "أعد الناتج بالشكل التالي:\n"
            "Summary:\n...\nBullet Points:\n- ...\nAction Items:\n- ..."
        )
    else:
        prompt = (
            "The following text is a transcript of a meeting or conversation. "
            "It may contain a mix of English and Egyptian Arabic (code-switching) — "
            "treat Arabic phrases with the same importance as English ones; if needed, "
            "convey their meaning in English while preserving the original intent.\n\n"
            "Organize this transcript into structured notes with EXACTLY these three sections:\n"
            "Summary: a short paragraph summarizing the main topic.\n"
            "Bullet Points: the key points discussed, as a bullet list.\n"
            "Action Items: concrete tasks or follow-ups mentioned, as a bullet list "
            "(write 'None mentioned' if there are none).\n\n"
            f"Transcript:\n{chunk_text}\n\n"
            "Structured Notes:"
        )
    return prompt

# ------------------------------------------------------------
# 8. GENERATE STRUCTURED NOTES (per chunk, then merge)
# ------------------------------------------------------------
def generate_notes_for_chunk(gen_pipeline, chunk, lang):
    prompt = build_prompt(chunk, lang)
    result = gen_pipeline(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.3
    )
    return result[0]["generated_text"].strip()

def merge_chunk_notes(gen_pipeline, chunk_notes, lang):
    """When the transcript needed multiple chunks, ask the model to merge partial
    notes into one de-duplicated final version."""
    combined = "\n\n".join(chunk_notes)

    if lang == "ar":
        prompt = (
            "فيما يلي ملاحظات جزئية مستخرجة من أجزاء مختلفة من نفس التسجيل. "
            "ادمجها في نسخة نهائية واحدة بدون تكرار، بنفس الشكل التالي:\n"
            "Summary:\n...\nBullet Points:\n- ...\nAction Items:\n- ...\n\n"
            f"الملاحظات الجزئية:\n{combined}"
        )
    else:
        prompt = (
            "Below are partial structured notes extracted from different parts of the same "
            "recording. Merge them into a single final version with no duplicate points, "
            "using EXACTLY this format:\n"
            "Summary: ...\nBullet Points:\n- ...\nAction Items:\n- ...\n\n"
            f"Partial notes:\n{combined}"
        )

    result = gen_pipeline(
        prompt,
        max_new_tokens=350,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.3
    )
    return result[0]["generated_text"].strip()

# ------------------------------------------------------------
# 9. MAIN PIPELINE
# ------------------------------------------------------------
def main():
    # Step 1: Read
    raw_transcript = read_transcript(TRANSCRIPT_FILE)
    print("Raw transcript loaded. Length (chars):", len(raw_transcript))

    # Step 2: Clean
    cleaned_transcript = clean_transcript(raw_transcript)
    print("\nCleaned transcript preview:\n", cleaned_transcript[:300], "...\n")

    # Step 3: Detect language mix -> choose model
    lang = detect_language_mix(cleaned_transcript)
    print(f"Detected dominant language: {'Arabic' if lang == 'ar' else 'English'}")

    # Step 4: Load model
    gen_pipeline, tokenizer, model_name = load_model(lang)

    # Step 5: Chunk if needed
    chunks = chunk_text(cleaned_transcript, tokenizer)
    print(f"Transcript split into {len(chunks)} chunk(s).")

    # Step 6: Generate notes per chunk
    chunk_notes = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {i}/{len(chunks)} ...")
        chunk_notes.append(generate_notes_for_chunk(gen_pipeline, chunk, lang))

    # Step 7: Merge if more than one chunk
    if len(chunk_notes) > 1:
        print("Merging chunk-level notes into final structured notes ...")
        final_notes = merge_chunk_notes(gen_pipeline, chunk_notes, lang)
    else:
        final_notes = chunk_notes[0]

    # Step 8: Print nicely
    print("\n" + "=" * 60)
    print("FINAL STRUCTURED NOTES")
    print("=" * 60)
    print(final_notes)
    print("=" * 60)

    # Step 9: Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_notes)
    print(f"\nSaved structured notes to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()