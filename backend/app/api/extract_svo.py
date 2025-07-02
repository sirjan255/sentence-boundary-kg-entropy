"""
FastAPI API: Extract SVO Triplets from Text using spaCy

- Accepts a plain text file (or raw text string) uploaded by the user.
- Extracts SVO (subject, verb, object) triplets from each sentence using spaCy.
- Returns the triplets as a JSON array (each with subject, verb, object, and sentence).
- Does NOT require or save any data to local files.

How to use in frontend:
    - POST multipart/form-data with a file field named "text" (or raw text via "text" field).
    - Receives JSON response: [{subject, verb, object, sentence}, ...]
    - We can also accept a string field named "text" for direct text input.
"""

import spacy
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

def extract_svo_from_sentence(sent, nlp):
    """
    Extracts SVO triplets from a spacy-parsed sentence.
    Returns a list of (subject, verb, object) tuples.
    """
    doc = nlp(sent)
    svos = []
    for token in doc:
        if token.pos_ == "VERB":
            subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.rights if w.dep_ in ("dobj", "obj", "pobj", "attr")]
            for subj in subjects:
                for obj in objects:
                    subj_text = " ".join([t.text for t in subj.subtree])
                    obj_text = " ".join([t.text for t in obj.subtree])
                    svos.append((subj_text.strip(), token.lemma_.strip(), obj_text.strip()))
    return svos

# Lazy-load the spaCy model for performance
_nlp_model = None
def get_nlp(model="en_core_web_sm"):
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load(model)
        except OSError:
            from spacy.cli import download
            download(model)
            _nlp_model = spacy.load(model)
    return _nlp_model

@router.post("/extract_svo/")
async def extract_svo(
    text: UploadFile = File(None),
    raw_text: str = Form(None),
    model: str = Form("en_core_web_sm")
):
    """
    Extract SVO triplets from an uploaded text file or provided raw text.
    Returns: JSON list of {subject, verb, object, sentence}
    """
    try:
        # Handle input: file upload or raw text field
        if text is not None:
            file_content = await text.read()
            content = file_content.decode("utf-8").strip()
        elif raw_text is not None:
            content = raw_text.strip()
        else:
            raise HTTPException(status_code=400, detail="No text input provided.")

        nlp = get_nlp(model)
        doc = nlp(content)
        sentences = [sent.text.strip() for sent in doc.sents]

        all_triplets = []
        for sent in sentences:
            svos = extract_svo_from_sentence(sent, nlp)
            for subj, verb, obj in svos:
                all_triplets.append({
                    "subject": subj,
                    "verb": verb,
                    "object": obj,
                    "sentence": sent
                })

        return JSONResponse(all_triplets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVO extraction failed: {str(e)}")