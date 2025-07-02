"""
FastAPI API: Generate Starting Nodes from Sample Text via SVO Extraction

- Accepts a text file (e.g. sample_text.txt) from the frontend.
- For each sentence, extracts SVO triplets and emits the subject of the first SVO in that sentence.
- Returns a JSON array of starting nodes (subject entities), one per sentence.
- Does NOT read or write any local files.

Frontend usage:
    - POST multipart/form-data with a file field named "text" (the sample text)
    - Receives JSON array: ["subject1", "subject2", ...] 
"""

import spacy
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

def extract_svo_from_sentence(sent, nlp):
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
                    svos.append((
                        subj_text.strip(),
                        token.lemma_.strip(),
                        obj_text.strip()
                    ))
    return svos

# Lazy-load spaCy model for performance
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

@router.post("/generate_nodes_to_start/")
async def generate_nodes_to_start(
    text: UploadFile = File(None),
    raw_text: str = Form(None),
    model: str = Form("en_core_web_sm")
):
    """
    Generate starting nodes (subjects of first SVO per sentence) from sample text.
    Returns: JSON list of subjects, one per sentence
    """
    try:
        # Accept either an uploaded file or a raw text field
        if text is not None:
            file_content = await text.read()
            content = file_content.decode("utf-8").strip()
        elif raw_text is not None:
            content = raw_text.strip()
        else:
            raise HTTPException(status_code=400, detail="No text input provided.")

        nlp = get_nlp(model)
        doc = nlp(content)
        starting_nodes = []
        for sent in doc.sents:
            svos = extract_svo_from_sentence(sent.text, nlp)
            if svos:
                starting_nodes.append(svos[0][0])  # Subject of first SVO in this sentence

        return JSONResponse(starting_nodes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Starting node extraction failed: {str(e)}")