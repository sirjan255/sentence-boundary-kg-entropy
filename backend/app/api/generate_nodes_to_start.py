import spacy
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Tuple, Optional

router = APIRouter()

# --- SVO Extraction ---
def extract_svo_from_sentence(sent: str, nlp) -> List[Tuple[str, str, str]]:
    doc = nlp(sent)
    svos = []
    
    for token in doc:
        if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
            subjects = [
                w for w in token.lefts
                if w.dep_ in ("nsubj", "nsubjpass", "agent", "expl", "appos", "compound")
                or (w.pos_ in ("NOUN", "PROPN") and w.dep_ not in ("amod", "det"))
            ]
            objects = [
                w for w in token.rights
                if w.dep_ in ("dobj", "obj", "pobj", "attr", "oprd", "dative", "prep")
                or (w.pos_ in ("NOUN", "PROPN") and w.dep_ not in ("amod", "det"))
            ]

            for subj in subjects:
                subj_text = " ".join([t.text for t in subj.subtree])
                for obj in objects:
                    obj_text = " ".join([t.text for t in obj.subtree])
                    svos.append((
                        subj_text.strip(),
                        token.lemma_.strip(),
                        obj_text.strip()
                    ))

    # Sort to prefer longer or earlier subjects
    svos.sort(key=lambda x: (-len(x[0]), x[0].count(' '), doc.text.find(x[0])))
    return svos

# --- Starting Node Extraction ---
def extract_node_from_sentence(sent: str, nlp) -> Optional[str]:
    doc = nlp(sent)

    # 1. Try SVO Extraction
    svos = extract_svo_from_sentence(sent, nlp)
    if svos:
        subj = svos[0][0].strip(' ,.;:')
        # Reject weak/generic pronouns
        if subj.lower() not in {"it", "they", "we", "this", "he", "she"} and len(subj) > 2:
            return subj

    # 2. Named Entity fallback
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP", "PRODUCT", "WORK_OF_ART"}:
            return ent.text.strip(' ,.;:')

    # 3. Noun chunk fallback
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip(' ,.;:')
        if chunk.root.pos_ in ("NOUN", "PROPN") and len(chunk_text) > 2:
            return chunk_text

    # 4. Proper noun fallback
    for token in doc:
        if token.pos_ in ("PROPN", "NOUN") and token.text.lower() not in {"it", "they", "we", "this"}:
            return token.text.strip(' ,.;:')

    return None

# --- NLP Model Loader ---
_nlp_model = None
def get_nlp(model: str = "en_core_web_sm"):
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load(model)
            if 'sentencizer' not in _nlp_model.pipe_names:
                _nlp_model.add_pipe('sentencizer')
        except OSError:
            from spacy.cli import download
            download(model)
            _nlp_model = spacy.load(model)
            if 'sentencizer' not in _nlp_model.pipe_names:
                _nlp_model.add_pipe('sentencizer')
    return _nlp_model

# --- FastAPI Route ---
@router.post("/generate_nodes_to_start/")
async def generate_nodes_to_start(
    text: UploadFile = File(None),
    raw_text: str = Form(None),
    model: str = Form("en_core_web_sm"),
    debug: bool = Form(False)
):
    try:
        if text is not None:
            content = (await text.read()).decode("utf-8").strip()
        elif raw_text is not None:
            content = raw_text.strip()
        else:
            raise HTTPException(status_code=400, detail="No text input provided.")

        nlp = get_nlp(model)
        doc = nlp(content)

        starting_nodes = []

        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence:
                continue

            try:
                node = extract_node_from_sentence(sentence, nlp)
            except Exception as e:
                if debug:
                    print(f"Error processing sentence: {sentence} → {e}")
                node = None

            if node:
                node_clean = ' '.join(node.split()).strip(' ,.;:')
                if len(node_clean) > 2 and node_clean.lower() not in {"it", "they", "we", "this", "he", "she"}:
                    if node_clean not in starting_nodes:
                        starting_nodes.append(node_clean)
                        if debug:
                            print(f"Added node: {node_clean}")

        return JSONResponse(starting_nodes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node extraction failed: {str(e)}")
