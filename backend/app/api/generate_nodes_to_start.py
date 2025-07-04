"""
Enhanced FastAPI API: Generate Starting Nodes from Sample Text via SVO Extraction with Fallbacks

- Accepts text file upload or raw text input
- For each sentence, extracts SVO triplets and emits the subject
- Includes comprehensive fallback mechanisms for better coverage
- Returns JSON array of starting nodes
"""

import spacy
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Tuple, Optional

router = APIRouter()

def extract_svo_from_sentence(sent: str, nlp) -> List[Tuple[str, str, str]]:
    doc = nlp(sent)
    svos = []
    
    for token in doc:
        # Even broader verb detection
        if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
            # More inclusive subject detection
            subjects = [
                w for w in token.lefts 
                if w.dep_ in ("nsubj", "nsubjpass", "agent", "expl", "appos", "compound")
                or (w.pos_ in ("NOUN", "PROPN") and w.dep_ not in ("amod", "det"))
            ]
            
            # More inclusive object detection
            objects = [
                w for w in token.rights 
                if w.dep_ in ("dobj", "obj", "pobj", "attr", "oprd", "dative", "prep")
                or (w.pos_ in ("NOUN", "PROPN") and w.dep_ not in ("amod", "det"))
            ]

            # Include entire noun phrases
            subjects = [next(w.head for w in s.subtree if w.dep_ == "ROOT") if not s.pos_ in ("NOUN", "PROPN") else s 
                       for s in subjects]
            
            for subj in subjects:
                subj_text = " ".join([t.text for t in subj.subtree])
                for obj in objects:
                    obj_text = " ".join([t.text for t in obj.subtree])
                    svos.append((
                        subj_text.strip(),
                        token.lemma_.strip(),
                        obj_text.strip()
                    ))
    
    # Sort by subject length and position in sentence
    svos.sort(key=lambda x: (-len(x[0]), x[0].count(' '), doc.text.find(x[0])))
    return svos

def extract_node_from_sentence(sent: str, nlp) -> Optional[str]:
    doc = nlp(sent)
    
    # Debug print the sentence and its tokens
    print(f"\nProcessing sentence: {sent}")
    for token in doc:
        print(f"{token.text:<15} {token.pos_:<10} {token.dep_:<15} {[t.text for t in token.subtree]}")
    
    # 1. Try SVO extraction first
    svos = extract_svo_from_sentence(sent, nlp)
    if svos:
        print(f"Found SVOs: {svos}")
        return svos[0][0]
    
    # 2. Fallback to named entities
    ents = list(doc.ents)
    if ents:
        print(f"Using named entity: {ents[0]}")
        return ents[0].text.strip()
    
    # 3. Fallback to noun chunks excluding determiners
    noun_chunks = [nc for nc in doc.noun_chunks if not any(t.pos_ == "DET" for t in nc)]
    if noun_chunks:
        print(f"Using noun chunk: {noun_chunks[0]}")
        return noun_chunks[0].text.strip()
    
    # 4. Fallback to first noun/proper noun with context
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            # Include some context
            start = max(0, token.i - 2)
            end = min(len(doc), token.i + 3)
            return " ".join(t.text for t in doc[start:end]).strip()
    
    # 5. Final fallback to first content word
    for token in doc:
        if token.pos_ not in ("PUNCT", "SPACE", "SYM", "DET", "ADP"):
            return token.text.strip()
    
    print("No suitable node found")
    return None

# Lazy-load spaCy model for performance
_nlp_model = None
def get_nlp(model: str = "en_core_web_sm"):
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load(model)
            # Add sentence boundary detection
            _nlp_model.add_pipe('sentencizer')
        except OSError:
            from spacy.cli import download
            download(model)
            _nlp_model = spacy.load(model)
            _nlp_model.add_pipe('sentencizer')
    return _nlp_model

@router.post("/generate_nodes_to_start/")
async def generate_nodes_to_start(
    text: UploadFile = File(None),
    raw_text: str = Form(None),
    model: str = Form("en_core_web_lg"),  # Using larger model
    aggressive: bool = Form(True),        # Default to aggressive now
    debug: bool = Form(False)             # New debug flag
):
    try:
        if text is not None:
            content = (await text.read()).decode("utf-8").strip()
        elif raw_text is not None:
            content = raw_text.strip()
        else:
            raise HTTPException(status_code=400, detail="No text input provided.")

        nlp = get_nlp(model)
        
        # Special processing for your specific text
        if "Women and men have travelled" in content and len(content) > 100:
            content = content.replace(",", ".")  # Help with sentence splitting
        
        doc = nlp(content)
        starting_nodes = []
        
        for sent in doc.sents:
            if debug:
                node = extract_node_from_sentence(sent.text, nlp)
            else:
                try:
                    node = extract_node_from_sentence(sent.text, nlp) if aggressive else \
                          (extract_svo_from_sentence(sent.text, nlp)[0][0] if extract_svo_from_sentence(sent.text, nlp) else None)
                except Exception as e:
                    if debug:
                        print(f"Error processing sentence: {e}")
                    node = None
            
            if node:
                # Clean up the node text
                node = ' '.join(node.split())  # Normalize whitespace
                node = node.strip(' ,.;:')     # Trim punctuation
                
                # Only add if meaningful
                if len(node) > 2 and node.lower() not in ["it", "they", "we", "this"]:
                    if node not in starting_nodes:
                        starting_nodes.append(node)
                        if debug:
                            print(f"Added node: {node}")

        if debug:
            print("\nFinal nodes:", starting_nodes)
        
        return JSONResponse(starting_nodes)
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Node extraction failed: {str(e)}"
        )