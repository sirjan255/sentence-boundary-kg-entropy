import spacy

def extract_svo_from_sentence(sent, nlp):
    """
    Extracts SVO triplets from a spacy-parsed sentence.
    Returns a list of (subject, verb, object) tuples.
    """
    doc = nlp(sent)
    svos = []

    for token in doc:
        # Find verbs
        if token.pos_ == "VERB":
            # Find subjects
            subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            # Find objects
            objects = [w for w in token.rights if w.dep_ in ("dobj", "obj", "pobj", "attr")]

            # If multiple subjects/objects, create all combinations
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

def extract_svo_from_text(text, model="en_core_web_sm"):
    """
    Extracts SVO triplets from all sentences in the input text.
    Returns a list of [subject, verb, object, sentence] rows.
    """
    try:
        nlp = spacy.load(model)
    except OSError:
        from spacy.cli import download
        download(model)
        nlp = spacy.load(model)

    doc = nlp(text)
    results = []
    for sent in doc.sents:
        svos = extract_svo_from_sentence(sent.text, nlp)
        for subj, verb, obj in svos:
            results.append([subj, verb, obj, sent.text.strip()])
    return results