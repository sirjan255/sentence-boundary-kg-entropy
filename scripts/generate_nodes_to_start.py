"""
Script to generate data/nodes_to_start.txt from data/sample.txt
by extracting one entity per sentence using the repo's SVO extraction logic.
"""

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

if __name__ == "__main__":
    model = "en_core_web_sm"
    try:
        nlp = spacy.load(model)
    except OSError:
        from spacy.cli import download
        download(model)
        nlp = spacy.load(model)

    with open("data/sample.txt", "r", encoding="utf-8") as fin, open("data/nodes_to_start.txt", "w", encoding="utf-8") as fout:
        text = fin.read()
        for sent in nlp(text).sents:
            svos = extract_svo_from_sentence(sent.text, nlp)
            if svos:
                fout.write(svos[0][0] + "\n")  # Write the subject of the first SVO as the starting node