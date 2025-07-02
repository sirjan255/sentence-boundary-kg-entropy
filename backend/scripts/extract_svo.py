import argparse
import spacy
import csv
import os

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

def main():
    parser = argparse.ArgumentParser(description="Extract SVO triplets from a text file using spaCy.")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file for triplets.")
    parser.add_argument("--model", type=str, default="en_core_web_sm", help="spaCy language model to use.")
    args = parser.parse_args()

    # Load spacy model
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"spaCy model '{args.model}' not found. Downloading...")
        from spacy.cli import download
        download(args.model)
        nlp = spacy.load(args.model)

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        return

    # Read text
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Split into sentences with spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    all_triplets = []

    # Extract SVOs
    for sent in sentences:
        svos = extract_svo_from_sentence(sent, nlp)
        for subj, verb, obj in svos:
            all_triplets.append([subj, verb, obj, sent])

    # Write to CSV
    with open(args.output, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subject", "verb", "object", "sentence"])
        for row in all_triplets:
            writer.writerow(row)

    print(f"Extracted {len(all_triplets)} SVO triplets from {len(sentences)} sentences.")
    print(f"Triplets saved to {args.output}")

if __name__ == "__main__":
    main()