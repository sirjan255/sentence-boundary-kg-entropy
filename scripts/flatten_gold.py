import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', required=True, help='Path to gold boundaries JSON (dicts)')
    parser.add_argument('--output', required=True, help='Path for output boundaries JSON (flattened)')
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)

    flat_gold = set()
    if isinstance(gold, dict):
        for v in gold.values():
            if isinstance(v, list):
                flat_gold.update(v)
            elif isinstance(v, dict) and "nodes_in_sentence" in v:
                flat_gold.update(v["nodes_in_sentence"])
            else:
                flat_gold.add(v)
    elif isinstance(gold, list):
        flat_gold.update(gold)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(list(flat_gold), f, indent=2)

    print(f"Flattened gold boundaries written to {args.output}")

if __name__ == "__main__":
    main()