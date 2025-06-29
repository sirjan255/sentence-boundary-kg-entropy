import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted', required=True, help='Path to predicted boundaries JSON (dicts)')
    parser.add_argument('--output', required=True, help='Path for output boundaries JSON (strings)')
    args = parser.parse_args()

    with open(args.predicted, "r", encoding="utf-8") as f:
        pred = json.load(f)

    # Convert: key -> list of dicts -> key -> list of node strings
    converted = {k: [d["node"] for d in v if isinstance(d, dict) and "node" in d] for k, v in pred.items()}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"Converted predicted boundaries saved to {args.output}")

if __name__ == "__main__":
    main()