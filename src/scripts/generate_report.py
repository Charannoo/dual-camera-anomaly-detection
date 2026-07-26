"""
Stage 8 – Report Artifact Generator
Runs evaluate() for every configured category, collects metrics,
and prints a Markdown-formatted results table suitable for a project report.
"""
import os
import sys
import yaml
import torch

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    categories = config["dataset"]["categories"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    features_dir = config["cache"]["features_dir"]

    # Collect results by running evaluate() per category
    from src.scripts.evaluate import evaluate

    rows = []
    for cat in categories:
        # Only run if cached features and checkpoint exist
        train_pt = os.path.join(features_dir, f"{cat}_train.pt")
        test_pt  = os.path.join(features_dir, f"{cat}_test.pt")
        ckpt     = os.path.join(checkpoint_dir, f"{cat}_fusion.pt")
        if not (os.path.exists(train_pt) and os.path.exists(test_pt) and os.path.exists(ckpt)):
            print(f"[SKIP] {cat}: missing cache or checkpoint.")
            continue
        print(f"\n{'='*50}")
        result = evaluate(cat)
        rows.append({
            "Category":    cat,
            "I-AUROC":     f"{result['i_auroc']:.4f}",
            "P-AUROC":     f"{result['p_auroc']:.4f}",
            "Latency(ms)": f"{result['latency_ms']:.1f}",
            "Params":      f"{result['fusion_params']:,}",
        })

    if not rows:
        print("No results to report. Run extract_features.py and train_fusion.py first.")
        sys.exit(1)

    # Print Markdown table
    print("\n\n" + "="*60)
    print("FINAL RESULTS TABLE (for project report)")
    print("="*60)
    header = ["Category", "I-AUROC", "P-AUROC", "Latency(ms)", "Params"]
    col_w  = [max(len(h), max(len(r[h]) for r in rows)) for h in header]
    sep    = "| " + " | ".join("-" * w for w in col_w) + " |"
    hdr    = "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_w)) + " |"
    print(hdr)
    print(sep)
    for r in rows:
        print("| " + " | ".join(r[h].ljust(w) for h, w in zip(header, col_w)) + " |")

    # Summarise artefact files
    print("\n\nArtefact files saved:")
    for root, dirs, files in os.walk("outputs"):
        for f in files:
            path = os.path.join(root, f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {path}  ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
