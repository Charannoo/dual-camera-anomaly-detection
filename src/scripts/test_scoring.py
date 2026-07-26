import os
import torch
from src.models.fusion import CrossAttentionFusion
from src.models.scoring import ResidualScorer

def test_anomaly_scoring(category="cable_gland"):
    # Load cached features
    train_data = torch.load(f"outputs/features/{category}_train.pt")
    test_data = torch.load(f"outputs/features/{category}_test.pt")
    
    # Load trained model checkpoint
    model = CrossAttentionFusion()
    checkpoint_path = f"outputs/checkpoints/{category}_fusion.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded trained model weights from {checkpoint_path}")
    else:
        print("Trained checkpoint not found. Using randomly initialized model (sanity check will still run).")
        
    model.eval()
    
    # Instantiate and fit scorer
    scorer = ResidualScorer()
    scorer.fit(train_data["rgb_feats"], train_data["depth_feats"], model)
    
    # Score test data
    results = scorer.score(test_data["rgb_feats"], test_data["depth_feats"], model)
    scores = results["image_score"]
    labels = test_data["labels"]
    
    print("\n--- Test Sample Scores ---")
    normal_scores = []
    defective_scores = []
    
    for idx in range(len(scores)):
        score = scores[idx].item()
        lbl = labels[idx].item()
        status = "DEFECTIVE" if lbl == 1 else "NORMAL"
        print(f"Sample {idx:02d} | Label: {status} ({lbl}) | Anomaly Score: {score:.4f}")
        if lbl == 1:
            defective_scores.append(score)
        else:
            normal_scores.append(score)
            
    mean_normal = sum(normal_scores) / len(normal_scores)
    mean_defective = sum(defective_scores) / len(defective_scores)
    print("\n--- Summary ---")
    print(f"Average Normal Score: {mean_normal:.4f}")
    print(f"Average Defective Score: {mean_defective:.4f}")
    
    if mean_defective > mean_normal:
        print("SANITY CHECK PASSED: Defective samples score higher on average than normal samples.")
    else:
        print("SANITY CHECK FAILED: Normal samples scored higher or equal to defective ones.")

if __name__ == "__main__":
    test_anomaly_scoring()
