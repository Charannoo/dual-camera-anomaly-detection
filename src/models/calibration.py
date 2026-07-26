import math

class ScoreCalibrator:
    def __init__(self):
        self.mean = None
        self.std = None
        self.anchor = None

    def fit(self, train_scores):
        if hasattr(train_scores, 'numpy'):
            train_scores = train_scores.numpy()
        train_scores = list(train_scores)
        n = len(train_scores)
        self.mean = sum(train_scores) / n
        variance = sum((s - self.mean) ** 2 for s in train_scores) / n
        self.std = max(variance ** 0.5, 1e-6)
        self.anchor = self.mean + 3 * self.std
        print(f"Calibrator fitted: mean={self.mean:.4f}, std={self.std:.4f}, anchor={self.anchor:.4f}")

    def calibrate(self, score):
        if self.anchor is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        x = (score - self.anchor) / self.std
        return 1.0 / (1.0 + math.exp(-x))


if __name__ == "__main__":
    import random
    random.seed(42)

    print("Testing ScoreCalibrator...")
    calibrator = ScoreCalibrator()

    normal_scores = [random.gauss(15.0, 3.0) for _ in range(50)]
    calibrator.fit(normal_scores)

    # Normal-range scores should have low confidence
    test_normal = [calibrator.calibrate(s) for s in normal_scores[:5]]
    print(f"Normal scores: {[f'{s:.4f}' for s in normal_scores[:5]]}")
    print(f"Normal confidence: {[f'{c:.4f}' for c in test_normal]}")
    assert all(c < 0.2 for c in test_normal), "Normal scores should have low confidence"

    # Scores far above anchor should approach 1.0
    high_score = calibrator.anchor + 5 * calibrator.std
    high_conf = calibrator.calibrate(high_score)
    print(f"High score ({high_score:.2f}): confidence = {high_conf:.4f}")
    assert high_conf > 0.99, "High scores should approach 1.0"

    print("ScoreCalibrator test passed.")
