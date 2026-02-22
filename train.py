"""
Random Forest Training Script for Carbon Gate
Real ML training using scikit-learn
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time


def train_random_forest(n_samples=100000, n_features=50, n_estimators=200):
    """
    Train a Random Forest classifier on synthetic data
    This is real ML that will consume CPU resources
    """
    print("help plsssssssssss")
    print("=" * 80)
    print("CARBON GATE DEMO - Random Forest Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {n_features}")
    print(f"  Estimators: {n_estimators}")
    print()

    # Generate synthetic dataset
    print("[1/4] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=5,
        random_state=42,
    )
    print(f"  Dataset shape: {X.shape}")
    print(f"  Classes: {len(np.unique(y))}")

    # Split data
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")

    # Train model
    print(f"\n[3/4] Training Random Forest ({n_estimators} trees)...")
    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=10,
        n_jobs=1,  # CARBON-OPT: Limit to 1 worker to reduce CPU overhead and improve energy efficiency
        random_state=42,
        verbose=0,  # CARBON-OPT: Disable verbose output to reduce I/O overhead
    )

    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Evaluate
    print("\n[4/4] Evaluating model...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 80)
    print("[SUCCESS] Training Complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Test Accuracy: {accuracy:.2%}")
    print(f"  Trees: {n_estimators}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nEstimated CPU hours: {training_time / 3600:.4f}")
    print(f"Estimated carbon emissions: ~{training_time * 0.0005:.4f} kgCO2eq")
    print()


if __name__ == "__main__":
    print(
        """
    This training job will:
    1. Trigger the Carbon Gate GitHub Action on PR
    2. Estimate carbon emissions based on carbon-gate.yml config
    3. Post a detailed report to the PR
    4. Block/warn based on configured thresholds
    
    """
    )

    train_random_forest(n_samples=100000, n_features=50, n_estimators=200)
