"""
Data Splitting and Preprocessing Utilities
Strictly follows Paper Section 3.1: Data Split & Anti-Leakage Protocol
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ChronologicalSplitter:
    """
    Time-series aware data splitter - strict anti-leakage protocol

    Paper quote:
    "Let the full daily series contain T trading timestamps ordered by time.
     We adopt a time-series-aware split without shuffling:
     Train = {t1, ..., t⌊0.6T⌋}
     Validation = {t⌊0.6T⌋+1, ..., t⌊0.8T⌋}
     Test = {t⌊0.8T⌋+1, ..., tT}"
    """

    def __init__(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Verify ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"

    def split(self, data):
        """
        Split data chronologically

        Args:
            data: DataFrame, time-sorted data

        Returns:
            train_data, val_data, test_data: Three DataFrames
        """
        T = len(data)

        # Calculate split points (using floor operation, consistent with paper)
        train_end = int(np.floor(T * self.train_ratio))
        val_end = int(np.floor(T * (self.train_ratio + self.val_ratio)))

        # Strict chronological split
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()

        # Print split information
        print("\n" + "="*70)
        print("📊 Data Split Information (Chronological Split)")
        print("="*70)
        print(f"Total samples: {T}")
        print(f"\nTraining set:")
        print(f"  - Samples: {len(train_data)} ({len(train_data)/T*100:.1f}%)")
        print(f"  - Date range: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"\nValidation set:")
        print(f"  - Samples: {len(val_data)} ({len(val_data)/T*100:.1f}%)")
        print(f"  - Date range: {val_data.index[0]} to {val_data.index[-1]}")
        print(f"\nTest set:")
        print(f"  - Samples: {len(test_data)} ({len(test_data)/T*100:.1f}%)")
        print(f"  - Date range: {test_data.index[0]} to {test_data.index[-1]}")
        print("="*70)

        return train_data, val_data, test_data

    def verify_no_overlap(self, train_data, val_data, test_data):
        """Verify no overlap between datasets"""
        train_idx = set(train_data.index)
        val_idx = set(val_data.index)
        test_idx = set(test_data.index)

        assert len(train_idx & val_idx) == 0, "Training and validation sets overlap!"
        assert len(train_idx & test_idx) == 0, "Training and test sets overlap!"
        assert len(val_idx & test_idx) == 0, "Validation and test sets overlap!"

        print("✅ Dataset verification passed: No overlap")
        return True


class AntiLeakagePreprocessor:
    """
    Anti-leakage preprocessor

    Paper quote:
    "All scalers/encoders/feature constructors are fit only on the
     training slice and then frozen for validation and test periods."
    """

    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.fitted = False
        self.feature_names = None

    def fit_transform(self, train_data, features):
        """
        Fit scaler ONLY on training data

        Args:
            train_data: Training DataFrame
            features: List of features to normalize

        Returns:
            Normalized training data (numpy array)
        """
        print("\n" + "="*70)
        print("⚙️  Preprocessor Training (fit only on training set)")
        print("="*70)
        print(f"Number of features: {len(features)}")
        print(f"Feature list: {features}")

        # Fit only on training data
        self.scaler.fit(train_data[features])
        self.fitted = True
        self.feature_names = features

        # Transform training data
        transformed = self.scaler.transform(train_data[features])

        print("✅ Preprocessor training complete")
        print(f"   Data range: [{transformed.min():.4f}, {transformed.max():.4f}]")
        print("="*70)

        return transformed

    def transform(self, data, features):
        """
        Transform validation/test data (NO fit!)

        Args:
            data: Validation or test DataFrame
            features: Feature list

        Returns:
            Normalized data (numpy array)
        """
        if not self.fitted:
            raise RuntimeError("❌ Error: Must call fit_transform() on training data first!")

        if features != self.feature_names:
            raise ValueError(f"❌ Feature mismatch! Expected: {self.feature_names}, got: {features}")

        # Only transform, using training statistics
        transformed = self.scaler.transform(data[features])

        return transformed

    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        if not self.fitted:
            raise RuntimeError("❌ Error: Preprocessor not fitted!")

        return self.scaler.inverse_transform(data)


def verify_leakage_guards(train_data, val_data, test_data, features):
    """
    Verify anti-leakage measures

    Paper quote:
    "Leakage guards: (1) All rolling statistics at time tk are computed
     using {t1,...,tk} only, never using future information"
    """
    print("\n" + "="*70)
    print("🔍 Verifying Anti-Leakage Measures")
    print("="*70)

    checks = []

    # Check 1: Dataset chronological order
    train_max_date = train_data.index.max()
    val_min_date = val_data.index.min()
    val_max_date = val_data.index.max()
    test_min_date = test_data.index.min()

    check1 = train_max_date < val_min_date
    checks.append(("Training max date < Validation min date", check1))

    check2 = val_max_date < test_min_date
    checks.append(("Validation max date < Test min date", check2))

    # Check 2: Dataset size reasonableness
    total = len(train_data) + len(val_data) + len(test_data)
    train_ratio = len(train_data) / total
    val_ratio = len(val_data) / total
    test_ratio = len(test_data) / total

    check3 = abs(train_ratio - 0.6) < 0.05
    checks.append((f"Training ratio ≈ 60% (actual: {train_ratio*100:.1f}%)", check3))

    check4 = abs(val_ratio - 0.2) < 0.05
    checks.append((f"Validation ratio ≈ 20% (actual: {val_ratio*100:.1f}%)", check4))

    check5 = abs(test_ratio - 0.2) < 0.05
    checks.append((f"Test ratio ≈ 20% (actual: {test_ratio*100:.1f}%)", check5))

    # Print check results
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and result

    print("="*70)

    if all_passed:
        print("✅ All anti-leakage checks passed!")
    else:
        print("❌ Some checks failed, please verify data split!")

    return all_passed


if __name__ == "__main__":
    """Test code"""
    print("Data Utilities Module Test")
    print("Please run from main program for testing")
