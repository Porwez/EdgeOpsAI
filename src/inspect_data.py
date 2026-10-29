
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/ai4i2020.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET OVERVIEW =====")
    print(f"Dataset shape: {df.shape}")

    print("\n===== COLUMN NAMES =====")
    for column in df.columns:
        print(f"- {column}")

    print("\n===== FIRST 5 ROWS =====")
    print(df.head().to_string(index=False))

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    if "Machine failure" not in df.columns:
        raise ValueError(
            "Expected target column 'Machine failure' was not found."
        )

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["Machine failure"].value_counts().sort_index())

    print("\n===== TARGET PERCENTAGES =====")
    print(
        (df["Machine failure"].value_counts(normalize=True)
         .sort_index() * 100).round(2)
    )


if __name__ == "__main__":
    main()
