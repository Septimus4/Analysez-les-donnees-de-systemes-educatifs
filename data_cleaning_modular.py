from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json

import pandas as pd
from IPython.display import display
import yaml

# ---------------------------------------------------------------------------
#  Paths (edit if needed)
# ---------------------------------------------------------------------------

SOURCE_DIR = Path("./Projet+Python_Dataset_Edstats_csv")
OUTPUT_DIR = Path("./cleaned_data")

# ---------------------------------------------------------------------------
#  Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CleaningOptions:
    """Toggles & thresholds controlling the cleaning steps."""

    drop_columns_threshold: float = 0.5
    amputate_rows_threshold: float = 0.05
    remove_duplicates: bool = True
    remove_outliers: bool = True
    numeric_cols: Sequence[str] | None = None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "CleaningOptions":
        if cfg is None:
            return cls()
        return cls(**cfg)


@dataclass
class FileConfig:
    filename: str
    critical_columns: Sequence[str] | None = None
    options: CleaningOptions = field(default_factory=CleaningOptions)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "FileConfig":
        return cls(
            filename=cfg["filename"],
            critical_columns=cfg.get("critical_columns"),
            options=CleaningOptions.from_mapping(cfg.get("options")),
        )

# ---------------------------------------------------------------------------
#  Cleaning engine
# ---------------------------------------------------------------------------

class DataCleaner:
    """Cleans a DataFrame via chained steps."""

    def __init__(
            self,
            *,
            critical_columns: Sequence[str] | None = None,
            options: CleaningOptions | None = None,
    ) -> None:
        self.critical_columns: list[str] = list(critical_columns or [])
        self.opt = options or CleaningOptions()

    # -- Public API ---------------------------------------------------------

    def clean(self, df: pd.DataFrame, *, name: str = "<DataFrame>") -> pd.DataFrame:
        print(f"\n=== Cleaning: {name} ===")
        print(f"Initial shape: {df.shape}")
        display(df.head(2))

        df = (
            df.pipe(self._drop_high_missing_cols)
            .pipe(self._handle_critical_rows)
            .pipe(self._handle_missing_values)
        )

        if self.opt.remove_duplicates:
            df = self._drop_duplicates(df)
        if self.opt.remove_outliers:
            df = self._remove_outliers(df)

        print(f"\n=== Summary for {name} ===")
        print(f"Final shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        display(df.head(2))
        return df

    def _drop_high_missing_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_ratio = df.isna().mean()
        print("\nMissing percentage per column:")
        print(missing_ratio.sort_values(ascending=False).apply(lambda x: f"{x*100:.1f}%"))
        drop_cols = [
            c for c, r in missing_ratio.items()
            if r > self.opt.drop_columns_threshold and c not in self.critical_columns
        ]
        if drop_cols:
            pct = int(self.opt.drop_columns_threshold * 100)
            print(f"\nDropping columns with >{pct}% missing: {drop_cols}")
            df = df.drop(columns=drop_cols)
            print(f"Columns after drop: {list(df.columns)}")
        else:
            print("\nNo columns to drop based on missing threshold.")
        return df

    def _handle_critical_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.critical_columns:
            print("\nNo critical columns set.")
            return df
        present = [c for c in self.critical_columns if c in df.columns]
        if not present:
            print("⚠️  No critical columns present – skipping row filter.")
            return df
        print("\n-- Missing data in critical columns --")
        for col in present:
            n_missing = df[col].isna().sum()
            pct_missing = 100 * n_missing / len(df) if len(df) > 0 else 0
            print(f"'{col}': {n_missing} missing ({pct_missing:.2f}%)")
        before = len(df)
        df = df.dropna(subset=present, how="all").copy()
        after = len(df)
        print(f"Rows before: {before}, after dropping rows missing all critical columns: {after} (dropped {before - after})")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        limit = int(self.opt.amputate_rows_threshold * len(df))
        print(f"\n-- Handling missing values in non-critical columns --")
        for col in df.columns:
            if col in self.critical_columns:
                continue
            n_missing = df[col].isna().sum()
            pct_missing = 100 * n_missing / len(df) if len(df) > 0 else 0
            if n_missing == 0:
                continue
            if n_missing <= limit:
                print(f"'{col}': dropping {n_missing} rows ({pct_missing:.2f}%, <= threshold).")
                df = df[df[col].notna()].copy()
                print(f"Rows after dropping missing in '{col}': {len(df)}")
                continue
            # Imputation path
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value: Any = df[col].median(skipna=True)
            else:
                modes = df[col].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else pd.NA
            if pd.isna(fill_value):
                print(f"'{col}': column is all-NaN — cannot impute. Skipping.")
                continue
            df.loc[df[col].isna(), col] = fill_value
            print(f"'{col}': imputed {n_missing} values ({pct_missing:.2f}%) with {fill_value}.")
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        dup_count = df.duplicated().sum()
        if dup_count:
            before = len(df)
            df = df.drop_duplicates().copy()
            print(f"Removed {dup_count} duplicate rows. Rows before: {before}, after: {len(df)}")
        else:
            print("No duplicates found.")
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = (
            list(self.opt.numeric_cols)
            if self.opt.numeric_cols is not None
            else df.select_dtypes("number").columns
        )
        for col in numeric_cols:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            n_out = mask.sum()
            if n_out:
                before = len(df)
                df = df[~mask].copy()
                print(f"Column '{col}': removed {n_out} outliers. Rows before: {before}, after: {len(df)}")
            else:
                print(f"Column '{col}': No outliers detected.")
        return df

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_configs(path: Path) -> list[FileConfig]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml not installed; install or use JSON config")
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Config must be a list of file configurations")
    return [FileConfig.from_mapping(item) for item in data]

# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main(files_info: Sequence[FileConfig]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for cfg in files_info:
        file_path = SOURCE_DIR / cfg.filename
        print(f"\n--- Processing {file_path} ---")
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(
            critical_columns=cfg.critical_columns,
            options=cfg.options,
        )
        cleaned = cleaner.clean(df, name=cfg.filename)
        out_path = OUTPUT_DIR / f"{file_path.stem}_cleaned.csv"
        cleaned.to_csv(out_path, index=False)
        print(f"Saved cleaned file → {out_path}")

# ---------------------------------------------------------------------------
#  CLI wrapper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean CSV files as defined in a JSON/YAML config list.",
        epilog="Example: python data_cleaning_modular.py config.yaml",
    )
    parser.add_argument("config", type=Path, help="Path to config list (JSON/YAML)")
    args = parser.parse_args()
    main(_load_configs(args.config))
