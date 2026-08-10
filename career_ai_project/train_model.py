"""
train_model.py
Huấn luyện Random Forest offline từ career_prediction_multi_industry.csv.

Chạy:
    python train_model.py

Kết quả lưu vào models/:
    - rf_model.pkl
    - field_encoder.pkl
    - career_encoder.pkl
    - skills_encoder.pkl   (MultiLabelBinarizer cho cột Skills)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "career_prediction_multi_industry.csv"
MODELS_DIR = BASE_DIR / "models"

# Features số — phải khớp model.py khi inference
NUMERIC_FEATURE_COLUMNS = [
    "Field",
    "Professional Skills",
    "Communication Skills",
    "Problem Solving Skills",
    "Teamwork Skills",
    "Projects",
    "Internships",
]
TARGET_COLUMN = "Career"
SKILLS_COLUMN = "Skills"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Đọc và kiểm tra dữ liệu đầu vào."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {csv_path}."
        )

    df = pd.read_csv(csv_path)
    required_cols = NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {missing}")

    if SKILLS_COLUMN not in df.columns:
        df[SKILLS_COLUMN] = ""

    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        raise ValueError("Sau khi dropna, dữ liệu trống — không thể train.")

    df.loc[:, SKILLS_COLUMN] = df[SKILLS_COLUMN].fillna("").astype(str)
    return df


def encode_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, LabelEncoder, LabelEncoder, MultiLabelBinarizer]:
    """
    Encode Field, Career và Skills.
    Trả về X (numeric + skill binary), y, và 3 encoder.
    """
    field_encoder = LabelEncoder()
    career_encoder = LabelEncoder()
    skills_encoder = MultiLabelBinarizer()

    work = df.copy()
    work.loc[:, "Field"] = field_encoder.fit_transform(work["Field"].astype(str))
    y = career_encoder.fit_transform(work[TARGET_COLUMN].astype(str))

    skill_lists = work[SKILLS_COLUMN].apply(parse_skills_cell)
    skill_matrix = skills_encoder.fit_transform(skill_lists)

    numeric = work[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=float)
    x = np.hstack([numeric, skill_matrix.astype(float)])

    print(f"Số skill unique (vocabulary): {len(skills_encoder.classes_)}")
    print(f"Kích thước X: {x.shape} (numeric={numeric.shape[1]} + skills={skill_matrix.shape[1]})")

    return x, pd.Series(y), field_encoder, career_encoder, skills_encoder


def train_random_forest(
    x: np.ndarray, y: pd.Series
) -> tuple[RandomForestClassifier, float]:
    """Chia train/test, huấn luyện RandomForest và trả về model + accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Độ chính xác trên tập test: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, acc

def parse_skills_cell(value: object) -> list[str]:
    """Tách chuỗi Skills 'Python, Docker, AWS' thành list token."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def save_artifacts(
    model: RandomForestClassifier,
    field_encoder: LabelEncoder,
    career_encoder: LabelEncoder,
    skills_encoder: MultiLabelBinarizer,
    models_dir: Path,
) -> None:
    """Xuất model và các encoder ra file .pkl."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "rf_model.pkl")
    joblib.dump(field_encoder, models_dir / "field_encoder.pkl")
    joblib.dump(career_encoder, models_dir / "career_encoder.pkl")
    joblib.dump(skills_encoder, models_dir / "skills_encoder.pkl")

    print(f"\nĐã lưu artifacts vào: {models_dir}")
    print("  - rf_model.pkl")
    print("  - field_encoder.pkl")
    print("  - career_encoder.pkl")
    print("  - skills_encoder.pkl")


def main() -> None:
    print("=== Career Guidance AI — Train Random Forest ===")
    print(f"Đọc dữ liệu từ: {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"Số mẫu sau làm sạch: {len(df)}")
    print(f"Số nghề (Career): {df[TARGET_COLUMN].nunique()}")
    print(f"Số Field: {df['Field'].nunique()}")

    x, y, field_encoder, career_encoder, skills_encoder = encode_features(df)
    model, _ = train_random_forest(x, y)
    save_artifacts(
        model,
        field_encoder,
        career_encoder,
        skills_encoder,
        MODELS_DIR,
    )
    print("\nHoàn tất huấn luyện.")


if __name__ == "__main__":
    main()
