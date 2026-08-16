""" 
Model: Random Forest
Features: Field (OneHot) + soft/numeric scores + Skills cột CSV (MultiLabelBinarizer) 
"""

from __future__ import annotations 
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "career_prediction_multi_industry.csv"
MODELS_DIR = BASE_DIR / "models"

FIELD_COLUMN = "Field"
NUMERIC_FEATURE_COLUMNS = [
    "Professional Skills",
    "Communication Skills",
    "Problem Solving Skills",
    "Teamwork Skills",
    "Projects",
    "Internships",
]
TARGET_COLUMN = "Career"
SKILLS_COLUMN = "Skills"

RF_PARAMS = {
    "n_estimators": 400,
    "random_state": 42,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}


def parse_skills_cell(value: object) -> list[str]:
    """Tách chuỗi Skills CSV thành list token."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}.")

    df = pd.read_csv(csv_path)
    required_cols = [FIELD_COLUMN] + NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {missing}")

    if SKILLS_COLUMN not in df.columns:
        df[SKILLS_COLUMN] = ""

    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        raise ValueError("Sau khi dropna, dữ liệu trống — không thể train.")

    df = df.assign(**{SKILLS_COLUMN: df[SKILLS_COLUMN].fillna("").astype(str)})
    return df


def prepare_skill_lists(df: pd.DataFrame) -> list[list[str]]:
    """Parse cột Skills CSV (không gắn thêm domain Career/Field)."""
    return [parse_skills_cell(v) for v in df[SKILLS_COLUMN].tolist()]


def encode_features(
    df: pd.DataFrame,
    skill_lists: list[list[str]],
) -> tuple[np.ndarray, pd.Series, OneHotEncoder, LabelEncoder, MultiLabelBinarizer]:
    """
    Trả về X, y và các encoder.
    X = [Field one-hot | numeric 6 | skills multi-hot]
    """
    field_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    career_encoder = LabelEncoder()
    skills_encoder = MultiLabelBinarizer()

    field_matrix = field_encoder.fit_transform(df[[FIELD_COLUMN]].astype(str))
    y = career_encoder.fit_transform(df[TARGET_COLUMN].astype(str))
    skills_encoder.fit(skill_lists)
    skill_matrix = skills_encoder.transform(skill_lists).astype(float)
    numeric = df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=float)

    x = np.hstack([field_matrix.astype(float), numeric, skill_matrix])
    print(f"Số Field (one-hot): {field_matrix.shape[1]}")
    print(f"Số skill vocabulary (từ CSV): {len(skills_encoder.classes_)}")
    print(f"Kích thước X: {x.shape}")
    return x, pd.Series(y), field_encoder, career_encoder, skills_encoder


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


def evaluate_holdout(x: np.ndarray, y: pd.Series) -> dict:
    """Holdout 20%: top-1 / top-3 / top-5 của RF thuần."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = make_rf()
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)
    pred = model.predict(x_test)
    metrics = {
        "rf_top1": float(accuracy_score(y_test, pred)),
        "rf_top3": float(top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)),
        "rf_top5": float(top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)),
    }

    print("\n=== Đánh giá holdout (20%) — RF thuần ===")
    print(
        f"   top1={metrics['rf_top1'] * 100:.2f}%  "
        f"top3={metrics['rf_top3'] * 100:.2f}%  "
        f"top5={metrics['rf_top5'] * 100:.2f}%"
    )
    return {"metrics": metrics, "pipeline": "csv_skills_rf_only"}


def save_artifacts(
    model: RandomForestClassifier,
    field_encoder: OneHotEncoder,
    career_encoder: LabelEncoder,
    skills_encoder: MultiLabelBinarizer,
    meta: dict,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "rf_model.pkl")
    joblib.dump(field_encoder, models_dir / "field_encoder.pkl")
    joblib.dump(career_encoder, models_dir / "career_encoder.pkl")
    joblib.dump(skills_encoder, models_dir / "skills_encoder.pkl")
    joblib.dump(meta, models_dir / "model_meta.pkl")
    print(f"\nĐã lưu artifacts vào: {models_dir}")


def main() -> None:
    print("=== Career Guidance AI — Train (CSV Skills + RF) ===")
    print(f"Đọc dữ liệu từ: {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"Số mẫu sau làm sạch: {len(df)}")
    print(f"Số nghề (Career): {df[TARGET_COLUMN].nunique()}")
    print(f"Số Field: {df[FIELD_COLUMN].nunique()}")

    skill_lists = prepare_skill_lists(df)
    empty = sum(1 for s in skill_lists if not s)
    avg_len = float(np.mean([len(s) for s in skill_lists])) if skill_lists else 0.0
    print(f"Skills trống: {empty} | avg skills/row: {avg_len:.1f}")

    x, y, field_encoder, career_encoder, skills_encoder = encode_features(df, skill_lists)
    meta = evaluate_holdout(x, y)

    m = meta["metrics"]
    print("\n=== Tóm tắt Accuracy (%) ===")
    print(
        f"RF thuần: "
        f"top1={m['rf_top1'] * 100:.2f}% | "
        f"top3={m['rf_top3'] * 100:.2f}% | "
        f"top5={m['rf_top5'] * 100:.2f}%"
    )

    print("\nHuấn luyện model cuối trên toàn bộ dữ liệu...")
    final_model = make_rf()
    final_model.fit(x, y)

    save_artifacts(
        final_model,
        field_encoder,
        career_encoder,
        skills_encoder,
        meta,
        MODELS_DIR,
    )
    print("\nHoàn tất huấn luyện.")


if __name__ == "__main__":
    main()
