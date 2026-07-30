"""
train_model.py
Phase 2 — Huấn luyện mô hình Random Forest offline cho Career Guidance AI.

Chạy:
    python train_model.py

Kết quả lưu vào thư mục models/:
    - rf_model.pkl
    - field_encoder.pkl
    - career_encoder.pkl
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đường dẫn gốc dự án (thư mục chứa file này)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "career_prediction.csv"
MODELS_DIR = BASE_DIR / "models"

# Thứ tự features phải khớp với app.py khi inference
FEATURE_COLUMNS = [
    "Field",
    "Coding Skills",
    "Communication Skills",
    "Problem Solving Skills",
    "Teamwork Skills",
    "Projects",
    "Internships",
]
TARGET_COLUMN = "Career"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Đọc và kiểm tra dữ liệu đầu vào."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {csv_path}. "
            "Hãy đặt career_prediction.csv vào thư mục data/."
        )

    df = pd.read_csv(csv_path)
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {missing}")

    # Loại bỏ dòng thiếu giá trị trên features / target
    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        raise ValueError("Sau khi dropna, dữ liệu trống — không thể train.")

    return df


def encode_categorical(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Encode biến phân loại Field (feature) và Career (target).
    Trả về DataFrame đã encode kèm 2 LabelEncoder để lưu lại.
    """
    field_encoder = LabelEncoder()
    career_encoder = LabelEncoder()

    df = df.copy()
    df["Field"] = field_encoder.fit_transform(df["Field"].astype(str))
    df[TARGET_COLUMN] = career_encoder.fit_transform(df[TARGET_COLUMN].astype(str))

    return df, field_encoder, career_encoder


def train_random_forest(
    x: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, float]:
    """Chia train/test, huấn luyện RandomForest và trả về model + accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2, #20% test
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Độ chính xác trên tập test: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, acc


def save_artifacts(
    model: RandomForestClassifier,
    field_encoder: LabelEncoder,
    career_encoder: LabelEncoder,
    models_dir: Path,
) -> None:
    """Xuất model và các encoder ra file .pkl."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "rf_model.pkl")
    joblib.dump(field_encoder, models_dir / "field_encoder.pkl")
    joblib.dump(career_encoder, models_dir / "career_encoder.pkl")

    print(f"\nĐã lưu artifacts vào: {models_dir}")
    print("  - rf_model.pkl")
    print("  - field_encoder.pkl")
    print("  - career_encoder.pkl")


def main() -> None:
    print("=== Career Guidance AI — Train Random Forest ===")
    print(f"Đọc dữ liệu từ: {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"Số mẫu sau làm sạch: {len(df)}")

    df_encoded, field_encoder, career_encoder = encode_categorical(df)

    x = df_encoded[FEATURE_COLUMNS]
    y = df_encoded[TARGET_COLUMN]

    model, _ = train_random_forest(x, y)
    save_artifacts(model, field_encoder, career_encoder, MODELS_DIR)
    print("\nHoàn tất huấn luyện.")


if __name__ == "__main__":
    main()
