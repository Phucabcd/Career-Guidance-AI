"""
train_model.py
Huấn luyện Random Forest offline từ career_prediction_IT_900rows.csv.

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
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "career_prediction_IT_900rows.csv"
MODELS_DIR = BASE_DIR / "models"

# Features số — phải khớp model.py khi inference
NUMERIC_FEATURE_COLUMNS = [
    "Field",
    "Coding Skills",
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

    # 1. Encode Field và Career ra mảng Numpy riêng (không ghi đè trực tiếp vào DataFrame string)
    field_encoded = field_encoder.fit_transform(df["Field"].astype(str))
    y = career_encoder.fit_transform(df[TARGET_COLUMN].astype(str))

    # 2. Encode Skills
    skill_lists = df[SKILLS_COLUMN].apply(parse_skills_cell)
    skill_matrix = skills_encoder.fit_transform(skill_lists)

    # 3. Lấy các cột numeric còn lại (trừ cột Field ra)
    other_numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c != "Field"]
    numeric_other = df[other_numeric_cols].to_numpy(dtype=float)

    # 4. Ghép ma trận X: [Field (1 col) | Các cột numeric khác (6 cols) | Skill matrix]
    field_col = field_encoded.reshape(-1, 1).astype(float)
    x = np.hstack([field_col, numeric_other, skill_matrix.astype(float)])

    print(f"Số skill unique (vocabulary): {len(skills_encoder.classes_)}")
    print(f"Kích thước X: {x.shape} (Field=1 + numeric khác={numeric_other.shape[1]} + skills={skill_matrix.shape[1]})")

    return x, pd.Series(y), field_encoder, career_encoder, skills_encoder


def train_random_forest(
    x: np.ndarray, y: pd.Series
) -> tuple[RandomForestClassifier, float]:
    """Chia train/test và huấn luyện RandomForest với vòng lặp giám sát (Monitoring Loop)."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # 1. Tính Class Weights chuẩn hóa (tránh UserWarning)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    # 2. Tạo tập test có nhiễu nhẹ (Noise) cho các cột kỹ năng mềm để test Robustness
    x_test_noisy = x_test.copy()
    # Thêm Gaussian noise vào các cột điểm số (từ chỉ số 1 đến 6)
    x_test_noisy[:, 1:7] += np.random.normal(0, 0.5, size=x_test_noisy[:, 1:7].shape)

    # 3. Khởi tạo Random Forest hỗ trợ warm_start
    model = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1,
    )

    print("\n=== Bắt đầu In-Training Monitoring ===")
    step = 20
    field_col_idx = 0  # Cột Field hiện tại nằm ở vị trí đầu tiên trong numeric

    for n_trees in range(step, 201, step):
        model.set_params(n_estimators=n_trees)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)
        pred_noisy = model.predict(x_test_noisy)

        # Tính toán các chỉ số
        acc = accuracy_score(y_test, pred)
        acc_noisy = accuracy_score(y_test, pred_noisy)
        robustness_gap = acc - acc_noisy

        # Reliability (Brier Score)
        y_test_dummies = pd.get_dummies(y_test).reindex(columns=model.classes_, fill_value=0).values
        brier = np.mean([brier_score_loss(y_test_dummies[:, k], proba[:, k]) for k in range(len(model.classes_))])

        # Fairness Gap theo Field
        unique_fields = np.unique(x_test[:, field_col_idx])
        acc_per_field = [
            accuracy_score(y_test[x_test[:, field_col_idx] == f], pred[x_test[:, field_col_idx] == f])
            for f in unique_fields
            if sum(x_test[:, field_col_idx] == f) > 0
        ]
        fairness_gap = (max(acc_per_field) - min(acc_per_field)) if acc_per_field else 0.0

        print(
            f"Trees: {n_trees:3d} | Acc: {acc:.3f} | "
            f"Robustness Gap: {robustness_gap:.3f} | Brier: {brier:.3f} | Fairness Gap: {fairness_gap:.3f}"
        )

    y_pred = model.predict(x_test)
    final_acc = accuracy_score(y_test, y_pred)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, final_acc

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
