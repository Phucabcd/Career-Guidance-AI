"""
train_model.py
Huấn luyện Random Forest cho Career Guidance AI — kết hợp:
- Feature engineering của bản MỚI: Field dùng OneHotEncoder (thay vì LabelEncoder
  ordinal như bản cũ), đủ 6 cột numeric (có "Professional Skills"), Skills từ
  cột CSV qua MultiLabelBinarizer.
- Vòng lặp In-Training Monitoring của bản CŨ: Robustness Gap, Brier Score
  (reliability), Fairness Gap theo Field — có sửa lại để tương thích với Field
  đã bị one-hot (không còn là 1 cột số duy nhất), và bổ sung thêm Top-3/Top-5
  Accuracy vào cùng vòng lặp.

Chạy:
    python train_model.py

Kết quả lưu vào models/:
    - rf_model.pkl          (model cuối, train trên 100% dữ liệu)
    - field_encoder.pkl     (OneHotEncoder)
    - career_encoder.pkl    (LabelEncoder)
    - skills_encoder.pkl    (MultiLabelBinarizer)
    - model_meta.pkl        (các chỉ số monitoring ở bước cuối)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

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

MONITOR_STEP = 20
MONITOR_MAX_TREES = 200
NOISE_STD = 0.5


def parse_skills_cell(value: object) -> list[str]:
    """Tách chuỗi Skills CSV 'Python, Docker, AWS' thành list token."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Đọc và kiểm tra dữ liệu đầu vào."""
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
) -> tuple[
    np.ndarray, pd.Series, np.ndarray, OneHotEncoder, LabelEncoder, MultiLabelBinarizer, int
]:
    """
    X = [Field one-hot | numeric 6 | skills multi-hot]

    Trả về thêm:
    - field_raw: giá trị Field gốc (string), cùng thứ tự hàng với X. Vì Field
      đã bị one-hot nên không thể suy ngược nhãn gốc từ chính X -> cần giữ lại
      riêng để tính Fairness Gap sau này.
    - numeric_start: vị trí cột bắt đầu của block "numeric 6" trong X, dùng để
      biết chính xác cần thêm nhiễu (noise) vào đâu khi test Robustness.
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
    field_raw = df[FIELD_COLUMN].astype(str).to_numpy()
    numeric_start = field_matrix.shape[1]

    print(f"Số Field (one-hot): {field_matrix.shape[1]}")
    print(f"Số skill vocabulary (từ CSV): {len(skills_encoder.classes_)}")
    print(f"Kích thước X: {x.shape}")
    return x, pd.Series(y), field_raw, field_encoder, career_encoder, skills_encoder, numeric_start


def train_random_forest(
    x: np.ndarray,
    y: pd.Series,
    field_raw: np.ndarray,
    numeric_start: int,
) -> dict:
    """
    Chia train/test (chỉ dùng để ĐÁNH GIÁ, không dùng để train model cuối),
    rồi huấn luyện Random Forest bằng warm_start với vòng lặp
    In-Training Monitoring, in ra sau mỗi bước:
        - Top-1 / Top-3 / Top-5 Accuracy
        - Robustness Gap (acc gốc vs acc khi thêm nhiễu Gaussian vào block
          numeric 6 cột: Professional/Communication/Problem Solving/Teamwork
          Skills, Projects, Internships)
        - Reliability qua Brier Score
        - Fairness Gap giữa các nhóm Field (so sánh trực tiếp trên tên Field
          gốc, thay vì trên cột đã one-hot)
    Trả về dict metrics ở bước cuối cùng.
    """
    x_train, x_test, y_train, y_test, field_train, field_test = train_test_split(
        x,
        y,
        field_raw,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    # Class weight thủ công để tránh UserWarning khi dùng chung warm_start + class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    # Tập test có nhiễu nhẹ, CHỈ áp vào đúng block numeric (không đụng vào
    # block Field one-hot hay block skills multi-hot)
    numeric_end = numeric_start + len(NUMERIC_FEATURE_COLUMNS)
    x_test_noisy = x_test.copy()
    x_test_noisy[:, numeric_start:numeric_end] += np.random.normal(
        0, NOISE_STD, size=x_test_noisy[:, numeric_start:numeric_end].shape
    )

    model = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1,
    )

    print("\n=== Bắt đầu In-Training Monitoring ===")
    last_metrics: dict = {}

    for n_trees in range(MONITOR_STEP, MONITOR_MAX_TREES + 1, MONITOR_STEP):
        model.set_params(n_estimators=n_trees)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)
        pred_noisy = model.predict(x_test_noisy)

        acc = accuracy_score(y_test, pred)
        acc_noisy = accuracy_score(y_test, pred_noisy)
        robustness_gap = acc - acc_noisy

        top3 = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
        top5 = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)

        y_test_dummies = (
            pd.get_dummies(y_test).reindex(columns=model.classes_, fill_value=0).values
        )
        brier = np.mean(
            [brier_score_loss(y_test_dummies[:, k], proba[:, k]) for k in range(len(model.classes_))]
        )

        unique_fields = np.unique(field_test)
        acc_per_field = [
            accuracy_score(y_test[field_test == f], pred[field_test == f])
            for f in unique_fields
            if np.sum(field_test == f) > 0
        ]
        fairness_gap = (max(acc_per_field) - min(acc_per_field)) if acc_per_field else 0.0

        last_metrics = {
            "n_trees": n_trees,
            "acc_top1": float(acc),
            "acc_top3": float(top3),
            "acc_top5": float(top5),
            "robustness_gap": float(robustness_gap),
            "brier_score": float(brier),
            "fairness_gap": float(fairness_gap),
        }

        print(
            f"Trees: {n_trees:3d} | Top1: {acc:.3f} | Top3: {top3:.3f} | Top5: {top5:.3f} | "
            f"Robustness Gap: {robustness_gap:.3f} | Brier: {brier:.3f} | Fairness Gap: {fairness_gap:.3f}"
        )

    print("\nClassification report (test set, bước cuối):")
    print(classification_report(y_test, model.predict(x_test), zero_division=0))

    return last_metrics


def train_final_model(x: np.ndarray, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    """
    Huấn luyện model CUỐI CÙNG (để deploy) trên 100% dữ liệu — model dùng cho
    monitoring ở trên chỉ train trên 80% (train split) nên không dùng để lưu.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


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
    print("  - rf_model.pkl")
    print("  - field_encoder.pkl")
    print("  - career_encoder.pkl")
    print("  - skills_encoder.pkl")
    print("  - model_meta.pkl")


def main() -> None:
    print("=== Career Guidance AI — Train Random Forest (+ Monitoring) ===")
    print(f"Đọc dữ liệu từ: {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"Số mẫu sau làm sạch: {len(df)}")
    print(f"Số nghề (Career): {df[TARGET_COLUMN].nunique()}")
    print(f"Số Field: {df[FIELD_COLUMN].nunique()}")

    skill_lists = prepare_skill_lists(df)
    empty = sum(1 for s in skill_lists if not s)
    avg_len = float(np.mean([len(s) for s in skill_lists])) if skill_lists else 0.0
    print(f"Skills trống: {empty} | avg skills/row: {avg_len:.1f}")

    x, y, field_raw, field_encoder, career_encoder, skills_encoder, numeric_start = encode_features(
        df, skill_lists
    )

    meta = train_random_forest(x, y, field_raw, numeric_start)

    print("\n=== Metrics ở bước cuối (n_estimators lớn nhất) ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\nHuấn luyện model cuối trên toàn bộ dữ liệu để deploy...")
    final_model = train_final_model(x, y, n_estimators=MONITOR_MAX_TREES)

    save_artifacts(final_model, field_encoder, career_encoder, skills_encoder, meta, MODELS_DIR)
    print("\nHoàn tất huấn luyện.")


if __name__ == "__main__":
    main()