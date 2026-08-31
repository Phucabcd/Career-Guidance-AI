""" 
Model: Random Forest
Features: Field (OneHot) + soft/numeric scores + Skills cột CSV (MultiLabelBinarizer) 
"""

from __future__ import annotations 
from pathlib import Path

import joblib
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight
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


def evaluate_holdout(x: np.ndarray, y: pd.Series, field_encoder: OneHotEncoder, df: pd.DataFrame) -> dict:
    """Holdout 20% với In-Training Monitoring cho Reliability, Robustness & Fairness"""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    
    # 1. Chuẩn bị tập Dữ liệu Nhiễu (Noisy Data) để test Robustness
    # Numeric features nằm ngay sau phần One-hot của Field
    num_fields = len(field_encoder.categories_[0])
    x_test_noisy = x_test.copy()
    # Thêm nhiễu Gaussian nhẹ (mean=0, std=0.5) vào 6 cột kỹ năng mềm
    noise = np.random.normal(0, 0.5, size=x_test_noisy[:, num_fields:num_fields+6].shape)
    x_test_noisy[:, num_fields:num_fields+6] += noise

    # Tính toán trước trọng số chuẩn cho toàn bộ tập y_train
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    # 2. Cấu hình Model với warm_start=True
    model = RandomForestClassifier(
        n_estimators=0,          # Bắt đầu với 0 cây
        warm_start=True,         # Giữ lại rừng cũ khi gọi fit() lần sau
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1
    )

    max_estimators = RF_PARAMS["n_estimators"]
    step = 20 # Cứ mỗi 20 cây sẽ test 1 lần
    
    best_acc = 0
    patience_counter = 0
    max_patience = 3 # Dừng sớm nếu Robustness giảm liên tục 3 lần
    
    print("\n=== Bắt đầu Training Loop (Monitoring Fairness, Robustness, Reliability) ===")
    
    for i in range(step, max_estimators + step, step):
        model.set_params(n_estimators=i)
        model.fit(x_train, y_train) # Chỉ train thêm 20 cây mới
        
        # Dự đoán trên tập chuẩn và tập nhiễu
        proba = model.predict_proba(x_test)
        pred = model.predict(x_test)
        pred_noisy = model.predict(x_test_noisy)
        
        # --- TÍNH TOÁN CÁC CHỈ SỐ AI AN TOÀN ---
        
        # A. General Metrics
        acc = accuracy_score(y_test, pred)
        
        # B. Robustness (Độ sụt giảm Accuracy khi dữ liệu đầu vào bị nhiễu)
        acc_noisy = accuracy_score(y_test, pred_noisy)
        robustness_gap = acc - acc_noisy
        
        # C. Reliability / Calibration (Mô hình có "tự tin thái quá" khi đoán sai không?)
        # Brier score càng thấp càng tốt (phản ánh độ tin cậy của xác suất dự đoán)
        y_test_dummies = pd.get_dummies(y_test).reindex(columns=model.classes_, fill_value=0).values
        brier = np.mean([brier_score_loss(y_test_dummies[:, k], proba[:, k]) for k in range(len(model.classes_))])
        
        # D. Fairness Check (Demographic Parity cơ bản qua cột Field)
        # Kiểm tra tỷ lệ mô hình dự đoán đúng giữa các nhóm Field khác nhau có bị lệch không
        field_idx = np.argmax(x_test[:, :num_fields], axis=1) # Lấy lại ID của Field từ one-hot
        unique_fields = np.unique(field_idx)
        acc_per_field = [accuracy_score(y_test.iloc[field_idx == f], pred[field_idx == f]) for f in unique_fields]
        fairness_gap = np.max(acc_per_field) - np.min(acc_per_field) if len(acc_per_field) > 0 else 0

        print(f"Trees: {i:3d} | Acc: {acc:.3f} | Robustness Gap: {robustness_gap:.3f} | Brier: {brier:.3f} | Fairness Gap: {fairness_gap:.3f}")
        
        # --- EARLY STOPPING LOGIC (Dựa trên Robustness) ---
        # Nếu mô hình bắt đầu học thuộc lòng (overfit tập sạch) -> Robustness Gap sẽ tăng mạnh
        if robustness_gap > 0.05: # Ví dụ: Sai số giữa tập sạch và tập nhiễu vượt 5%
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f">>> CẢNH BÁO: Kích hoạt Early Stopping ở cụm {i} cây do Robustness giảm mạnh (Overfitting).")
                break
        else:
            patience_counter = 0 # Reset nếu model ổn định trở lại
            
        # (Tùy chọn) Lưu lại best_model theo Acc hoặc tổng hợp các tiêu chí
        if acc > best_acc:
            best_acc = acc
            best_model_weights = copy.deepcopy(model)

    print("\n=== Hoàn tất Training Loop ===")
    
    # Sử dụng model tốt nhất để đánh giá cuối cùng
    model = best_model_weights if 'best_model_weights' in locals() else model
    final_proba = model.predict_proba(x_test)
    
    metrics = {
        "rf_top1": float(accuracy_score(y_test, model.predict(x_test))),
        "rf_top3": float(top_k_accuracy_score(y_test, final_proba, k=3, labels=model.classes_)),
        "rf_top5": float(top_k_accuracy_score(y_test, final_proba, k=5, labels=model.classes_)),
    }
    
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
    meta = evaluate_holdout(x, y, field_encoder, df)

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
