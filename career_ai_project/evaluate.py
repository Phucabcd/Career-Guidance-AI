import numpy as np

from model import call_gemini_extractor, load_ml_artifacts, build_extractor_prompt, predict_top_careers

#test Reliability call gemini 
def test_reliability_calll_gemini(n_runs: int = 5):
    # Load the ML artifacts
    
    model, fields, careers, skills = load_ml_artifacts()
    
    list_of_all_fields = list(fields.classes_)
    list_of_all_careers = list(careers.classes_)
    list_of_all_skills = list(skills.classes_)


    # Sample input data for testing
    sample_input = {
        "Em là sinh viên năm cuối ngành Khoa học dữ liệu với GPA 3.5.?": "",
        "Em sử dụng thành thạo Python, Pandas, NumPy và SQL.": "",
        "Em đã thực hiện nhiều bài toán phân tích dữ liệu và xây dựng mô hình dự đoán bằng scikit-learn.": "",
        "Em thích khám phá dữ liệu, trực quan hóa bằng Power BI và Tableau.": "",
        "Em có tư duy phân tích tốt, cẩn thận và yêu thích giải quyết các bài toán thực tế từ dữ liệu.": ""
    }

    results = []
    for i in range(n_runs):
        data = call_gemini_extractor(
            sample_input,
            list_of_all_careers,
            list_of_all_fields,
            list_of_all_skills,
        )
        results.append({
            "run": i + 1,
            "Field": data.get("Field"),
            "Coding Skills": data.get("Coding Skills"),
            "Communication Skills": data.get("Communication Skills"),
            "Problem Solving Skills": data.get("Problem Solving Skills"),
            "Teamwork Skills": data.get("Teamwork Skills"),
        })
        print(f"Run {i+1}: {results[-1]}")

    return results

    
#test Bias of gemini
def test_bias_of_gemini():
    return ""
   
    
#test Robustness user input
def test_robustness_user_input():
    return ""


if __name__ == "__main__":
    test_reliability_calll_gemini(n_runs=5)