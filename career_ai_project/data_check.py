import pandas as pd 

df = pd.read_csv("D:\Tu_duy_TTNT\career_ai_project\data\career_prediction_IT_900rows.csv")

get_career = df["Career"].nunique()
print(get_career)