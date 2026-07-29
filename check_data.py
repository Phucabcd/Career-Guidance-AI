import pandas as pd

jobs = pd.read_csv("csv/career_path_in_all_field.csv")

print("Số dòng, số cột:", jobs.shape)
print()
print("Số lượng missing value mỗi cột:")
print(jobs.isnull().sum())
print()
print("Số dòng bị trùng lặp hoàn toàn:", jobs.duplicated().sum())