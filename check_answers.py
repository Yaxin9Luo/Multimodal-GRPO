import pandas as pd

# Load dataset
df = pd.read_parquet('ViRL39K/39Krelease.parquet')

# Print sample data
print("Sample questions and answers:")
for i in range(10):
    print(f"Question {i}: {df.iloc[i]['question']}")
    print(f"Answer {i}: {df.iloc[i]['answer']}")
    print("-" * 50) 