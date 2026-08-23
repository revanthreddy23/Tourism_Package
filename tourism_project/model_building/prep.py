import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv")

# 2. Remove unnecessary unique identifier or artifact index columns
cols_to_drop = ["CustomerID", "Unnamed: 0"]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 3. Clean categorical text data (strip leading/trailing whitespaces)
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].str.strip()

# 4. Fix Gender typos ('Fe Male' -> 'Female')
if "Gender" in df.columns:
    gender_mapping = {"Fe Male": "Female", "female": "Female", "male": "Male"}
    df["Gender"] = df["Gender"].replace(gender_mapping)

# 5. Harmonize MaritalStatus redundancies ('Unmarried' -> 'Single')
if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# 6. Define features (X) and target variable (y)
target_col = "ProdTaken"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

X = df.drop(columns=[target_col])
y = df[target_col]

# 7. Stratified train-test split (keeps the imbalanced ProdTaken ratio consistent)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Save split files locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Training samples: {Xtrain.shape[0]}, Testing samples: {Xtest.shape[0]}")
