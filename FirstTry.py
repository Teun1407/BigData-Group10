import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# Function to create the decade feature
def extract_decade(X):
    X = X.copy()
    X['decade'] = (X['startYear'] // 10) * 10  # Group by decades
    return X[['decade']]

# Define preprocessing for numerical features
num_features = ["startYear", "endYear", "runtimeMinutes", "numVotes"]
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values
    ("scaler", StandardScaler())  # Scale features
])

# Define decade transformation
decade_transformer = FunctionTransformer(extract_decade)

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("decade", decade_transformer, ["startYear"])  # Add new decade feature
])

# Full preprocessing pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

# Load dataset
df = pd.read_csv("C:/Users/Gebruiker/Documents/UVA/Vakken met code/BD/BigData-Group10/imdb/train-3.csv")

# Convert '\N' to NaN
df.replace("\\N", np.nan, inplace=True)

# Apply pipeline (excluding categorical columns for now)
df_transformed = pipeline.fit_transform(df)

# Convert back to DataFrame
processed_df = pd.DataFrame(df_transformed, columns=[
    "startYear", "endYear", "runtimeMinutes", "numVotes", "decade"
])

# Display processed data
print(processed_df.head())

