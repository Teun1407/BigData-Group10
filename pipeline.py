import pandas as pd
import numpy as np
import duckdb
import unicodedata
import re
from textblob import TextBlob

def preprocess_imdb_data(data_path, directors_path, writers_path):
    """
    General preprocessing pipeline for IMDB data.
    
    Arguments:
    - data_path: Path to the train/test/validation data CSV file.
    - directors_path: Path to the directing.json file.
    - writers_path: Path to the writing.json file.
    
    Returns:
    - Cleaned Pandas DataFrame ready for model training or prediction.
    """
    
    # Step 1: Load main dataset
    df = pd.read_csv(data_path)

    # Step 2: Load JSON files (Directors & Writers)
    df_directors = pd.read_json(directors_path)
    df_writers = pd.read_json(writers_path)

    # Step 3: Rename columns for consistency
    df_directors.rename(columns={"movie": "tconst", "director": "director_id"}, inplace=True)
    df_writers.rename(columns={"movie": "tconst", "writer": "writer_id"}, inplace=True)

    # Step 4: Convert nested JSON fields into strings
    df_directors["director_id"] = df_directors["director_id"].astype(str)
    df_writers["writer_id"] = df_writers["writer_id"].astype(str)

    # Step 5: Merge main dataset with Directors & Writers using DuckDB
    con = duckdb.connect()
    con.register("movies", df)
    con.register("directors", df_directors)
    con.register("writers", df_writers)

    query = """
    SELECT 
        movies.*, 
        directors.director_id, 
        writers.writer_id
    FROM movies
    LEFT JOIN directors ON movies.tconst = directors.tconst
    LEFT JOIN writers ON movies.tconst = writers.tconst
    """

    df = con.execute(query).fetchdf()
    con.close()

    # Step 6: Create column Year from startYear and endYear
    df['startYear'] = df['startYear'].replace('\\N', np.nan).astype(float)
    df['endYear'] = df['endYear'].replace('\\N', np.nan).astype(float)
    df['Year'] = df['startYear'].fillna(df['endYear'])

    # Step 7: Clean title names
    def normalize_text(text):
        if pd.isna(text):  
            return ""
        text = str(text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')  # Remove accents
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text.strip()

    def clean_titles(row):
        primary = row['primaryTitle'] if pd.notna(row['primaryTitle']) else ''
        original = row['originalTitle'] if pd.notna(row['originalTitle']) else ''

        if not primary:
            primary = original

        cleaned_title = normalize_text(primary)

        return cleaned_title if cleaned_title else "Unknown Title"

    df['primaryTitle'] = df.apply(clean_titles, axis=1)
    df.rename(columns={'primaryTitle': 'movieTitle'}, inplace=True)

    # Step 8: Compute Title Uniqueness Score
    title_counts = df["movieTitle"].value_counts()
    df["title_uniqueness"] = df["movieTitle"].apply(lambda x: 1 / title_counts[x] if title_counts[x] > 1 else 1)

    # Step 9: Compute Sentiment Score
    df["sentiment_score"] = df["movieTitle"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

    # Step 10: Count words in each title
    df["word_count"] = df["movieTitle"].apply(lambda x: len(x.split()))

    # Step 11: Compute title word length standard deviation
    df["title_word_length_std"] = df["movieTitle"].apply(lambda x: np.std([len(word) for word in x.split()]) if len(x.split()) > 1 else 0)
    
    # Step 12: Drop unnecessary columns
    columns_to_drop = ["originalTitle", "endYear", "startYear", "Unnamed: 0"]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Step 13: Handle missing values
    numeric_columns = ["runtimeMinutes", "numVotes"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")  
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())  

    df["director_id"] = df["director_id"].fillna("unknown")
    df["writer_id"] = df["writer_id"].fillna("unknown")

    # Step 14: Ensure correct data types
    df["Year"] = df["Year"].astype(int)
    df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
    df["numVotes"] = df["numVotes"].astype(int)

    # Step 15: Ensure each tconst is unique
    df = df.groupby("tconst").first().reset_index()
    
    return df
