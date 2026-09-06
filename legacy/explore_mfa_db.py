import sqlite3
import pandas as pd
from pathlib import Path
import ast
import re

def inspect_mfa_db(db_path: str):
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n🔍 Inspecting MFA database: {db_path}\n")

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    if not tables:
        print("⚠️ No tables found. Is this even an MFA DB?")
        return

    print("📋 Tables found:")
    for t in tables:
        print(f"  - {t}")
    print("\n")

    # Show schema + preview for each table
    for table in tables:
        print(f"=== 🧱 {table} ===")
        # Schema
        cursor.execute(f"PRAGMA table_info({table});")
        schema = cursor.fetchall()
        print("Columns:")
        for col in schema:
            print(f"  - {col[1]} ({col[2]})")
        print()

        # Preview first 5 rows
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5;", conn)
            print(df)
        except Exception as e:
            print(f"⚠️ Couldn't preview {table}: {e}")
        print("-" * 60 + "\n")

    conn.close()

def safe_eval_words(raw_str, debug = False):
    """Convert the weird array(...) syntax into a JSON-like string and safely eval it."""
    if not isinstance(raw_str, str):
        return []
    # Replace numpy array() syntax with Python lists
    cleaned = re.sub(r"\}\s*\{", r"}, {", raw_str)
    cleaned = re.sub(r"array\((\[.*?\])(?:,\s*dtype=.*?)?\)", r"\1", cleaned)

    try:
        val = ast.literal_eval(cleaned)
        if isinstance(val, list):
            return val
        return []
    except Exception as e:
        return []

def build_phone_alignment_csv(db_path: str, dataset_csv: str, output_csv: str):
    conn = sqlite3.connect(db_path)

    print("🔍 Loading MFA tables...")
    phone_intervals = pd.read_sql_query("SELECT * FROM phone_interval", conn)
    phones = pd.read_sql_query("SELECT id AS phone_id, phone FROM phone", conn)
    utterances = pd.read_sql_query("SELECT id AS utterance_id, file_id FROM utterance", conn)
    files = pd.read_sql_query("SELECT id AS file_id, name AS file_name FROM file", conn)
    word_intervals = pd.read_sql_query("SELECT id AS word_interval_id, word_id FROM word_interval", conn)
    words = pd.read_sql_query("SELECT id AS word_id, word FROM word", conn)
    conn.close()

    # Join everything
    df = (
        phone_intervals
        .merge(phones, on="phone_id", how="left")
        .merge(word_intervals, on="word_interval_id", how="left")
        .merge(words, on="word_id", how="left")
        .merge(utterances, on="utterance_id", how="left")
        .merge(files, on="file_id", how="left")
    )

    df = df[[
        "file_name", "id", "phone", "begin", "end",
        "phone_goodness", "word"
    ]].rename(columns={"phone_goodness": "phone_goodness_mfa", 'id': 'phone_interval_id'})

    print("✅ Loaded MFA alignments:", len(df), "phone intervals")

    # Load dataset
    print("🔍 Loading dataset with human annotations...")
    data = pd.read_csv(dataset_csv, dtype={'id': object})
    if "words" not in data.columns:
        raise ValueError("Dataset must include a 'words' column with phone-level scores")

    phone_prof_map = {}
    for i, row in data.iterrows():
        file_id = str(row.get("id") or row.get("file_name"))
        if i == 1:
            print(file_id)
        parsed = safe_eval_words(row["words"], debug = bool(i == 1))
        if i == 1:
            print(parsed)
        phone_scores = []
        for w in parsed:
            phones = w.get("phones", [])
            scores = w.get("phones-accuracy", [])
            text = w.get("text", "")
            if (i == 1):
                print(phones, scores, text)
            for p, s in zip(phones, scores):
                phone_scores.append((p, float(s), text))
        phone_prof_map[file_id] = phone_scores

    print(f"✅ Built phone_prof_map for {len(phone_prof_map)} files")

    # Match MFA phones to prof phones — align by label, not by order
    prof_scores, prof_words = [], []
    for _, row in df.iterrows():
        fname = str(row["file_name"])
        ph = str(row["phone"]).upper()
        entries = phone_prof_map.get(fname, [])
        match = next(((s, w) for (p, s, w) in entries if p.upper() == ph), (None, None))
        prof_scores.append(match[0])

    df["phone_goodness_prof"] = prof_scores

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Saved merged CSV to: {output_csv}")
    print(f"Columns: {list(df.columns)} | Rows: {len(df)}")

def correlation ():
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv('data/joined.csv')
    df['gop'] = df['phone_goodness_mfa'] / (df['end'] - df['begin'])

    sns.regplot(
        data=df,
        x="gop",
        y="phone_goodness_prof",
        scatter_kws={"alpha":0.3},
        line_kws={"color":"red"}
    )
    plt.title("Correlation between MFA log-score and Professional Phone Goodness")
    plt.show()

if __name__ == "__main__":
    correlation()