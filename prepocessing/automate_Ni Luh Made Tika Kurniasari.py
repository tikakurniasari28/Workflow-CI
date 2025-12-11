"""
Automated Preprocessing Script
Author : Ni Luh Made Tika Kurniasari
Description :
    Script ini bertugas melakukan preprocessing otomatis 
    sesuai tahapan yang dibuat saat eksperimen manual.
    Output: dataset hasil preprocessing siap untuk training.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(path="train.csv"):
    """Membaca dataset mentah."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {path}")
    df = pd.read_csv(path)
    print("Dataset berhasil dimuat.")
    return df


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess_data(df):
    """Melakukan preprocessing lengkap sesuai eksperimen manual."""

    # --- Menghapus kolom yang tidak diperlukan ---
    drop_cols = ["Name", "Ticket", "Cabin"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # --- Handle Missing Values ---
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # --- Encoding kategorikal ---
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # --- Handle missing lainnya ---
    df = df.fillna(0)

    # --- Scaling fitur numerik ---
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Preprocessing selesai.")

    return df


# ============================================================
# 3. SIMPAN DATA HASIL
# ============================================================

def save_processed_data(df, output_path="prepocessing/titanic_processed.csv"):
    """Menyimpan dataset hasil preprocessing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Hasil preprocessing berhasil disimpan di: {output_path}")


# ============================================================
# 4. MAIN EXECUTION (untuk GitHub Actions)
# ============================================================

if __name__ == "__main__":
    print("=== AUTOMATED PREPROCESSING STARTED ===")

    df_raw = load_dataset()
    df_processed = preprocess_data(df_raw)
    save_processed_data(df_processed)

    print("=== AUTOMATED PREPROCESSING FINISHED ===")
