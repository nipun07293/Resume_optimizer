import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def train_and_save():
    print("\nSTEP 1: Initializing Enterprise Training Pipeline...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "resume_data.csv")
    save_path = os.path.join(current_dir, "saved_resume_model.keras")

    if not os.path.exists(data_path):
        print(f"ERROR: Dataset NOT FOUND at {data_path}!")
        return

    print("STEP 2: Loading & Cleaning dataset...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['matched_score'])
    
    if len(df) == 0:
        print("ERROR: 0 valid rows of data to train on!")
        return
    
    text_cols = ['skills', 'responsibilities', 'career_objective', 'skills_required', 'responsibilities.1']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    df['resume_text'] = df['skills'] + " " + df['responsibilities'] + " " + df['career_objective']
    df['job_text'] = df['skills_required'] + " " + df.get('responsibilities.1', '')

    df['resume_text'] = df['resume_text'].str.replace(r'\W+', ' ', regex=True).str.lower()
    df['job_text'] = df['job_text'].str.replace(r'\W+', ' ', regex=True).str.lower()

    print("\nSTEP 3: Loading AI Transformer Model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\nSTEP 4: Encoding Resumes & Jobs...")
    res_emb = encoder.encode(df['resume_text'].tolist(), show_progress_bar=True)
    job_emb = encoder.encode(df['job_text'].tolist(), show_progress_bar=True)
    
    print("\nSTEP 4.5: Generating Interaction Features (Absolute Diff & Product)...")
    diff_emb = np.abs(res_emb - job_emb)
    prod_emb = res_emb * job_emb
    
    X = np.hstack((res_emb, job_emb, diff_emb, prod_emb))
    y = df['matched_score'].values

    print(f"\nSTEP 5: Training Deep Neural Network (Input dimension: {X.shape[1]})...")
    model = build_model(X.shape[1])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12, restore_best_weights=True)
    
    model.fit(X, y, epochs=150, batch_size=32, callbacks=[early_stop], verbose=1)

    print("\nSTEP 6: Saving the Model...")
    model.save(save_path)
    print(f"SUCCESS! Advanced Model trained and saved to: {save_path}")

if __name__ == "__main__":
    train_and_save()