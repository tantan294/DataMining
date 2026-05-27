# src/classifier.py
"""
Module phan lop Neural Network (TensorFlow/Keras):
- MLP va Deep Dense NN
- Phan loai do kho: Easy / Medium / Hard
- Danh gia Accuracy, Precision, Recall, F1
- Ve confusion matrix va learning curves
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

# ---- Constants ----
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
IMG_DIR   = os.path.join(BASE_DIR, 'report', 'images')

def build_mlp(input_dim: int, n_classes: int = 3):
    """Mo hinh MLP don gian."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ], name='MLP')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_deep_nn(input_dim: int, n_classes: int = 3):
    """Mo hinh Deep Dense NN."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ], name='DeepDenseNN')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val,
                epochs: int = 50, batch_size: int = 256):
    """Huan luyen model."""
    import tensorflow as tf
    from tensorflow import keras
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=4, factor=0.5, min_lr=1e-6, monitor='val_loss'
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, label_names=None):
    """Danh gia mo hinh."""
    if label_names is None:
        label_names = ['Easy', 'Medium', 'Hard']
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return {
        'accuracy': acc, 'f1': f1,
        'confusion_matrix': cm, 'report': report,
        'y_pred': y_pred, 'y_proba': y_pred_proba
    }

def run_classification(tfidf_matrix, df):
    """Pipeline phan lop hoan chinh."""
    import tensorflow as tf
    from scipy.sparse import issparse
    
    print(f"\n[NN] Starting classification pipeline...")
    print(f"  Dataset: {len(df)} samples")
    print(f"  Labels: {df['difficulty'].value_counts().to_dict()}")
    
    # Chuan bi data
    if issparse(tfidf_matrix):
        X = tfidf_matrix.toarray().astype(np.float32)
    else:
        X = tfidf_matrix.astype(np.float32)
    
    y = df['difficulty_encoded'].values.astype(np.int32)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    input_dim = X_train.shape[1]
    
    results = {}
    histories = {}
    
    # Model 1: MLP
    print("\n--- Model 1: MLP ---")
    mlp = build_mlp(input_dim)
    mlp.summary()
    hist_mlp = train_model(mlp, X_train, y_train, X_val, y_val, epochs=50)
    results['MLP'] = evaluate_model(mlp, X_test, y_test)
    histories['MLP'] = hist_mlp
    mlp.save(os.path.join(MODEL_DIR, 'mlp_model.keras'))
    
    # Model 2: Deep NN
    print("\n--- Model 2: Deep Dense NN ---")
    dnn = build_deep_nn(input_dim)
    dnn.summary()
    hist_dnn = train_model(dnn, X_train, y_train, X_val, y_val, epochs=50)
    results['DeepNN'] = evaluate_model(dnn, X_test, y_test)
    histories['DeepNN'] = hist_dnn
    dnn.save(os.path.join(MODEL_DIR, 'deepnn_model.keras'))
    
    # Chon model tot nhat
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n[NN] Best model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")
    
    # Save best model label
    import json
    meta = {
        'best_model': best_name,
        'input_dim': int(input_dim),
        'label_map': {'Easy': 0, 'Medium': 1, 'Hard': 2}
    }
    with open(os.path.join(MODEL_DIR, 'nn_metadata.json'), 'w') as f:
        json.dump(meta, f)
    
    return results, histories, (X_train, X_val, X_test, y_train, y_val, y_test), best_name


if __name__ == '__main__':
    from scipy.sparse import load_npz
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    tfidf = load_npz(os.path.join(BASE_DIR, 'data', 'processed', 'tfidf_matrix.npz'))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'unified_data.csv'))
    run_classification(tfidf, df)
