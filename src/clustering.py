# src/clustering.py
"""
Module phan cum EM (GaussianMixture):
- Giam chieu TF-IDF bang TruncatedSVD
- Thu nhieu gia tri k
- Danh gia Silhouette, BIC, AIC
- Chon k toi uu tu dong
- Luu model va ket qua
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import issparse, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ---- Constants ----
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
IMG_DIR   = os.path.join(BASE_DIR, 'report', 'images')

for d in [MODEL_DIR, IMG_DIR]:
    os.makedirs(d, exist_ok=True)

def reduce_dimensions(tfidf_matrix, n_components: int = 100, save_path: str = None):
    """Giam chieu bang TruncatedSVD (LSA)."""
    print(f"[SVD] Reducing to {n_components} components...")
    if issparse(tfidf_matrix):
        X = tfidf_matrix
    else:
        X = tfidf_matrix
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    X_reduced = normalize(X_reduced)
    
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    
    if save_path:
        joblib.dump(svd, save_path)
    
    return X_reduced, svd

def evaluate_em(X_reduced, k_values: list = None):
    """
    Thu nhieu gia tri k, tinh Silhouette, BIC, AIC.
    Returns DataFrame ket qua va best_k.
    """
    if k_values is None:
        k_values = [3, 5, 7, 8, 10, 12, 15]
    
    print(f"[EM] Evaluating k = {k_values}...")
    results = []
    
    for k in k_values:
        print(f"  k={k}...", end=' ')
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='diag',
                max_iter=200,
                n_init=3,
                random_state=42
            )
            gmm.fit(X_reduced)
            labels = gmm.predict(X_reduced)
            
            bic  = gmm.bic(X_reduced)
            aic  = gmm.aic(X_reduced)
            
            # Silhouette on subset if large
            if len(X_reduced) > 5000:
                idx = np.random.choice(len(X_reduced), 5000, replace=False)
                sil = silhouette_score(X_reduced[idx], labels[idx])
            else:
                sil = silhouette_score(X_reduced, labels)
            
            print(f"BIC={bic:.1f} AIC={aic:.1f} Sil={sil:.4f}")
            results.append({'k': k, 'bic': bic, 'aic': aic, 'silhouette': sil, 'model': gmm})
        except Exception as e:
            print(f"FAILED: {e}")
    
    results_df = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != 'model'} for r in results])
    
    # Chon k toi uu: nho BIC nhat
    best_idx = results_df['bic'].idxmin()
    best_k   = int(results_df.loc[best_idx, 'k'])
    best_gmm = results[best_idx]['model']
    
    print(f"\n[EM] Best k={best_k} (BIC={results_df.loc[best_idx,'bic']:.1f}, "
          f"Silhouette={results_df.loc[best_idx,'silhouette']:.4f})")
    
    return results_df, best_k, best_gmm

def get_cluster_keywords(tfidf_matrix, labels, vectorizer, top_n: int = 15):
    """Lay top keywords moi cum tu TF-IDF."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    n_clusters = len(set(labels))
    cluster_keywords = {}
    
    if issparse(tfidf_matrix):
        mat = tfidf_matrix.toarray()
    else:
        mat = tfidf_matrix
    
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            cluster_keywords[c] = []
            continue
        cluster_mean = mat[mask].mean(axis=0)
        top_idx = cluster_mean.argsort()[::-1][:top_n]
        cluster_keywords[c] = feature_names[top_idx].tolist()
    
    return cluster_keywords

def train_final_em(X_reduced, best_k: int):
    """Train mo hinh GMM cuoi cung voi k toi uu."""
    print(f"[EM] Training final model with k={best_k}...")
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type='diag',
        max_iter=300,
        n_init=5,
        random_state=42
    )
    gmm.fit(X_reduced)
    labels = gmm.predict(X_reduced)
    proba  = gmm.predict_proba(X_reduced)
    return gmm, labels, proba

def run_clustering(tfidf_matrix, vectorizer, df=None):
    """Pipeline phan cum hoan chinh."""
    # Reduce dimensions
    svd_path = os.path.join(MODEL_DIR, 'svd_model.pkl')
    X_reduced, svd = reduce_dimensions(tfidf_matrix, n_components=100, save_path=svd_path)
    
    # Evaluate
    results_df, best_k, _ = evaluate_em(X_reduced)
    
    # Train final
    gmm, labels, proba = train_final_em(X_reduced, best_k)
    
    # Save models
    joblib.dump(gmm, os.path.join(MODEL_DIR, 'gmm_model.pkl'))
    np.save(os.path.join(PROC_DIR, 'cluster_labels.npy'), labels)
    np.save(os.path.join(PROC_DIR, 'X_reduced.npy'), X_reduced)
    np.save(os.path.join(PROC_DIR, 'cluster_proba.npy'), proba)
    
    # Get keywords
    keywords = get_cluster_keywords(tfidf_matrix, labels, vectorizer)
    
    # Print summary
    print("\n[Cluster Summary]")
    unique, counts = np.unique(labels, return_counts=True)
    for c, cnt in zip(unique, counts):
        kws = ', '.join(keywords.get(c, [])[:8])
        print(f"  Cluster {c:2d} ({cnt:5d} items): {kws}")
    
    return gmm, svd, labels, proba, X_reduced, keywords, results_df, best_k


if __name__ == '__main__':
    from scipy.sparse import load_npz
    tfidf = load_npz(os.path.join(PROC_DIR, 'tfidf_matrix.npz'))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    run_clustering(tfidf, vectorizer)
