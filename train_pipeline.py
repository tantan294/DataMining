#!/usr/bin/env python3
"""
train_pipeline.py - Script chay toan bo pipeline:
1. Tien xu ly du lieu
2. Phan cum EM
3. Huan luyen Neural Network
4. Luat ket hop
5. Sinh bieu do va bao cao
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---- Setup paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

KHAN_FILE     = os.path.join(BASE_DIR, 'youtube_khan_academy.csv')
COURSERA_FILE = os.path.join(BASE_DIR, 'courses_en.csv')
IMG_DIR       = os.path.join(BASE_DIR, 'report', 'images')
PROC_DIR      = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR     = os.path.join(BASE_DIR, 'models')

for d in [IMG_DIR, PROC_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Style ----
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
          '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

def save_fig(name: str, dpi: int = 150):
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [Saved] {path}")

# ==============================================================================
# STEP 1: PREPROCESSING
# ==============================================================================
def step1_preprocessing():
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING")
    print("="*70)
    
    from src.preprocessing import run_preprocessing
    
    df, tfidf_matrix, vectorizer, label_map = run_preprocessing(
        khan_file=KHAN_FILE,
        coursera_file=COURSERA_FILE,
        khan_sample=20000,
        coursera_sample=15000,
        tfidf_features=5000
    )
    return df, tfidf_matrix, vectorizer

# ==============================================================================
# STEP 2: DATA EXPLORATION (Bieu do khao sat)
# ==============================================================================
def step2_exploration(df: pd.DataFrame):
    print("\n" + "="*70)
    print("STEP 2: DATA EXPLORATION - Generating Charts")
    print("="*70)
    
    # Load raw data for full stats
    khan_raw = pd.read_csv(KHAN_FILE, low_memory=False, nrows=52000)
    coursera_raw = pd.read_csv(COURSERA_FILE, low_memory=False)
    
    # ---- 2.1 Dataset Overview ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
    
    # Records by source
    src_counts = df['source'].value_counts()
    axes[0].bar(src_counts.index, src_counts.values, color=COLORS[:2])
    axes[0].set_title('Records by Source')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(src_counts.values):
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontsize=11)
    
    # Difficulty distribution
    diff_counts = df['difficulty'].value_counts()
    wedge_props = {'linewidth': 2, 'edgecolor': 'white'}
    axes[1].pie(diff_counts.values, labels=diff_counts.index,
                autopct='%1.1f%%', colors=COLORS[:3], wedgeprops=wedge_props)
    axes[1].set_title('Difficulty Distribution')
    
    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        axes[2].barh(missing.index, missing.values, color=COLORS[3])
        axes[2].set_title('Missing Values')
    else:
        axes[2].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=14, color='green')
        axes[2].set_title('Missing Values')
    
    plt.tight_layout()
    save_fig('01_dataset_overview.png')
    
    # ---- 2.2 Khan Academy: View Count Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Khan Academy Video Statistics', fontsize=16, fontweight='bold')
    
    khan_df = df[df['source'] == 'khan'].copy()
    
    # View count histogram
    vc = khan_raw['view_count'].dropna()
    vc = vc[vc < vc.quantile(0.95)]  # Remove outliers
    axes[0].hist(vc, bins=50, color=COLORS[0], alpha=0.8, edgecolor='white')
    axes[0].set_title('View Count Distribution (Khan Academy)')
    axes[0].set_xlabel('View Count')
    axes[0].set_ylabel('Frequency')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Like count
    lc = khan_raw['like_count'].dropna()
    lc = lc[lc < lc.quantile(0.95)]
    axes[1].hist(lc, bins=50, color=COLORS[1], alpha=0.8, edgecolor='white')
    axes[1].set_title('Like Count Distribution (Khan Academy)')
    axes[1].set_xlabel('Like Count')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    save_fig('02_khan_stats.png')
    
    # ---- 2.3 Coursera Category Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Coursera Course Distribution', fontsize=16, fontweight='bold')
    
    # Category bar chart
    cat_counts = coursera_raw['category'].value_counts().head(15)
    axes[0].barh(cat_counts.index[::-1], cat_counts.values[::-1], color=COLORS)
    axes[0].set_title('Top 15 Coursera Categories')
    axes[0].set_xlabel('Number of Courses')
    
    # Language distribution (if available)
    if 'language' in coursera_raw.columns:
        lang_counts = coursera_raw['language'].value_counts().head(10)
        axes[1].bar(lang_counts.index, lang_counts.values, color=COLORS)
        axes[1].set_title('Course Language Distribution')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        # Skills word count distribution
        coursera_raw['skills_count'] = coursera_raw['skills'].fillna('').apply(
            lambda x: len(str(x).split(','))
        )
        axes[1].hist(coursera_raw['skills_count'], bins=30, color=COLORS[2], alpha=0.8)
        axes[1].set_title('Number of Skills per Course')
        axes[1].set_xlabel('Skills Count')
    
    plt.tight_layout()
    save_fig('03_coursera_distribution.png')
    
    # ---- 2.4 Difficulty by Source ----
    fig, ax = plt.subplots(figsize=(10, 6))
    diff_by_source = pd.crosstab(df['source'], df['difficulty'])
    diff_by_source.plot(kind='bar', ax=ax, color=COLORS[:3])
    ax.set_title('Difficulty Distribution by Source', fontsize=14, fontweight='bold')
    ax.set_xlabel('Source')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Difficulty')
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.tight_layout()
    save_fig('04_difficulty_by_source.png')
    
    # ---- 2.5 Khan Academy Readability Scores ----
    readability_cols = [c for c in khan_raw.columns if 'gunning' in c.lower() or 
                        'flesch' in c.lower() or 'smog' in c.lower()]
    if readability_cols:
        fig, axes = plt.subplots(1, min(3, len(readability_cols)), figsize=(15, 5))
        if len(readability_cols) == 1:
            axes = [axes]
        fig.suptitle('Khan Academy Readability Scores', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(readability_cols[:3]):
            data = pd.to_numeric(khan_raw[col], errors='coerce').dropna()
            data = data[(data > 0) & (data < 30)]
            if i < len(axes):
                axes[i].hist(data, bins=40, color=COLORS[i], alpha=0.8, edgecolor='white')
                axes[i].set_title(col.replace('_', ' ').title())
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        save_fig('05_readability_scores.png')
    
    # ---- 2.6 Text Length Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Text Length Distribution', fontsize=16, fontweight='bold')
    
    df['text_len'] = df['clean_text'].str.split().str.len()
    
    for i, src in enumerate(['khan', 'coursera']):
        src_df = df[df['source'] == src]['text_len'].dropna()
        axes[i].hist(src_df, bins=50, color=COLORS[i], alpha=0.8, edgecolor='white')
        axes[i].set_title(f'{src.title()} - Text Word Count')
        axes[i].set_xlabel('Word Count')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(src_df.median(), color='red', linestyle='--', 
                        label=f'Median: {src_df.median():.0f}')
        axes[i].legend()
    
    plt.tight_layout()
    save_fig('06_text_length.png')
    
    # ---- 2.7 Correlation Heatmap (Khan Academy numeric) ----
    numeric_cols = [c for c in khan_raw.columns if khan_raw[c].dtype in ['float64', 'int64']]
    numeric_cols = [c for c in numeric_cols if 'sentiment' in c or 'readability' in c.lower() 
                    or 'gunning' in c or 'flesch' in c or 'view' in c or 'like' in c][:12]
    
    if len(numeric_cols) >= 4:
        corr_df = khan_raw[numeric_cols].apply(pd.to_numeric, errors='coerce').corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, square=True, linewidths=0.5,
                    annot_kws={'size': 8})
        ax.set_title('Correlation Heatmap - Khan Academy Numeric Features', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_fig('07_correlation_heatmap.png')
    
    print(f"\n[Exploration] All charts saved to {IMG_DIR}")

# ==============================================================================
# STEP 3: EM CLUSTERING
# ==============================================================================
def step3_clustering(tfidf_matrix, vectorizer, df):
    print("\n" + "="*70)
    print("STEP 3: EM CLUSTERING")
    print("="*70)
    
    from src.clustering import run_clustering
    gmm, svd, labels, proba, X_reduced, keywords, results_df, best_k = \
        run_clustering(tfidf_matrix, vectorizer, df)
    
    # ---- Plot BIC/AIC ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('EM Clustering Evaluation', fontsize=16, fontweight='bold')
    
    axes[0].plot(results_df['k'], results_df['bic'], 'o-', color=COLORS[0], lw=2, ms=8, label='BIC')
    axes[0].plot(results_df['k'], results_df['aic'], 's--', color=COLORS[1], lw=2, ms=8, label='AIC')
    axes[0].axvline(best_k, color='red', linestyle=':', lw=2, label=f'Best k={best_k}')
    axes[0].set_title('BIC & AIC by Number of Clusters')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Score (lower is better)')
    axes[0].legend()
    
    axes[1].plot(results_df['k'], results_df['silhouette'], 'D-', color=COLORS[2], lw=2, ms=8)
    axes[1].axvline(best_k, color='red', linestyle=':', lw=2, label=f'Best k={best_k}')
    axes[1].set_title('Silhouette Score by k')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score (higher is better)')
    axes[1].legend()
    
    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    axes[2].bar(unique, counts, color=COLORS[:len(unique)])
    axes[2].set_title(f'Cluster Size Distribution (k={best_k})')
    axes[2].set_xlabel('Cluster ID')
    axes[2].set_ylabel('Number of Items')
    
    plt.tight_layout()
    save_fig('08_em_evaluation.png')
    
    # ---- PCA 2D ----
    from sklearn.decomposition import PCA
    print("  [PCA] Reducing to 2D...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_reduced[:min(10000, len(X_reduced))])
    labels_sub = labels[:min(10000, len(labels))]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sub,
                         cmap='tab20', alpha=0.4, s=8)
    ax.set_title(f'PCA 2D - EM Clusters (k={best_k})', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    save_fig('09_pca_clusters.png')
    
    # ---- t-SNE 2D (on smaller sample) ----
    from sklearn.manifold import TSNE
    n_tsne = min(5000, len(X_reduced))
    idx = np.random.choice(len(X_reduced), n_tsne, replace=False)
    print(f"  [t-SNE] Running on {n_tsne} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_reduced[idx])
    labels_tsne = labels[idx]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_tsne,
                         cmap='tab20', alpha=0.5, s=10)
    ax.set_title(f't-SNE 2D - EM Clusters (k={best_k})', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    save_fig('10_tsne_clusters.png')
    
    # ---- Cluster Keywords Table ----
    print("\n[Cluster Keywords Summary]")
    print(f"{'Cluster':>8} | {'Count':>6} | {'Top Keywords'}")
    print("-"*80)
    for c, cnt in zip(unique, counts):
        kws = ', '.join(keywords.get(c, [])[:6])
        print(f"{c:>8} | {cnt:>6} | {kws}")
    
    # Save keywords as JSON
    import json
    kw_path = os.path.join(PROC_DIR, 'cluster_keywords.json')
    with open(kw_path, 'w') as f:
        json.dump({str(k): v for k, v in keywords.items()}, f, indent=2)
    
    # ---- Cluster difficulty distribution ----
    df_labeled = df.copy()
    df_labeled = df_labeled.iloc[:len(labels)]
    df_labeled['cluster'] = labels
    
    fig, ax = plt.subplots(figsize=(14, 6))
    diff_cluster = pd.crosstab(df_labeled['cluster'], df_labeled['difficulty'])
    diff_cluster.plot(kind='bar', ax=ax, stacked=True, color=COLORS[:3])
    ax.set_title('Difficulty Distribution per Cluster', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Difficulty')
    plt.tight_layout()
    save_fig('11_cluster_difficulty.png')
    
    return gmm, svd, labels, keywords, results_df, best_k

# ==============================================================================
# STEP 4: NEURAL NETWORK
# ==============================================================================
def step4_neural_network(tfidf_matrix, df):
    print("\n" + "="*70)
    print("STEP 4: NEURAL NETWORK CLASSIFICATION")
    print("="*70)
    
    from src.classifier import run_classification
    results, histories, splits, best_name = run_classification(tfidf_matrix, df)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    
    # ---- Plot Training History ----
    for model_name, hist in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name} - Training History', fontsize=14, fontweight='bold')
        
        # Loss
        axes[0].plot(hist.history['loss'], label='Train Loss', color=COLORS[0], lw=2)
        axes[0].plot(hist.history['val_loss'], label='Val Loss', color=COLORS[1], lw=2, linestyle='--')
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy
        axes[1].plot(hist.history['accuracy'], label='Train Accuracy', color=COLORS[2], lw=2)
        axes[1].plot(hist.history['val_accuracy'], label='Val Accuracy', color=COLORS[3], lw=2, linestyle='--')
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        save_fig(f'12_{model_name.lower()}_training.png')
    
    # ---- Confusion Matrices ----
    label_names = ['Easy', 'Medium', 'Hard']
    
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    
    for i, (model_name, res) in enumerate(results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=label_names, yticklabels=label_names)
        axes[i].set_title(f'{model_name}\nAcc={res["accuracy"]:.4f} F1={res["f1"]:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.suptitle('Confusion Matrices - Neural Network Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('13_confusion_matrices.png')
    
    # ---- Model Comparison Bar Chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['accuracy', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    
    model_names = list(results.keys())
    for i, model_name in enumerate(model_names):
        values = [results[model_name]['accuracy'], results[model_name]['f1']]
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=COLORS[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Accuracy', 'F1-Score (weighted)'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.legend()
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='70% threshold')
    plt.tight_layout()
    save_fig('14_model_comparison.png')
    
    print(f"\n[NN Summary]")
    for name, res in results.items():
        print(f"  {name}: Accuracy={res['accuracy']:.4f}, F1={res['f1']:.4f}")
    print(f"  Best model: {best_name}")
    
    return results, histories

# ==============================================================================
# STEP 5: ASSOCIATION RULES
# ==============================================================================
def step5_association_rules(df):
    print("\n" + "="*70)
    print("STEP 5: ASSOCIATION RULES")
    print("="*70)
    
    # Load original data with skills
    orig_coursera = pd.read_csv(COURSERA_FILE, low_memory=False)
    
    # Merge skills back into df
    if 'skills' in orig_coursera.columns:
        coursera_mask = df['source'] == 'coursera'
        n_coursera = coursera_mask.sum()
        df_work = df.copy()
        orig_sample = orig_coursera.drop_duplicates(subset=['name'] if 'name' in orig_coursera.columns else None)
        orig_sample = orig_sample.head(n_coursera)
        df_work.loc[coursera_mask, 'skills'] = orig_sample['skills'].fillna('').values[:n_coursera]
    else:
        df_work = df.copy()
        df_work['skills'] = ''
    
    from src.association_rules import run_association_rules, get_top_rules
    rules, transactions, te = run_association_rules(df_work)
    
    if len(rules) == 0:
        print("  WARNING: No rules generated, skipping AR plots")
        return rules
    
    top_rules = get_top_rules(rules, 20)
    
    # ---- Plot Support vs Confidence scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold')
    
    scatter = axes[0].scatter(rules['support'], rules['confidence'],
                               c=rules['lift'], cmap='YlOrRd',
                               alpha=0.6, s=60, edgecolors='gray', lw=0.5)
    plt.colorbar(scatter, ax=axes[0], label='Lift')
    axes[0].set_title('Support vs Confidence (colored by Lift)')
    axes[0].set_xlabel('Support')
    axes[0].set_ylabel('Confidence')
    axes[0].axhline(0.5, color='blue', linestyle='--', alpha=0.3, label='Conf=0.5')
    axes[0].legend()
    
    # Top 20 rules by Lift
    top20 = rules.head(20).copy()
    y_pos = range(len(top20))
    rule_labels = [f"{r['antecedents_str'][:20]}→{r['consequents_str'][:15]}" 
                   for _, r in top20.iterrows()]
    
    bars = axes[1].barh(list(y_pos), top20['lift'].values, color=COLORS)
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(rule_labels, fontsize=7)
    axes[1].set_title('Top 20 Rules by Lift')
    axes[1].set_xlabel('Lift')
    plt.tight_layout()
    save_fig('15_association_rules.png')
    
    # ---- Network-style plot ----
    if len(top_rules) >= 5:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a scatter where each rule is a point
        ax.scatter(top_rules['support'], top_rules['confidence'],
                   s=top_rules['lift'] * 200, c=top_rules['lift'],
                   cmap='Reds', alpha=0.7, edgecolors='darkred', lw=1)
        
        for _, row in top_rules.head(10).iterrows():
            ax.annotate(
                f"{row['antecedents_str'][:15]}→{row['consequents_str'][:10]}",
                (row['support'], row['confidence']),
                fontsize=7, ha='center', va='bottom'
            )
        
        ax.set_title('Association Rules - Bubble Chart (size=lift)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        plt.tight_layout()
        save_fig('16_ar_bubble.png')
    
    print(f"\n[Top 20 Association Rules]")
    print(top_rules.to_string(index=False))
    
    return rules

# ==============================================================================
# STEP 6: RECOMMENDATION DEMO
# ==============================================================================
def step6_recommendation_demo():
    print("\n" + "="*70)
    print("STEP 6: RECOMMENDATION SYSTEM DEMO")
    print("="*70)
    
    sys.path.insert(0, BASE_DIR)
    from src.recommender import recommend_exercises
    
    test_queries = [
        "Introduction to Algebra: Solving Linear Equations",
        "Organic Chemistry: Reaction Mechanisms",
        "Machine Learning with Python: Neural Networks",
        "World War II: Causes and Consequences",
        "Microeconomics: Supply and Demand Analysis"
    ]
    
    results = []
    for q in test_queries:
        r = recommend_exercises(q, top_n=5, verbose=True)
        results.append(r)
        print()
    
    # ---- Demo Results Chart ----
    fig, ax = plt.subplots(figsize=(12, 8))
    
    query_labels = [q[:40]+'...' if len(q)>40 else q for q in test_queries]
    difficulty_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    topics = [r['topic'] for r in results]
    difficulties = [r['difficulty'] for r in results]
    clusters = [r['cluster'] if r['cluster'] is not None else -1 for r in results]
    
    colors_by_diff = {'Easy': '#55A868', 'Medium': '#4C72B0', 'Hard': '#C44E52'}
    bar_colors = [colors_by_diff.get(d, '#8C8C8C') for d in difficulties]
    
    y_pos = range(len(query_labels))
    bars = ax.barh(list(y_pos), [r['metadata']['n_recommendations'] for r in results],
                   color=bar_colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(query_labels, fontsize=9)
    ax.set_title('Recommendation System Demo Results', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Recommendations')
    
    # Add annotations
    for i, (bar, r) in enumerate(zip(bars, results)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"Topic: {r['topic']}, Diff: {r['difficulty']}, Cluster: {r['cluster']}",
                va='center', fontsize=8)
    
    # Legend
    patches = [mpatches.Patch(color=colors_by_diff[d], label=d) 
               for d in ['Easy', 'Medium', 'Hard']]
    ax.legend(handles=patches, title='Difficulty')
    
    plt.tight_layout()
    save_fig('17_recommendation_demo.png')
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    start = time.time()
    print("\n" + "#"*70)
    print("#  DO AN KHAI PHA DU LIEU - HE THONG GOI Y BAI TAP")
    print("#  Chay toan bo pipeline...")
    print("#"*70)
    
    # Step 1: Preprocessing
    df, tfidf_matrix, vectorizer = step1_preprocessing()
    
    # Step 2: Exploration
    step2_exploration(df)
    
    # Step 3: EM Clustering
    gmm, svd, labels, keywords, eval_results, best_k = step3_clustering(tfidf_matrix, vectorizer, df)
    
    # Step 4: Neural Network
    nn_results, histories = step4_neural_network(tfidf_matrix, df)
    
    # Step 5: Association Rules
    rules = step5_association_rules(df)
    
    # Step 6: Recommendation Demo
    demo_results = step6_recommendation_demo()
    
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE! Total time: {elapsed/60:.1f} minutes")
    print(f"Charts saved to: {IMG_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Run Streamlit: streamlit run app.py")
    print(f"{'='*70}\n")
    
    # Save summary
    summary = {
        'dataset': {
            'total_records': len(df),
            'khan_records': int((df['source'] == 'khan').sum()),
            'coursera_records': int((df['source'] == 'coursera').sum()),
            'difficulty_distribution': df['difficulty'].value_counts().to_dict()
        },
        'clustering': {
            'best_k': int(best_k),
            'n_components_svd': 100,
            'best_bic': float(eval_results['bic'].min()),
            'best_silhouette': float(eval_results.loc[eval_results['bic'].idxmin(), 'silhouette'])
        },
        'neural_network': {
            m: {
                'accuracy': round(float(r['accuracy']), 4),
                'f1': round(float(r['f1']), 4)
            } for m, r in nn_results.items()
        },
        'association_rules': {
            'total_rules': len(rules)
        }
    }
    
    with open(os.path.join(BASE_DIR, 'data', 'processed', 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSUMMARY:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
