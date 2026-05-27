# src/association_rules.py
"""
Module luat ket hop:
- Tao transactions tu Coursera skills va Khan Academy categories
- FP-Growth (mlxtend)
- Tinh Support, Confidence, Lift
- Xuat top 20 luat tot nhat
"""

import os
import numpy as np
import pandas as pd
import ast
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ---- Constants ----
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
IMG_DIR   = os.path.join(BASE_DIR, 'report', 'images')

def parse_skills(skills_str: str) -> list:
    """Parse skills string thanh list."""
    if not isinstance(skills_str, str) or not skills_str.strip():
        return []
    # Try ast.literal_eval
    try:
        items = ast.literal_eval(skills_str)
        if isinstance(items, list):
            return [str(i).strip().lower() for i in items if str(i).strip()]
    except Exception:
        pass
    # Comma separated
    items = [i.strip().lower() for i in re.split(r'[,;]', skills_str) if i.strip()]
    return items

def extract_category_tags(title: str, category: str) -> list:
    """Trich keywords tu title + category cua Khan Academy."""
    tags = []
    if isinstance(category, str) and category.strip():
        tags.append(category.strip().lower())
    
    # Extract common subject keywords from title
    subjects = ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'probability',
                'physics', 'chemistry', 'biology', 'science', 'history', 'economics',
                'finance', 'programming', 'computer', 'english', 'grammar', 'literature',
                'art', 'music', 'philosophy', 'psychology', 'sociology',
                'trigonometry', 'linear', 'organic', 'molecular', 'genetics']
    
    title_lower = str(title).lower()
    for subj in subjects:
        if subj in title_lower:
            tags.append(subj)
    
    return list(set(tags))

def build_transactions(df: pd.DataFrame) -> list:
    """
    Tao danh sach transactions:
    - Coursera: dung cot skills
    - Khan: dung category + title keywords
    """
    transactions = []
    
    for _, row in df.iterrows():
        items = []
        
        if row.get('source') == 'coursera':
            # Parse skills
            if 'skills' in df.columns:
                items = parse_skills(str(row.get('skills', '')))
            if not items:
                # Fallback: dung category
                cat = str(row.get('category', '')).strip().lower()
                if cat and cat != 'nan':
                    items = [cat]
        
        elif row.get('source') == 'khan':
            items = extract_category_tags(
                row.get('title', ''), 
                row.get('category', '')
            )
        
        # Them difficulty
        diff = str(row.get('difficulty', '')).strip().lower()
        if diff in ['easy', 'medium', 'hard']:
            items.append(f'difficulty:{diff}')
        
        if len(items) >= 2:
            transactions.append(list(set(items)))
    
    print(f"[AR] Total transactions: {len(transactions)}")
    return transactions

def run_fpgrowth(transactions: list, min_support: float = 0.005, min_confidence: float = 0.3):
    """Chay FP-Growth va trich luat ket hop."""
    print(f"[AR] Running FP-Growth (min_support={min_support}, min_confidence={min_confidence})...")
    
    # Encode
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)
    
    print(f"  Transaction matrix: {te_df.shape}")
    print(f"  Unique items: {len(te.columns_)}")
    
    # FP-Growth
    try:
        freq_itemsets = fpgrowth(te_df, min_support=min_support, use_colnames=True)
        print(f"  Frequent itemsets: {len(freq_itemsets)}")
        
        if len(freq_itemsets) == 0:
            print("  Lowering min_support to 0.002...")
            freq_itemsets = fpgrowth(te_df, min_support=0.002, use_colnames=True)
    except Exception:
        # Fallback to Apriori
        print("  FP-Growth failed, using Apriori...")
        freq_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
    
    if len(freq_itemsets) == 0:
        print("  WARNING: No frequent itemsets found!")
        return pd.DataFrame(), freq_itemsets, te
    
    # Generate rules
    rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_confidence)
    
    # Them conviction va zhang
    rules = rules[rules['lift'] >= 1.0].copy()
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
    
    # Sort by lift
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    
    print(f"  Total rules: {len(rules)}")
    if len(rules) > 0:
        print(f"  Top rule: {rules.iloc[0]['antecedents_str']} => {rules.iloc[0]['consequents_str']}")
        print(f"    Support={rules.iloc[0]['support']:.4f}, Confidence={rules.iloc[0]['confidence']:.4f}, Lift={rules.iloc[0]['lift']:.4f}")
    
    return rules, freq_itemsets, te

def get_top_rules(rules: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Lay top N luat theo lift."""
    if len(rules) == 0:
        return pd.DataFrame()
    return rules.head(n)[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()

def recommend_by_rules(rules: pd.DataFrame, input_items: list, top_n: int = 10) -> pd.DataFrame:
    """
    Dua vao danh sach items dau vao, tim cac luat ket hop phu hop
    va tra ve cac consequents de xuất.
    """
    if len(rules) == 0 or not input_items:
        return pd.DataFrame()
    
    input_set = set([i.lower().strip() for i in input_items])
    matched = []
    
    for _, row in rules.iterrows():
        ant_set = set(row['antecedents'])
        # Kiem tra xem co overlapping khong
        if ant_set & input_set:
            overlap = len(ant_set & input_set) / len(ant_set)
            matched.append({
                'antecedents': row['antecedents_str'],
                'consequents': row['consequents_str'],
                'support': row['support'],
                'confidence': row['confidence'],
                'lift': row['lift'],
                'overlap': overlap,
                'score': row['lift'] * overlap * row['confidence']
            })
    
    if not matched:
        return pd.DataFrame()
    
    result = pd.DataFrame(matched).sort_values('score', ascending=False).head(top_n)
    return result

def run_association_rules(df: pd.DataFrame):
    """Pipeline luat ket hop hoan chinh."""
    # Build transactions
    transactions = build_transactions(df)
    
    # FP-Growth
    rules, freq_itemsets, te = run_fpgrowth(transactions, min_support=0.005, min_confidence=0.3)
    
    # Save
    if len(rules) > 0:
        rules.to_csv(os.path.join(PROC_DIR, 'association_rules.csv'), index=False)
        top_rules = get_top_rules(rules, 20)
        top_rules.to_csv(os.path.join(PROC_DIR, 'top20_rules.csv'), index=False)
        
        print("\n[Top 20 Association Rules]")
        print(top_rules.to_string(index=False))
    
    # Save transactions for later use
    joblib.dump(transactions, os.path.join(PROC_DIR, 'transactions.pkl'))
    joblib.dump(te, os.path.join(MODEL_DIR, 'transaction_encoder.pkl'))
    
    return rules, transactions, te


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(PROC_DIR, 'unified_data.csv'))
    # Load skills from original coursera file
    orig_coursera = pd.read_csv(os.path.join(BASE_DIR, 'courses_en.csv'))
    if 'skills' in orig_coursera.columns:
        # Merge skills back
        pass
    run_association_rules(df)
