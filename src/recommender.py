# src/recommender.py
"""
He thong goi y bai tap hoan chinh:
- recommend_exercises(text) -> list of recommendations
- Pipeline: text -> TF-IDF -> GMM cluster -> NN difficulty -> AR -> rank
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics.pairwise import cosine_similarity

# ---- Constants ----
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')

# ---- Load Models (cached) ----
_models = {}

def _load_models():
    """Load tat ca models can thiet."""
    global _models
    if _models:
        return _models
    
    print("[Recommender] Loading models...")
    
    # TF-IDF Vectorizer
    vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    if os.path.exists(vec_path):
        _models['vectorizer'] = joblib.load(vec_path)
    
    # GMM Model
    gmm_path = os.path.join(MODEL_DIR, 'gmm_model.pkl')
    if os.path.exists(gmm_path):
        _models['gmm'] = joblib.load(gmm_path)
    
    # SVD Model
    svd_path = os.path.join(MODEL_DIR, 'svd_model.pkl')
    if os.path.exists(svd_path):
        _models['svd'] = joblib.load(svd_path)
    
    # NN Model (best)
    nn_meta_path = os.path.join(MODEL_DIR, 'nn_metadata.json')
    if os.path.exists(nn_meta_path):
        with open(nn_meta_path) as f:
            meta = json.load(f)
        best_model = meta.get('best_model', 'MLP')
        model_file = 'mlp_model.keras' if best_model == 'MLP' else 'deepnn_model.keras'
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            import tensorflow as tf
            _models['nn'] = tf.keras.models.load_model(model_path)
            _models['nn_meta'] = meta
    
    # Association Rules
    rules_path = os.path.join(PROC_DIR, 'association_rules.csv')
    if os.path.exists(rules_path):
        _models['rules'] = pd.read_csv(rules_path)
    
    # Dataset
    data_path = os.path.join(PROC_DIR, 'unified_data.csv')
    if os.path.exists(data_path):
        _models['data'] = pd.read_csv(data_path)
    
    # TF-IDF matrix
    mat_path = os.path.join(PROC_DIR, 'tfidf_matrix.npz')
    if os.path.exists(mat_path):
        from scipy.sparse import load_npz
        _models['tfidf_matrix'] = load_npz(mat_path)
    
    # Cluster labels
    labels_path = os.path.join(PROC_DIR, 'cluster_labels.npy')
    if os.path.exists(labels_path):
        _models['cluster_labels'] = np.load(labels_path)
    
    print(f"[Recommender] Loaded: {list(_models.keys())}")
    return _models

# ---- Exercise Templates ----
EXERCISE_TEMPLATES = {
    'math': [
        "Solve quadratic equations using the quadratic formula",
        "Find the derivative of polynomial functions",
        "Evaluate definite integrals using substitution",
        "Apply the Pythagorean theorem to find missing sides",
        "Solve systems of linear equations using elimination",
        "Calculate the area and perimeter of geometric shapes",
        "Simplify algebraic expressions with like terms",
        "Find the slope and y-intercept from two points",
    ],
    'science': [
        "Design a controlled experiment with hypothesis and variables",
        "Analyze data from a scientific study and draw conclusions",
        "Classify organisms using taxonomic hierarchy",
        "Balance chemical equations for common reactions",
        "Calculate velocity, acceleration, and force using Newton's laws",
        "Describe the water cycle and its environmental impact",
        "Explain the structure of DNA and protein synthesis",
        "Calculate pH and pOH of acidic and basic solutions",
    ],
    'programming': [
        "Implement a sorting algorithm (bubble sort, merge sort)",
        "Write a recursive function to calculate Fibonacci numbers",
        "Create a class with constructor, getters, and setters",
        "Build a REST API endpoint that handles GET/POST requests",
        "Implement a binary search tree with insert and search",
        "Write unit tests for a function using a testing framework",
        "Create a data pipeline that reads, transforms, and saves CSV data",
        "Implement a simple neural network from scratch",
    ],
    'history': [
        "Create a timeline of major events in World War II",
        "Compare and contrast two historical civilizations",
        "Analyze primary sources from the Industrial Revolution",
        "Write an essay on the causes of the French Revolution",
        "Trace the origins and impact of the Civil Rights Movement",
        "Explain how trade routes shaped ancient civilizations",
    ],
    'economics': [
        "Calculate supply and demand equilibrium price",
        "Analyze the impact of fiscal policy on GDP",
        "Evaluate the pros and cons of free trade agreements",
        "Apply compound interest formula to investment scenarios",
        "Draw and interpret production possibility curves",
        "Analyze market structures (monopoly vs. perfect competition)",
    ],
    'language': [
        "Identify and correct grammatical errors in a paragraph",
        "Write a persuasive essay with thesis, evidence, and conclusion",
        "Analyze figurative language in a poem or short story",
        "Summarize and paraphrase a technical article",
        "Practice conjugating verbs in different tenses",
        "Create an annotated bibliography for research sources",
    ],
    'general': [
        "Apply critical thinking to analyze a real-world problem",
        "Create a mind map connecting key concepts from the lesson",
        "Write a reflective journal entry about what you learned",
        "Design a project that applies concepts from this topic",
        "Teach the main concept to a peer (Feynman Technique)",
        "Create multiple-choice quiz questions for this topic",
        "Find a real-world application of this concept",
    ]
}

DIFFICULTY_EXERCISE_HINTS = {
    'Easy': {
        'prefix': "Beginner Practice:",
        'note': "Focus on recall and basic application"
    },
    'Medium': {
        'prefix': "Intermediate Challenge:",
        'note': "Apply concepts in new contexts"
    },
    'Hard': {
        'prefix': "Advanced Problem:",
        'note': "Synthesize multiple concepts and think critically"
    }
}

def _detect_topic(text: str) -> str:
    """Phat hien chu de chinh tu text."""
    text_lower = text.lower()
    
    topic_keywords = {
        'math': ['math', 'algebra', 'calculus', 'geometry', 'equation', 'function', 
                 'derivative', 'integral', 'matrix', 'vector', 'statistics', 'probability',
                 'trigonometry', 'polynomial', 'series', 'sequence'],
        'science': ['physics', 'chemistry', 'biology', 'science', 'experiment', 'reaction',
                    'molecule', 'atom', 'cell', 'organism', 'evolution', 'energy', 'force',
                    'wave', 'genetics', 'dna', 'protein'],
        'programming': ['programming', 'code', 'algorithm', 'python', 'java', 'javascript',
                        'function', 'class', 'data structure', 'machine learning', 'ai',
                        'neural', 'database', 'sql', 'api', 'software'],
        'history': ['history', 'war', 'revolution', 'civilization', 'ancient', 'medieval',
                    'empire', 'democracy', 'colonialism', 'cold war'],
        'economics': ['economics', 'finance', 'market', 'supply', 'demand', 'gdp', 'inflation',
                      'trade', 'investment', 'monetary', 'fiscal', 'budget'],
        'language': ['english', 'grammar', 'writing', 'essay', 'literature', 'language',
                     'reading', 'vocabulary', 'pronunciation', 'poetry', 'rhetoric'],
    }
    
    scores = {topic: 0 for topic in topic_keywords}
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                scores[topic] += 1
    
    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        return 'general'
    return best_topic

def _normalize_text(text: str) -> str:
    """Chuan hoa text nhanh."""
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    for r in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(r, quiet=True)
    
    _lem = WordNetLemmatizer()
    _stop = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [_lem.lemmatize(t) for t in tokens if t not in _stop and len(t) > 2]
    return ' '.join(tokens)

def recommend_exercises(
    input_text: str,
    top_n: int = 10,
    difficulty_filter: str = None,
    verbose: bool = True
) -> dict:
    """
    Ham goi y bai tap chinh.
    
    Args:
        input_text: Tieu de hoac noi dung video bai giang
        top_n: So luong goi y
        difficulty_filter: 'Easy'/'Medium'/'Hard' hoac None (auto detect)
        verbose: In thong tin
    
    Returns:
        dict voi keys: topic, difficulty, cluster, exercises, similar_content, metadata
    """
    models = _load_models()
    
    if verbose:
        print(f"\n[Recommend] Input: '{input_text[:80]}...' " if len(input_text) > 80 else f"\n[Recommend] Input: '{input_text}'")
    
    # 1. Tien xu ly input
    clean_input = _normalize_text(input_text)
    
    # 2. Transform TF-IDF
    cluster_id = None
    predicted_difficulty = 'Medium'
    similar_items = []
    
    if 'vectorizer' in models:
        tfidf_vec = models['vectorizer'].transform([clean_input])
        
        # 3. Predict cluster (GMM)
        if 'gmm' in models and 'svd' in models:
            try:
                X_reduced = models['svd'].transform(tfidf_vec)
                from sklearn.preprocessing import normalize
                X_reduced = normalize(X_reduced)
                cluster_id = int(models['gmm'].predict(X_reduced)[0])
                cluster_proba = models['gmm'].predict_proba(X_reduced)[0]
                if verbose:
                    print(f"  Cluster: {cluster_id} (confidence: {cluster_proba[cluster_id]:.3f})")
            except Exception as e:
                if verbose:
                    print(f"  Clustering error: {e}")
        
        # 4. Predict difficulty (NN)
        if 'nn' in models and difficulty_filter is None:
            try:
                X_nn = tfidf_vec.toarray().astype(np.float32)
                proba = models['nn'].predict(X_nn, verbose=0)[0]
                label_map_inv = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
                predicted_difficulty = label_map_inv[int(np.argmax(proba))]
                if verbose:
                    print(f"  Difficulty: {predicted_difficulty} (E:{proba[0]:.2f}, M:{proba[1]:.2f}, H:{proba[2]:.2f})")
            except Exception as e:
                if verbose:
                    print(f"  NN error: {e}")
        elif difficulty_filter:
            predicted_difficulty = difficulty_filter
        
        # 5. Tim noi dung tuong tu trong dataset (cosine similarity)
        if 'data' in models and 'tfidf_matrix' in models and cluster_id is not None:
            try:
                sim = cosine_similarity(tfidf_vec, models['tfidf_matrix'])[0]
                
                # Loc theo cluster
                data_df = models['data']
                cluster_labels = models.get('cluster_labels', np.zeros(len(data_df)))
                
                # Uu tien items cung cluster
                cluster_mask = cluster_labels == cluster_id
                
                sim_cluster = sim.copy()
                sim_cluster[~cluster_mask] *= 0.5  # giam trong so items khac cluster
                
                # Loc theo difficulty
                diff_mask = data_df['difficulty'] == predicted_difficulty
                sim_cluster[~diff_mask] *= 0.7
                
                top_indices = sim_cluster.argsort()[::-1][:top_n]
                
                similar_items = []
                for idx in top_indices:
                    if idx < len(data_df):
                        row = data_df.iloc[idx]
                        similar_items.append({
                            'title': str(row.get('title', ''))[:100],
                            'category': str(row.get('category', '')),
                            'difficulty': str(row.get('difficulty', '')),
                            'source': str(row.get('source', '')),
                            'similarity': float(sim[idx]),
                            'score': float(sim_cluster[idx])
                        })
            except Exception as e:
                if verbose:
                    print(f"  Similarity error: {e}")
    
    # 6. Detect topic
    topic = _detect_topic(input_text)
    
    # 7. Tao bai tap goi y
    difficulty = difficulty_filter or predicted_difficulty
    
    topic_exercises = EXERCISE_TEMPLATES.get(topic, EXERCISE_TEMPLATES['general'])
    general_exercises = EXERCISE_TEMPLATES['general']
    
    # Mix exercises based on difficulty
    all_exercises = topic_exercises + general_exercises
    np.random.seed(hash(input_text) % 2**32)
    np.random.shuffle(all_exercises)
    
    diff_hint = DIFFICULTY_EXERCISE_HINTS.get(difficulty, DIFFICULTY_EXERCISE_HINTS['Medium'])
    
    exercises = []
    for i, ex in enumerate(all_exercises[:top_n]):
        exercises.append({
            'rank': i + 1,
            'exercise': f"{diff_hint['prefix']} {ex}",
            'difficulty': difficulty,
            'topic': topic,
            'note': diff_hint['note'],
            'score': round(1.0 - i * 0.08, 2)
        })
    
    # 8. AR expansion (neu co rules)
    ar_recommendations = []
    if 'rules' in models and len(models['rules']) > 0:
        try:
            from src.association_rules import recommend_by_rules
            input_items = [topic, difficulty.lower()]
            ar_recs = recommend_by_rules(models['rules'], input_items, top_n=5)
            if len(ar_recs) > 0:
                ar_recommendations = ar_recs.to_dict('records')
        except Exception as e:
            pass
    
    result = {
        'input': input_text,
        'topic': topic,
        'difficulty': difficulty,
        'cluster': cluster_id,
        'exercises': exercises,
        'similar_content': similar_items,
        'ar_recommendations': ar_recommendations,
        'metadata': {
            'n_recommendations': len(exercises),
            'n_similar': len(similar_items),
            'model_used': 'Full Pipeline' if 'nn' in models else 'Fallback'
        }
    }
    
    if verbose:
        print(f"\n=== RECOMMENDATIONS for '{input_text[:50]}' ===")
        print(f"Topic: {topic} | Difficulty: {difficulty} | Cluster: {cluster_id}")
        print(f"\nTop {min(5, len(exercises))} Exercises:")
        for ex in exercises[:5]:
            print(f"  [{ex['rank']}] {ex['exercise'][:80]} (score={ex['score']})")
        print(f"\nSimilar Content ({len(similar_items)} items found)")
    
    return result


def batch_recommend(inputs: list, top_n: int = 5) -> list:
    """Goi y cho nhieu inputs."""
    return [recommend_exercises(text, top_n=top_n, verbose=False) for text in inputs]


if __name__ == '__main__':
    # Test
    test_queries = [
        "Introduction to Algebra: Solving Linear Equations",
        "Organic Chemistry: Reaction Mechanisms and Functional Groups",
        "Machine Learning with Python: Neural Networks",
        "World War II: Causes and Consequences",
        "Microeconomics: Supply and Demand Analysis"
    ]
    
    for q in test_queries:
        result = recommend_exercises(q, top_n=5, verbose=True)
        print()
