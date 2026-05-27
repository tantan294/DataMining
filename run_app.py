# run_app.py - Helper script de chay Streamlit app
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_trained():
    """Kiem tra da train chua."""
    required = [
        os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'),
        os.path.join(BASE_DIR, 'models', 'gmm_model.pkl'),
        os.path.join(BASE_DIR, 'data', 'processed', 'unified_data.csv'),
    ]
    return all(os.path.exists(p) for p in required)

if __name__ == '__main__':
    if not check_trained():
        print("Models chua san sang. Dang chay pipeline training...")
        subprocess.run([sys.executable, 'train_pipeline.py'], cwd=BASE_DIR)
    
    print("Starting Streamlit app...")
    subprocess.run(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py',
         '--server.port', '8501',
         '--server.headless', 'false'],
        cwd=BASE_DIR
    )
