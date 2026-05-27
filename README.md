# README.md
# 🎓 Hệ thống Gợi ý Bài tập Dựa trên Video Bài giảng
## Đồ án môn Khai phá Dữ liệu

---

## 📋 Tổng quan

Hệ thống gợi ý bài tập cho nền tảng học trực tuyến sử dụng pipeline khai phá dữ liệu hoàn chỉnh:

| Bước | Kỹ thuật | Mô tả |
|------|----------|-------|
| 1 | **Tiền xử lý** | Lowercase, remove stopwords, lemmatization, TF-IDF |
| 2 | **EM Clustering** | GaussianMixture, SVD, Silhouette/BIC/AIC |
| 3 | **Neural Network** | MLP + Deep Dense NN, phân loại Easy/Medium/Hard |
| 4 | **Association Rules** | FP-Growth, Support/Confidence/Lift |
| 5 | **Recommendation** | Pipeline tổng hợp → gợi ý bài tập |

---

## 📊 Dataset

| Dataset | File | Số bản ghi | Mô tả |
|---------|------|-----------|-------|
| Khan Academy | `youtube_khan_academy.csv` | ~52,657 | Video metadata, readability scores |
| Coursera | `courses_en.csv` | ~41,264 | Course info, skills, categories |

---

## 🚀 Hướng dẫn chạy

### Bước 1: Cài môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy pipeline training
```bash
python train_pipeline.py
```
Pipeline sẽ:
- Tiền xử lý và merge 2 dataset
- Sinh 7+ biểu đồ khảo sát dữ liệu
- Phân cụm EM với k tối ưu
- Huấn luyện MLP và Deep Dense NN
- Tính luật kết hợp FP-Growth
- Demo hệ thống gợi ý

### Bước 3: Chạy Streamlit Demo
```bash
streamlit run app.py
```
Mở trình duyệt: http://localhost:8501

---

## 📁 Cấu trúc thư mục

```
Cuoiky/
├── youtube_khan_academy.csv     # Dataset Khan Academy
├── courses_en.csv               # Dataset Coursera
├── courses_en.json              # Dataset Coursera (JSON)
├── train_pipeline.py            # Script chạy toàn bộ pipeline
├── app.py                       # Streamlit demo app
├── run_app.py                   # Helper script
├── requirements.txt
├── src/
│   ├── preprocessing.py         # Module tiền xử lý
│   ├── clustering.py            # Module EM clustering
│   ├── classifier.py            # Module Neural Network
│   ├── association_rules.py     # Module FP-Growth
│   └── recommender.py           # Module gợi ý
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_em_clustering.ipynb
│   ├── 04_neural_network.ipynb
│   ├── 05_association_rules.ipynb
│   └── 06_recommendation_system.ipynb
├── data/
│   ├── raw/
│   └── processed/
│       ├── unified_data.csv
│       ├── tfidf_matrix.npz
│       ├── cluster_labels.npy
│       ├── cluster_keywords.json
│       ├── association_rules.csv
│       └── pipeline_summary.json
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svd_model.pkl
│   ├── gmm_model.pkl
│   ├── mlp_model.keras
│   ├── deepnn_model.keras
│   └── nn_metadata.json
└── report/
    └── images/                  # Tất cả biểu đồ
        ├── 01_dataset_overview.png
        ├── 02_khan_stats.png
        ├── ... (17+ images)
```

---

## 💡 Sử dụng API gợi ý

```python
from src.recommender import recommend_exercises

# Gợi ý bài tập cho một video
result = recommend_exercises(
    "Introduction to Calculus: Derivatives and Limits",
    top_n=10
)

print(f"Topic: {result['topic']}")
print(f"Difficulty: {result['difficulty']}")
print(f"Cluster: {result['cluster']}")
for ex in result['exercises']:
    print(f"  [{ex['rank']}] {ex['exercise']} (score={ex['score']})")
```

---

## 📈 Kết quả

- **EM Clustering**: k tối ưu được chọn tự động theo BIC
- **Neural Network**: Accuracy ≥ 70% trên test set
- **Association Rules**: Tạo ra ≥ 20 luật kết hợp có ý nghĩa
- **Biểu đồ**: 17+ biểu đồ được lưu vào `report/images/`

---

## 🛠️ Công nghệ sử dụng

- **Python 3.11**
- **scikit-learn**: TF-IDF, SVD, GaussianMixture, metrics
- **TensorFlow/Keras**: MLP, Dense NN
- **mlxtend**: FP-Growth, Association Rules
- **NLTK**: NLP, lemmatization, stopwords
- **Streamlit**: Web demo interface
- **Plotly/Matplotlib/Seaborn**: Visualization

---

*Đồ án Khai phá Dữ liệu - Đề tài: Gợi ý bài tập trong hệ thống học trực tuyến dựa trên video bài giảng*
