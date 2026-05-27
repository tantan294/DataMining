"""
app.py - Streamlit Demo App
He thong goi y bai tap dua tren video bai giang
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st

# ---- Setup paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
IMG_DIR   = os.path.join(BASE_DIR, 'report', 'images')

# ---- Page Config ----
st.set_page_config(
    page_title="ExerciseAI - Gợi ý Bài tập Thông minh",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif !important; }
    
    .main { background-color: #0e1117; }
    
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(100, 149, 237, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #e0f7fa, #4fc3f7, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        color: #90caf9;
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #252540);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(100,149,237,0.25);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover { transform: translateY(-2px); }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fc3f7;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #90a4ae;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .exercise-card {
        background: linear-gradient(135deg, #1a237e20, #0d47a120);
        border: 1px solid rgba(66, 165, 245, 0.4);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.2s;
        cursor: default;
    }
    
    .exercise-card:hover {
        background: linear-gradient(135deg, #1a237e35, #0d47a135);
        border-color: rgba(66, 165, 245, 0.7);
        transform: translateX(4px);
    }
    
    .exercise-rank {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4fc3f7;
        float: left;
        margin-right: 0.8rem;
    }
    
    .exercise-text {
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .exercise-meta {
        font-size: 0.78rem;
        color: #78909c;
        margin-top: 0.3rem;
    }
    
    .difficulty-easy {
        background: linear-gradient(135deg, #00600020, #00800020);
        border-color: rgba(76, 175, 80, 0.5) !important;
    }
    
    .difficulty-medium {
        background: linear-gradient(135deg, #e65100 20, #ff6f0020);
        border-color: rgba(255, 152, 0, 0.5) !important;
    }
    
    .difficulty-hard {
        background: linear-gradient(135deg, #b71c1c20, #c6282820);
        border-color: rgba(244, 67, 54, 0.5) !important;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    
    .badge-easy { background: #1b5e20; color: #a5d6a7; }
    .badge-medium { background: #e65100; color: #ffcc80; }
    .badge-hard { background: #b71c1c; color: #ef9a9a; }
    .badge-topic { background: #1a237e; color: #90caf9; }
    .badge-cluster { background: #311b92; color: #ce93d8; }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e0e0;
        border-bottom: 2px solid #4fc3f7;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .pipeline-step {
        background: linear-gradient(135deg, #0d1b2a, #1b2838);
        border-left: 4px solid #4fc3f7;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #b0bec5;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e1e2e !important;
        border: 1px solid rgba(100,149,237,0.4) !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #7c4dff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 77, 255, 0.4) !important;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #1e1e2e, #252540);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(100,149,237,0.2);
        font-size: 0.85rem;
        color: #90a4ae;
    }
    
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #0e1117 100%);
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-green { background-color: #4caf50; }
    .dot-orange { background-color: #ff9800; }
    .dot-red { background-color: #f44336; }
    
    .similar-item {
        background: #1a1a2e;
        border: 1px solid rgba(100,149,237,0.2);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    
    .progress-bar-container {
        background: #1e1e2e;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 0.2rem 0;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff);
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
@st.cache_data
def load_processed_data():
    """Load processed data (cached)."""
    path = os.path.join(PROC_DIR, 'unified_data.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_summary():
    """Load pipeline summary."""
    path = os.path.join(PROC_DIR, 'pipeline_summary.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_cluster_keywords():
    path = os.path.join(PROC_DIR, 'cluster_keywords.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_association_rules():
    path = os.path.join(PROC_DIR, 'association_rules.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_cluster_labels():
    path = os.path.join(PROC_DIR, 'cluster_labels.npy')
    if os.path.exists(path):
        return np.load(path)
    return np.array([])

def check_models_ready():
    """Kiem tra model da san sang chua."""
    required = [
        os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'),
        os.path.join(MODEL_DIR, 'gmm_model.pkl'),
        os.path.join(PROC_DIR, 'unified_data.csv'),
    ]
    return all(os.path.exists(p) for p in required)

def get_difficulty_badge(difficulty: str) -> str:
    cls_map = {'Easy': 'easy', 'Medium': 'medium', 'Hard': 'hard'}
    cls = cls_map.get(difficulty, 'medium')
    emoji = {'Easy': '🟢', 'Medium': '🟡', 'Hard': '🔴'}.get(difficulty, '⚪')
    return f'<span class="badge badge-{cls}">{emoji} {difficulty}</span>'

def difficulty_color(diff: str) -> str:
    return {'Easy': '#4caf50', 'Medium': '#ff9800', 'Hard': '#f44336'}.get(diff, '#90a4ae')

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🎓</span>
        <h2 style="color: #4fc3f7; margin: 0.5rem 0;">ExerciseAI</h2>
        <p style="color: #78909c; font-size: 0.85rem;">Data Mining Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model Status
    models_ready = check_models_ready()
    status_dot = "dot-green" if models_ready else "dot-orange"
    status_text = "Models Ready ✓" if models_ready else "Training Required"
    st.markdown(f"""
    <div class="sidebar-info">
        <b>System Status:</b><br>
        <span class="status-dot {status_dot}"></span>{status_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    st.markdown('<p style="color: #78909c; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Navigation</p>', unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠 Dashboard", "🔍 Phân tích Video", "💡 Gợi ý Bài tập", 
         "📊 Kết quả Training", "📚 Khám phá Dataset"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Pipeline info
    st.markdown("""
    <div class="sidebar-info">
        <b style="color: #4fc3f7;">Pipeline:</b><br>
        <div class="pipeline-step">1️⃣ TF-IDF Vectorization</div>
        <div class="pipeline-step">2️⃣ EM Clustering (GMM)</div>
        <div class="pipeline-step">3️⃣ Neural Network (Difficulty)</div>
        <div class="pipeline-step">4️⃣ Association Rules</div>
        <div class="pipeline-step">5️⃣ Recommendation</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not models_ready:
        st.divider()
        st.warning("⚠️ Chạy pipeline trước khi sử dụng:")
        st.code("python train_pipeline.py", language="bash")

# ==============================================================================
# PAGES
# ==============================================================================

# ---- PAGE 1: DASHBOARD ----
if page == "🏠 Dashboard":
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎓 ExerciseAI</div>
        <div class="hero-subtitle">
            Hệ thống gợi ý bài tập thông minh dựa trên nội dung video bài giảng<br>
            <small>Sử dụng EM Clustering · Neural Network · Association Rules</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_processed_data()
    summary = load_summary()
    
    # ---- Metrics Row ----
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(df) if len(df) > 0 else summary.get('dataset', {}).get('total_records', 0)
    khan_n = int((df['source'] == 'khan').sum()) if len(df) > 0 else summary.get('dataset', {}).get('khan_records', 0)
    coursera_n = int((df['source'] == 'coursera').sum()) if len(df) > 0 else summary.get('dataset', {}).get('coursera_records', 0)
    best_k = summary.get('clustering', {}).get('best_k', 'N/A')
    best_acc = max([v.get('accuracy', 0) for v in summary.get('neural_network', {}).values()], default=0)
    
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Records</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{khan_n:,}</div>
            <div class="metric-label">Khan Academy Videos</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{coursera_n:,}</div>
            <div class="metric-label">Coursera Courses</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_k}</div>
            <div class="metric-label">Optimal Clusters</div>
        </div>""", unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_acc:.1%}</div>
            <div class="metric-label">NN Accuracy</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    if len(df) > 0:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('<div class="section-header">📊 Phân bố Độ khó</div>', unsafe_allow_html=True)
            diff_counts = df['difficulty'].value_counts()
            fig = px.pie(
                values=diff_counts.values,
                names=diff_counts.index,
                color=diff_counts.index,
                color_discrete_map={'Easy': '#4caf50', 'Medium': '#ff9800', 'Hard': '#f44336'},
                hole=0.45
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                legend=dict(font=dict(color='#e0e0e0')),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.update_traces(textinfo='percent+label', textfont_size=13)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown('<div class="section-header">📚 Phân bố theo Nguồn & Độ khó</div>', unsafe_allow_html=True)
            diff_src = pd.crosstab(df['source'], df['difficulty']).reset_index()
            diff_src_melted = diff_src.melt(id_vars='source', var_name='Difficulty', value_name='Count')
            fig2 = px.bar(
                diff_src_melted, x='source', y='Count', color='Difficulty',
                color_discrete_map={'Easy': '#4caf50', 'Medium': '#ff9800', 'Hard': '#f44336'},
                barmode='group', text='Count'
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                xaxis=dict(color='#e0e0e0'),
                yaxis=dict(color='#e0e0e0', gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(font=dict(color='#e0e0e0')),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Category distribution
        st.markdown('<div class="section-header">🏷️ Top Categories</div>', unsafe_allow_html=True)
        cat_counts = df['category'].value_counts().head(20)
        fig3 = px.bar(
            x=cat_counts.values, y=cat_counts.index,
            orientation='h',
            color=cat_counts.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Count', 'y': 'Category'}
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(color='#e0e0e0', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(color='#e0e0e0'),
            showlegend=False,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Saved charts gallery
        saved_charts = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')] if os.path.exists(IMG_DIR) else []
        if saved_charts:
            st.markdown('<div class="section-header">🖼️ Biểu đồ đã tạo</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, chart in enumerate(sorted(saved_charts)[:9]):
                with cols[i % 3]:
                    chart_path = os.path.join(IMG_DIR, chart)
                    st.image(chart_path, caption=chart.replace('.png', '').replace('_', ' ').title(),
                             use_column_width=True)
    else:
        st.info("ℹ️ Chưa có dữ liệu. Hãy chạy `python train_pipeline.py` trước.")
        
        # Show dataset preview
        st.markdown('<div class="section-header">📂 Files Dataset</div>', unsafe_allow_html=True)
        files = ['youtube_khan_academy.csv', 'courses_en.csv']
        for f in files:
            fpath = os.path.join(BASE_DIR, f)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                st.markdown(f"✅ `{f}` ({size_mb:.1f} MB)")
            else:
                st.markdown(f"❌ `{f}` (not found)")

# ---- PAGE 2: VIDEO ANALYSIS ----
elif page == "🔍 Phân tích Video":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title" style="font-size: 2rem;">🔍 Phân tích Video Bài giảng</div>
        <div class="hero-subtitle">Nhập tiêu đề hoặc nội dung video để phân tích cluster và độ khó</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input
    col_input, col_settings = st.columns([3, 1])
    
    with col_input:
        video_input = st.text_area(
            "Nhập tiêu đề / nội dung video bài giảng:",
            placeholder="Ví dụ: Introduction to Calculus - Derivatives and Limits",
            height=100,
            key="video_input"
        )
    
    with col_settings:
        st.markdown('<br>', unsafe_allow_html=True)
        top_n = st.slider("Số gợi ý:", 3, 20, 10)
        diff_filter = st.selectbox("Lọc độ khó:", ["Auto (AI)", "Easy", "Medium", "Hard"])
    
    # Prebuilt examples
    st.markdown("**💡 Ví dụ nhanh:**")
    example_cols = st.columns(5)
    examples = [
        "Introduction to Algebra: Solving Linear Equations",
        "Organic Chemistry: Reaction Mechanisms", 
        "Machine Learning: Neural Networks",
        "World War II Causes and Effects",
        "Supply and Demand in Microeconomics"
    ]
    for i, (col, ex) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(f"📝 Ex {i+1}", key=f"ex_{i}", help=ex):
                st.session_state['video_input_example'] = ex
                video_input = ex
    
    # Check if example was selected
    if 'video_input_example' in st.session_state and not video_input:
        video_input = st.session_state['video_input_example']
    
    analyze_btn = st.button("🚀 Phân tích ngay", type="primary")
    
    if analyze_btn and video_input.strip():
        if not check_models_ready():
            st.error("⚠️ Models chưa sẵn sàng. Hãy chạy: `python train_pipeline.py`")
        else:
            with st.spinner("🔄 Đang phân tích..."):
                try:
                    from src.recommender import recommend_exercises
                    diff_val = None if diff_filter == "Auto (AI)" else diff_filter
                    result = recommend_exercises(
                        video_input, top_n=top_n,
                        difficulty_filter=diff_val, verbose=False
                    )
                    
                    # Display results
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value">🏷️</div>
                            <div class="metric-label">Topic Detected</div>
                            <div style="color: #4fc3f7; font-weight: 600; margin-top: 0.5rem; font-size: 1.1rem;">{result['topic'].title()}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    with col_res2:
                        diff_emoji = {'Easy': '🟢', 'Medium': '🟡', 'Hard': '🔴'}
                        diff = result['difficulty']
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value">{diff_emoji.get(diff, '⚪')}</div>
                            <div class="metric-label">Difficulty Level</div>
                            <div style="color: {difficulty_color(diff)}; font-weight: 600; margin-top: 0.5rem; font-size: 1.1rem;">{diff}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    with col_res3:
                        cluster = result['cluster']
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value">🔵</div>
                            <div class="metric-label">Cluster ID</div>
                            <div style="color: #ce93d8; font-weight: 600; margin-top: 0.5rem; font-size: 1.1rem;">Cluster {cluster}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    # Exercises
                    st.markdown('<div class="section-header">📝 Bài tập được gợi ý</div>', unsafe_allow_html=True)
                    
                    for ex in result['exercises']:
                        diff_cls = ex['difficulty'].lower()
                        st.markdown(f"""
                        <div class="exercise-card difficulty-{diff_cls}">
                            <span class="exercise-rank">#{ex['rank']}</span>
                            <div class="exercise-text">{ex['exercise']}</div>
                            <div class="exercise-meta">
                                {get_difficulty_badge(ex['difficulty'])}
                                <span class="badge badge-topic">{ex['topic']}</span>
                                <span style="color: #78909c; margin-left: 0.5rem; font-size: 0.78rem;">
                                    Score: {ex['score']:.2f} | {ex['note']}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Similar content
                    if result['similar_content']:
                        st.markdown('<div class="section-header">🔗 Nội dung Tương tự</div>', unsafe_allow_html=True)
                        sim_df = pd.DataFrame(result['similar_content'])
                        if len(sim_df) > 0:
                            fig_sim = px.scatter(
                                sim_df, x='similarity', y='score',
                                color='difficulty',
                                hover_data=['title', 'category', 'source'],
                                color_discrete_map={'Easy': '#4caf50', 'Medium': '#ff9800', 'Hard': '#f44336'},
                                title='Cosine Similarity of Similar Content'
                            )
                            fig_sim.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#e0e0e0'),
                                xaxis=dict(color='#e0e0e0', gridcolor='rgba(255,255,255,0.1)'),
                                yaxis=dict(color='#e0e0e0', gridcolor='rgba(255,255,255,0.1)'),
                                legend=dict(font=dict(color='#e0e0e0')),
                                height=350
                            )
                            st.plotly_chart(fig_sim, use_container_width=True)
                            
                            for item in result['similar_content'][:5]:
                                diff_clr = difficulty_color(item['difficulty'])
                                st.markdown(f"""
                                <div class="similar-item">
                                    <b style="color: #e0e0e0;">{item['title'][:80]}</b>
                                    <span style="float: right; color: {diff_clr};">{item['difficulty']}</span>
                                    <br>
                                    <span style="color: #78909c;">{item['source'].title()} · {item['category'][:40]}</span>
                                    <span style="float: right; color: #4fc3f7;">Sim: {item['similarity']:.3f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.code(str(e))

# ---- PAGE 3: RECOMMENDATION ----
elif page == "💡 Gợi ý Bài tập":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title" style="font-size: 2rem;">💡 Hệ thống Gợi ý Bài tập</div>
        <div class="hero-subtitle">Nhập nhiều video và nhận gợi ý bài tập cho từng video</div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 Nhập thủ công", "📁 Upload File"])
    
    with tab1:
        inputs = st.text_area(
            "Nhập các tiêu đề video (mỗi dòng một video):",
            placeholder="Introduction to Calculus\nOrganic Chemistry Basics\nWorld History: Roman Empire",
            height=150
        )
        
        col_s, col_n, col_d = st.columns(3)
        with col_s:
            source_filter = st.selectbox("Nguồn:", ["All", "Khan Academy", "Coursera"])
        with col_n:
            n_recs = st.slider("Số gợi ý mỗi video:", 3, 15, 7)
        with col_d:
            diff_mode = st.selectbox("Độ khó:", ["Auto (AI)", "Easy", "Medium", "Hard"])
        
        if st.button("🚀 Gợi ý tất cả", type="primary"):
            if not inputs.strip():
                st.warning("Vui lòng nhập ít nhất một tiêu đề video.")
            elif not check_models_ready():
                st.error("⚠️ Models chưa sẵn sàng. Hãy chạy `python train_pipeline.py`")
            else:
                input_list = [line.strip() for line in inputs.strip().split('\n') if line.strip()]
                
                from src.recommender import recommend_exercises
                diff_val = None if diff_mode == "Auto (AI)" else diff_mode
                
                for i, query in enumerate(input_list):
                    with st.spinner(f"Đang xử lý: {query[:50]}..."):
                        try:
                            result = recommend_exercises(query, top_n=n_recs, 
                                                        difficulty_filter=diff_val, verbose=False)
                            
                            with st.expander(f"🎬 [{i+1}] {query[:60]}", expanded=(i == 0)):
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric("Topic", result['topic'].title())
                                with cols[1]:
                                    st.metric("Difficulty", result['difficulty'])
                                with cols[2]:
                                    st.metric("Cluster", f"#{result['cluster']}")
                                
                                st.markdown("**Bài tập đề xuất:**")
                                for ex in result['exercises']:
                                    st.markdown(f"**{ex['rank']}.** {ex['exercise']} *(Score: {ex['score']})*")
                        except Exception as e:
                            st.error(f"Lỗi với '{query}': {e}")
    
    with tab2:
        uploaded = st.file_uploader(
            "Upload file CSV hoặc TXT (mỗi dòng một video):",
            type=['csv', 'txt'],
            help="CSV cần có cột 'title' hoặc TXT với mỗi dòng là một tiêu đề"
        )
        
        if uploaded is not None:
            if uploaded.name.endswith('.txt'):
                content = uploaded.read().decode('utf-8')
                input_list = [line.strip() for line in content.split('\n') if line.strip()]
            else:
                upload_df = pd.read_csv(uploaded)
                title_col = next((c for c in ['title', 'name', 'Title', 'Name'] 
                                  if c in upload_df.columns), None)
                if title_col:
                    input_list = upload_df[title_col].dropna().tolist()
                else:
                    input_list = upload_df.iloc[:, 0].dropna().tolist()
            
            st.success(f"✅ Đọc được {len(input_list)} video titles")
            st.dataframe(pd.DataFrame({'title': input_list}), height=200)
            
            if st.button("🚀 Xử lý tất cả", type="primary") and check_models_ready():
                from src.recommender import batch_recommend
                with st.spinner(f"Đang xử lý {len(input_list)} videos..."):
                    results = batch_recommend(input_list[:50], top_n=5)
                
                # Summary table
                summary_data = [{
                    'Video': r['input'][:50],
                    'Topic': r['topic'],
                    'Difficulty': r['difficulty'],
                    'Cluster': r['cluster'],
                    'Recommendations': r['metadata']['n_recommendations']
                } for r in results]
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                # Download
                export_rows = []
                for r in results:
                    for ex in r['exercises']:
                        export_rows.append({
                            'video': r['input'],
                            'topic': r['topic'],
                            'difficulty': r['difficulty'],
                            'exercise': ex['exercise'],
                            'score': ex['score']
                        })
                
                export_df = pd.DataFrame(export_rows)
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download kết quả CSV",
                    data=csv_data,
                    file_name="recommendations.csv",
                    mime="text/csv"
                )

# ---- PAGE 4: TRAINING RESULTS ----
elif page == "📊 Kết quả Training":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title" style="font-size: 2rem;">📊 Kết quả Training</div>
        <div class="hero-subtitle">Chi tiết về quá trình huấn luyện các mô hình</div>
    </div>
    """, unsafe_allow_html=True)
    
    summary = load_summary()
    keywords = load_cluster_keywords()
    rules = load_association_rules()
    
    tab_cluster, tab_nn, tab_ar = st.tabs(["🔵 EM Clustering", "🧠 Neural Network", "🔗 Association Rules"])
    
    with tab_cluster:
        st.markdown('<div class="section-header">Kết quả Phân cụm EM (GaussianMixture)</div>', unsafe_allow_html=True)
        
        cluster_info = summary.get('clustering', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số cụm tối ưu (k)", cluster_info.get('best_k', 'N/A'))
        with col2:
            st.metric("SVD Components", cluster_info.get('n_components_svd', 100))
        with col3:
            sil = cluster_info.get('best_silhouette', None)
            st.metric("Best Silhouette", f"{sil:.4f}" if sil else 'N/A')
        
        if keywords:
            st.markdown('<div class="section-header">🏷️ Top Keywords mỗi Cluster</div>', unsafe_allow_html=True)
            
            n_clusters = len(keywords)
            cols_per_row = 3
            rows = (n_clusters + cols_per_row - 1) // cols_per_row
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    cluster_idx = row * cols_per_row + col_idx
                    cluster_key = str(cluster_idx)
                    if cluster_key in keywords:
                        kws = keywords[cluster_key]
                        with cols[col_idx]:
                            kw_html = ' '.join([f'<span class="badge badge-topic">{k}</span>' for k in kws[:8]])
                            st.markdown(f"""
                            <div style="background: #1a1a2e; border: 1px solid rgba(100,149,237,0.3); 
                                        border-radius: 10px; padding: 0.8rem; margin: 0.3rem 0;">
                                <b style="color: #4fc3f7;">Cluster {cluster_idx}</b><br>
                                {kw_html}
                            </div>
                            """, unsafe_allow_html=True)
        
        # Show saved cluster images
        cluster_imgs = ['09_pca_clusters.png', '10_tsne_clusters.png', 
                        '08_em_evaluation.png', '11_cluster_difficulty.png']
        available = [f for f in cluster_imgs if os.path.exists(os.path.join(IMG_DIR, f))]
        
        if available:
            st.markdown('<div class="section-header">📈 Biểu đồ Clustering</div>', unsafe_allow_html=True)
            img_cols = st.columns(2)
            for i, img_name in enumerate(available):
                with img_cols[i % 2]:
                    st.image(os.path.join(IMG_DIR, img_name), 
                             caption=img_name.replace('.png', '').replace('_', ' ').title(),
                             use_column_width=True)
    
    with tab_nn:
        st.markdown('<div class="section-header">Kết quả Neural Network</div>', unsafe_allow_html=True)
        
        nn_results = summary.get('neural_network', {})
        if nn_results:
            # Comparison chart
            model_names = list(nn_results.keys())
            accuracies = [nn_results[m]['accuracy'] for m in model_names]
            f1s = [nn_results[m]['f1'] for m in model_names]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies,
                                  marker_color='#4fc3f7', text=[f'{v:.4f}' for v in accuracies],
                                  textposition='outside'))
            fig.add_trace(go.Bar(name='F1-Score', x=model_names, y=f1s,
                                  marker_color='#7c4dff', text=[f'{v:.4f}' for v in f1s],
                                  textposition='outside'))
            fig.update_layout(
                barmode='group', title='Model Performance Comparison',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'), yaxis=dict(range=[0, 1.1]),
                xaxis=dict(color='#e0e0e0'), legend=dict(font=dict(color='#e0e0e0'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show saved NN images
        nn_imgs = [f for f in os.listdir(IMG_DIR) if 'training' in f or 'confusion' in f or 'comparison' in f] \
                  if os.path.exists(IMG_DIR) else []
        
        if nn_imgs:
            img_cols = st.columns(2)
            for i, img in enumerate(sorted(nn_imgs)):
                with img_cols[i % 2]:
                    st.image(os.path.join(IMG_DIR, img),
                             caption=img.replace('.png', '').replace('_', ' ').title(),
                             use_column_width=True)
    
    with tab_ar:
        st.markdown('<div class="section-header">Kết quả Luật kết hợp (FP-Growth)</div>', unsafe_allow_html=True)
        
        if len(rules) > 0:
            st.metric("Tổng số luật", len(rules))
            
            # Top 20 rules table
            st.markdown("**Top 20 Luật tốt nhất (theo Lift):**")
            top_rules = rules.head(20)[['antecedents_str', 'consequents_str', 
                                         'support', 'confidence', 'lift']].copy()
            top_rules.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift']
            top_rules['Lift'] = top_rules['Lift'].round(3)
            top_rules['Support'] = top_rules['Support'].round(4)
            top_rules['Confidence'] = top_rules['Confidence'].round(3)
            st.dataframe(top_rules, use_container_width=True)
            
            # AR scatter plot
            fig_ar = px.scatter(
                rules.head(100), x='support', y='confidence',
                size='lift', color='lift',
                color_continuous_scale='Reds',
                title='Support vs Confidence (size=Lift)',
                hover_data=['antecedents_str', 'consequents_str']
            )
            fig_ar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                xaxis=dict(color='#e0e0e0', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(color='#e0e0e0', gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_ar, use_container_width=True)
            
            # AR images
            ar_imgs = ['15_association_rules.png', '16_ar_bubble.png']
            available_ar = [f for f in ar_imgs if os.path.exists(os.path.join(IMG_DIR, f))]
            if available_ar:
                ar_cols = st.columns(len(available_ar))
                for i, img in enumerate(available_ar):
                    with ar_cols[i]:
                        st.image(os.path.join(IMG_DIR, img), use_column_width=True)
        else:
            st.info("Chưa có kết quả Association Rules. Chạy pipeline trước.")

# ---- PAGE 5: EXPLORE DATASET ----
elif page == "📚 Khám phá Dataset":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title" style="font-size: 2rem;">📚 Khám phá Dataset</div>
        <div class="hero-subtitle">Duyệt và tìm kiếm trong dữ liệu đã xử lý</div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_processed_data()
    
    if len(df) == 0:
        st.info("Chưa có dữ liệu. Chạy pipeline trước.")
    else:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            source_filter = st.selectbox("Nguồn:", ["All"] + sorted(df['source'].unique().tolist()))
        with col2:
            diff_filter = st.selectbox("Độ khó:", ["All"] + sorted(df['difficulty'].unique().tolist()))
        with col3:
            search_query = st.text_input("Tìm kiếm:", placeholder="Tìm theo tiêu đề...")
        with col4:
            n_display = st.selectbox("Hiển thị:", [50, 100, 200, 500])
        
        # Apply filters
        filtered = df.copy()
        if source_filter != "All":
            filtered = filtered[filtered['source'] == source_filter]
        if diff_filter != "All":
            filtered = filtered[filtered['difficulty'] == diff_filter]
        if search_query:
            mask = filtered['title'].str.contains(search_query, case=False, na=False)
            filtered = filtered[mask]
        
        st.markdown(f"**Kết quả: {len(filtered):,} bản ghi** (hiển thị {min(n_display, len(filtered))})")
        
        display_cols = ['title', 'category', 'difficulty', 'source', 'view_count']
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        # Colorize difficulty
        def color_difficulty(val):
            colors = {'Easy': 'background-color: #1b5e20; color: #a5d6a7',
                      'Medium': 'background-color: #e65100; color: #ffcc80',
                      'Hard': 'background-color: #b71c1c; color: #ef9a9a'}
            return colors.get(val, '')
        
        styled_df = filtered[display_cols].head(n_display)
        st.dataframe(
            styled_df.style.applymap(color_difficulty, subset=['difficulty']),
            use_container_width=True,
            height=450
        )
        
        # Stats
        st.markdown('<div class="section-header">📊 Thống kê nhanh</div>', unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_diff = px.histogram(filtered, x='difficulty', color='difficulty',
                                     color_discrete_map={'Easy': '#4caf50', 'Medium': '#ff9800', 'Hard': '#f44336'},
                                     title='Phân bố Độ khó (đã lọc)')
            fig_diff.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#e0e0e0'), showlegend=False)
            st.plotly_chart(fig_diff, use_container_width=True)
        
        with col_s2:
            if 'view_count' in filtered.columns:
                vc = filtered['view_count'].clip(upper=filtered['view_count'].quantile(0.95))
                fig_vc = px.histogram(filtered, x='view_count', nbins=50, title='Phân bố View Count',
                                       color_discrete_sequence=['#4fc3f7'])
                fig_vc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color='#e0e0e0'))
                st.plotly_chart(fig_vc, use_container_width=True)
        
        # Download filtered
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download filtered data", data=csv,
                           file_name="filtered_data.csv", mime="text/csv")

# ---- FOOTER ----
st.divider()
st.markdown("""
<div style="text-align: center; color: #546e7a; font-size: 0.8rem; padding: 1rem;">
    <span style="color: #4fc3f7;">🎓 ExerciseAI</span> · Đồ án Khai phá Dữ liệu · 
    Pipeline: TF-IDF → EM Clustering → Neural Network → Association Rules → Recommendation<br>
    Dataset: Khan Academy (52K videos) + Coursera (41K courses)
</div>
""", unsafe_allow_html=True)
