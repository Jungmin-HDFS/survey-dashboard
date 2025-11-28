"""
청년 특강 설문조사 실시간 대시보드
=====================================
구글 폼 응답을 실시간으로 분석하고 시각화하는 Streamlit 대시보드

기능:
- 7점 리커트 척도 문항 분포 시각화
- 인구통계학적 특성 분석
- 주관식 응답 워드클라우드
- 자동 새로고침 (30초)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# 통계 분석 모듈 import
try:
    from analysis import (
        run_linear_regression,
        run_logistic_regression, 
        run_factor_analysis,
        run_topic_modeling,
        generate_demo_data_for_analysis
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="청년 특강 설문조사 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 커스텀 CSS - 깔끔한 학술/정책 연구 스타일
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* 헤더 스타일 */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.15);
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* 섹션 카드 */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* 실시간 배지 */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
    }
    
    /* Streamlit 기본 요소 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 데이터 로드 함수
# ============================================================

# 문항 정보 정의
LIKERT_QUESTIONS = {
    'Q1': '내가 가진 강점과 약점을 알고 있다',
    'Q2': '향후 3년 내의 단기적 인생 목표를 갖고 있다',
    'Q3': '나는 졸업 후 원하는 일을 시작할 자신이 있다',
    'Q4': '나는 졸업 후 직업을 가질 필요가 없다',
    'Q5': '진로를 위한 구체적인 준비를 하고 있다',
    'Q6': '내 인생의 전망은 밝다',
    'Q7': '세상에는 많은 기회가 있다'
}

LIKERT_SHORT = {
    'Q1': '자기인식',
    'Q2': '목표설정',
    'Q3': '취업자신감',
    'Q4': '무직업필요',
    'Q5': '진로준비',
    'Q6': '인생전망',
    'Q7': '기회인식'
}

@st.cache_data(ttl=30)  # 30초마다 캐시 갱신
def load_data_from_sheet(sheet_url: str) -> pd.DataFrame:
    """구글 시트에서 데이터 로드 (공개 시트용)"""
    try:
        # 구글 시트 URL을 CSV export URL로 변환
        if '/edit' in sheet_url:
            csv_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
            csv_url = csv_url.replace('/edit?usp=sharing', '/export?format=csv')
        elif 'spreadsheets/d/' in sheet_url:
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
        else:
            csv_url = sheet_url
        
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def generate_demo_data(n: int = 100) -> pd.DataFrame:
    """데모용 샘플 데이터 생성 - 분석 모듈 연동"""
    # 분석 모듈이 있으면 그쪽 함수 사용 (더 정교한 패턴 포함)
    if ANALYSIS_AVAILABLE:
        return generate_demo_data_for_analysis(n)
    
    # 분석 모듈 없을 때 기본 데이터 생성
    np.random.seed(42)
    
    # 리커트 척도 응답 생성 (약간의 패턴 포함)
    data = {
        'timestamp': pd.date_range(start='2024-01-15 09:00', periods=n, freq='5min'),
        'Q1': np.random.choice([4, 5, 6, 7], n, p=[0.15, 0.35, 0.35, 0.15]),  # 자기인식 높음
        'Q2': np.random.choice([2, 3, 4, 5, 6], n, p=[0.1, 0.2, 0.3, 0.25, 0.15]),  # 목표설정 보통
        'Q3': np.random.choice([3, 4, 5, 6], n, p=[0.2, 0.35, 0.3, 0.15]),  # 취업자신감 보통
        'Q4': np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),  # 무직업필요 낮음
        'Q5': np.random.choice([2, 3, 4, 5], n, p=[0.15, 0.3, 0.35, 0.2]),  # 진로준비 보통
        'Q6': np.random.choice([3, 4, 5, 6], n, p=[0.15, 0.3, 0.35, 0.2]),  # 인생전망 보통~긍정
        'Q7': np.random.choice([4, 5, 6, 7], n, p=[0.1, 0.3, 0.4, 0.2]),  # 기회인식 긍정
        'gender': np.random.choice(['남', '여'], n, p=[0.45, 0.55]),
        'age': np.random.choice([21, 22, 23, 24, 25, 26], n, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05]),
        'semester': np.random.choice([5, 6, 7, 8], n, p=[0.2, 0.3, 0.3, 0.2]),
    }
    
    # 주관식 응답 샘플
    open_responses = [
        "대학원 진학이 취업에 도움이 될까요?",
        "비전공자도 연구원이 될 수 있나요?",
        "유학 vs 국내 대학원 어떤 게 나을까요?",
        "박사 졸업 후 진로가 궁금합니다",
        "정책연구 분야 전망이 어떤가요?",
        "워라밸이 어떤지 궁금해요",
        "연구원 연봉 수준이 궁금합니다",
        "석사만으로도 연구직 가능한가요?",
        "문과생도 정책연구 할 수 있나요?",
        "AI 시대에 정책연구자 역할은?",
        "해외 취업 기회가 있을까요?",
        "인턴 경험이 중요한가요?",
        "논문 실적이 얼마나 필요한가요?",
        "통계 분석 능력 어느 정도 필요해요?",
        "청년 정책의 미래가 궁금합니다",
        "",  # 일부 미응답
        "연구소 생활은 어떤가요?",
        "경제학과인데 정책연구 가능할까요?",
        "사회학 전공자 수요가 있나요?",
        "공무원 vs 연구원 어떤 게 나을까요?",
    ]
    
    data['Q11'] = np.random.choice(open_responses, n)
    
    return pd.DataFrame(data)

# ============================================================
# 시각화 함수
# ============================================================

def create_likert_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """리커트 척도 문항별 분포 차트"""
    
    colors = ['#dc2626', '#ea580c', '#f59e0b', '#84cc16', '#22c55e', '#0ea5e9', '#6366f1']
    
    fig = go.Figure()
    
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']
    y_labels = [LIKERT_SHORT[q] for q in questions]
    
    for i, score in enumerate(range(1, 8)):
        percentages = []
        for q in questions:
            if q in df.columns:
                pct = (df[q] == score).sum() / len(df) * 100
            else:
                pct = 0
            percentages.append(pct)
        
        fig.add_trace(go.Bar(
            name=f'{score}점',
            y=y_labels,
            x=percentages,
            orientation='h',
            marker_color=colors[i],
            text=[f'{p:.0f}%' if p >= 5 else '' for p in percentages],
            textposition='inside',
            textfont=dict(color='white', size=11),
            hovertemplate='%{y}: %{x:.1f}%<extra>%{fullData.name}</extra>'
        ))
    
    fig.update_layout(
        barmode='stack',
        height=350,
        margin=dict(l=0, r=20, t=30, b=0),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title='',
            font=dict(size=11)
        ),
        xaxis=dict(
            title='응답 비율 (%)',
            range=[0, 100],
            showgrid=True,
            gridcolor='#f1f5f9'
        ),
        yaxis=dict(
            title='',
            autorange='reversed'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Noto Sans KR')
    )
    
    return fig

def create_mean_score_chart(df: pd.DataFrame) -> go.Figure:
    """문항별 평균 점수 차트"""
    
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']
    means = []
    stds = []
    labels = []
    
    for q in questions:
        if q in df.columns:
            means.append(df[q].mean())
            stds.append(df[q].std())
        else:
            means.append(0)
            stds.append(0)
        labels.append(LIKERT_SHORT[q])
    
    # 색상: 4점 기준으로 긍정/부정 구분 (Q4는 역코딩 고려)
    colors = []
    for i, (q, m) in enumerate(zip(questions, means)):
        if q == 'Q4':  # Q4는 낮을수록 긍정적
            colors.append('#22c55e' if m < 4 else '#f59e0b' if m < 5 else '#ef4444')
        else:
            colors.append('#22c55e' if m >= 5 else '#f59e0b' if m >= 4 else '#ef4444')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        marker_color=colors,
        text=[f'{m:.2f}' for m in means],
        textposition='outside',
        textfont=dict(size=12, color='#1e3a5f'),
        error_y=dict(
            type='data',
            array=stds,
            visible=True,
            color='#94a3b8',
            thickness=1.5
        ),
        hovertemplate='%{x}<br>평균: %{y:.2f}<extra></extra>'
    ))
    
    # 중립선 (4점)
    fig.add_hline(y=4, line_dash="dash", line_color="#94a3b8", 
                  annotation_text="중립(4점)", annotation_position="right")
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(title='', tickangle=0),
        yaxis=dict(title='평균 점수', range=[1, 7.5], dtick=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Noto Sans KR'),
        showlegend=False
    )
    
    return fig

def create_demographic_charts(df: pd.DataFrame) -> tuple:
    """인구통계 차트들"""
    
    # 성별 분포
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        fig_gender = go.Figure(data=[go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            hole=0.5,
            marker_colors=['#3b82f6', '#ec4899'],
            textinfo='label+percent',
            textfont=dict(size=12)
        )])
        fig_gender.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            paper_bgcolor='white'
        )
    else:
        fig_gender = go.Figure()
    
    # 학기 분포
    if 'semester' in df.columns:
        semester_counts = df['semester'].value_counts().sort_index()
        fig_semester = go.Figure(data=[go.Bar(
            x=[f'{s}학기' for s in semester_counts.index],
            y=semester_counts.values,
            marker_color='#6366f1',
            text=semester_counts.values,
            textposition='outside'
        )])
        fig_semester.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title=''),
            yaxis=dict(title='', showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        fig_semester = go.Figure()
    
    # 나이 분포
    if 'age' in df.columns:
        age_counts = df['age'].value_counts().sort_index()
        fig_age = go.Figure(data=[go.Bar(
            x=[f'{a}세' for a in age_counts.index],
            y=age_counts.values,
            marker_color='#14b8a6',
            text=age_counts.values,
            textposition='outside'
        )])
        fig_age.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title=''),
            yaxis=dict(title='', showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        fig_age = go.Figure()
    
    return fig_gender, fig_semester, fig_age

def create_wordcloud(texts: list) -> plt.Figure:
    """주관식 응답 워드클라우드 생성"""
    
    # 텍스트 전처리
    all_text = ' '.join([str(t) for t in texts if pd.notna(t) and str(t).strip()])
    
    if not all_text.strip():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, '응답 데이터가 없습니다', ha='center', va='center', fontsize=14, color='#94a3b8')
        ax.axis('off')
        return fig
    
    # 불용어 정의
    stopwords = {'있나요', '어떤', '있을까요', '할까요', '할까', '어떤가요', '어떻게', 
                 '무엇', '뭐', '왜', '어디', '언제', '누구', '어느', '얼마나', '얼마',
                 '있는', '하는', '되는', '있는지', '인가요', '인지', '은', '는', '이', '가',
                 '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '도', '만', '까지'}
    
    try:
        # 한글 폰트 설정
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        
        wordcloud = WordCloud(
            font_path=font_path,
            width=800,
            height=400,
            background_color='white',
            colormap='Blues',
            max_words=50,
            min_font_size=12,
            max_font_size=80,
            stopwords=stopwords,
            relative_scaling=0.5,
            prefer_horizontal=0.7
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
        
    except Exception as e:
        # 폰트가 없는 경우 대체 시각화
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 단어 빈도 계산
        words = all_text.split()
        word_freq = Counter([w for w in words if w not in stopwords and len(w) > 1])
        top_words = word_freq.most_common(15)
        
        if top_words:
            words_list, counts = zip(*top_words)
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(words_list)))
            bars = ax.barh(range(len(words_list)), counts, color=colors)
            ax.set_yticks(range(len(words_list)))
            ax.set_yticklabels(words_list)
            ax.invert_yaxis()
            ax.set_xlabel('빈도')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, '분석할 텍스트가 부족합니다', ha='center', va='center')
            ax.axis('off')
        
        return fig

def extract_keywords(texts: list) -> list:
    """주요 키워드 추출"""
    all_text = ' '.join([str(t) for t in texts if pd.notna(t) and str(t).strip()])
    
    # 주요 키워드 패턴
    keywords = {
        '대학원': 0, '유학': 0, '박사': 0, '석사': 0, '취업': 0,
        '연봉': 0, '워라밸': 0, '전망': 0, '연구': 0, '정책': 0,
        '인턴': 0, '경험': 0, '논문': 0, '통계': 0, '분석': 0,
        '해외': 0, '공무원': 0, '연구원': 0, '진로': 0, '미래': 0
    }
    
    for keyword in keywords:
        keywords[keyword] = all_text.count(keyword)
    
    # 빈도 기준 정렬
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    return [(k, v) for k, v in sorted_keywords if v > 0][:10]

# ============================================================
# 메인 앱
# ============================================================

def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>📊 청년 특강 설문조사 대시보드</h1>
                <p>청년 진로인식 및 고용전망 실시간 분석</p>
            </div>
            <div class="live-badge">
                <div class="live-dot"></div>
                실시간 업데이트
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 - 데이터 소스 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        data_source = st.radio(
            "데이터 소스",
            ["데모 데이터", "구글 시트 연결"],
            index=0
        )
        
        if data_source == "구글 시트 연결":
            sheet_url = st.text_input(
                "구글 시트 URL",
                placeholder="https://docs.google.com/spreadsheets/u/0/d/1D9WSEOpED13_NyFbhVbRye-Y70tpUEUDggma2_kxhNU"
                #placeholder="https://docs.google.com/spreadsheets/d/..."
            )
            st.caption("시트는 '링크가 있는 모든 사용자'에게 공개되어야 합니다.")
            
            st.markdown("---")
            st.markdown("**컬럼 매핑**")
            st.caption("시트의 컬럼명이 다른 경우 매핑하세요")
            
            col_mapping = {}
            expected_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'gender', 'age', 'semester', 'Q11']
            
            for col in expected_cols:
                col_mapping[col] = st.text_input(f"{col}", value=col, key=f"map_{col}")
        
        st.markdown("---")
        auto_refresh = st.checkbox("자동 새로고침 (30초)", value=True)
        
        if auto_refresh:
            st.markdown("""
            <script>
                setTimeout(function() {
                    window.location.reload();
                }, 30000);
            </script>
            """, unsafe_allow_html=True)
    
    # 데이터 로드
    if data_source == "데모 데이터":
        df = generate_demo_data(100)  # 분석에 충분한 표본 크기
    else:
        if sheet_url:
            df = load_data_from_sheet(sheet_url)
            if df is not None:
                # 컬럼 매핑 적용
                df = df.rename(columns={v: k for k, v in col_mapping.items() if v != k})
        else:
            st.info("👆 사이드바에서 구글 시트 URL을 입력하세요")
            df = generate_demo_data(100)
    
    if df is None:
        st.error("데이터를 불러올 수 없습니다.")
        return
    
    # ============================================================
    # 상단 메트릭 카드
    # ============================================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">총 응답자 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_positive = df[['Q1', 'Q2', 'Q3', 'Q5', 'Q6', 'Q7']].mean().mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_positive:.1f}</div>
            <div class="metric-label">긍정 문항 평균 (7점 만점)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'Q11' in df.columns:
            open_response_rate = (df['Q11'].notna() & (df['Q11'] != '')).sum() / len(df) * 100
        else:
            open_response_rate = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{open_response_rate:.0f}%</div>
            <div class="metric-label">주관식 응답률</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'gender' in df.columns:
            female_ratio = (df['gender'] == '여').sum() / len(df) * 100
        else:
            female_ratio = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{female_ratio:.0f}%</div>
            <div class="metric-label">여성 응답자 비율</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================================
    # 메인 콘텐츠 - 탭 구성
    # ============================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 리커트 척도 분석", 
        "👥 응답자 특성", 
        "📊 선형회귀 분석",
        "🎯 로지스틱 회귀",
        "🔬 요인분석",
        "📑 토픽 모델링"
    ])
    
    with tab1:
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">문항별 응답 분포</div>', unsafe_allow_html=True)
            fig_dist = create_likert_distribution_chart(df)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">문항별 평균 점수</div>', unsafe_allow_html=True)
            fig_mean = create_mean_score_chart(df)
            st.plotly_chart(fig_mean, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 문항 해석 가이드
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 문항 전체 내용</div>', unsafe_allow_html=True)
        
        for q_id, q_text in LIKERT_QUESTIONS.items():
            if q_id in df.columns:
                mean_val = df[q_id].mean()
                color = '#22c55e' if (q_id != 'Q4' and mean_val >= 5) or (q_id == 'Q4' and mean_val < 4) else '#f59e0b' if mean_val >= 4 else '#ef4444'
                st.markdown(f"**{q_id}.** {q_text} — <span style='color:{color}'>평균 {mean_val:.2f}</span>", unsafe_allow_html=True)
        
        st.caption("※ Q4(졸업 후 직업을 가질 필요가 없다)는 역코딩 문항으로, 낮을수록 긍정적")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">성별 분포</div>', unsafe_allow_html=True)
            fig_gender, _, _ = create_demographic_charts(df)
            st.plotly_chart(fig_gender, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">이수 학기</div>', unsafe_allow_html=True)
            _, fig_semester, _ = create_demographic_charts(df)
            st.plotly_chart(fig_semester, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">연령 분포</div>', unsafe_allow_html=True)
            _, _, fig_age = create_demographic_charts(df)
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================
    # Tab 3: 선형회귀 분석
    # ============================================================
    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 다중 선형회귀 분석 결과</div>', unsafe_allow_html=True)
        st.markdown("""
        **분석 설계**: 성별, 이수학기, 연령을 독립변수(X)로, Q1~Q7 각각을 종속변수(Y)로 하는 7개의 회귀모델
        """)
        
        if ANALYSIS_AVAILABLE:
            with st.spinner('선형회귀 분석 중...'):
                lr_results = run_linear_regression(df)
            
            if 'error' in lr_results:
                st.warning(lr_results['error'])
            else:
                st.markdown(f"**분석 대상**: {lr_results['n_obs']}명")
                st.markdown("---")
                
                # 회귀계수 테이블
                summary_df = pd.DataFrame(lr_results['summary_table'])
                
                # 스타일링된 테이블 표시
                st.markdown("#### 회귀계수 (Coefficients)")
                
                # st.dataframe 사용
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.caption("유의수준: †p < .10, *p < .05, **p < .01, ***p < .001 | 성별: 여성=1, 남성=0 (더미코딩)")
                
                # 해석 가이드
                st.markdown("#### 📝 해석 가이드")
                st.info("""
                - **양(+)의 계수**: 해당 변수가 증가하면 종속변수(Q점수)도 증가
                - **음(-)의 계수**: 해당 변수가 증가하면 종속변수(Q점수)가 감소
                - **R²**: 모델의 설명력 (0~1, 높을수록 좋음)
                - **Adj. R²**: 독립변수 수를 고려한 조정된 설명력
                """)
        else:
            st.warning("분석 모듈(statsmodels)이 설치되지 않았습니다. `pip install statsmodels`를 실행하세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================
    # Tab 4: 로지스틱 회귀 분석
    # ============================================================
    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 로지스틱 회귀분석 결과</div>', unsafe_allow_html=True)
        st.markdown("""
        **분석 설계**: Q1~Q7 응답을 독립변수(X)로, 성별(여성=1)을 종속변수(Y)로 하는 이항 로지스틱 회귀
        """)
        
        if ANALYSIS_AVAILABLE:
            with st.spinner('로지스틱 회귀 분석 중...'):
                logit_results = run_logistic_regression(df)
            
            if 'error' in logit_results:
                st.warning(logit_results['error'])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("분석 대상", f"{logit_results['n_obs']}명")
                with col2:
                    st.metric("Pseudo R²", f"{logit_results['pseudo_r2']:.3f}")
                with col3:
                    st.metric("AIC", f"{logit_results['aic']:.1f}")
                
                st.markdown("---")
                
                # 결과 테이블
                st.markdown("#### 회귀계수 및 승산비 (Odds Ratio)")
                
                table_data = []
                for var, stats in logit_results['coefficients'].items():
                    table_data.append({
                        '변수': var,
                        'B (계수)': f"{stats['coef']:.3f}",
                        'SE': f"{stats['se']:.3f}",
                        'z': f"{stats['z']:.2f}",
                        'p-value': f"{stats['p_value']:.3f}{stats['significance']}",
                        'OR (승산비)': f"{stats['odds_ratio']:.3f}",
                        '95% CI': f"[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]"
                    })
                
                result_df = pd.DataFrame(table_data)
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # 해석 문구
                st.markdown("#### 📝 주요 발견 및 해석")
                
                for interp in logit_results['interpretation']:
                    if "유의" in interp:
                        st.success(f"✓ {interp}")
                    else:
                        st.info(f"• {interp}")
                
                # 승산비 시각화
                st.markdown("#### 승산비 Forest Plot")
                
                or_data = []
                for var, stats in logit_results['coefficients'].items():
                    if var.startswith('Q'):
                        or_data.append({
                            'variable': var,
                            'or': stats['odds_ratio'],
                            'ci_lower': stats['ci_lower'],
                            'ci_upper': stats['ci_upper'],
                            'significant': stats['p_value'] < 0.05
                        })
                
                if or_data:
                    or_df = pd.DataFrame(or_data)
                    
                    fig = go.Figure()
                    
                    # 신뢰구간 라인
                    for i, row in or_df.iterrows():
                        color = '#22c55e' if row['significant'] else '#94a3b8'
                        fig.add_trace(go.Scatter(
                            x=[row['ci_lower'], row['ci_upper']],
                            y=[row['variable'], row['variable']],
                            mode='lines',
                            line=dict(color=color, width=2),
                            showlegend=False
                        ))
                    
                    # 승산비 점
                    colors = ['#22c55e' if row['significant'] else '#94a3b8' for _, row in or_df.iterrows()]
                    fig.add_trace(go.Scatter(
                        x=or_df['or'],
                        y=or_df['variable'],
                        mode='markers',
                        marker=dict(size=12, color=colors),
                        name='승산비',
                        hovertemplate='%{y}: OR=%{x:.2f}<extra></extra>'
                    ))
                    
                    # 기준선 (OR=1)
                    fig.add_vline(x=1, line_dash="dash", line_color="#ef4444", 
                                  annotation_text="OR=1 (기준선)")
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis_title="승산비 (Odds Ratio)",
                        yaxis_title="",
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.caption("""
                **해석 가이드**
                - OR > 1: 해당 변수가 높을수록 여성일 확률 증가
                - OR < 1: 해당 변수가 높을수록 여성일 확률 감소
                - OR = 1: 성별과 무관
                - 95% CI가 1을 포함하지 않으면 통계적으로 유의
                """)
        else:
            st.warning("분석 모듈(statsmodels)이 설치되지 않았습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================
    # Tab 5: 요인분석
    # ============================================================
    with tab5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 요인분석 결과</div>', unsafe_allow_html=True)
        
        if ANALYSIS_AVAILABLE:
            with st.spinner('요인분석 중...'):
                fa_results = run_factor_analysis(df)
            
            if 'error' in fa_results:
                st.warning(fa_results['error'])
            else:
                st.markdown(f"**분석 대상**: {fa_results['n_obs']}명")
                
                # EFA 섹션
                st.markdown("---")
                st.markdown("### 1️⃣ 탐색적 요인분석 (EFA)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    kmo = fa_results['efa'].get('kmo')
                    kmo_str = f"{kmo:.3f}" if kmo else "N/A"
                    st.metric("KMO 적합도", kmo_str)
                    if kmo and kmo >= 0.7:
                        st.caption("✓ 양호 (≥0.7)")
                    elif kmo and kmo >= 0.6:
                        st.caption("△ 보통 (≥0.6)")
                    else:
                        st.caption("✗ 부적합 (<0.6)")
                
                with col2:
                    bartlett_p = fa_results['efa'].get('bartlett_p')
                    st.metric("Bartlett 검정 p", f"{bartlett_p:.4f}" if bartlett_p else "N/A")
                    if bartlett_p and bartlett_p < 0.05:
                        st.caption("✓ 유의 (요인분석 적합)")
                
                with col3:
                    n_factors = fa_results['efa']['n_factors_optimal']
                    st.metric("최적 요인 수", f"{n_factors}개")
                    st.caption("(고유값 > 1 기준)")
                
                # 스크리 도표
                st.markdown("#### 스크리 도표 (Scree Plot)")
                eigenvalues = fa_results['efa']['eigenvalues']
                
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Scatter(
                    x=list(range(1, len(eigenvalues) + 1)),
                    y=eigenvalues,
                    mode='lines+markers',
                    marker=dict(size=10, color='#1e3a5f'),
                    line=dict(color='#1e3a5f', width=2),
                    name='고유값'
                ))
                fig_scree.add_hline(y=1, line_dash="dash", line_color="#ef4444",
                                    annotation_text="Kaiser 기준 (고유값=1)")
                fig_scree.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_title="요인 번호",
                    yaxis_title="고유값 (Eigenvalue)",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_scree, use_container_width=True)
                
                # EFA 요인적재량
                st.markdown("#### EFA 요인적재량 (Varimax 회전)")
                loadings_df = pd.DataFrame(fa_results['efa']['loadings'])
                
                # 히트맵으로 표시
                fig_loadings = go.Figure(data=go.Heatmap(
                    z=loadings_df.values,
                    x=loadings_df.columns,
                    y=loadings_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(loadings_df.values, 3),
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    hovertemplate='%{y} → %{x}: %{z:.3f}<extra></extra>'
                ))
                fig_loadings.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_title="요인",
                    yaxis_title="문항"
                )
                st.plotly_chart(fig_loadings, use_container_width=True)
                
                # 설명된 분산
                variance = fa_results['efa']['variance_explained']
                cumvar = fa_results['efa']['cumulative_variance']
                st.markdown(f"**설명된 총 분산**: {cumvar[-1]*100:.1f}%")
                
                # CFA 섹션
                st.markdown("---")
                st.markdown("### 2️⃣ 확인적 요인분석 (CFA)")
                st.markdown("""
                **사전 지정된 요인구조**:
                - **요인1 (자기효능)**: Q1(자기인식), Q2(목표설정), Q5(진로준비)
                - **요인2 (미래전망)**: Q3(취업자신감), Q6(인생전망), Q7(기회인식)
                - **요인3 (직업무관심)**: Q4(무직업필요)
                """)
                
                # CFA 결과 테이블 - st.dataframe 사용
                cfa_table = fa_results['cfa']['loading_table']
                
                # 데이터프레임으로 변환
                cfa_df = pd.DataFrame(cfa_table)
                cfa_df['요인적재량'] = cfa_df['요인적재량'].apply(lambda x: f"{x:.3f}")
                cfa_df['Cronbach α'] = cfa_df['Cronbach α'].apply(lambda x: f"{x:.3f}" if x else "-")
                
                st.dataframe(cfa_df, use_container_width=True, hide_index=True)
                
                # 적합도 지수
                st.markdown("#### 모형 적합도 지수")
                fit = fa_results['cfa']['fit_indices']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSR", f"{fit['RMSR']:.3f}")
                    st.caption("< 0.08 권장")
                with col2:
                    st.metric("GFI (근사)", f"{fit['GFI_approx']:.3f}")
                    st.caption("> 0.90 권장")
                
                st.info(fit['note'])
                
        else:
            st.warning("분석 모듈(factor_analyzer)이 설치되지 않았습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================
    # Tab 6: 토픽 모델링
    # ============================================================
    with tab6:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📑 LDA 토픽 모델링 결과</div>', unsafe_allow_html=True)
        st.markdown("""
        **분석 방법**: Q11 주관식 응답에 대한 Latent Dirichlet Allocation (LDA) 토픽 모델링
        """)
        
        if ANALYSIS_AVAILABLE and 'Q11' in df.columns:
            texts = df['Q11'].dropna().tolist()
            
            with st.spinner('토픽 모델링 분석 중...'):
                lda_results = run_topic_modeling(texts, n_topics=5)
            
            if 'error' in lda_results:
                st.warning(lda_results['error'])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("분석 문서 수", f"{lda_results['n_documents']}개")
                with col2:
                    st.metric("추출 토픽 수", f"{lda_results['n_topics']}개")
                with col3:
                    st.metric("고유 단어 수", f"{lda_results['n_features']}개")
                
                st.markdown("---")
                
                # 토픽별 결과
                st.markdown("### 🏷️ 토픽별 분류 결과")
                
                for topic in lda_results['topics']:
                    with st.expander(f"**토픽 {topic['topic_id']}**: {topic['label']}", expanded=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("**주요 키워드**")
                            # 키워드를 텍스트로 표시
                            keyword_text = " · ".join([word for word, weight in topic['keywords']])
                            st.write(keyword_text)
                            
                            st.markdown(f"**문서 수**: {topic['document_count']}개")
                        
                        with col2:
                            st.markdown("**🏆 가장 적합도 높은 문항**")
                            best_doc = topic['best_document']
                            st.success(f"「{best_doc['text']}」")
                            st.caption(f"토픽 적합도: {best_doc['score']:.3f}")
                
                # 토픽 분포 시각화
                st.markdown("---")
                st.markdown("### 📊 토픽 분포")
                
                topic_dist = lda_results['topic_distribution']
                topic_labels = [f"토픽 {k+1}" for k in sorted(topic_dist.keys())]
                topic_counts = [topic_dist[k] for k in sorted(topic_dist.keys())]
                
                fig_topic = go.Figure(data=[go.Pie(
                    labels=topic_labels,
                    values=topic_counts,
                    hole=0.4,
                    marker_colors=['#1e3a5f', '#2d5a87', '#4a7c9b', '#6b9eb8', '#8ec0d6'],
                    textinfo='label+percent',
                    textposition='outside'
                )])
                fig_topic.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=False
                )
                st.plotly_chart(fig_topic, use_container_width=True)
                
                st.caption("""
                **분석 방법론**
                - 텍스트 전처리: 한글 형태소 추출, 불용어 제거
                - 모델: Latent Dirichlet Allocation (LDA)
                - 토픽 수: 문서 수 기반 자동 조정 (최대 5개)
                """)
        
        elif 'Q11' not in df.columns:
            st.warning("주관식 응답(Q11) 데이터가 없습니다.")
        else:
            st.warning("분석 모듈(scikit-learn)이 설치되지 않았습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
