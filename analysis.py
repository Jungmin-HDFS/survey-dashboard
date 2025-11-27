"""
통계 분석 모듈
================
선형회귀, 로지스틱 회귀, 요인분석, 토픽모델링 분석 함수
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 선형회귀 분석
# ============================================================

def run_linear_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    성별, 이수학기, 연령을 독립변수로, Q1~Q7 각각을 종속변수로 하는
    7개의 선형회귀 모델 실행
    
    Returns:
        결과 딕셔너리: coefficients, std_errors, t_values, p_values, r_squared 등
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels 패키지가 필요합니다'}
    
    # 데이터 준비
    df_clean = df.copy()
    
    # 성별 더미 코딩 (여성=1, 남성=0)
    df_clean['gender_dummy'] = (df_clean['gender'] == '여').astype(int)
    
    # 독립변수
    X_cols = ['gender_dummy', 'semester', 'age']
    Y_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']
    
    # 결측치 제거
    analysis_cols = X_cols + Y_cols
    df_analysis = df_clean[analysis_cols].dropna()
    
    if len(df_analysis) < 10:
        return {'error': '분석에 필요한 데이터가 부족합니다 (최소 10개 필요)'}
    
    # 독립변수 행렬 (상수항 포함)
    X = sm.add_constant(df_analysis[X_cols])
    
    results = {
        'models': {},
        'summary_table': [],
        'n_obs': len(df_analysis)
    }
    
    var_names = ['(상수)', '성별(여=1)', '이수학기', '연령']
    
    for y_col in Y_cols:
        y = df_analysis[y_col]
        model = sm.OLS(y, X).fit()
        
        model_result = {
            'coefficients': model.params.values,
            'std_errors': model.bse.values,
            't_values': model.tvalues.values,
            'p_values': model.pvalues.values,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_stat': model.fvalue,
            'f_pvalue': model.f_pvalue
        }
        results['models'][y_col] = model_result
    
    # 요약 테이블 생성 (회귀계수 + 유의성)
    summary_rows = []
    for i, var in enumerate(var_names):
        row = {'변수': var}
        for y_col in Y_cols:
            coef = results['models'][y_col]['coefficients'][i]
            pval = results['models'][y_col]['p_values'][i]
            
            # 유의성 표시
            stars = ''
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            elif pval < 0.1:
                stars = '†'
            
            row[y_col] = f"{coef:.3f}{stars}"
        summary_rows.append(row)
    
    # R² 행 추가
    r2_row = {'변수': 'R²'}
    adj_r2_row = {'변수': 'Adj. R²'}
    f_row = {'변수': 'F통계량'}
    
    for y_col in Y_cols:
        r2_row[y_col] = f"{results['models'][y_col]['r_squared']:.3f}"
        adj_r2_row[y_col] = f"{results['models'][y_col]['adj_r_squared']:.3f}"
        f_val = results['models'][y_col]['f_stat']
        f_pval = results['models'][y_col]['f_pvalue']
        f_stars = '***' if f_pval < 0.001 else '**' if f_pval < 0.01 else '*' if f_pval < 0.05 else ''
        f_row[y_col] = f"{f_val:.2f}{f_stars}"
    
    summary_rows.extend([r2_row, adj_r2_row, f_row])
    results['summary_table'] = summary_rows
    
    return results


# ============================================================
# 2. 로지스틱 회귀 분석
# ============================================================

def run_logistic_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Q1~Q7을 독립변수로, 성별을 종속변수로 하는 로지스틱 회귀분석
    
    Returns:
        결과 딕셔너리: coefficients, odds_ratios, p_values, confidence_intervals 등
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels 패키지가 필요합니다'}
    
    df_clean = df.copy()
    
    # 종속변수: 성별 (여성=1, 남성=0)
    df_clean['gender_binary'] = (df_clean['gender'] == '여').astype(int)
    
    # 독립변수
    X_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']
    
    # 결측치 제거
    analysis_cols = X_cols + ['gender_binary']
    df_analysis = df_clean[analysis_cols].dropna()
    
    if len(df_analysis) < 20:
        return {'error': '분석에 필요한 데이터가 부족합니다 (최소 20개 필요)'}
    
    X = sm.add_constant(df_analysis[X_cols])
    y = df_analysis['gender_binary']
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        return {'error': f'모델 추정 실패: {str(e)}'}
    
    # 승산비 계산
    odds_ratios = np.exp(model.params)
    
    # 신뢰구간 (95%)
    conf_int = model.conf_int()
    conf_int_or = np.exp(conf_int)
    
    results = {
        'n_obs': len(df_analysis),
        'pseudo_r2': model.prsquared,
        'log_likelihood': model.llf,
        'aic': model.aic,
        'bic': model.bic,
        'coefficients': {},
        'interpretation': []
    }
    
    var_names = ['(상수)'] + X_cols
    
    for i, var in enumerate(var_names):
        coef = model.params.iloc[i]
        se = model.bse.iloc[i]
        z_val = model.tvalues.iloc[i]
        p_val = model.pvalues.iloc[i]
        or_val = odds_ratios.iloc[i]
        ci_lower = conf_int_or.iloc[i, 0]
        ci_upper = conf_int_or.iloc[i, 1]
        
        # 유의성 표시
        stars = ''
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        elif p_val < 0.1:
            stars = '†'
        
        results['coefficients'][var] = {
            'coef': coef,
            'se': se,
            'z': z_val,
            'p_value': p_val,
            'odds_ratio': or_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significance': stars
        }
        
        # 해석 문구 생성 (Q 변수만)
        if var.startswith('Q') and p_val < 0.1:
            direction = "높을수록" if coef > 0 else "낮을수록"
            effect = "여성일 확률이 높습니다" if coef > 0 else "여성일 확률이 낮습니다"
            
            interpretation = f"{var} 점수가 {direction} {effect} "
            interpretation += f"(OR={or_val:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], p={p_val:.3f})"
            
            if p_val < 0.05:
                interpretation += " [통계적으로 유의]"
            else:
                interpretation += " [경계 유의]"
            
            results['interpretation'].append(interpretation)
    
    if not results['interpretation']:
        results['interpretation'].append("통계적으로 유의한 예측변수가 없습니다 (p < 0.1).")
    
    return results


# ============================================================
# 3. 요인분석
# ============================================================

def run_factor_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Q1~Q7에 대한 탐색적 요인분석(EFA) 및 확인적 요인분석(CFA)
    
    CFA 요인구조:
    - 요인1 (자기효능): Q1, Q2, Q5
    - 요인2 (미래전망): Q3, Q6, Q7
    - 요인3 (직업무관심): Q4
    
    Returns:
        EFA 결과, CFA 결과, 적합도 지수
    """
    try:
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    except ImportError:
        return {'error': 'factor_analyzer 패키지가 필요합니다', 'efa': None, 'cfa': None}
    
    Q_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']
    df_analysis = df[Q_cols].dropna()
    
    if len(df_analysis) < 30:
        return {'error': '분석에 필요한 데이터가 부족합니다 (최소 30개 필요)'}
    
    X = df_analysis.values
    
    results = {
        'n_obs': len(df_analysis),
        'efa': {},
        'cfa': {}
    }
    
    # ============ EFA ============
    
    # KMO 검정
    try:
        kmo_all, kmo_model = calculate_kmo(X)
        results['efa']['kmo'] = kmo_model
    except:
        results['efa']['kmo'] = None
    
    # Bartlett 검정
    try:
        chi_square, p_value = calculate_bartlett_sphericity(X)
        results['efa']['bartlett_chi2'] = chi_square
        results['efa']['bartlett_p'] = p_value
    except:
        results['efa']['bartlett_chi2'] = None
        results['efa']['bartlett_p'] = None
    
    # 최적 요인 수 결정 (고유값 > 1 기준)
    fa_initial = FactorAnalyzer(rotation=None, n_factors=7, method='principal')
    fa_initial.fit(X)
    eigenvalues = fa_initial.get_eigenvalues()[0]
    
    n_factors_optimal = sum(eigenvalues > 1)
    n_factors_optimal = max(1, min(n_factors_optimal, 3))  # 1~3 범위로 제한
    
    results['efa']['eigenvalues'] = eigenvalues.tolist()
    results['efa']['n_factors_optimal'] = n_factors_optimal
    
    # 최적 요인 수로 EFA 실행 (Varimax 회전)
    fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors_optimal, method='principal')
    fa.fit(X)
    
    loadings = pd.DataFrame(
        fa.loadings_,
        index=Q_cols,
        columns=[f'요인{i+1}' for i in range(n_factors_optimal)]
    )
    
    results['efa']['loadings'] = loadings.to_dict()
    results['efa']['variance_explained'] = fa.get_factor_variance()[1].tolist()  # 비율
    results['efa']['cumulative_variance'] = fa.get_factor_variance()[2].tolist()  # 누적
    
    # ============ CFA ============
    # Note: 실제 CFA는 semopy 또는 lavaan 필요. 여기서는 간이 CFA 구현
    
    # 요인구조 정의
    cfa_structure = {
        '자기효능': ['Q1', 'Q2', 'Q5'],
        '미래전망': ['Q3', 'Q6', 'Q7'],
        '직업무관심': ['Q4']
    }
    
    # 각 요인에 대해 내적 일관성 및 요인적재량 계산
    cfa_results = {}
    
    for factor_name, items in cfa_structure.items():
        factor_data = df_analysis[items]
        
        # Cronbach's Alpha
        if len(items) > 1:
            item_vars = factor_data.var()
            total_var = factor_data.sum(axis=1).var()
            n_items = len(items)
            alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
        else:
            alpha = None
        
        # 요인점수 계산 (평균)
        factor_score = factor_data.mean(axis=1)
        
        # 각 문항과 요인점수의 상관 (요인적재량 대용)
        loadings_cfa = {}
        for item in items:
            corr = factor_data[item].corr(factor_score)
            loadings_cfa[item] = corr
        
        cfa_results[factor_name] = {
            'items': items,
            'cronbach_alpha': alpha,
            'loadings': loadings_cfa,
            'mean': factor_score.mean(),
            'std': factor_score.std()
        }
    
    results['cfa']['factors'] = cfa_results
    
    # 간이 적합도 지수 계산
    # 실제 CFA fit indices는 SEM 패키지 필요, 여기서는 근사값 제공
    
    # 모형 기반 상관행렬 vs 관측 상관행렬 비교
    observed_corr = df_analysis.corr()
    
    # 요인구조 기반 재현 상관행렬 (간이 계산)
    factor_scores = pd.DataFrame()
    for factor_name, items in cfa_structure.items():
        factor_scores[factor_name] = df_analysis[items].mean(axis=1)
    
    # 잔차 계산
    residuals = []
    for i, q1 in enumerate(Q_cols):
        for j, q2 in enumerate(Q_cols):
            if i < j:
                obs_corr = observed_corr.loc[q1, q2]
                # 같은 요인인지 확인
                same_factor = False
                for items in cfa_structure.values():
                    if q1 in items and q2 in items:
                        same_factor = True
                        break
                
                if same_factor:
                    # 같은 요인이면 높은 상관 기대
                    expected = 0.5
                else:
                    # 다른 요인이면 낮은 상관 기대
                    expected = 0.2
                
                residuals.append((obs_corr - expected) ** 2)
    
    rmsr = np.sqrt(np.mean(residuals))
    
    # 적합도 지수 (근사값)
    n = len(df_analysis)
    p = len(Q_cols)
    
    results['cfa']['fit_indices'] = {
        'RMSR': rmsr,
        'GFI_approx': max(0, 1 - rmsr),  # 근사 GFI
        'note': '* 정확한 CFI, TLI, RMSEA는 SEM 패키지(semopy) 필요'
    }
    
    # CFA 요인적재량 테이블
    cfa_loading_table = []
    for factor_name, factor_info in cfa_results.items():
        for item, loading in factor_info['loadings'].items():
            cfa_loading_table.append({
                '요인': factor_name,
                '문항': item,
                '요인적재량': loading,
                'Cronbach α': factor_info['cronbach_alpha']
            })
    
    results['cfa']['loading_table'] = cfa_loading_table
    
    return results


# ============================================================
# 4. 토픽 모델링 (LDA)
# ============================================================

def run_topic_modeling(texts: List[str], n_topics: int = 5) -> Dict[str, Any]:
    """
    주관식 응답에 대한 LDA 토픽 모델링
    
    Args:
        texts: 텍스트 리스트
        n_topics: 토픽 수 (기본값 5)
    
    Returns:
        토픽별 키워드, 대표 문서 등
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
    except ImportError:
        return {'error': 'scikit-learn 패키지가 필요합니다'}
    
    # 빈 텍스트 제거
    valid_texts = [str(t).strip() for t in texts if pd.notna(t) and str(t).strip()]
    
    if len(valid_texts) < 10:
        return {'error': '분석에 필요한 텍스트가 부족합니다 (최소 10개 필요)'}
    
    # 한국어 불용어
    korean_stopwords = [
        '있나요', '어떤', '있을까요', '할까요', '할까', '어떤가요', '어떻게',
        '무엇', '뭐', '왜', '어디', '언제', '누구', '어느', '얼마나', '얼마',
        '있는', '하는', '되는', '있는지', '인가요', '인지', '은', '는', '이', '가',
        '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '도', '만', '까지',
        '그리고', '그러나', '하지만', '그래서', '따라서', '또한', '및', '등',
        '수', '것', '때', '중', '더', '잘', '좀', '많이', '정말', '너무',
        '궁금합니다', '궁금해요', '싶습니다', '싶어요', '알고', '알려', '주세요'
    ]
    
    # CountVectorizer로 문서-단어 행렬 생성
    vectorizer = CountVectorizer(
        max_features=100,
        min_df=2,
        max_df=0.9,
        token_pattern=r'[가-힣]+',  # 한글만 추출
        stop_words=korean_stopwords
    )
    
    try:
        doc_term_matrix = vectorizer.fit_transform(valid_texts)
    except ValueError as e:
        return {'error': f'텍스트 벡터화 실패: 유효한 단어가 부족합니다'}
    
    feature_names = vectorizer.get_feature_names_out()
    
    if len(feature_names) < 5:
        return {'error': '분석에 필요한 고유 단어가 부족합니다'}
    
    # LDA 모델 학습
    n_topics = min(n_topics, len(valid_texts) // 3, 5)  # 토픽 수 조정
    n_topics = max(2, n_topics)
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    
    doc_topic_dist = lda.fit_transform(doc_term_matrix)
    
    results = {
        'n_topics': n_topics,
        'n_documents': len(valid_texts),
        'n_features': len(feature_names),
        'topics': [],
        'perplexity': lda.perplexity(doc_term_matrix)
    }
    
    # 각 토픽별 결과
    for topic_idx, topic in enumerate(lda.components_):
        # 상위 키워드
        top_word_indices = topic.argsort()[-10:][::-1]
        top_words = [(feature_names[i], topic[i]) for i in top_word_indices]
        
        # 해당 토픽에 가장 적합한 문서
        topic_scores = doc_topic_dist[:, topic_idx]
        best_doc_idx = topic_scores.argmax()
        best_doc_score = topic_scores[best_doc_idx]
        best_doc_text = valid_texts[best_doc_idx]
        
        # 토픽 라벨 자동 생성 (상위 3개 키워드 조합)
        topic_label = ', '.join([w for w, _ in top_words[:3]])
        
        results['topics'].append({
            'topic_id': topic_idx + 1,
            'label': topic_label,
            'keywords': top_words[:7],
            'best_document': {
                'text': best_doc_text,
                'score': best_doc_score
            },
            'document_count': sum(doc_topic_dist.argmax(axis=1) == topic_idx)
        })
    
    # 토픽 분포 요약
    topic_assignments = doc_topic_dist.argmax(axis=1)
    topic_dist = pd.Series(topic_assignments).value_counts().sort_index()
    results['topic_distribution'] = topic_dist.to_dict()
    
    return results


# ============================================================
# 데모 데이터 생성 (테스트용)
# ============================================================

def generate_demo_data_for_analysis(n: int = 100) -> pd.DataFrame:
    """분석 테스트용 데모 데이터 생성"""
    np.random.seed(42)
    
    # 성별에 따른 응답 패턴 차이 반영
    data = []
    
    for i in range(n):
        gender = np.random.choice(['남', '여'], p=[0.45, 0.55])
        age = np.random.choice([21, 22, 23, 24, 25, 26], p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])
        semester = np.random.choice([5, 6, 7, 8], p=[0.2, 0.3, 0.3, 0.2])
        
        # 성별에 따른 기본 경향
        base = 0.3 if gender == '여' else 0
        
        row = {
            'gender': gender,
            'age': age,
            'semester': semester,
            'Q1': min(7, max(1, int(np.random.normal(5 + base, 1.2)))),
            'Q2': min(7, max(1, int(np.random.normal(4 + base * 0.5, 1.3)))),
            'Q3': min(7, max(1, int(np.random.normal(4.2 + base * 0.3, 1.1)))),
            'Q4': min(7, max(1, int(np.random.normal(2.5 - base * 0.5, 1.0)))),
            'Q5': min(7, max(1, int(np.random.normal(3.8 + base * 0.4, 1.2)))),
            'Q6': min(7, max(1, int(np.random.normal(4.5 + base * 0.2, 1.1)))),
            'Q7': min(7, max(1, int(np.random.normal(5 + base * 0.3, 1.0)))),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 주관식 응답
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
        "연구소 생활은 어떤가요?",
        "경제학과인데 정책연구 가능할까요?",
        "사회학 전공자 수요가 있나요?",
        "공무원 vs 연구원 어떤 게 나을까요?",
        "대학원 장학금 받기 쉬운가요?",
        "연구 주제는 어떻게 정하나요?",
        "국책연구원 입사 과정이 궁금해요",
        "박사과정 중 취업 준비 어떻게 하나요?",
        "연구원으로서 보람있는 순간은?",
        "정책연구의 사회적 영향력이 궁금해요",
        "",  # 일부 미응답
    ]
    
    df['Q11'] = np.random.choice(open_responses, n)
    
    return df


if __name__ == "__main__":
    # 테스트
    df = generate_demo_data_for_analysis(100)
    
    print("=" * 50)
    print("1. 선형회귀 분석")
    print("=" * 50)
    lr_results = run_linear_regression(df)
    if 'error' not in lr_results:
        print(f"관측치 수: {lr_results['n_obs']}")
        for row in lr_results['summary_table']:
            print(row)
    else:
        print(lr_results['error'])
    
    print("\n" + "=" * 50)
    print("2. 로지스틱 회귀 분석")
    print("=" * 50)
    logit_results = run_logistic_regression(df)
    if 'error' not in logit_results:
        print(f"Pseudo R²: {logit_results['pseudo_r2']:.3f}")
        for interp in logit_results['interpretation']:
            print(interp)
    else:
        print(logit_results['error'])
    
    print("\n" + "=" * 50)
    print("3. 요인분석")
    print("=" * 50)
    fa_results = run_factor_analysis(df)
    if 'error' not in fa_results:
        print(f"KMO: {fa_results['efa']['kmo']:.3f}")
        print(f"최적 요인 수: {fa_results['efa']['n_factors_optimal']}")
    else:
        print(fa_results['error'])
    
    print("\n" + "=" * 50)
    print("4. 토픽 모델링")
    print("=" * 50)
    lda_results = run_topic_modeling(df['Q11'].tolist())
    if 'error' not in lda_results:
        for topic in lda_results['topics']:
            print(f"토픽 {topic['topic_id']}: {topic['label']}")
            print(f"  대표 문서: {topic['best_document']['text']}")
    else:
        print(lda_results['error'])
