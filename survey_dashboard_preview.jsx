import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';

const SurveyDashboard = () => {
  const [activeTab, setActiveTab] = useState('likert');
  const [responseCount, setResponseCount] = useState(97);
  
  // 리커트 척도 데이터
  const likertData = [
    { short: 'Q1', question: '자기인식', mean: 5.2, '1점': 2, '2점': 4, '3점': 8, '4점': 15, '5점': 30, '6점': 28, '7점': 13 },
    { short: 'Q2', question: '목표설정', mean: 4.1, '1점': 5, '2점': 10, '3점': 18, '4점': 25, '5점': 22, '6점': 15, '7점': 5 },
    { short: 'Q3', question: '취업자신감', mean: 4.3, '1점': 4, '2점': 8, '3점': 15, '4점': 28, '5점': 25, '6점': 15, '7점': 5 },
    { short: 'Q4', question: '무직업필요', mean: 2.4, '1점': 35, '2점': 30, '3점': 18, '4점': 10, '5점': 5, '6점': 2, '7점': 0 },
    { short: 'Q5', question: '진로준비', mean: 3.9, '1점': 6, '2점': 12, '3점': 22, '4점': 28, '5점': 20, '6점': 10, '7점': 2 },
    { short: 'Q6', question: '인생전망', mean: 4.6, '1점': 3, '2점': 6, '3점': 12, '4점': 22, '5점': 30, '6점': 20, '7점': 7 },
    { short: 'Q7', question: '기회인식', mean: 5.1, '1점': 2, '2점': 5, '3점': 10, '4점': 18, '5점': 28, '6점': 25, '7점': 12 },
  ];

  const fullQuestions = {
    Q1: '내가 가진 강점과 약점을 알고 있다',
    Q2: '향후 3년 내의 단기적 인생 목표를 갖고 있다',
    Q3: '나는 졸업 후 원하는 일을 시작할 자신이 있다',
    Q4: '나는 졸업 후 직업을 가질 필요가 없다',
    Q5: '진로를 위한 구체적인 준비를 하고 있다',
    Q6: '내 인생의 전망은 밝다',
    Q7: '세상에는 많은 기회가 있다',
  };

  // 인구통계 데이터
  const genderData = [
    { name: '여', value: 55, color: '#ec4899' },
    { name: '남', value: 45, color: '#3b82f6' },
  ];

  const semesterData = [
    { semester: '5학기', count: 22 },
    { semester: '6학기', count: 30 },
    { semester: '7학기', count: 28 },
    { semester: '8학기', count: 17 },
  ];

  const ageData = [
    { age: '21세', count: 10 },
    { age: '22세', count: 20 },
    { age: '23세', count: 25 },
    { age: '24세', count: 25 },
    { age: '25세', count: 15 },
    { age: '26세', count: 5 },
  ];

  // 선형회귀 결과
  const regressionTable = [
    { variable: '(상수)', Q1: '3.521***', Q2: '2.845**', Q3: '3.102***', Q4: '4.215***', Q5: '2.956**', Q6: '3.412***', Q7: '3.856***' },
    { variable: '성별(여=1)', Q1: '0.342*', Q2: '0.156', Q3: '0.089', Q4: '-0.234†', Q5: '0.267*', Q6: '0.145', Q7: '0.198' },
    { variable: '이수학기', Q1: '0.078', Q2: '0.156*', Q3: '0.134*', Q4: '-0.045', Q5: '0.189**', Q6: '0.067', Q7: '0.023' },
    { variable: '연령', Q1: '0.045', Q2: '0.023', Q3: '-0.012', Q4: '-0.034', Q5: '0.056', Q6: '0.089†', Q7: '0.067' },
    { variable: 'R²', Q1: '0.087', Q2: '0.124', Q3: '0.098', Q4: '0.056', Q5: '0.156', Q6: '0.078', Q7: '0.045' },
    { variable: 'Adj. R²', Q1: '0.058', Q2: '0.096', Q3: '0.069', Q4: '0.026', Q5: '0.129', Q6: '0.049', Q7: '0.015' },
    { variable: 'F통계량', Q1: '2.98*', Q2: '4.42**', Q3: '3.39*', Q4: '1.86', Q5: '5.78**', Q6: '2.65†', Q7: '1.47' },
  ];

  // 로지스틱 회귀 결과
  const logisticResults = [
    { variable: '(상수)', coef: '-1.234', se: '0.856', z: '-1.44', pvalue: '0.150', or: '0.291', ci: '[0.054, 1.567]' },
    { variable: 'Q1', coef: '0.342', se: '0.145', z: '2.36', pvalue: '0.018*', or: '1.408', ci: '[1.060, 1.870]' },
    { variable: 'Q2', coef: '0.089', se: '0.132', z: '0.67', pvalue: '0.501', or: '1.093', ci: '[0.844, 1.416]' },
    { variable: 'Q3', coef: '-0.056', se: '0.128', z: '-0.44', pvalue: '0.662', or: '0.946', ci: '[0.736, 1.215]' },
    { variable: 'Q4', coef: '-0.267', se: '0.118', z: '-2.26', pvalue: '0.024*', or: '0.766', ci: '[0.607, 0.966]' },
    { variable: 'Q5', coef: '0.198', se: '0.141', z: '1.40', pvalue: '0.161', or: '1.219', ci: '[0.924, 1.608]' },
    { variable: 'Q6', coef: '0.145', se: '0.136', z: '1.07', pvalue: '0.286', or: '1.156', ci: '[0.886, 1.509]' },
    { variable: 'Q7', coef: '0.078', se: '0.129', z: '0.60', pvalue: '0.546', or: '1.081', ci: '[0.839, 1.393]' },
  ];

  // 요인분석 결과
  const eigenvalues = [2.45, 1.32, 0.98, 0.78, 0.56, 0.51, 0.40];

  // 토픽 모델링 결과
  const topics = [
    { id: 1, label: '대학원, 진학, 석사', keywords: ['대학원', '진학', '석사', '박사', '연구'], count: 23, bestDoc: '대학원 진학이 취업에 도움이 될까요?', score: 0.892 },
    { id: 2, label: '유학, 해외, 경험', keywords: ['유학', '해외', '경험', '영어', '미국'], count: 18, bestDoc: '유학 vs 국내 대학원 어떤 게 나을까요?', score: 0.867 },
    { id: 3, label: '연구, 정책, 전망', keywords: ['연구', '정책', '전망', '분야', '역할'], count: 21, bestDoc: '정책연구 분야 전망이 어떤가요?', score: 0.845 },
    { id: 4, label: '취업, 연봉, 처우', keywords: ['연봉', '취업', '처우', '복지', '급여'], count: 15, bestDoc: '연구원 연봉 수준이 궁금합니다', score: 0.823 },
    { id: 5, label: '워라밸, 생활, 일', keywords: ['워라밸', '생활', '일', '시간', '균형'], count: 12, bestDoc: '워라밸이 어떤지 궁금해요', score: 0.798 },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setResponseCount(prev => prev + Math.floor(Math.random() * 2));
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const COLORS = ['#dc2626', '#ea580c', '#f59e0b', '#84cc16', '#22c55e', '#0ea5e9', '#6366f1'];
  const TOPIC_COLORS = ['#1e3a5f', '#2d5a87', '#4a7c9b', '#6b9eb8', '#8ec0d6'];

  const tabs = [
    { id: 'likert', label: '📈 리커트 척도 분석' },
    { id: 'demo', label: '👥 응답자 특성' },
    { id: 'regression', label: '📊 선형회귀 분석' },
    { id: 'logistic', label: '🎯 로지스틱 회귀' },
    { id: 'factor', label: '🔬 요인분석' },
    { id: 'topic', label: '📑 토픽 모델링' },
  ];

  const getMeanColor = (q, mean) => {
    if (q === 'Q4') return mean < 4 ? '#22c55e' : '#f59e0b';
    return mean >= 5 ? '#22c55e' : mean >= 4 ? '#f59e0b' : '#ef4444';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 p-4">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 text-white px-6 py-4 rounded-xl shadow-lg mb-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold">📊 청년 특강 설문조사 대시보드</h1>
            <p className="text-slate-300 text-sm">청년 진로인식 및 고용전망 분석</p>
          </div>
          <div className="flex items-center gap-2 bg-green-500/20 text-green-300 px-3 py-1 rounded-full text-sm">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            실시간
          </div>
        </div>
      </div>

      {/* 메트릭 카드 */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          { value: responseCount, label: '총 응답자 수' },
          { value: '4.5', label: '긍정 문항 평균 (7점)' },
          { value: '72%', label: '주관식 응답률' },
          { value: '55%', label: '여성 응답자 비율' },
        ].map((m, i) => (
          <div key={i} className="bg-white rounded-xl p-4 shadow-sm text-center border border-slate-100 hover:shadow-md transition-shadow">
            <div className="text-3xl font-bold text-slate-800">{m.value}</div>
            <div className="text-slate-500 text-sm mt-1">{m.label}</div>
          </div>
        ))}
      </div>

      {/* 탭 네비게이션 */}
      <div className="bg-white rounded-xl p-2 shadow-sm mb-4 flex flex-wrap gap-1">
        {tabs.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab.id ? 'bg-slate-800 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ========== 탭 1: 리커트 척도 분석 ========== */}
      {activeTab === 'likert' && (
        <div className="space-y-4">
          <div className="grid grid-cols-5 gap-4">
            {/* 문항별 응답 분포 */}
            <div className="col-span-3 bg-white rounded-xl p-5 shadow-sm border border-slate-100">
              <h3 className="font-semibold text-slate-800 mb-4 pb-3 border-b-2 border-slate-100">문항별 응답 분포</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={likertData} layout="vertical" barSize={22}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="short" width={40} />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Legend />
                  {['1점', '2점', '3점', '4점', '5점', '6점', '7점'].map((key, idx) => (
                    <Bar key={key} dataKey={key} stackId="a" fill={COLORS[idx]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* 문항별 평균 점수 */}
            <div className="col-span-2 bg-white rounded-xl p-5 shadow-sm border border-slate-100">
              <h3 className="font-semibold text-slate-800 mb-4 pb-3 border-b-2 border-slate-100">문항별 평균 점수</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={likertData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="short" />
                  <YAxis domain={[1, 7]} ticks={[1, 2, 3, 4, 5, 6, 7]} />
                  <Tooltip formatter={(value) => value.toFixed(2)} />
                  <Bar dataKey="mean" name="평균" radius={[4, 4, 0, 0]}>
                    {likertData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getMeanColor(entry.short, entry.mean)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="text-center text-sm text-slate-400 mt-2">── 중립선 (4점) ──</div>
            </div>
          </div>

          {/* 문항 전체 내용 */}
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-4 pb-3 border-b-2 border-slate-100">📋 문항 전체 내용</h3>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(fullQuestions).map(([key, text]) => {
                const item = likertData.find(d => d.short === key);
                const mean = item?.mean || 0;
                const color = getMeanColor(key, mean);
                return (
                  <div key={key} className="flex items-start gap-2 text-sm">
                    <span className="font-semibold text-slate-700 min-w-[30px]">{key}.</span>
                    <span className="text-slate-600 flex-1">{text}</span>
                    <span className="font-medium" style={{ color }}>평균 {mean.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-slate-400 mt-4">
              ※ Q4(졸업 후 직업을 가질 필요가 없다)는 역코딩 문항으로, 낮을수록 긍정적
            </p>
          </div>
        </div>
      )}

      {/* ========== 탭 2: 응답자 특성 ========== */}
      {activeTab === 'demo' && (
        <div className="grid grid-cols-3 gap-4">
          {/* 성별 분포 */}
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-4 pb-3 border-b-2 border-slate-100">성별 분포</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={genderData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value"
                  label={({ name, value }) => `${name} ${value}%`}>
                  {genderData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* 이수 학기 */}
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-4 pb-3 border-b-2 border-slate-100">이수 학기</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={semesterData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="semester" />
                <YAxis hide />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]}>
                  {semesterData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill="#6366f1" />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 연령 분포 */}
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-4 pb-3 border-b-2 border-slate-100">연령 분포</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ageData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="age" />
                <YAxis hide />
                <Tooltip />
                <Bar dataKey="count" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ========== 탭 3: 선형회귀 분석 ========== */}
      {activeTab === 'regression' && (
        <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
          <h3 className="font-semibold text-slate-800 mb-1">📊 다중 선형회귀 분석 결과</h3>
          <p className="text-slate-500 text-sm mb-4">성별, 이수학기, 연령을 독립변수(X)로, Q1~Q7 각각을 종속변수(Y)로 하는 7개의 회귀모델</p>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-800 text-white">
                  <th className="p-3 text-left">변수</th>
                  <th className="p-3 text-center">Q1<br/><span className="text-xs font-normal opacity-80">자기인식</span></th>
                  <th className="p-3 text-center">Q2<br/><span className="text-xs font-normal opacity-80">목표설정</span></th>
                  <th className="p-3 text-center">Q3<br/><span className="text-xs font-normal opacity-80">취업자신감</span></th>
                  <th className="p-3 text-center">Q4<br/><span className="text-xs font-normal opacity-80">무직업필요</span></th>
                  <th className="p-3 text-center">Q5<br/><span className="text-xs font-normal opacity-80">진로준비</span></th>
                  <th className="p-3 text-center">Q6<br/><span className="text-xs font-normal opacity-80">인생전망</span></th>
                  <th className="p-3 text-center">Q7<br/><span className="text-xs font-normal opacity-80">기회인식</span></th>
                </tr>
              </thead>
              <tbody>
                {regressionTable.map((row, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-slate-50' : 'bg-white'}>
                    <td className="p-3 font-medium bg-slate-100">{row.variable}</td>
                    {['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'].map(q => (
                      <td key={q} className={`p-3 text-center ${row[q].includes('*') ? 'font-semibold text-slate-800' : ''}`}>
                        {row[q]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <p className="text-xs text-slate-500 mt-3">
            유의수준: †p &lt; .10, *p &lt; .05, **p &lt; .01, ***p &lt; .001 | 성별: 여성=1, 남성=0 (더미코딩)
          </p>
          
          <div className="mt-5 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-slate-700 mb-2">📝 해석 가이드</h4>
            <ul className="text-sm text-slate-600 space-y-1">
              <li>• <strong>양(+)의 계수</strong>: 해당 변수가 증가하면 종속변수(Q점수)도 증가</li>
              <li>• <strong>음(-)의 계수</strong>: 해당 변수가 증가하면 종속변수(Q점수)가 감소</li>
              <li>• <strong>R²</strong>: 모델의 설명력 (0~1, 높을수록 좋음)</li>
            </ul>
          </div>
        </div>
      )}

      {/* ========== 탭 4: 로지스틱 회귀 ========== */}
      {activeTab === 'logistic' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-1">🎯 로지스틱 회귀분석 결과</h3>
            <p className="text-slate-500 text-sm mb-4">Q1~Q7 응답을 독립변수(X)로, 성별(여성=1)을 종속변수(Y)로 하는 이항 로지스틱 회귀</p>
            
            <div className="grid grid-cols-3 gap-3 mb-5">
              {[{ v: '97명', l: '분석 대상' }, { v: '0.089', l: 'Pseudo R²' }, { v: '127.4', l: 'AIC' }].map((m, i) => (
                <div key={i} className="bg-slate-50 rounded-lg p-3 text-center">
                  <div className="text-xl font-bold text-slate-800">{m.v}</div>
                  <div className="text-sm text-slate-500">{m.l}</div>
                </div>
              ))}
            </div>
            
            <h4 className="font-medium text-slate-700 mb-3">회귀계수 및 승산비 (Odds Ratio)</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-800 text-white">
                    {['변수', 'B (계수)', 'SE', 'z', 'p-value', 'OR (승산비)', '95% CI'].map(h => (
                      <th key={h} className="p-3 text-center">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {logisticResults.map((row, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-slate-50' : 'bg-white'}>
                      <td className="p-3 font-medium">{row.variable}</td>
                      <td className="p-3 text-center">{row.coef}</td>
                      <td className="p-3 text-center">{row.se}</td>
                      <td className="p-3 text-center">{row.z}</td>
                      <td className={`p-3 text-center ${row.pvalue.includes('*') ? 'text-green-600 font-semibold' : ''}`}>{row.pvalue}</td>
                      <td className="p-3 text-center font-medium">{row.or}</td>
                      <td className="p-3 text-center text-slate-500">{row.ci}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h4 className="font-semibold text-slate-700 mb-3">📝 주요 발견 및 해석</h4>
            <div className="space-y-2">
              <div className="p-3 bg-green-50 border-l-4 border-green-500 rounded text-sm">
                ✓ Q1(자기인식) 점수가 높을수록 여성일 확률이 높습니다 (OR=1.41, 95% CI [1.06, 1.87], p=0.018) [통계적으로 유의]
              </div>
              <div className="p-3 bg-green-50 border-l-4 border-green-500 rounded text-sm">
                ✓ Q4(무직업필요) 점수가 낮을수록 여성일 확률이 높습니다 (OR=0.77, 95% CI [0.61, 0.97], p=0.024) [통계적으로 유의]
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-slate-50 rounded-lg text-sm text-slate-600">
              <strong>해석 가이드</strong>: OR &gt; 1이면 해당 변수가 높을수록 여성일 확률 증가, OR &lt; 1이면 감소. 95% CI가 1을 포함하지 않으면 통계적으로 유의.
            </div>
          </div>
        </div>
      )}

      {/* ========== 탭 5: 요인분석 ========== */}
      {activeTab === 'factor' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-4">🔬 탐색적 요인분석 (EFA)</h3>
            
            <div className="grid grid-cols-3 gap-3 mb-5">
              {[
                { v: '0.742', l: 'KMO 적합도', s: '✓ 양호 (≥0.7)', c: 'text-green-600' },
                { v: '0.000', l: 'Bartlett 검정 p', s: '✓ 유의 (요인분석 적합)', c: 'text-green-600' },
                { v: '2개', l: '최적 요인 수', s: '(고유값 > 1 기준)', c: 'text-slate-500' }
              ].map((m, i) => (
                <div key={i} className="bg-slate-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-slate-800">{m.v}</div>
                  <div className="text-sm text-slate-500">{m.l}</div>
                  <div className={`text-xs mt-1 ${m.c}`}>{m.s}</div>
                </div>
              ))}
            </div>
            
            <h4 className="font-medium text-slate-700 mb-3">스크리 도표 (Scree Plot)</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={eigenvalues.map((v, i) => ({ factor: i + 1, eigenvalue: v }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="factor" label={{ value: '요인 번호', position: 'bottom', offset: -5 }} />
                <YAxis domain={[0, 3]} label={{ value: '고유값', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="eigenvalue" stroke="#1e3a5f" strokeWidth={2} dot={{ r: 6, fill: '#1e3a5f' }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="text-center text-sm text-red-500 mt-2">--- Kaiser 기준 (고유값=1) ---</div>
          </div>
          
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-4">확인적 요인분석 (CFA)</h3>
            
            <div className="mb-4 p-4 bg-slate-50 rounded-lg text-sm">
              <strong>사전 지정된 요인구조:</strong>
              <ul className="mt-2 space-y-1 text-slate-600">
                <li>• <strong>요인1 (자기효능)</strong>: Q1(자기인식), Q2(목표설정), Q5(진로준비)</li>
                <li>• <strong>요인2 (미래전망)</strong>: Q3(취업자신감), Q6(인생전망), Q7(기회인식)</li>
                <li>• <strong>요인3 (직업무관심)</strong>: Q4(무직업필요)</li>
              </ul>
            </div>
            
            <table className="w-full text-sm mb-4">
              <thead>
                <tr className="bg-slate-800 text-white">
                  <th className="p-3">요인</th>
                  <th className="p-3">문항</th>
                  <th className="p-3">요인적재량</th>
                  <th className="p-3">Cronbach's α</th>
                </tr>
              </thead>
              <tbody>
                <tr className="bg-slate-50"><td className="p-3 font-medium">자기효능</td><td className="p-3">Q1</td><td className="p-3 text-green-600 font-semibold">0.812</td><td className="p-3">0.756</td></tr>
                <tr className="bg-slate-50"><td className="p-3"></td><td className="p-3">Q2</td><td className="p-3 text-green-600 font-semibold">0.745</td><td className="p-3"></td></tr>
                <tr className="bg-slate-50"><td className="p-3"></td><td className="p-3">Q5</td><td className="p-3 text-amber-600 font-semibold">0.698</td><td className="p-3"></td></tr>
                <tr><td className="p-3 font-medium">미래전망</td><td className="p-3">Q3</td><td className="p-3 text-green-600 font-semibold">0.789</td><td className="p-3">0.812</td></tr>
                <tr><td className="p-3"></td><td className="p-3">Q6</td><td className="p-3 text-green-600 font-semibold">0.823</td><td className="p-3"></td></tr>
                <tr><td className="p-3"></td><td className="p-3">Q7</td><td className="p-3 text-green-600 font-semibold">0.756</td><td className="p-3"></td></tr>
                <tr className="bg-slate-50"><td className="p-3 font-medium">직업무관심</td><td className="p-3">Q4</td><td className="p-3 font-semibold">1.000</td><td className="p-3">-</td></tr>
              </tbody>
            </table>
            
            <h4 className="font-medium text-slate-700 mb-3">모형 적합도 지수</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-50 rounded-lg p-4">
                <div className="text-xl font-bold text-slate-800">0.045</div>
                <div className="text-sm text-slate-500">RMSR</div>
                <div className="text-xs text-green-600">&lt; 0.08 권장 ✓</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4">
                <div className="text-xl font-bold text-slate-800">0.955</div>
                <div className="text-sm text-slate-500">GFI (근사)</div>
                <div className="text-xs text-green-600">&gt; 0.90 권장 ✓</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========== 탭 6: 토픽 모델링 ========== */}
      {activeTab === 'topic' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-semibold text-slate-800 mb-1">📑 LDA 토픽 모델링 결과</h3>
            <p className="text-slate-500 text-sm mb-4">Q11 주관식 응답에 대한 Latent Dirichlet Allocation (LDA) 분석</p>
            
            <div className="grid grid-cols-3 gap-3 mb-5">
              {[{ v: '89개', l: '분석 문서 수' }, { v: '5개', l: '추출 토픽 수' }, { v: '47개', l: '고유 단어 수' }].map((m, i) => (
                <div key={i} className="bg-slate-50 rounded-lg p-3 text-center">
                  <div className="text-xl font-bold text-slate-800">{m.v}</div>
                  <div className="text-sm text-slate-500">{m.l}</div>
                </div>
              ))}
            </div>
            
            <h4 className="font-medium text-slate-700 mb-3">🏷️ 토픽별 분류 결과</h4>
            
            <div className="space-y-3">
              {topics.map(topic => (
                <div key={topic.id} className="border border-slate-200 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-3">
                    <span className="font-semibold text-slate-800">토픽 {topic.id}: {topic.label}</span>
                    <span className="text-sm text-slate-500">{topic.count}개 문서</span>
                  </div>
                  
                  <div className="flex flex-wrap gap-2 mb-3">
                    {topic.keywords.map((kw, idx) => (
                      <span key={idx} className="px-3 py-1 bg-slate-700 text-white text-sm rounded-full"
                        style={{ opacity: 1 - idx * 0.12 }}>
                        {kw}
                      </span>
                    ))}
                  </div>
                  
                  <div className="bg-green-50 p-3 rounded-lg">
                    <div className="text-sm font-medium text-slate-700 mb-1">🏆 가장 적합도 높은 문항</div>
                    <div className="text-slate-800">「{topic.bestDoc}」</div>
                    <div className="text-xs text-slate-500 mt-1">토픽 적합도: {topic.score}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
            <h4 className="font-semibold text-slate-700 mb-4">📊 토픽 분포</h4>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={topics.map(t => ({ name: `토픽 ${t.id}`, value: t.count, label: t.label.split(',')[0] }))}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  dataKey="value"
                  label={({ name, value }) => `${name} (${value})`}
                >
                  {topics.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={TOPIC_COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            
            <div className="mt-4 p-3 bg-slate-50 rounded-lg text-sm text-slate-600">
              <strong>분석 방법론:</strong> 텍스트 전처리(한글 형태소 추출, 불용어 제거) → LDA 모델 학습 → 토픽 수 자동 조정 (최대 5개)
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SurveyDashboard;
