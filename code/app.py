import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# =============================================
# 기본 설정
# =============================================
st.set_page_config(page_title="스마트폰 사용과 공부", page_icon="📱", layout="wide")

CSV_PATH = "kyrbs2025_smartphone_study.csv"
RANDOM_STATE = 42
FINAL_MODEL_NAME = "그래디언트 부스팅"   # 최종 모델 고정

FEATURES = [
    "smartphone_avg_minutes",
    "weekday_phone_minutes",
    "weekend_phone_minutes",
    "grade",
    "sex",
    "economic_status",
]
TARGET = "study_avg_minutes"

FEATURE_KR = {
    "smartphone_avg_minutes": "스마트폰 평균(분)",
    "weekday_phone_minutes": "평일 스마트폰(분)",
    "weekend_phone_minutes": "주말 스마트폰(분)",
    "grade": "학년",
    "sex": "성별",
    "economic_status": "경제 수준",
}

# 튜닝할 하이퍼파라미터 격자 (2 x 3 x 2 x 2 = 24가지)
PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0],
}

# =============================================
# 데이터 (①, ② 2015 설문 요약표)
# =============================================
survey = pd.DataFrame({
    "구분": ["성인: 스마트폰이 공부(하던 일)에 방해된다",
             "청소년: 스마트폰 과다 사용으로 성적이 떨어졌다"],
    "사례수(명)": [13687, 2523],
    "1 전혀 그렇지 않다(%)": [11.3, 26.9],
    "2 그렇지 않다(%)": [46.6, 51.7],
    "3 그렇다(%)": [31.1, 19.0],
    "4 매우 그렇다(%)": [11.0, 2.3],
    "3+4 (%)": [42.1, 21.4],
    "1+2 (%)": [57.9, 78.6],
    "5점평균(점)": [2.4, 2.0],
    "100점평균(점)": [47.3, 32.3],
})
short_names = ["성인(공부 방해)", "청소년(성적 하락)"]
choice_cols = ["1 전혀 그렇지 않다(%)", "2 그렇지 않다(%)",
               "3 그렇다(%)", "4 매우 그렇다(%)"]


# =============================================
# 데이터 ③ 불러오기 / 전처리
# =============================================
@st.cache_data
def load_raw():
    return pd.read_csv(CSV_PATH, encoding="utf-8-sig")


@st.cache_data
def preprocess(raw):
    info = {"원본 행 수": len(raw)}
    df = raw.copy()

    info["결측치 개수"] = int(df.isnull().sum().sum())
    df = df.dropna()

    time_cols = ["smartphone_avg_minutes", "study_avg_minutes",
                 "weekday_phone_minutes", "weekend_phone_minutes",
                 "weekday_study_minutes", "weekend_study_minutes"]
    bad = ((df[time_cols] < 0) | (df[time_cols] > 1440)).any(axis=1)
    info["이상치 행 수(0 미만 또는 1440 초과)"] = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)

    info["처리 후 행 수"] = len(df)
    return df, info


# =============================================
# 모델링
# =============================================
def rmse(y, p):
    return np.sqrt(mean_squared_error(y, p))


def get_models():
    """베이스라인 1개 + 후보 모델 3개 (튜닝 전 기본 설정)"""
    baseline = DummyRegressor(strategy="mean")
    candidates = {
        "선형회귀": make_pipeline(StandardScaler(), LinearRegression()),
        "랜덤포레스트": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1),
        "그래디언트 부스팅": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE),
    }
    return baseline, candidates


@st.cache_data(show_spinner="모델을 학습하고 튜닝하는 중입니다... (1~3분 걸릴 수 있어요)")
def run_modeling(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    baseline, candidates = get_models()
    all_models = {"베이스라인(평균 예측)": baseline, **candidates}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- (1) 후보 모델 비교: 5겹 교차검증 ----
    rows = []
    for name, model in all_models.items():
        cv_rmse = -cross_val_score(
            model, X_train, y_train, cv=kf,
            scoring="neg_root_mean_squared_error").mean()
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
        rows.append({"모델": name, "CV RMSE(분)": cv_rmse, "CV R²": cv_r2})
    cv_table = pd.DataFrame(rows)

    cand_table = cv_table[cv_table["모델"].isin(candidates.keys())] \
        .sort_values("CV RMSE(분)").reset_index(drop=True)
    cand_table["순위"] = cand_table.index + 1
    my_rank = int(cand_table.loc[cand_table["모델"] == FINAL_MODEL_NAME, "순위"].iloc[0])

    # ---- (2) 최종 모델(그래디언트 부스팅) 하이퍼파라미터 튜닝 ----
    gs = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    grid_df = pd.DataFrame(gs.cv_results_["params"])
    grid_df["CV RMSE(분)"] = -gs.cv_results_["mean_test_score"]
    grid_df = grid_df.sort_values("CV RMSE(분)").reset_index(drop=True)

    best_params = gs.best_params_
    tuned_model = gs.best_estimator_

    # 튜닝 전(기본 설정) 모델
    default_model = candidates[FINAL_MODEL_NAME]
    default_model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)

    # ---- (3) 테스트셋 평가: 베이스라인 / 튜닝 전 / 튜닝 후 ----
    eval_list = [
        ("베이스라인(평균 예측)", baseline),
        ("그래디언트 부스팅 (튜닝 전)", default_model),
        ("그래디언트 부스팅 (튜닝 후)", tuned_model),
    ]
    result = {}
    for name, model in eval_list:
        pred = model.predict(X_test)
        result[name] = {
            "MAE(분)": mean_absolute_error(y_test, pred),
            "RMSE(분)": rmse(y_test, pred),
            "R²": r2_score(y_test, pred),
        }
    test_table = pd.DataFrame(result).T.reset_index().rename(columns={"index": "모델"})

    final_pred = tuned_model.predict(X_test)

    # ---- (4) 변수 중요도 ----
    n = min(3000, len(X_test))
    perm = permutation_importance(
        tuned_model, X_test.iloc[:n], y_test.iloc[:n], n_repeats=5,
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({
        "변수": [FEATURE_KR[f] for f in FEATURES],
        "중요도(RMSE 증가량)": perm.importances_mean,
    }).sort_values("중요도(RMSE 증가량)")

    return {
        "cv_table": cv_table,
        "cand_table": cand_table,
        "my_rank": my_rank,
        "grid_df": grid_df,
        "best_params": best_params,
        "test_table": test_table,
        "y_test": y_test.values,
        "final_pred": final_pred,
        "importance": imp,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# =============================================
# 사이드바: 페이지 선택
# =============================================
PAGES = [
    "1. 프로젝트 개요",
    "2. 데이터 ①② 2015년 설문",
    "3. 데이터 ③ 탐색과 전처리",
    "4. 베이스라인과 후보 모델 비교",
    "5. 하이퍼파라미터 튜닝",
    "6. 최종 모델과 평가지표",
    "7. 결론과 한계",
]
st.sidebar.title("📱 스마트폰과 공부")
page = st.sidebar.radio("페이지 선택", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("구술평가 대비: 각 페이지의 '구술 포인트'를 자기 말로 설명해 보세요.")

try:
    raw3 = load_raw()
    df3, info3 = preprocess(raw3)
    data_ok = True
except FileNotFoundError:
    data_ok = False


def need_data():
    if not data_ok:
        st.error(f"'{CSV_PATH}' 파일을 app.py와 같은 폴더에 넣어주세요.")
        st.stop()


# =============================================
# 페이지 1: 개요
# =============================================
if page == PAGES[0]:
    st.title("📱 스마트폰 사용은 공부에 방해가 될까?")
    st.subheader("프로젝트 개요")

    st.markdown("""
**사용한 데이터 3개**

| 번호 | 데이터 | 시점 | 성격 |
|---|---|---|---|
| ① | 청소년: 스마트폰 지나친 사용으로 학교성적이 떨어졌다 | 2015 | 인식 설문(요약표) |
| ② | 성인: 스마트폰 사용이 하고 있는 일(공부)에 방해가 된다 | 2015 | 인식 설문(요약표) |
| ③ | 청소년건강행태조사 `kyrbs2025_smartphone_study.csv` | 2025 | 학생 개인별 시간(분) 데이터 53,164행 |

**분석 흐름**
1. ①②로 "사람들이 스마트폰이 공부에 방해된다고 *느끼는지*" 확인
2. ③으로 "스마트폰 사용 시간 등으로 공부 시간을 *예측*할 수 있는지" 모델링
3. 베이스라인 → 후보 모델 3개 비교 → **그래디언트 부스팅 선택** → 하이퍼파라미터 튜닝 → 평가지표로 해석
""")

    st.info("""
**모델링 문제 정의**
- 문제 유형: **회귀** (공부 시간은 연속적인 숫자)
- 목표(y): `study_avg_minutes` (하루 평균 공부 시간, 분)
- 입력(X): 스마트폰 평균/평일/주말 시간, 학년, 성별, 경제 수준
- 제외한 변수: `weekday_study_minutes`, `weekend_study_minutes`
  → 목표 변수를 계산하는 재료라서 넣으면 **데이터 누수**
""")

    st.markdown("""
**🎤 구술 포인트**
- 왜 이 문제를 회귀로 정했는가?
- 왜 평일/주말 공부 시간은 입력에서 뺐는가? (데이터 누수)
""")


# =============================================
# 페이지 2: 설문 ①②
# =============================================
elif page == PAGES[1]:
    st.title("데이터 ①② : 2015년 인식 설문")
    st.dataframe(survey.set_index("구분"), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["응답 분포", "긍정 vs 부정", "평균 점수"])

    with tab1:
        fig = go.Figure()
        colors = ["#9ECAE1", "#C6DBEF", "#FDAE6B", "#E6550D"]
        for col, color in zip(choice_cols, colors):
            fig.add_trace(go.Bar(
                y=short_names, x=survey[col], name=col, orientation="h",
                marker_color=color,
                text=survey[col].apply(lambda v: f"{v:.1f}%"),
                textposition="inside"))
        fig.update_layout(barmode="stack", title="응답 항목별 비율",
                          xaxis=dict(range=[0, 100], ticksuffix="%"),
                          legend=dict(orientation="h", y=-0.3), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=short_names, y=survey["1+2 (%)"], name="부정(1+2)",
                             marker_color="#6BAED6",
                             text=survey["1+2 (%)"], textposition="outside"))
        fig.add_trace(go.Bar(x=short_names, y=survey["3+4 (%)"], name="긍정(3+4)",
                             marker_color="#FD8D3C",
                             text=survey["3+4 (%)"], textposition="outside"))
        fig.update_layout(barmode="group", title="긍정·부정 응답 비교",
                          yaxis=dict(range=[0, 100], ticksuffix="%"), height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure(go.Scatter(
            x=short_names, y=survey["5점평균(점)"], mode="markers+lines+text",
            text=survey["5점평균(점)"].apply(lambda v: f"{v:.1f}점"),
            textposition="top center", marker=dict(size=14)))
        fig.add_hline(y=2.5, line_dash="dash", line_color="gray",
                      annotation_text="중간값 2.5점")
        fig.update_layout(title="평균 점수 (4점 척도)",
                          yaxis=dict(range=[0, 4], dtick=0.5),
                          showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("※ 원자료 파일에는 '5점평균'으로 표기되어 있지만 보기가 1~4이므로 "
               "실제로는 4점 척도 평균입니다. 청소년 3+4는 세부 합계(21.3%)와 "
               "원자료 값(21.4%)이 반올림 때문에 0.1%p 다릅니다.")

    st.warning("""
**해석 시 주의**
- 질문이 다릅니다. (공부 방해 vs 성적 하락)
- 대상이 다릅니다. (성인 vs 청소년)
- 인식 조사이므로 **인과관계를 증명할 수 없습니다.**
- 2015년 자료라 2025년 데이터 ③과 직접 비교는 어렵습니다.
""")


# =============================================
# 페이지 3: 데이터 ③ 탐색
# =============================================
elif page == PAGES[2]:
    st.title("데이터 ③ : 탐색과 전처리")
    need_data()

    st.subheader("① 데이터 개요")
    c1, c2 = st.columns(2)
    c1.metric("행 개수", f"{raw3.shape[0]:,}")
    c2.metric("열 개수", raw3.shape[1])
    st.dataframe(raw3.head(10), use_container_width=True)

    st.subheader("② 결측치·이상치 처리")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("원본 행", f"{info3['원본 행 수']:,}")
    c2.metric("결측치", info3["결측치 개수"])
    c3.metric("이상치 행", info3["이상치 행 수(0 미만 또는 1440 초과)"])
    c4.metric("처리 후 행", f"{info3['처리 후 행 수']:,}")
    st.markdown("""
- **결측치**: 빈 값이 있는 행은 삭제
- **이상치 기준**: 하루는 1440분(24시간)이므로 0 미만 또는 1440 초과는 불가능한 값으로 보고 삭제
- 1000분 넘는 스마트폰 사용처럼 *가능은 하지만 매우 극단적인* 값은 남겼습니다.
""")

    st.subheader("③ 기술통계")
    st.dataframe(df3.describe().T, use_container_width=True)

    st.subheader("④ 시각화")
    tab1, tab2, tab3 = st.tabs(["산점도", "학년별 평균", "상관관계"])

    with tab1:
        samp = df3.sample(min(5000, len(df3)), random_state=RANDOM_STATE)
        fig = px.scatter(samp, x="smartphone_avg_minutes", y="study_avg_minutes",
                         opacity=0.3,
                         labels={"smartphone_avg_minutes": "스마트폰(분)",
                                 "study_avg_minutes": "공부(분)"},
                         title="스마트폰 사용 vs 공부 시간 (5,000명 샘플)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        g = df3.groupby("grade")[["smartphone_avg_minutes", "study_avg_minutes"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g["grade"], y=g["smartphone_avg_minutes"], name="스마트폰"))
        fig.add_trace(go.Bar(x=g["grade"], y=g["study_avg_minutes"], name="공부"))
        fig.update_layout(barmode="group", title="학년별 평균(분)",
                          xaxis_title="학년(1~6: 중1~고3)", yaxis_title="분")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = df3[FEATURES + [TARGET]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="상관계수 히트맵")
        st.plotly_chart(fig, use_container_width=True)
        r = df3["smartphone_avg_minutes"].corr(df3[TARGET])
        st.markdown(f"스마트폰 평균 시간과 공부 시간의 상관계수: **{r:.3f}**")

    st.markdown("""
**🎤 구술 포인트**
- 이상치를 어떤 기준으로 처리했고 왜 그렇게 했는가?
- 상관계수가 0에 가까우면 모델링 결과가 어떨 것 같은가?
""")


# =============================================
# 페이지 4: 베이스라인 + 후보 모델 비교
# =============================================
elif page == PAGES[3]:
    st.title("베이스라인과 후보 모델 비교")
    need_data()

    res = run_modeling(df3)

    st.subheader("① 데이터 분할")
    c1, c2 = st.columns(2)
    c1.metric("학습용(80%)", f"{res['n_train']:,}")
    c2.metric("테스트용(20%)", f"{res['n_test']:,}")
    st.caption("테스트 데이터는 마지막 평가에만 사용하고, 모델 비교는 "
               "학습 데이터 안에서 5겹 교차검증으로 했습니다.")

    st.subheader("② 베이스라인")
    st.markdown("""
**베이스라인 = 학습 데이터 공부 시간의 평균값을 모든 학생에게 똑같이 예측**하는 모델입니다.
입력 변수를 전혀 쓰지 않으므로, 우리가 고른 모델이 이보다 **좋아야만** 의미가 있습니다.
""")

    st.subheader("③ 후보 모델 3개 (최대 3개)")
    st.markdown("""
| 후보 | 선택 이유 | 주요 하이퍼파라미터(기본 설정) |
|---|---|---|
| 선형회귀 | 가장 단순하고 해석이 쉬운 기준 모델 | (표준화 후 적용) 없음 |
| 랜덤포레스트 | 비선형 관계와 변수 간 상호작용을 잡을 수 있음 | n_estimators=100, max_depth=8, min_samples_leaf=20 |
| 그래디언트 부스팅 | 이전 트리의 오차를 보완하며 순차 학습 | n_estimators=100, learning_rate=0.1, max_depth=3 |
""")

    st.subheader("④ 교차검증 비교 결과 (5-fold)")
    cv = res["cv_table"].copy()
    st.dataframe(
        cv.style.format({"CV RMSE(분)": "{:.2f}", "CV R²": "{:.4f}"})
          .highlight_min(subset=["CV RMSE(분)"], color="#c6efce"),
        use_container_width=True, hide_index=True)

    fig = px.bar(cv, x="모델", y="CV RMSE(분)", text=cv["CV RMSE(분)"].round(1),
                 title="모델별 교차검증 RMSE (낮을수록 좋음)")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    ct = res["cand_table"]
    st.success(f"✅ 최종 모델: **{FINAL_MODEL_NAME}** (후보 3개 중 CV RMSE {res['my_rank']}위)")
    if res["my_rank"] != 1:
        top = ct.iloc[0]
        gb = ct[ct["모델"] == FINAL_MODEL_NAME].iloc[0]
        gap = gb["CV RMSE(분)"] - top["CV RMSE(분)"]
        st.warning(
            f"CV RMSE 1위는 **{top['모델']}**({top['CV RMSE(분)']:.2f}분)이고, "
            f"{FINAL_MODEL_NAME}는 {gb['CV RMSE(분)']:.2f}분으로 {gap:.2f}분 차이입니다. "
            "차이가 작다면 '튜닝으로 더 개선할 여지가 있다'는 점을 선택 이유로 설명할 수 있습니다.")

    st.markdown("""
**🎤 구술 포인트**
- 베이스라인은 무엇이고 왜 필요한가?
- 후보 3개를 어떤 기준으로 골랐고, 어떻게 비교했는가? (같은 데이터 분할·같은 지표·교차검증)
- 왜 그래디언트 부스팅을 최종 모델로 골랐는가? (위 표의 숫자를 근거로 말하기)
""")


# =============================================
# 페이지 5: 하이퍼파라미터 튜닝
# =============================================
elif page == PAGES[4]:
    st.title("하이퍼파라미터 튜닝 (그래디언트 부스팅)")
    need_data()

    res = run_modeling(df3)

    st.subheader("① 하이퍼파라미터란?")
    st.markdown("""
모델이 **학습하기 전에 사람이 정해 주는 설정값**입니다. (데이터로부터 자동으로 배우는 값이 아님)

| 하이퍼파라미터 | 뜻 | 값이 커지면 |
|---|---|---|
| `n_estimators` | 만드는 트리의 개수 | 더 많이 학습, 과적합 위험↑, 느려짐 |
| `learning_rate` | 각 트리가 예측에 반영되는 비율 | 빠르게 학습하지만 불안정할 수 있음 |
| `max_depth` | 각 트리의 최대 깊이 | 복잡한 관계를 잡지만 과적합 위험↑ |
| `subsample` | 각 트리를 만들 때 쓰는 데이터 비율 | 1.0이면 전부 사용, 작으면 무작위성↑(과적합 완화) |
""")

    st.subheader("② 탐색 방법: GridSearchCV")
    total = 1
    for v in PARAM_GRID.values():
        total *= len(v)
    st.markdown(f"""
정해 둔 값들의 **모든 조합({total}가지)을 3겹 교차검증**으로 평가하고, CV RMSE가 가장 낮은 조합을 고릅니다.
""")
    st.json(PARAM_GRID)

    st.subheader("③ 탐색 결과 (CV RMSE 낮은 순 상위 10개)")
    gdf = res["grid_df"]
    st.dataframe(
        gdf.head(10).style.format({"CV RMSE(분)": "{:.3f}"}),
        use_container_width=True, hide_index=True)

    bp = res["best_params"]
    st.success(
        f"🏆 최적 조합: n_estimators={bp['n_estimators']}, "
        f"learning_rate={bp['learning_rate']}, max_depth={bp['max_depth']}, "
        f"subsample={bp['subsample']}")

    st.subheader("④ 하이퍼파라미터별 영향")
    param_choice = st.selectbox("살펴볼 하이퍼파라미터", list(PARAM_GRID.keys()))
    by_param = gdf.groupby(param_choice)["CV RMSE(분)"].mean().reset_index()
    by_param[param_choice] = by_param[param_choice].astype(str)
    fig = px.bar(by_param, x=param_choice, y="CV RMSE(분)",
                 text=by_param["CV RMSE(분)"].round(2),
                 title=f"{param_choice} 값별 평균 CV RMSE (다른 값들은 평균)")
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[by_param["CV RMSE(분)"].min() * 0.98,
                            by_param["CV RMSE(분)"].max() * 1.01])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("막대 높이 차이가 작다면, 이 하이퍼파라미터는 성능에 큰 영향을 주지 않는다는 뜻입니다. "
               "(y축이 잘려 있어 차이가 과장되어 보일 수 있어요.)")

    st.markdown("""
**🎤 구술 포인트**
- 하이퍼파라미터와 일반 파라미터(학습으로 정해지는 값)의 차이는?
- 왜 테스트 데이터가 아니라 교차검증으로 튜닝했는가? (테스트 데이터로 고르면 시험 문제를 미리 보는 것)
- learning_rate를 낮추면 n_estimators는 어떻게 해야 할까?
""")


# =============================================
# 페이지 6: 최종 모델 + 평가지표
# =============================================
elif page == PAGES[5]:
    st.title("최종 모델과 평가지표")
    need_data()

    res = run_modeling(df3)
    bp = res["best_params"]
    st.header(f"🏆 최종 모델: {FINAL_MODEL_NAME} (튜닝 후)")

    st.subheader("① 핵심 원리")
    st.markdown("""
작은 트리(약한 학습기)를 **하나씩 순서대로** 만들면서, 이전까지 모델이 틀린 부분(**잔차 = 실제값 − 예측값**)을
다음 트리가 줄이도록 학습합니다.

1. 처음엔 평균값으로 예측
2. 잔차를 계산
3. 잔차를 맞추는 작은 트리를 만들어 `learning_rate`만큼만 더함
4. 2~3을 `n_estimators`번 반복

→ 여러 개의 약한 모델이 모여 점점 오차를 줄여 가는 방식입니다. (랜덤포레스트는 트리를 *병렬*로 만들어 평균, 부스팅은 *순차*로 보완)
""")

    st.subheader("② 최종 하이퍼파라미터 (튜닝 결과)")
    hp = pd.DataFrame({
        "하이퍼파라미터": list(bp.keys()),
        "최종 값": list(bp.values()),
        "기본값(튜닝 전)": [100, 0.1, 3, 1.0],
    })
    hp["기본값(튜닝 전)"] = hp["기본값(튜닝 전)"].astype(str)
    hp["최종 값"] = hp["최종 값"].astype(str)
    st.dataframe(hp, use_container_width=True, hide_index=True)

    st.subheader("③ 테스트 데이터 평가 (베이스라인 vs 튜닝 전 vs 튜닝 후)")
    tt = res["test_table"]
    st.dataframe(
        tt.style.format({"MAE(분)": "{:.2f}", "RMSE(분)": "{:.2f}", "R²": "{:.4f}"}),
        use_container_width=True, hide_index=True)

    base_row = tt.iloc[0]
    fin_row = tt.iloc[2]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{fin_row['MAE(분)']:.1f}분",
              f"{fin_row['MAE(분)'] - base_row['MAE(분)']:.1f}분 (베이스라인 대비)",
              delta_color="inverse")
    c2.metric("RMSE", f"{fin_row['RMSE(분)']:.1f}분",
              f"{fin_row['RMSE(분)'] - base_row['RMSE(분)']:.1f}분 (베이스라인 대비)",
              delta_color="inverse")
    c3.metric("R²", f"{fin_row['R²']:.3f}",
              f"{fin_row['R²'] - base_row['R²']:.3f} (베이스라인 대비)")

    st.subheader("④ 평가지표의 뜻")
    st.markdown("""
| 지표 | 뜻 | 읽는 법 |
|---|---|---|
| **MAE** | 예측과 실제 차이의 절댓값 평균 | "평균적으로 몇 분 틀리는가". 직관적이고 이상치에 덜 민감 |
| **RMSE** | 오차를 제곱해 평균낸 뒤 제곱근 | 큰 오차에 더 큰 벌점. 단위가 분이라 해석 가능 |
| **R²** | 모델이 설명하는 변동의 비율 | 1에 가까울수록 좋고, 0이면 평균 예측과 같음, 음수면 평균 예측보다 나쁨 |
""")

    st.subheader("⑤ 이 지표를 고른 이유")
    st.markdown("""
- **RMSE**: 모델 비교·튜닝의 기준. 단위가 '분'이라 의미가 직관적이고, 크게 틀리는 예측을 더 강하게 벌주기 때문
- **MAE**: RMSE와 함께 보면 극단값이 오차에 얼마나 영향을 주는지 알 수 있기 때문
- **R²**: 베이스라인(평균 예측) 대비 얼마나 설명력이 있는지 한눈에 보기 위해
""")

    st.subheader("⑥ 결과 시각화")
    tab1, tab2 = st.tabs(["실제 vs 예측", "변수 중요도"])
    with tab1:
        n = min(3000, len(res["y_test"]))
        plot_df = pd.DataFrame({"실제(분)": res["y_test"][:n],
                                "예측(분)": res["final_pred"][:n]})
        fig = px.scatter(plot_df, x="실제(분)", y="예측(분)", opacity=0.3,
                         title="실제 vs 예측 (테스트 3,000명 샘플)")
        fig.add_shape(type="line", x0=0, y0=0, x1=1000, y1=1000,
                      line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("점선(y=x)에 가까울수록 정확한 예측입니다.")
    with tab2:
        imp = res["importance"]
        fig = px.bar(imp, x="중요도(RMSE 증가량)", y="변수", orientation="h",
                     title="Permutation Importance (변수를 섞었을 때 RMSE가 얼마나 나빠지는가)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**🎤 구술 포인트**
- 그래디언트 부스팅은 어떤 원리로 예측하는가? (잔차를 순차적으로 줄임)
- 튜닝 전후 성능이 얼마나 달라졌고, 그 의미는?
- R²가 낮다면 그 의미는? (입력 변수만으로는 공부 시간을 잘 설명하지 못한다)
- MAE와 RMSE가 다른 이유는?
""")


# =============================================
# 페이지 7: 결론
# =============================================
else:
    st.title("결론과 한계")

    st.markdown("""
**알 수 있는 점**
- 2015년 설문(①②)에서는 성인의 42.1%, 청소년의 21.4%가 스마트폰이 공부(성적)에 방해된다고 응답했습니다.
- 2025년 데이터(③)로 만든 그래디언트 부스팅 모델의 성능은 6페이지의 **베이스라인 대비 개선 정도**와 **R²** 로 판단합니다.
- 튜닝 전후의 차이는 6페이지 표에서 확인할 수 있습니다.

**한계**
- ①②는 인식 설문, ③은 시간 데이터이며 시점이 달라서 직접 비교할 수 없습니다.
- 스마트폰 사용과 공부 시간의 **상관·예측 관계가 곧 인과관계는 아닙니다.**
- 시간 데이터는 자기보고식이라 부정확할 수 있습니다.
- 입력 변수가 6개뿐이라 공부 시간에 영향을 주는 다른 요인(학업 수준, 학원, 수면 등)을 담지 못합니다.
- 공부 시간은 '양'일 뿐 학습의 '질'이나 성적을 보여주지 않습니다.
- 하이퍼파라미터 격자가 작아(24가지) 더 좋은 조합이 있을 수 있습니다.
""")

    st.subheader("✍️ 나의 결론 써보기")
    st.text_area("자료에서 알 수 있는 점과 한계를 직접 써보세요.", height=150,
                 placeholder="예: 모델 결과 ...라는 점을 알 수 있다. 하지만 ...라는 한계가 있다.")
