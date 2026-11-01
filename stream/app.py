import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------
st.set_page_config(
    page_title="스마트폰 사용과 공부",
    page_icon="📱",
    layout="wide"
)

st.title("📱 스마트폰 사용은 공부에 방해가 될까?")
st.caption("자료 출처: 제공된 2015년 조사 자료 2종 (인식 조사)")

# ---------------------------------------------
# 데이터
# ---------------------------------------------
data = {
    "구분": [
        "성인: 스마트폰이 하던 일(공부)에 방해된다",
        "청소년: 스마트폰 과다 사용으로 성적이 떨어졌다"
    ],
    "응답자 수(명)": [13687, 2523],
    "1 전혀 그렇지 않다(%)": [11.3, 26.9],
    "2 그렇지 않다(%)": [46.6, 51.7],
    "3 그렇다(%)": [31.1, 19.0],
    "4 매우 그렇다(%)": [11.0, 2.3],
    "긍정(3+4)(%)": [42.1, 21.4],
    "부정(1+2)(%)": [57.9, 78.6],
    "4점 평균(점)": [2.4, 2.0],
    "100점 환산(점)": [47.3, 32.3],
}

raw = pd.DataFrame(data)
short_names = ["성인(방해)", "청소년(성적 하락)"]

choice_cols = [
    "1 전혀 그렇지 않다(%)",
    "2 그렇지 않다(%)",
    "3 그렇다(%)",
    "4 매우 그렇다(%)"
]

# ---------------------------------------------
# 결측치·이상치 처리
# ---------------------------------------------
rows_before, cols_before = raw.shape

df = raw.copy()

# 결측치: 빈 값이 있는 행 삭제
missing_total = int(df.isnull().sum().sum())
df = df.dropna()

# 이상치: 비율 열이 0~100%를 벗어나거나, 평균이 1~4점을 벗어나면 삭제
pct_cols = [c for c in df.columns if c.endswith("(%)")]
bad_pct = ((df[pct_cols] < 0) | (df[pct_cols] > 100)).any(axis=1)
bad_mean = (df["4점 평균(점)"] < 1) | (df["4점 평균(점)"] > 4)
bad_rows = bad_pct | bad_mean
outlier_total = int(bad_rows.sum())
df = df[~bad_rows].reset_index(drop=True)

rows_after, cols_after = df.shape

# ---------------------------------------------
# 그래프에 쓸 수치 계산
# ---------------------------------------------
adult = df.iloc[0]
teen = df.iloc[1]

adult_pos_sum = round(adult["3 그렇다(%)"] + adult["4 매우 그렇다(%)"], 1)  # 42.1
teen_pos_sum = round(teen["3 그렇다(%)"] + teen["4 매우 그렇다(%)"], 1)    # 21.3
adult_pos = adult["긍정(3+4)(%)"]   # 원자료 42.1
teen_pos = teen["긍정(3+4)(%)"]     # 원자료 21.4
adult_neg = adult["부정(1+2)(%)"]
teen_neg = teen["부정(1+2)(%)"]
adult_mean = adult["4점 평균(점)"]
teen_mean = teen["4점 평균(점)"]

# Plotly 그래프 공통 설정
chart_layout = {
    "font": {"family": "Arial, sans-serif", "size": 14},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": {"l": 30, "r": 30, "t": 45, "b": 45},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": -0.3},
    "hovermode": "closest"
}

# ---------------------------------------------
# 1. 데이터 개요
# ---------------------------------------------
st.header("1. 데이터 개요")

c1, c2 = st.columns(2)
c1.metric("행 개수", rows_before)
c2.metric("열 개수", cols_before)

overview = pd.DataFrame({
    "자료형": raw.dtypes.astype(str),
    "빈 값 개수": raw.isnull().sum()
})
st.dataframe(overview, use_container_width=True)

# ---------------------------------------------
# 2. 결측치·이상치 처리
# ---------------------------------------------
st.header("2. 결측치·이상치 처리")

st.markdown(f"""
- **결측치:** 열마다 빈 값을 세어 보니 총 {missing_total}개 → 빈 값이 있으면 해당 행 삭제
- **이상치:** 비율은 0~100%, 평균은 1~4점 범위를 벗어나는 값이 {outlier_total}개 → 있으면 해당 행 삭제
- **참고:** 자료가 2행짜리 요약표라서 처리할 값이 없었고, 확인한 결과를 아래에 표시했습니다.
""")

p1, p2, p3 = st.columns(3)
p1.metric("처리 전 행 수", rows_before)
p2.metric("처리 후 행 수", rows_after)
p3.metric("삭제된 행 수", rows_before - rows_after)
st.caption(f"열 수: 처리 전 {cols_before}개 → 처리 후 {cols_after}개")

# ---------------------------------------------
# 3. 자료 표
# ---------------------------------------------
st.header("3. 자료 표")

st.dataframe(
    df.set_index("구분"),
    use_container_width=True
)

# ---------------------------------------------
# 4. 핵심 지표
# ---------------------------------------------
st.header("4. 핵심 지표 비교")

col1, col2 = st.columns(2)

with col1:
    st.subheader("성인")
    st.metric("긍정 응답(3+4)", f"{adult_pos_sum}%")
    st.metric("4점 척도 평균", f"{adult_mean}점")

with col2:
    st.subheader("청소년")
    st.metric("긍정 응답(3+4)", f"{teen_pos_sum}%")
    st.metric("4점 척도 평균", f"{teen_mean}점")

st.caption(
    "※ 청소년 긍정 응답은 원자료의 세부 비율 합계 "
    "19.0% + 2.3% = 21.3%입니다. "
    "제공된 원자료의 21.4%와 차이가 있어 확인이 필요합니다."
)

# ---------------------------------------------
# 5. 그래프
# ---------------------------------------------
st.header("5. 그래프")

tab1, tab2, tab3 = st.tabs([
    "응답 분포",
    "긍정 vs 부정",
    "평균 점수"
])

# ---------------------------------------------
# 탭 1: 응답 분포 (누적 가로 막대)
# ---------------------------------------------
with tab1:
    st.subheader("응답 분포 비교")

    fig = go.Figure()
    colors = ["#9ECAE1", "#C6DBEF", "#FDAE6B", "#E6550D"]

    for col, color in zip(choice_cols, colors):
        fig.add_trace(go.Bar(
            y=short_names,
            x=df[col],
            name=col,
            orientation="h",
            marker_color=color,
            text=df[col].apply(lambda v: f"{v:.1f}%"),
            textposition="inside",
            hovertemplate="%{y}<br>" + col + ": %{x:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        **chart_layout,
        barmode="stack",
        title="응답 항목별 비율",
        xaxis_title="비율(%)",
        yaxis_title="조사 대상",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**찾아낸 것:** 성인은 '그렇다' 이상이 {adult_pos_sum}%로 "
        f"청소년 {teen_pos_sum}%보다 {round(adult_pos_sum - teen_pos_sum, 1)}%p 높고, "
        f"청소년은 '그렇지 않다' 이하가 {round(teen['1 전혀 그렇지 않다(%)'] + teen['2 그렇지 않다(%)'], 1)}%로 가장 많습니다. "
        f"(근거: 성인 {adult_pos_sum}% vs 청소년 {teen_pos_sum}%, 세부 응답 비율 합계 기준)"
    )

# ---------------------------------------------
# 탭 2: 긍정 vs 부정 (묶음 세로 막대)
# ---------------------------------------------
with tab2:
    st.subheader("긍정 응답과 부정 응답 비교")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=short_names,
        y=df["부정(1+2)(%)"],
        name="부정 응답",
        marker_color="#6BAED6",
        text=df["부정(1+2)(%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{x}<br>부정 응답: %{y:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=short_names,
        y=df["긍정(3+4)(%)"],
        name="긍정 응답(원자료)",
        marker_color="#FD8D3C",
        text=df["긍정(3+4)(%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{x}<br>긍정 응답: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        **chart_layout,
        barmode="group",
        title="긍정 응답과 부정 응답",
        xaxis_title="조사 대상",
        yaxis_title="비율(%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        height=430
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**찾아낸 것:** 두 집단 모두 부정 응답이 긍정 응답보다 많지만, "
        f"청소년의 부정 응답({teen_neg}%)이 성인({adult_neg}%)보다 "
        f"{round(teen_neg - adult_neg, 1)}%p 더 높습니다. "
        f"(근거: 긍정 응답 성인 {adult_pos}% vs 청소년 {teen_pos}%, 원자료 기준)"
    )

# ---------------------------------------------
# 탭 3: 평균 점수 (막대)
# ---------------------------------------------
with tab3:
    st.subheader("4점 척도 평균 비교")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=short_names,
        y=df["4점 평균(점)"],
        marker_color=["#74C476", "#FD8D3C"],
        text=df["4점 평균(점)"].apply(lambda v: f"{v:.1f}점"),
        textposition="outside",
        hovertemplate="%{x}<br>평균: %{y:.1f}점<extra></extra>"
    ))

    fig.add_hline(
        y=2.5, line_dash="dash", line_color="gray",
        annotation_text="중간값 2.5", annotation_position="top left"
    )

    fig.update_layout(
        **chart_layout,
        title="스마트폰 사용에 대한 인식 평균",
        xaxis_title="조사 대상",
        yaxis_title="4점 척도 평균(점)",
        yaxis=dict(range=[0, 4], dtick=1),
        showlegend=False,
        height=430
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**찾아낸 것:** 두 집단 모두 평균이 중간값 2.5점 아래지만, "
        f"성인({adult_mean}점)이 청소년({teen_mean}점)보다 {round(adult_mean - teen_mean, 1)}점 높아 "
        f"방해 정도를 더 크게 느낍니다. "
        f"(근거: 100점 환산 성인 {adult['100점 환산(점)']}점 vs 청소년 {teen['100점 환산(점)']}점)"
    )
    st.caption("※ 평균 점수가 높을수록 해당 문항에 동의하는 정도가 높습니다. (1=전혀 그렇지 않다 ~ 4=매우 그렇다)")

# ---------------------------------------------
# 6. 해석할 때 주의할 점
# ---------------------------------------------
st.header("6. 해석할 때 주의할 점")

st.warning("""
- 두 문항의 질문 내용이 다릅니다. (공부 방해 vs 성적 하락)
- 조사 대상이 다릅니다. (성인 vs 청소년)
- 인식 조사이므로 스마트폰 사용이 공부 시간을 줄인다는 인과관계를 증명할 수 없습니다.
- 2015년 자료이므로 현재의 상황과 다를 수 있습니다.
- 긍정 응답 비율은 세부 항목의 합계와 원자료 기재값이 일부 다르므로 확인이 필요합니다.
""")

st.subheader("✍️ 나의 결론 써보기")

st.text_area(
    "자료에서 알 수 있는 점과 한계를 직접 써보세요.",
    height=150,
    placeholder="예: 조사 결과에서 알 수 있는 점은 ...이다. "
                "하지만 이 자료는 ...라는 한계가 있다."
)
