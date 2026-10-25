
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    "5점 평균(점)": [2.4, 2.0],
    "100점 환산(점)": [47.3, 32.3],
}

df = pd.DataFrame(data)

short_names = ["성인(방해)", "청소년(성적 하락)"]

choice_cols = [
    "1 전혀 그렇지 않다(%)",
    "2 그렇지 않다(%)",
    "3 그렇다(%)",
    "4 매우 그렇다(%)"
]

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
# 1. 자료 표
# ---------------------------------------------
st.header("1. 자료 표")

st.dataframe(
    df.set_index("구분"),
    use_container_width=True
)

# ---------------------------------------------
# 2. 핵심 지표
# ---------------------------------------------
st.header("2. 핵심 지표 비교")

col1, col2 = st.columns(2)

with col1:
    st.subheader("성인")
    st.metric("긍정 응답(3+4)", "42.1%")
    st.metric("5점 척도 평균", "2.4점")

with col2:
    st.subheader("청소년")
    st.metric("긍정 응답(3+4)", "21.3%")
    st.metric("5점 척도 평균", "2.0점")

st.caption(
    "※ 청소년 긍정 응답은 원자료의 세부 비율 합계 "
    "19.0% + 2.3% = 21.3%입니다. "
    "제공된 원자료의 21.4%와 차이가 있어 확인이 필요합니다."
)

# ---------------------------------------------
# 3. 그래프
# ---------------------------------------------
st.header("3. 그래프")

tab1, tab2, tab3 = st.tabs([
    "응답 분포",
    "긍정 vs 부정",
    "평균 점수"
])

# ---------------------------------------------
# 탭 1: 응답 분포 (100% 누적 가로 막대)
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
            hovertemplate=(
                "%{y}<br>" + col + ": %{x:.1f}%<extra></extra>"
            )
        ))

    fig.update_layout(
        **chart_layout,
        barmode="stack",
        title="응답 항목별 비율",
        xaxis_title="비율(%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "성인은 '그렇다'와 '매우 그렇다'의 합이 42.1%이며, "
        "청소년은 세부 응답 비율 기준 21.3%입니다."
    )

# ---------------------------------------------
# 탭 2: 긍정 vs 부정
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

# ---------------------------------------------
# 탭 3: 평균 점수
# ---------------------------------------------
with tab3:
    st.subheader("5점 척도 평균 비교")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=short_names,
        y=df["5점 평균(점)"],
        marker_color=["#74C476", "#FD8D3C"],
        text=df["5점 평균(점)"].apply(lambda v: f"{v:.1f}점"),
        textposition="outside",
        hovertemplate="%{x}<br>평균: %{y:.1f}점<extra></extra>"
    ))

    fig.update_layout(
        **chart_layout,
        title="스마트폰 사용에 대한 인식 평균",
        xaxis_title="조사 대상",
        yaxis_title="5점 척도 평균",
        yaxis=dict(range=[0, 5], dtick=1),
        showlegend=False,
        height=430
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "※ 평균 점수가 높을수록 해당 문항에 동의하는 정도가 높습니다."
    )

# ---------------------------------------------
# 4. 해석할 때 주의할 점
# ---------------------------------------------
st.header("4. 해석할 때 주의할 점")

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
