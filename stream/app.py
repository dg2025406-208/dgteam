
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------
st.set_page_config(
    page_title="스마트폰 사용과 공부",
    page_icon="📱",
    layout="wide"
)

st.title("📱 스마트폰 사용은 공부에 방해가 될까?")
st.caption("자료 출처: 제공된 2015년 성인·청소년 조사 CSV 파일")

BASE_DIR = Path(__file__).parent

adult_file = BASE_DIR / (
    "성인_스마트폰_사용이_지금_하고_있는일_공부_에_방해가_된다_20261023093236.csv"
)

teen_file = BASE_DIR / (
    "청소년_스마트폰의_지나친_사용으로_학교성적이_떨어졌다_20261023093223.csv"
)

# ---------------------------------------------
# 2. 원본 데이터 읽기
# ---------------------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

try:
    adult_raw = load_csv(str(adult_file))
    teen_raw = load_csv(str(teen_file))

except FileNotFoundError as e:
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.write("app.py와 CSV 파일 2개가 같은 폴더에 있는지 확인하세요.")
    st.code(str(e))
    st.stop()

# ---------------------------------------------
# 3. 데이터 개요
# ---------------------------------------------
st.header("1. 데이터 개요")

overview_tabs = st.tabs(["성인 자료", "청소년 자료"])

for tab, raw in zip(
    overview_tabs,
    [adult_raw, teen_raw]
):
    with tab:
        c1, c2 = st.columns(2)
        c1.metric("행 개수", raw.shape[0])
        c2.metric("열 개수", raw.shape[1])

        overview = pd.DataFrame({
            "항목": raw.columns.astype(str),
            "자료형": raw.dtypes.astype(str).values,
            "빈 값 개수": raw.isnull().sum().values
        })

        st.dataframe(
            overview,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"원본 전체 빈 값: {int(raw.isnull().sum().sum())}개"
        )

# ---------------------------------------------
# 4. 원본 표
# ---------------------------------------------
st.header("2. 원본 CSV 자료")

with st.expander("성인 원본 표", expanded=True):
    st.dataframe(
        adult_raw,
        use_container_width=True,
        hide_index=True
    )

with st.expander("청소년 원본 표", expanded=True):
    st.dataframe(
        teen_raw,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------
# 5. 분석용 자료 만들기
# ---------------------------------------------
st.header("3. 결측치·이상치 처리")

# 원본의 실제 조사 결과 행은 마지막 행
# 첫 번째 열은 조사 대상, 나머지 열은 조사 수치
def extract_result(raw, group):
    if raw.empty or len(raw.columns) != 10:
        raise ValueError(
            f"{group} 자료의 행 또는 열 구조를 확인해 주세요."
        )

    row = raw.iloc[-1]

    result = {
        "조사 대상": group,
        "응답자 수(명)": row.iloc[1],
        "전혀 그렇지 않다(%)": row.iloc[2],
        "그렇지 않다(%)": row.iloc[3],
        "그렇다(%)": row.iloc[4],
        "매우 그렇다(%)": row.iloc[5],
        "긍정 응답(원자료, %)": row.iloc[6],
        "부정 응답(원자료, %)": row.iloc[7],
        "5점 평균(점)": row.iloc[8],
        "100점 평균(점)": row.iloc[9]
    }

    numeric_columns = list(result.keys())[1:]

    for col in numeric_columns:
        result[col] = pd.to_numeric(
            result[col], errors="coerce"
        )

    return result


try:
    original_results = pd.DataFrame([
        extract_result(adult_raw, "성인"),
        extract_result(teen_raw, "청소년")
    ])
except ValueError as e:
    st.error(str(e))
    st.stop()

numeric_columns = [
    "응답자 수(명)",
    "전혀 그렇지 않다(%)",
    "그렇지 않다(%)",
    "그렇다(%)",
    "매우 그렇다(%)",
    "긍정 응답(원자료, %)",
    "부정 응답(원자료, %)",
    "5점 평균(점)",
    "100점 평균(점)"
]

percentage_columns = [
    "전혀 그렇지 않다(%)",
    "그렇지 않다(%)",
    "그렇다(%)",
    "매우 그렇다(%)",
    "긍정 응답(원자료, %)",
    "부정 응답(원자료, %)",
    "100점 평균(점)"
]

# 분석용 수치 변환 후 결측치 확인
conversion_missing = int(
    original_results[numeric_columns].isna().sum().sum()
)

# 원본 조사 결과는 보존하고, 별도 복사본에서 처리
analysis_df = original_results.copy()

# 이상치 조건
bad_percentage = (
    (analysis_df[percentage_columns] < 0) |
    (analysis_df[percentage_columns] > 100)
).any(axis=1)

bad_count = (
    (analysis_df["응답자 수(명)"] < 0) |
    analysis_df["응답자 수(명)"].isna()
)

bad_mean = (
    (analysis_df["5점 평균(점)"] < 1) |
    (analysis_df["5점 평균(점)"] > 5) |
    analysis_df["5점 평균(점)"].isna()
)

bad_rows = bad_percentage | bad_count | bad_mean

# 결측치가 있거나 이상치로 확인된 집단은 분석에서 제외
missing_rows = analysis_df[numeric_columns].isna().any(axis=1)
excluded_rows = bad_rows | missing_rows

analysis_df = analysis_df.loc[
    ~excluded_rows
].reset_index(drop=True)

# 처리 결과 표시
c1, c2, c3 = st.columns(3)

c1.metric(
    "원본 조사 집단",
    len(original_results)
)

c2.metric(
    "분석 가능 집단",
    len(analysis_df)
)

c3.metric(
    "분석 제외 집단",
    int(excluded_rows.sum())
)

st.markdown(f"""
- **원본 빈 값:** 성인 {int(adult_raw.isnull().sum().sum())}개,
  청소년 {int(teen_raw.isnull().sum().sum())}개
- **수치 변환 후 결측치:** {conversion_missing}개
- **이상치 검사:** 비율 0~100%, 응답자 수 0명 이상,
  5점 평균 1~5점 범위를 확인했습니다.
- **처리 방법:** 분석에 필요한 값이 없거나 범위를 벗어난 집단은
  그래프 분석에서 제외했습니다.
- **원본 보존:** 원본 표는 수정하거나 삭제하지 않았습니다.
""")

if conversion_missing == 0 and int(excluded_rows.sum()) == 0:
    st.success(
        "검사 결과, 분석용 자료에서 결측치나 범위를 벗어난 값이 발견되지 않았습니다."
    )

st.subheader("분석용 통합 표")

st.dataframe(
    analysis_df,
    use_container_width=True,
    hide_index=True
)

if len(analysis_df) < 2:
    st.warning(
        "두 집단 모두 분석할 수 있어야 비교 그래프를 그릴 수 있습니다. "
        "원본 CSV를 확인해 주세요."
    )
    st.stop()

# ---------------------------------------------
# 6. 핵심 지표
# ---------------------------------------------
st.header("4. 핵심 지표 비교")

adult = analysis_df[
    analysis_df["조사 대상"] == "성인"
].iloc[0]

teen = analysis_df[
    analysis_df["조사 대상"] == "청소년"
].iloc[0]

c1, c2 = st.columns(2)

with c1:
    st.subheader("성인")
    st.metric(
        "긍정 응답",
        f"{adult['긍정 응답(원자료, %)']:.1f}%"
    )
    st.metric(
        "5점 평균",
        f"{adult['5점 평균(점)']:.1f}점"
    )
    st.caption(
        f"응답자 {int(adult['응답자 수(명)']):,}명"
    )

with c2:
    st.subheader("청소년")
    st.metric(
        "긍정 응답",
        f"{teen['긍정 응답(원자료, %)']:.1f}%"
    )
    st.metric(
        "5점 평균",
        f"{teen['5점 평균(점)']:.1f}점"
    )
    st.caption(
        f"응답자 {int(teen['응답자 수(명)']):,}명"
    )

# ---------------------------------------------
# 7. 그래프
# ---------------------------------------------
st.header("5. 그래프 분석")

chart_layout = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": {"family": "Arial, sans-serif", "size": 14},
    "margin": {"l": 35, "r": 35, "t": 60, "b": 60},
    "height": 430
}

groups = ["성인", "청소년"]

tabs = st.tabs([
    "응답 분포",
    "긍정·부정 비교",
    "평균 점수"
])

# ---------------------------------------------
# 그래프 1: 누적 가로 막대그래프
# ---------------------------------------------
with tabs[0]:
    st.subheader("응답 항목별 비율")

    fig = go.Figure()

    response_cols = [
        ("전혀 그렇지 않다(%)", "#9ECAE1"),
        ("그렇지 않다(%)", "#C6DBEF"),
        ("그렇다(%)", "#FDAE6B"),
        ("매우 그렇다(%)", "#E6550D")
    ]

    for col, color in response_cols:
        fig.add_trace(go.Bar(
            y=analysis_df["조사 대상"],
            x=analysis_df[col],
            name=col,
            orientation="h",
            marker_color=color,
            text=analysis_df[col].map(
                lambda v: f"{v:.1f}%"
            ),
            textposition="inside",
            hovertemplate=(
                "%{y}<br>" + col +
                ": %{x:.1f}%<extra></extra>"
            )
        ))

    fig.update_layout(
        **chart_layout,
        barmode="stack",
        title="성인과 청소년의 응답 분포",
        xaxis_title="응답 비율(%)",
        yaxis_title="조사 대상",
        xaxis=dict(range=[0, 100], ticksuffix="%")
    )

    st.plotly_chart(fig, use_container_width=True)

    adult_agree = (
        adult["그렇다(%)"] + adult["매우 그렇다(%)"]
    )
    teen_agree = (
        teen["그렇다(%)"] + teen["매우 그렇다(%)"]
    )

    st.markdown(
        f"**찾아낸 것:** 성인은 '그렇다' 이상 응답이 "
        f"{adult_agree:.1f}%로 청소년의 "
        f"{teen_agree:.1f}%보다 높다."
    )

    st.caption(
        f"근거: 성인 {adult_agree:.1f}%, "
        f"청소년 {teen_agree:.1f}% "
        "(세부 항목 합계 기준)"
    )

# ---------------------------------------------
# 그래프 2: 묶은 세로 막대그래프
# ---------------------------------------------
with tabs[1]:
    st.subheader("긍정 응답과 부정 응답 비교")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=analysis_df["조사 대상"],
        y=analysis_df["부정 응답(원자료, %)"],
        name="부정 응답",
        marker_color="#6BAED6",
        text=analysis_df["부정 응답(원자료, %)"].map(
            lambda v: f"{v:.1f}%"
        ),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=analysis_df["조사 대상"],
        y=analysis_df["긍정 응답(원자료, %)"],
        name="긍정 응답",
        marker_color="#FD8D3C",
        text=analysis_df["긍정 응답(원자료, %)"].map(
            lambda v: f"{v:.1f}%"
        ),
        textposition="outside"
    ))

    fig.update_layout(
        **chart_layout,
        barmode="group",
        title="성인과 청소년의 긍정·부정 응답",
        xaxis_title="조사 대상",
        yaxis_title="응답 비율(%)",
        yaxis=dict(range=[0, 100], ticksuffix="%")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**찾아낸 것:** 두 집단 모두 부정 응답이 "
        "긍정 응답보다 많다."
    )

    st.caption(
        f"근거: 성인 부정 {adult['부정 응답(원자료, %)']:.1f}%, "
        f"긍정 {adult['긍정 응답(원자료, %)']:.1f}% / "
        f"청소년 부정 {teen['부정 응답(원자료, %)']:.1f}%, "
        f"긍정 {teen['긍정 응답(원자료, %)']:.1f}%"
    )

# ---------------------------------------------
# 그래프 3: 점과 선 그래프
# ---------------------------------------------
with tabs[2]:
    st.subheader("5점 평균 비교")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=analysis_df["조사 대상"],
        y=analysis_df["5점 평균(점)"],
        mode="markers+lines+text",
        name="5점 평균",
        text=analysis_df["5점 평균(점)"].map(
            lambda v: f"{v:.1f}점"
        ),
        textposition="top center",
        marker=dict(size=13),
        line=dict(width=2)
    ))

    fig.add_hline(
        y=3,
        line_dash="dash",
        line_color="gray",
        annotation_text="중간값 3점"
    )

    fig.update_layout(
        **chart_layout,
        title="성인과 청소년의 5점 평균",
        xaxis_title="조사 대상",
        yaxis_title="5점 평균(점)",
        yaxis=dict(range=[0, 5], dtick=1),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    difference = (
        adult["5점 평균(점)"] -
        teen["5점 평균(점)"]
    )

    direction = (
        "높다" if difference > 0
        else "낮다" if difference < 0
        else "같다"
    )

    st.markdown(
        f"**찾아낸 것:** 성인의 5점 평균은 청소년보다 "
        f"{abs(difference):.1f}점 {direction}."
    )

    st.caption(
        f"근거: 성인 {adult['5점 평균(점)']:.1f}점, "
        f"청소년 {teen['5점 평균(점)']:.1f}점"
    )

# ---------------------------------------------
# 8. 분석의 한계
# ---------------------------------------------
st.header("6. 해석할 때 주의할 점")

st.warning("""
- 성인 문항은 스마트폰이 공부에 방해되는지 묻습니다.
- 청소년 문항은 스마트폰 과다 사용으로 성적이 떨어졌는지 묻습니다.
- 질문 내용과 조사 대상이 다르므로 두 결과를 완전히 같은 의미로 비교하기 어렵습니다.
- 인식 조사만으로 스마트폰 사용이 공부 시간을 줄이거나 성적을 떨어뜨린다는 인과관계를 증명할 수 없습니다.
- 2015년 자료이므로 현재 상황과 다를 수 있습니다.
- 긍정 응답의 원자료 값과 세부 항목 합계에는 반올림 차이가 있을 수 있습니다.
""")

st.subheader("✍️ 나의 결론 써보기")

st.text_area(
    "자료에서 알 수 있는 점과 한계를 작성해 보세요.",
    height=150,
    placeholder=(
        "예: 조사 결과에서 알 수 있는 점은 ...이다. "
        "하지만 두 문항의 질문 내용이 다르다는 한계가 있다."
    )
)
