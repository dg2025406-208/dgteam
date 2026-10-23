import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------
st.set_page_config(page_title="스마트폰 사용과 공부", layout="wide")

# 한글 폰트 설정 (환경에 따라 폰트가 없을 수 있음)
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.title("📱 스마트폰 사용은 공부에 방해가 될까?")
st.caption("자료 출처: 제공된 2015년 조사 자료 2종 (인식 조사)")

# ---------------------------------------------
# 데이터 (CSV 값을 직접 입력)
# ---------------------------------------------
data = {
    "구분": ["성인: 스마트폰이 하던 일(공부)에 방해된다",
             "청소년: 스마트폰 과다 사용으로 성적이 떨어졌다"],
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
choice_cols = ["1 전혀 그렇지 않다(%)", "2 그렇지 않다(%)",
               "3 그렇다(%)", "4 매우 그렇다(%)"]

# ---------------------------------------------
# 1. 표
# ---------------------------------------------
st.header("1. 자료 표")
st.dataframe(df.set_index("구분"), use_container_width=True)

# ---------------------------------------------
# 2. 핵심 지표
# ---------------------------------------------
st.header("2. 핵심 지표 비교")
col1, col2 = st.columns(2)
with col1:
    st.metric("성인 긍정 응답(3+4)", f"{df.loc[0, '긍정(3+4)(%)']}%")
    st.metric("성인 5점 평균", f"{df.loc[0, '5점 평균(점)']}점")
with col2:
    st.metric("청소년 긍정 응답(3+4)", f"{df.loc[1, '긍정(3+4)(%)']}%")
    st.metric("청소년 5점 평균", f"{df.loc[1, '5점 평균(점)']}점")

# ---------------------------------------------
# 3. 그래프
# ---------------------------------------------
st.header("3. 그래프")

tab1, tab2, tab3 = st.tabs(["응답 분포(누적 막대)", "긍정 vs 부정", "평균 점수"])

# 탭 1: 100% 누적 가로 막대
with tab1:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    left = [0, 0]
    colors = ["#9ecae1", "#c6dbef", "#fdae6b", "#e6550d"]
    for col, color in zip(choice_cols, colors):
        values = df[col].tolist()
        ax.barh(short_names, values, left=left, label=col, color=color)
        for i, v in enumerate(values):
            ax.text(left[i] + v / 2, i, f"{v}%", ha="center", va="center", fontsize=9)
        left = [l + v for l, v in zip(left, values)]
    ax.set_xlabel("비율(%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    st.pyplot(fig)

# 탭 2: 긍정 vs 부정 묶음 막대
with tab2:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(short_names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["부정(1+2)(%)"], width,
           label="부정(1+2)", color="#6baed6")
    ax.bar([i + width / 2 for i in x], df["긍정(3+4)(%)"], width,
           label="긍정(3+4)", color="#fd8d3c")
    for i in x:
        ax.text(i - width / 2, df["부정(1+2)(%)"][i] + 1,
                f"{df['부정(1+2)(%)'][i]}%", ha="center")
        ax.text(i + width / 2, df["긍정(3+4)(%)"][i] + 1,
                f"{df['긍정(3+4)(%)'][i]}%", ha="center")
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_names)
    ax.set_ylabel("비율(%)")
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)

# 탭 3: 평균 점수
with tab3:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(short_names, df["5점 평균(점)"], color=["#74c476", "#fd8d3c"])
    for i, v in enumerate(df["5점 평균(점)"]):
        ax.text(i, v + 0.05, f"{v}점", ha="center")
    ax.set_ylim(0, 5)
    ax.set_ylabel("5점 척도 평균")
    st.pyplot(fig)

# ---------------------------------------------
# 4. 해석 시 주의할 점
# ---------------------------------------------
st.header("4. 해석할 때 주의할 점")
st.warning(
    "- 두 문항의 질문 내용이 다릅니다. (방해 vs 성적 하락)\n"
    "- 조사 대상이 다릅니다. (성인 vs 청소년)\n"
    "- 인식 조사이므로 '오래 쓰면 공부 시간이 줄어든다'는 인과관계를 증명할 수 없습니다.\n"
    "- 2015년 자료이므로 현재와 다를 수 있습니다."
)

st.subheader("✍️ 나의 결론 써보기")
st.text_area("자료에서 알 수 있는 점과 한계를 직접 써보세요.", height=150)
