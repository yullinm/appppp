# app.py
# CogCompass: AI 연구 네비게이터(인지심리학)
# - Single-file Streamlit app
# - Sidebar: OpenAI API Key, local JSON persistence toggle
# - Core: research topic generator, trending paper fetch + LLM summaries, idea expansion + actionable next steps

import os
import re
import json
import time
import math
import uuid
import textwrap
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="CogCompass: AI 연구 네비게이터",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Style (readability-first)
# -----------------------------
CUSTOM_CSS = """
<style>
/* Improve readability */
html, body, [class*="css"]  {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(49, 51, 63, 0.7); font-size: 0.92rem; }
.badge {
  display:inline-block; padding:0.18rem 0.52rem; border-radius:999px;
  background: rgba(14, 17, 23, 0.06); margin-right:0.35rem; font-size:0.86rem;
}
.card {
  border: 1px solid rgba(49, 51, 63, 0.18);
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
  background: rgba(255,255,255,0.6);
}
hr { margin: 0.8rem 0; }
code { font-size: 0.92em; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Persistence
# -----------------------------
DEFAULT_STORE_PATH = "cogcompass_store.json"

def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def safe_read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_store(use_local: bool, path: str) -> Dict[str, Any]:
    if use_local:
        data = safe_read_json(path)
        return data if isinstance(data, dict) else {}
    # session store
    return st.session_state.get("store", {})

def save_store(use_local: bool, path: str, store: Dict[str, Any]) -> None:
    if use_local:
        safe_write_json(path, store)
    else:
        st.session_state["store"] = store

def ensure_store_schema(store: Dict[str, Any]) -> Dict[str, Any]:
    store = store or {}
    store.setdefault("created_at", _now_iso())
    store.setdefault("updated_at", _now_iso())
    store.setdefault("topics", [])       # generated topics
    store.setdefault("papers", [])       # fetched papers + summaries
    store.setdefault("notes", [])        # user notes
    store.setdefault("favorites", {"topics": [], "papers": []})
    return store


# -----------------------------
# OpenAI client (supports both new & legacy)
# -----------------------------
def openai_chat_completion(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.5,
    max_tokens: int = 900,
) -> str:
    """
    Returns assistant text.
    Tries OpenAI v1 SDK; falls back to REST if needed.
    """
    if not api_key:
        raise ValueError("OpenAI API Key가 필요합니다.")

    # Prefer new SDK if available
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # REST fallback
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI API 오류: {r.status_code} / {r.text[:300]}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


# -----------------------------
# External paper sources (no key)
# -----------------------------
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_semantic_scholar_papers(
    query: str,
    limit: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sort: str = "citationCount",  # citationCount | publicationDate
) -> List[Dict[str, Any]]:
    """
    Semantic Scholar Paper Search.
    """
    if not query.strip():
        return []

    params = {
        "query": query,
        "limit": min(max(limit, 1), 40),
        "fields": "title,abstract,year,authors,venue,citationCount,publicationDate,url,externalIds,isOpenAccess,openAccessPdf",
        "offset": 0,
        "sort": sort,
    }
    # year filter: Semantic Scholar supports year in query syntax. We apply query decoration.
    q = query.strip()
    if year_from and year_to:
        q = f"{q} year:{year_from}-{year_to}"
    elif year_from:
        q = f"{q} year:{year_from}-"
    elif year_to:
        q = f"{q} year:-{year_to}"
    params["query"] = q

    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    r = requests.get(url, params=params, timeout=30)
    if r.status_code >= 400:
        return []
    data = r.json()
    out = []
    for p in data.get("data", []) or []:
        out.append({
            "source": "SemanticScholar",
            "paperId": p.get("paperId"),
            "title": _clean_text(p.get("title", "")),
            "abstract": _clean_text(p.get("abstract", "")),
            "year": p.get("year"),
            "publicationDate": p.get("publicationDate"),
            "authors": [a.get("name") for a in (p.get("authors") or []) if a.get("name")],
            "venue": p.get("venue"),
            "citationCount": p.get("citationCount"),
            "url": p.get("url"),
            "isOpenAccess": p.get("isOpenAccess"),
            "openAccessPdf": (p.get("openAccessPdf") or {}).get("url"),
            "externalIds": p.get("externalIds") or {},
        })
    return out

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_arxiv_papers(
    query: str,
    limit: int = 10,
    sortBy: str = "submittedDate",  # relevance | submittedDate | lastUpdatedDate
    sortOrder: str = "descending",
) -> List[Dict[str, Any]]:
    """
    arXiv API (ATOM). Light parsing without external libs.
    """
    if not query.strip():
        return []

    # arXiv query: wrap keywords
    q = "all:" + " AND all:".join([re.sub(r"[^\w\-]+", " ", t).strip() for t in query.split() if t.strip()])
    params = {
        "search_query": q,
        "start": 0,
        "max_results": min(max(limit, 1), 40),
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    if r.status_code >= 400:
        return []
    xml = r.text

    # Minimal ATOM parsing using regex (good enough for a Streamlit MVP)
    entries = re.split(r"<entry>", xml)[1:]
    out = []
    for e in entries:
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        summary = re.search(r"<summary>(.*?)</summary>", e, re.S)
        published = re.search(r"<published>(.*?)</published>", e, re.S)
        link = re.search(r"<id>(.*?)</id>", e, re.S)
        authors = re.findall(r"<name>(.*?)</name>", e, re.S)

        t = _clean_text(title.group(1)) if title else ""
        s = _clean_text(summary.group(1)) if summary else ""
        pub = _clean_text(published.group(1)) if published else ""
        u = _clean_text(link.group(1)) if link else ""
        out.append({
            "source": "arXiv",
            "paperId": None,
            "title": t,
            "abstract": s,
            "year": int(pub[:4]) if len(pub) >= 4 and pub[:4].isdigit() else None,
            "publicationDate": pub,
            "authors": [_clean_text(a) for a in authors if _clean_text(a)],
            "venue": "arXiv",
            "citationCount": None,
            "url": u,
            "isOpenAccess": True,
            "openAccessPdf": u.replace("/abs/", "/pdf/") + ".pdf" if "/abs/" in u else None,
            "externalIds": {},
        })
    return out


# -----------------------------
# LLM prompt helpers (philosophy: actionable)
# -----------------------------
SYSTEM_CORE = """너는 'CogCompass: AI 연구 네비게이터(인지심리학)'의 연구 코파일럿이다.
사용자에게 '무엇을 해야 하는지'가 바로 보이도록, 항상 실행 가능한 형태로 제안한다.
금지: 추상적 조언만 하기, 뻔한 교과서식 설명, 과한 장황함.
필수: (1) 구체 연구질문/가설 (2) 최소 실행 실험/분석 계획 (3) 필요한 데이터/자극/측정 (4) 리스크/대안 (5) 다음 7일 TODO를 포함해라.
출력은 한국어로, 구조화된 불릿/섹션으로 간결히 작성한다.
"""

def build_topic_prompt(
    user_context: str,
    focus: str,
    methods: List[str],
    constraints: str,
    n: int,
) -> List[Dict[str, str]]:
    method_str = ", ".join(methods) if methods else "제약 없음"
    user_context = user_context.strip()
    constraints = constraints.strip()

    u = f"""
[사용자 맥락]
{user_context if user_context else "없음"}

[관심 초점]
{focus}

[선호 방법론/데이터]
{method_str}

[제약/현실 조건]
{constraints if constraints else "없음"}

요청: 인지심리학 연구 주제 {n}개를 '던져줘'. 단순 아이디어가 아니라, 바로 착수 가능한 수준으로.
각 주제마다 아래 포맷을 반드시 지켜라:

- 제목(짧게)
- 핵심 질문 1문장
- 가설 2개(검증가능)
- 최소 실험/분석 디자인(표본, 과제, 조작/측정, 분석)
- 필요한 리소스(도구/자극/데이터)
- 실패 가능 지점 & 플랜B
- 7일 실행 TODO(체크리스트)
- (선택) 관련 키워드(논문 검색용) 5개
"""
    return [
        {"role": "system", "content": SYSTEM_CORE},
        {"role": "user", "content": _clean_text(u)},
    ]

def build_paper_summary_prompt(paper: Dict[str, Any], user_goal: str) -> List[Dict[str, str]]:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    meta = {
        "source": paper.get("source"),
        "year": paper.get("year"),
        "publicationDate": paper.get("publicationDate"),
        "venue": paper.get("venue"),
        "citationCount": paper.get("citationCount"),
        "authors": (paper.get("authors") or [])[:6],
        "url": paper.get("url"),
        "openAccessPdf": paper.get("openAccessPdf"),
    }
    u = f"""
[사용자 목표]
{user_goal.strip() if user_goal.strip() else "인지심리학에서 실행 가능한 연구 아이디어 발굴"}

[논문 메타]
{json.dumps(meta, ensure_ascii=False)}

[제목]
{title}

[초록]
{abstract if abstract else "(초록 없음 — 제목/메타 기반으로 추정하되, 불확실성 명시)"}

요청: 이 논문을 '연구 실행 관점'에서 요약해라.
반드시 포함:
1) 한 줄 요약
2) 연구 질문/가설(추정 가능하나 근거/불확실성 표기)
3) 방법(과제/측정/분석) 핵심
4) 결과/기여(초록 기반)
5) 재현/확장 아이디어 3개(각각: 조작, 측정, 예상효과)
6) 내가 지금 당장 할 수 있는 다음 행동 5개(검색어/데이터/코드/실험)
형식: 번호 섹션 + 불릿, 한국어, 간결.
"""
    return [
        {"role": "system", "content": SYSTEM_CORE},
        {"role": "user", "content": _clean_text(u)},
    ]

def build_idea_expansion_prompt(
    seed: str,
    papers_context: List[Dict[str, Any]],
    desired_output: str,
) -> List[Dict[str, str]]:
    # Provide compact context (titles + 1-liners if available)
    ctx_lines = []
    for p in papers_context[:6]:
        t = p.get("title", "")
        s = ""
        if p.get("llm_summary"):
            # take first line
            s = p["llm_summary"].splitlines()[0][:180]
        ctx_lines.append(f"- {t} :: {s}".strip())
    ctx = "\n".join(ctx_lines) if ctx_lines else "(참고 논문 컨텍스트 없음)"

    u = f"""
[씨드 아이디어/문제의식]
{seed.strip()}

[참고 논문 컨텍스트(제목+요약 한줄)]
{ctx}

[원하는 결과물]
{desired_output.strip()}

요청: 위 씨드를 '연구 계획'으로 발전시켜라.
반드시 포함:
A) 메커니즘/구성개념 정의(작동 가정)
B) 테스트 가능한 가설 3~5개(각각: 조작/측정/기대방향)
C) 최소 실험 1개 + 확장 실험 1개(각각: 디자인, 표본, 자극, 절차, 품질관리)
D) 분석 계획(모델/검정, 주요 DV/IV, 사전 기준)
E) 파워/표본 크기 산정에 필요한 정보(추정치/가정)
F) 윤리/편향/교란요인 체크
G) 7일 실행 플랜(데일리 TODO)
H) 논문 검색 쿼리 5개(영문)
형식: 섹션 헤더 + 불릿, 한국어, 간결.
"""
    return [
        {"role": "system", "content": SYSTEM_CORE},
        {"role": "user", "content": _clean_text(u)},
    ]


# -----------------------------
# UI helpers
# -----------------------------
def render_paper_card(p: Dict[str, Any], idx: int, allow_actions: bool = True) -> None:
    title = p.get("title", "(제목 없음)")
    authors = ", ".join(p.get("authors") or []) if p.get("authors") else "저자 정보 없음"
    year = p.get("year") or ""
    venue = p.get("venue") or ""
    src = p.get("source") or ""
    cites = p.get("citationCount")
    url = p.get("url")
    pdf = p.get("openAccessPdf")

    badges = []
    if src: badges.append(f"<span class='badge'>{src}</span>")
    if year: badges.append(f"<span class='badge'>{year}</span>")
    if venue: badges.append(f"<span class='badge'>{venue}</span>")
    if cites is not None: badges.append(f"<span class='badge'>cites: {cites}</span>")
    if p.get("isOpenAccess"): badges.append("<span class='badge'>OpenAccess</span>")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**{idx+1}. {title}**")
    st.markdown("".join(badges), unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Authors: {authors}</div>", unsafe_allow_html=True)

    abs_txt = (p.get("abstract") or "").strip()
    if abs_txt:
        st.write(textwrap.shorten(abs_txt, width=450, placeholder=" …"))
    else:
        st.markdown("<span class='small-muted'>초록 없음</span>", unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 2])
    if url:
        cols[0].link_button("원문", url, use_container_width=True)
    else:
        cols[0].button("원문", disabled=True, use_container_width=True)
    if pdf:
        cols[1].link_button("PDF", pdf, use_container_width=True)
    else:
        cols[1].button("PDF", disabled=True, use_container_width=True)

    if allow_actions:
        if cols[2].button("⭐ 즐겨찾기", key=f"fav_p_{p.get('id')}_{idx}", use_container_width=True):
            add_favorite_paper(p)
            st.toast("즐겨찾기에 추가됨", icon="⭐")

        if cols[3].button("요약", key=f"sum_p_{p.get('id')}_{idx}", use_container_width=True):
            st.session_state["selected_paper_id"] = p.get("id")
            st.session_state["active_tab"] = "인기 논문"

    st.markdown("</div>", unsafe_allow_html=True)

def add_favorite_topic(topic: Dict[str, Any]) -> None:
    store = st.session_state["store_obj"]
    fav = store.setdefault("favorites", {}).setdefault("topics", [])
    # dedupe by id
    if not any(x.get("id") == topic.get("id") for x in fav):
        fav.append(topic)
        store["updated_at"] = _now_iso()
        st.session_state["store_obj"] = store

def add_favorite_paper(paper: Dict[str, Any]) -> None:
    store = st.session_state["store_obj"]
    fav = store.setdefault("favorites", {}).setdefault("papers", [])
    if not any(x.get("id") == paper.get("id") for x in fav):
        fav.append(paper)
        store["updated_at"] = _now_iso()
        st.session_state["store_obj"] = store

def upsert_paper_in_store(paper: Dict[str, Any]) -> None:
    store = st.session_state["store_obj"]
    papers = store.setdefault("papers", [])
    pid = paper.get("id")
    for i, p in enumerate(papers):
        if p.get("id") == pid:
            papers[i] = paper
            store["updated_at"] = _now_iso()
            st.session_state["store_obj"] = store
            return
    papers.append(paper)
    store["updated_at"] = _now_iso()
    st.session_state["store_obj"] = store

def upsert_topic_in_store(topic: Dict[str, Any]) -> None:
    store = st.session_state["store_obj"]
    topics = store.setdefault("topics", [])
    tid = topic.get("id")
    for i, t in enumerate(topics):
        if t.get("id") == tid:
            topics[i] = topic
            store["updated_at"] = _now_iso()
            st.session_state["store_obj"] = store
            return
    topics.append(topic)
    store["updated_at"] = _now_iso()
    st.session_state["store_obj"] = store


# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.title("🧠 CogCompass 설정")

api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="LLM 요약/아이디어 확장 기능에 필요합니다.")
model = st.sidebar.selectbox(
    "모델",
    options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
    index=0,
    help="계정에서 사용 가능한 모델을 선택하세요.",
)

use_local = st.sidebar.toggle("로컬 저장 사용", value=False, help="ON이면 로컬 JSON 파일에 저장합니다. OFF면 session_state만 사용.")
store_path = st.sidebar.text_input("로컬 저장 경로", value=DEFAULT_STORE_PATH, disabled=not use_local)

st.sidebar.markdown("---")
st.sidebar.caption("철학: ‘연구 주제 던지기 → 인기 논문 요약 → 생각 발전 → 다음 7일 실행’까지 한 번에.")

# Load store
_store = ensure_store_schema(load_store(use_local, store_path))
st.session_state["store_obj"] = _store  # working copy


# -----------------------------
# Header
# -----------------------------
left, right = st.columns([3, 2])
with left:
    st.title("CogCompass: AI 연구 네비게이터")
    st.markdown("<div class='small-muted'>인지심리학 연구를 ‘바로 실행’ 가능한 형태로 던져주는 앱</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**오늘의 워크플로우**")
    st.markdown(
        "- ① **주제 던지기**: 바로 착수 가능한 연구안 생성\n"
        "- ② **인기 논문**: 최신/인기 논문 수집 → 실행 관점 요약\n"
        "- ③ **생각 발전**: 씨드 + 논문 컨텍스트로 연구계획 완성"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Persist changes button
persist_col1, persist_col2, persist_col3 = st.columns([1, 1, 2])
with persist_col1:
    if st.button("💾 저장", use_container_width=True):
        save_store(use_local, store_path, st.session_state["store_obj"])
        st.toast("저장 완료", icon="💾")
with persist_col2:
    if st.button("🧹 초기화", use_container_width=True):
        st.session_state["store_obj"] = ensure_store_schema({})
        save_store(use_local, store_path, st.session_state["store_obj"])
        st.toast("저장소 초기화", icon="🧹")

with persist_col3:
    st.markdown(
        f"<div class='small-muted'>저장 방식: <b>{'로컬 JSON' if use_local else '세션(session_state)'}</b> · "
        f"마지막 업데이트: {st.session_state['store_obj'].get('updated_at','-')}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# -----------------------------
# Tabs
# -----------------------------
tab_names = ["주제 던지기", "인기 논문", "생각 발전", "라이브러리(저장됨)"]
default_tab = st.session_state.get("active_tab", "주제 던지기")
tab_index = tab_names.index(default_tab) if default_tab in tab_names else 0
tabs = st.tabs(tab_names)
# Note: Streamlit tabs don't support programmatic switching reliably; we emulate via session_state flags.

# -----------------------------
# Tab 1: Topic Generator
# -----------------------------
with tabs[0]:
    st.subheader("주제 던지기: ‘바로 시작 가능한’ 연구안 생성")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        user_context = st.text_area(
            "내 상황/맥락 (선택)",
            placeholder="예: 석사 1년차, 온라인 실험만 가능, 시선추적 없음, 표본 80명 정도...",
            height=120,
        )
        focus = st.text_input(
            "관심 초점",
            value="주의(attention)와 작업기억(working memory)의 상호작용",
            help="한 문장으로 적을수록 좋습니다.",
        )
    with c2:
        methods = st.multiselect(
            "선호 방법론/데이터",
            options=["온라인 행동실험", "실험실 행동실험", "설문/척도", "EEG/ERP", "fMRI", "안구추적", "계산모델/시뮬레이션", "메타분석", "기존 공개데이터 재분석"],
            default=["온라인 행동실험"],
        )
        constraints = st.text_area(
            "제약/현실 조건 (선택)",
            placeholder="예: 2주 내 파일럿 필요, 자극 제작 최소화, 한국어 과제 선호, 참가자 비용 제한...",
            height=120,
        )
    with c3:
        n_topics = st.number_input("생성 개수", min_value=3, max_value=10, value=5, step=1)
        temperature = st.slider("창의성(temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        generate_btn = st.button("🧠 연구 주제 던져줘", use_container_width=True)

    if generate_btn:
        if not api_key:
            st.error("OpenAI API Key를 사이드바에 입력하세요.")
        else:
            with st.spinner("연구 주제 생성 중..."):
                messages = build_topic_prompt(user_context, focus, methods, constraints, int(n_topics))
                try:
                    out = openai_chat_completion(
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        temperature=float(temperature),
                        max_tokens=1400,
                    )
                except Exception as e:
                    st.error(str(e))
                    out = ""

            if out:
                topic_obj = {
                    "id": str(uuid.uuid4()),
                    "created_at": _now_iso(),
                    "focus": focus,
                    "methods": methods,
                    "constraints": constraints,
                    "user_context": user_context,
                    "llm_output": out,
                }
                upsert_topic_in_store(topic_obj)
                save_store(use_local, store_path, st.session_state["store_obj"])

                st.markdown("### 결과")
                st.markdown(out)

                colA, colB = st.columns([1, 1])
                if colA.button("⭐ 이 결과 즐겨찾기", use_container_width=True):
                    add_favorite_topic(topic_obj)
                    save_store(use_local, store_path, st.session_state["store_obj"])
                    st.toast("즐겨찾기에 추가됨", icon="⭐")
                if colB.button("➡️ 생각 발전으로 가져가기", use_container_width=True):
                    st.session_state["seed_idea"] = out
                    st.session_state["active_tab"] = "생각 발전"
                    st.success("씨드를 저장했습니다. 상단의 ‘생각 발전’ 탭으로 이동해 계속 진행하세요.")

    # Recent topics
    st.markdown("---")
    st.markdown("#### 최근 생성한 주제")
    recent_topics = list(reversed(st.session_state["store_obj"].get("topics", [])))[0:5]
    if not recent_topics:
        st.markdown("<div class='small-muted'>아직 생성 기록이 없습니다.</div>", unsafe_allow_html=True)
    else:
        for t in recent_topics:
            with st.expander(f"🧩 {t.get('focus','(no focus)')} · {t.get('created_at','')}", expanded=False):
                st.markdown(t.get("llm_output", ""))
                c1, c2, c3 = st.columns([1, 1, 2])
                if c1.button("⭐ 즐겨찾기", key=f"fav_topic_{t.get('id')}", use_container_width=True):
                    add_favorite_topic(t)
                    save_store(use_local, store_path, st.session_state["store_obj"])
                    st.toast("즐겨찾기에 추가됨", icon="⭐")
                if c2.button("➡️ 씨드로 사용", key=f"use_seed_{t.get('id')}", use_container_width=True):
                    st.session_state["seed_idea"] = t.get("llm_output", "")
                    st.session_state["active_tab"] = "생각 발전"
                    st.success("씨드를 저장했습니다. ‘생각 발전’ 탭에서 이어가세요.")
                st.caption(f"Methods: {', '.join(t.get('methods') or [])}")


# -----------------------------
# Tab 2: Trending Papers + Summaries
# -----------------------------
with tabs[1]:
    st.subheader("인기 논문: 수집 → 실행 관점 요약 → 아이디어로 연결")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        paper_query = st.text_input(
            "검색 키워드(영문 권장)",
            value="attention working memory cognitive control",
            help="Semantic Scholar/arXiv에서 검색합니다. (예: 'visual attention suppression', 'task switching' 등)",
        )
        user_goal = st.text_input("요약을 어떤 목적에 맞출까?", value="2주 내 온라인 행동실험으로 확장 가능한 아이디어 찾기")
    with c2:
        source_choice = st.selectbox("소스", ["Semantic Scholar", "arXiv", "둘 다"], index=0)
        sort_choice = st.selectbox("정렬(SS)", ["citationCount", "publicationDate"], index=0)
    with c3:
        year_from = st.number_input("From(연도)", min_value=1990, max_value=2100, value=2022, step=1)
        year_to = st.number_input("To(연도)", min_value=1990, max_value=2100, value=2026, step=1)
    with c4:
        k = st.number_input("가져올 개수", min_value=5, max_value=20, value=10, step=1)
        fetch_btn = st.button("🔎 논문 가져오기", use_container_width=True)

    papers: List[Dict[str, Any]] = []
    if fetch_btn:
        with st.spinner("논문 수집 중..."):
            if source_choice in ("Semantic Scholar", "둘 다"):
                ss = fetch_semantic_scholar_papers(
                    paper_query,
                    limit=int(k),
                    year_from=int(year_from) if year_from else None,
                    year_to=int(year_to) if year_to else None,
                    sort=sort_choice,
                )
                papers.extend(ss)
            if source_choice in ("arXiv", "둘 다"):
                ax = fetch_arxiv_papers(paper_query, limit=int(k))
                papers.extend(ax)

        # assign internal ids and store
        normalized = []
        for p in papers:
            pid = p.get("paperId") or p.get("url") or (p.get("title","") + "_" + (p.get("publicationDate") or ""))
            p2 = dict(p)
            p2["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(pid)))
            p2.setdefault("fetched_at", _now_iso())
            normalized.append(p2)

        st.session_state["last_fetched_papers"] = normalized

        # upsert into store (without LLM summary yet)
        for p in normalized:
            upsert_paper_in_store(p)
        save_store(use_local, store_path, st.session_state["store_obj"])
        st.toast(f"{len(normalized)}개 저장됨", icon="📚")

    # Display last fetched
    st.markdown("---")
    fetched = st.session_state.get("last_fetched_papers", [])
    if not fetched:
        st.markdown("<div class='small-muted'>아직 수집된 논문이 없습니다. 위에서 검색해보세요.</div>", unsafe_allow_html=True)
    else:
        st.markdown("### 수집된 논문")
        for i, p in enumerate(fetched):
            render_paper_card(p, i, allow_actions=True)

    st.markdown("---")
    st.markdown("### 선택 논문 실행 관점 요약")

    selected_id = st.session_state.get("selected_paper_id")
    store_papers = st.session_state["store_obj"].get("papers", [])
    selected = next((p for p in store_papers if p.get("id") == selected_id), None) if selected_id else None

    if not selected:
        st.markdown("<div class='small-muted'>논문 카드에서 <b>요약</b>을 눌러 선택하세요.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"**선택됨:** {selected.get('title','')}")
        do_sum = st.button("🧾 LLM 요약 생성/갱신", use_container_width=True)

        if do_sum:
            if not api_key:
                st.error("OpenAI API Key를 사이드바에 입력하세요.")
            else:
                with st.spinner("요약 생성 중..."):
                    messages = build_paper_summary_prompt(selected, user_goal)
                    try:
                        summary = openai_chat_completion(
                            api_key=api_key,
                            model=model,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=1100,
                        )
                    except Exception as e:
                        st.error(str(e))
                        summary = ""

                if summary:
                    selected["llm_summary"] = summary
                    selected["summary_updated_at"] = _now_iso()
                    upsert_paper_in_store(selected)
                    save_store(use_local, store_path, st.session_state["store_obj"])
                    st.toast("요약 저장 완료", icon="✅")

        if selected.get("llm_summary"):
            st.markdown(selected["llm_summary"])
            c1, c2 = st.columns([1, 1])
            if c1.button("➡️ 생각 발전에 컨텍스트로 사용", use_container_width=True):
                st.session_state["selected_papers_for_expansion"] = [selected.get("id")]
                st.session_state["active_tab"] = "생각 발전"
                st.success("선택 논문을 컨텍스트로 저장했습니다. ‘생각 발전’ 탭에서 이어가세요.")
            if c2.button("⭐ 즐겨찾기", use_container_width=True):
                add_favorite_paper(selected)
                save_store(use_local, store_path, st.session_state["store_obj"])
                st.toast("즐겨찾기에 추가됨", icon="⭐")
        else:
            st.markdown("<div class='small-muted'>아직 요약이 없습니다. 위 버튼으로 생성하세요.</div>", unsafe_allow_html=True)


# -----------------------------
# Tab 3: Idea Expansion
# -----------------------------
with tabs[2]:
    st.subheader("생각 발전: 씨드 + 논문 컨텍스트 → 연구계획 & 7일 플랜")

    # Pick seed
    seed_default = st.session_state.get("seed_idea", "")
    seed = st.text_area(
        "씨드 아이디어 / 문제의식",
        value=seed_default,
        placeholder="주제 던지기 결과 일부를 붙여넣거나, 당신의 아이디어를 직접 입력하세요.",
        height=220,
    )

    # Select papers context
    st.markdown("#### 컨텍스트로 사용할 논문 선택 (저장된 논문 중)")
    store_papers = st.session_state["store_obj"].get("papers", [])
    paper_options = []
    for p in store_papers:
        label = f"{p.get('title','(no title)')[:80]}"
        paper_options.append((label, p.get("id")))

    preselected = st.session_state.get("selected_papers_for_expansion", [])
    selected_ids = st.multiselect(
        "논문 선택 (최대 6개 권장)",
        options=[pid for _, pid in paper_options],
        default=[pid for pid in preselected if pid in [x[1] for x in paper_options]],
        format_func=lambda pid: next((lbl for lbl, _pid in paper_options if _pid == pid), pid),
    )

    desired_output = st.text_input(
        "원하는 결과물 형태(선택)",
        value="온라인 행동실험 1개 + 확장 실험 1개가 포함된 연구 계획서",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        temp2 = st.slider("정교함(temperature)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    with c2:
        run_btn = st.button("🧭 계획으로 발전시키기", use_container_width=True)
    with c3:
        st.markdown("<div class='small-muted'>팁: ‘씨드’를 짧게 쓰지 말고, 이미 가진 가정/제약/목표를 같이 적으면 품질이 크게 올라갑니다.</div>", unsafe_allow_html=True)

    if run_btn:
        if not api_key:
            st.error("OpenAI API Key를 사이드바에 입력하세요.")
        elif not seed.strip():
            st.error("씨드 아이디어를 입력하세요.")
        else:
            # Build context papers
            ctx_papers = []
            for pid in selected_ids[:6]:
                p = next((x for x in store_papers if x.get("id") == pid), None)
                if p:
                    ctx_papers.append(p)

            with st.spinner("연구계획 생성 중..."):
                messages = build_idea_expansion_prompt(seed, ctx_papers, desired_output)
                try:
                    plan = openai_chat_completion(
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        temperature=float(temp2),
                        max_tokens=1600,
                    )
                except Exception as e:
                    st.error(str(e))
                    plan = ""

            if plan:
                note = {
                    "id": str(uuid.uuid4()),
                    "created_at": _now_iso(),
                    "type": "expanded_plan",
                    "seed": seed,
                    "paper_ids": selected_ids[:6],
                    "output": plan,
                }
                store = st.session_state["store_obj"]
                store.setdefault("notes", []).append(note)
                store["updated_at"] = _now_iso()
                st.session_state["store_obj"] = store
                save_store(use_local, store_path, st.session_state["store_obj"])
                st.toast("계획 저장 완료", icon="✅")

                st.markdown("### 결과: 연구계획")
                st.markdown(plan)

                cA, cB = st.columns([1, 1])
                if cA.button("📌 씨드로 재사용", use_container_width=True):
                    st.session_state["seed_idea"] = plan
                    st.toast("결과를 씨드로 저장했습니다.", icon="📌")
                if cB.button("💾 저장", use_container_width=True):
                    save_store(use_local, store_path, st.session_state["store_obj"])
                    st.toast("저장 완료", icon="💾")


# -----------------------------
# Tab 4: Library
# -----------------------------
with tabs[3]:
    st.subheader("라이브러리(저장됨): 주제 / 논문 / 플랜 / 즐겨찾기")

    store = st.session_state["store_obj"]
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.metric("주제", len(store.get("topics", [])))
    c2.metric("논문", len(store.get("papers", [])))
    c3.metric("노트/플랜", len(store.get("notes", [])))
    fav_count = len(store.get("favorites", {}).get("topics", [])) + len(store.get("favorites", {}).get("papers", []))
    c4.metric("즐겨찾기", fav_count)

    st.markdown("---")

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### ⭐ 즐겨찾기: 주제")
        fav_topics = store.get("favorites", {}).get("topics", [])
        if not fav_topics:
            st.markdown("<div class='small-muted'>즐겨찾기된 주제가 없습니다.</div>", unsafe_allow_html=True)
        else:
            for t in reversed(fav_topics[-20:]):
                with st.expander(f"⭐ {t.get('focus','(no focus)')} · {t.get('created_at','')}", expanded=False):
                    st.markdown(t.get("llm_output", ""))

        st.markdown("### 🧩 저장된 주제")
        topics = store.get("topics", [])
        if not topics:
            st.markdown("<div class='small-muted'>저장된 주제가 없습니다.</div>", unsafe_allow_html=True)
        else:
            for t in reversed(topics[-15:]):
                with st.expander(f"🧩 {t.get('focus','(no focus)')} · {t.get('created_at','')}", expanded=False):
                    st.markdown(t.get("llm_output", ""))
                    c1, c2 = st.columns([1, 1])
                    if c1.button("⭐ 즐겨찾기", key=f"fav_t_lib_{t.get('id')}", use_container_width=True):
                        add_favorite_topic(t)
                        save_store(use_local, store_path, st.session_state["store_obj"])
                        st.toast("즐겨찾기에 추가됨", icon="⭐")
                    if c2.button("➡️ 씨드로", key=f"seed_t_lib_{t.get('id')}", use_container_width=True):
                        st.session_state["seed_idea"] = t.get("llm_output", "")
                        st.session_state["active_tab"] = "생각 발전"
                        st.success("씨드를 저장했습니다. ‘생각 발전’ 탭에서 이어가세요.")

    with colR:
        st.markdown("### ⭐ 즐겨찾기: 논문")
        fav_papers = store.get("favorites", {}).get("papers", [])
        if not fav_papers:
            st.markdown("<div class='small-muted'>즐겨찾기된 논문이 없습니다.</div>", unsafe_allow_html=True)
        else:
            for i, p in enumerate(reversed(fav_papers[-20:])):
                render_paper_card(p, i, allow_actions=False)
                if p.get("llm_summary"):
                    with st.expander("요약 보기", expanded=False):
                        st.markdown(p["llm_summary"])

        st.markdown("### 🗒️ 노트/플랜")
        notes = store.get("notes", [])
        if not notes:
            st.markdown("<div class='small-muted'>아직 저장된 플랜/노트가 없습니다.</div>", unsafe_allow_html=True)
        else:
            for n in reversed(notes[-20:]):
                title = f"🗒️ {n.get('type','note')} · {n.get('created_at','')}"
                with st.expander(title, expanded=False):
                    st.markdown("**Seed**")
                    st.code((n.get("seed", "") or "")[:2000])
                    st.markdown("**Output**")
                    st.markdown(n.get("output", ""))


# -----------------------------
# Footer: auto-save working copy into session or file when local enabled
# -----------------------------
# We don't auto-save on every widget change to avoid excessive IO, but do a light save
# when local storage is enabled and enough time has passed.
if "last_autosave_ts" not in st.session_state:
    st.session_state["last_autosave_ts"] = 0.0

autosave_interval = 20.0  # seconds
if time.time() - st.session_state["last_autosave_ts"] > autosave_interval:
    # keep session store always, and optionally write local
    save_store(False, store_path, st.session_state["store_obj"])  # session copy
    if use_local:
        save_store(True, store_path, st.session_state["store_obj"])
    st.session_state["last_autosave_ts"] = time.time()
