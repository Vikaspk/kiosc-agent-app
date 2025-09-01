import re
import streamlit as st
import pandas as pd
from typing import Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="KIOSC Dataset Agent", layout="wide")
# ---------- Load dataset ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_excel("KIOSC_Cleaned_Merged.xlsx", sheet_name="Sheet1")

    # Convert any column with 'date' in its name
    for col in df.columns:
        if "date" in col.lower():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], unit="d", origin="1899-12-30", errors="coerce")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")

            # Format to readable DD-MM-YYYY
            df[col] = df[col].dt.strftime("%d-%m-%Y")

    return df


df = load_data()

# ---------- QA + Embedding models ----------
@st.cache_resource
def get_qa():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

qa = get_qa()
embedder = get_embedder()

# ---------- Column mappings ----------
KEYWORD_TO_COL = {
    "enjoy": "How much did you enjoy the sessions today?",
    "enjoyed": "How much did you enjoy the sessions today?",
    "enjoyment": "How much did you enjoy the sessions today?",
    "learn today": "How much do you think you have learnt today?",
    "learnt today": "How much do you think you have learnt today?",
    "gain knowledge": "How much do you think you have learnt today?"
}

def pick_metric_from_query(q: str) -> Optional[str]:
    q_low = q.lower()

    # enjoyment/learning mappings
    for kw, col in KEYWORD_TO_COL.items():
        if kw in q_low and col in df.columns:
            return col

    # date-related queries
    if any(word in q_low for word in ["date", "day", "when","session"]):
        if "response_date" in df.columns:
            return "response_date"

    # default fallback
    if "overall_experience" in df.columns:
        return "overall_experience"

    return None


# ---------- Dimension helpers ----------
def pick_dim_from_query(q: str) -> Optional[str]:
    q_low = q.lower()
    if "program" in q_low and "Program" in df.columns:
        return "Program"
    if "school" in q_low and "School" in df.columns:
        return "School"
    if "year band" in q_low and "year_band" in df.columns:
        return "year_band"
    if "year_num" in q_low and "year_num" in df.columns:
        return "year_num"
    if "grade" in q_low and "Grade Level" in df.columns:
        return "Grade Level"
    if "gender" in q_low and "Gender" in df.columns:
        return "Gender"
    if "delivered" in q_low and "Delivered" in df.columns:
        return "Delivered"
    if "attendance" in q_low and "Attendance" in df.columns:
        return "Attendance"
    if "term" in q_low and "response_term" in df.columns:
        return "response_term"
    if "year " in q_low and "response_year" in df.columns:
        return "response_year"
    return None

def parse_topn(q: str) -> Optional[int]:
    m = re.search(r"top\s+(\d+)", q.lower())
    return int(m.group(1)) if m else None

def parse_term(q: str) -> Optional[int]:
    m = re.search(r"term\s*(\d+)", q.lower())
    return int(m.group(1)) if m else None

def parse_year(q: str) -> Optional[int]:
    m = re.search(r"(?:year\s*[:= ]|\by\s*)(\d{4})", q.lower())
    return int(m.group(1)) if m else None

def filtered_df_from_query(q: str) -> pd.DataFrame:
    d = df.copy()
    t = parse_term(q)
    if t is not None and "response_term" in d.columns:
        d = d[d["response_term"] == t]
    y = parse_year(q)
    if y is not None and "response_year" in d.columns:
        d = d[d["response_year"] == y]
    return d

# ---------- Semantic retrieval ----------
def retrieve_relevant_rows(q: str, k: int = 40) -> pd.DataFrame:
    metric = pick_metric_from_query(q)
    cols = [metric] if metric else []
    if "School" in df.columns: cols.append("School")
    if "Program" in df.columns: cols.append("Program")

    subdf = filtered_df_from_query(q)
    if subdf.empty:
        return df.head(k)

    # Represent rows as text
    row_texts = subdf[cols].astype(str).agg(" | ".join, axis=1).tolist()
    row_embeddings = np.array(embedder.encode(row_texts, convert_to_tensor=False))

    q_emb = np.array(embedder.encode([q], convert_to_tensor=False))
    sims = cosine_similarity(q_emb, row_embeddings)[0]

    top_idx = np.argsort(sims)[::-1][:k]
    return subdf.iloc[top_idx]

# ---------- Dataset summary ----------
def make_dataset_summary() -> str:
    bullets = []
    bullets.append("**Available columns include:**")
    bullets.append(", ".join(df.columns))

    preview_html = df.head(3).to_html(index=False)
    return "📘 This dataset comes from KIOSC survey exports.\n\n" + "\n".join(bullets) + "\n\n**Preview (first 3 rows):**\n\n" + preview_html

# ---------- Visuals ----------
def make_grouped_visual(q: str):
    metric = pick_metric_from_query(q)
    if metric is None:
        st.warning("No metric column found for this query.")
        return

    d = filtered_df_from_query(q)
    dim = pick_dim_from_query(q)
    if dim is None:
        dim = d.columns[0]

    if metric == "response_date":
        # Show all session dates for each school/program
        tbl = (
            d.groupby(dim, dropna=False)[metric]
            .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
            .to_frame(name="session_dates")
        )

        st.subheader(f"Session dates by {dim}")
        st.dataframe(tbl)
    else:
        # Normal averaging flow
        tbl = (
            d.groupby(dim, dropna=False)[metric]
            .mean()
            .sort_values(ascending=False)
            .to_frame(name=f"avg_{metric}")
        )

        topn = parse_topn(q)
        if topn:
            tbl = tbl.head(topn)

        st.subheader(f"Average {metric} by {dim}")
        st.dataframe(tbl)
        st.bar_chart(tbl)


# ---------- QA ----------
def answer_with_llm(q: str):
    metric = pick_metric_from_query(q)
    cols_for_context = [metric] if metric else []
    if "School" in df.columns: cols_for_context.append("School")
    if "Program" in df.columns: cols_for_context.append("Program")
    if "response_date" in df.columns: cols_for_context.append("response_date")

    # Use semantic retrieval instead of naive head()
    context_df = retrieve_relevant_rows(q, k=40)

    # Ensure all date columns are formatted as strings
    for col in context_df.columns:
        if "date" in col.lower():
            context_df[col] = pd.to_datetime(context_df[col], errors="coerce").dt.strftime("%d-%m-%Y")

    context_text = context_df[cols_for_context].astype(str).to_string(index=False)

    if not context_text.strip():
        context_text = "No data available to answer this question."

    res = qa(question=q, context=context_text)

    if isinstance(res, dict) and "answer" in res:
        st.write("🤖 **Answer:**", res["answer"])
    else:
        st.write("🤖 **Answer:** (no answer)")

    with st.expander("Show context sample used"):
        st.code(context_text)




st.title(" KIOSC Dataset Agent")

mode = st.radio(
    "Choose a mode:",
    ["Ask", "Make a visual/table"],
    horizontal=True
)

if mode == "Explain dataset":
    st.markdown(make_dataset_summary())
else:
    q = st.text_input("Type your question:")
    if q:
        q_low = q.lower()

        if mode == "Make a visual/table" or any(k in q_low for k in ["chart", "visual", "graph", "bar", "table", "rank", "top "]):
            make_grouped_visual(q)

        elif "explain" in q_low or "columns" in q_low:
            st.markdown(make_dataset_summary())

        else:
            answer_with_llm(q)# 
