#!/usr/bin/env python3
# streamlit run thesaurus_browser.py
import json
import os
import urllib.parse
from typing import Dict, List, Tuple, Set

import networkx as nx
import pandas as pd
import streamlit as st

# Optional network preview
try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    HAVE_PYVIS = True
except Exception:
    HAVE_PYVIS = False

try:
    import altair as alt
    HAVE_ALTAIR = True
except Exception:
    HAVE_ALTAIR = False


DEFAULT_GRAPHML = "paper_graph.graphml"
LANG_CHOICES = ["en", "de", "fr", "label"]  # "label" = fallback if specific lang missing


# ----------------------------- helpers -----------------------------
@st.cache_data(show_spinner=True)
def load_graph(graphml_path: str) -> nx.MultiDiGraph:
    G = nx.read_graphml(graphml_path)
    # ensure MultiDiGraph
    if not isinstance(G, nx.MultiDiGraph):
        MG = nx.MultiDiGraph()
        MG.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            MG.add_edge(u, v, **d)
        G = MG
    return G


def node_kind(G: nx.MultiDiGraph, n: str) -> str:
    return str(G.nodes[n].get("kind", "")).lower()


def get_desc_label(data: Dict, lang_code: str) -> str:
    if lang_code and lang_code != "label":
        return str(data.get(f"label_{lang_code}", data.get("label", "")))
    return str(data.get("label", ""))


def list_nodes_by_kind(G: nx.MultiDiGraph, kind: str) -> List[str]:
    k = kind.lower()
    return [n for n, d in G.nodes(data=True) if str(d.get("kind", "")).lower() == k]


def outgoing_mentions(G: nx.MultiDiGraph, src: str) -> List[Tuple[str, dict]]:
    """All (target, edge_attrs) for MENTIONS edges from src → *."""
    out = []
    for _, v, d in G.out_edges(src, data=True):
        if str(d.get("type", "")).upper() == "MENTIONS":
            out.append((v, d))
    return out


def papers_via_descriptors(G: nx.MultiDiGraph, descriptor_ids: Set[str]) -> List[Tuple[str, dict]]:
    """Papers that mention any of the given descriptors; returns (paper_node, stats)."""
    papers = []
    for p in list_nodes_by_kind(G, "paper"):
        hits = []
        weight_sum = 0
        for _, v, d in G.out_edges(p, data=True):
            if str(d.get("type", "")).upper() != "MENTIONS":
                continue
            if v in descriptor_ids:
                w = d.get("weight", 0)
                try:
                    w = int(w)
                except Exception:
                    try:
                        w = float(w)
                    except Exception:
                        w = 0
                hits.append((v, d))
                weight_sum += w
        if hits:
            papers.append((p, {"weight_sum": weight_sum, "desc_count": len(hits), "details": hits}))
    return papers


def df_for_descriptors(G: nx.MultiDiGraph, desc_ids: List[str], lang_code: str) -> pd.DataFrame:
    rows = []
    for d_id in desc_ids:
        d = G.nodes[d_id]
        label = get_desc_label(d, lang_code)
        rows.append({"descriptor_id": d_id, "label": label})
    return pd.DataFrame(rows).sort_values(["label", "descriptor_id"]).reset_index(drop=True)


def df_for_papers(G: nx.MultiDiGraph, items: List[Tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    for pid, stats in items:
        n = G.nodes[pid]
        rows.append({
            "paper_node": pid,
            "title": n.get("title", n.get("label", "")),
            "journal": n.get("journal", ""),
            "date_published": n.get("date_published", ""),
            "descriptors_shared": stats["desc_count"],
            "weight_sum": stats["weight_sum"],
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["paper_node","title","journal","date_published","descriptors_shared","weight_sum"])
    return df.sort_values(["weight_sum", "descriptors_shared", "title"], ascending=[False, False, True]).reset_index(drop=True)


def build_preview_subgraph(G: nx.MultiDiGraph, file_node: str, selected_desc: List[str], top_papers: List[str]) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    H.add_node(file_node, **G.nodes[file_node])
    for d in selected_desc:
        H.add_node(d, **G.nodes[d])
        for _, v, ed in G.out_edges(file_node, data=True):
            if v == d and str(ed.get("type", "")).upper() == "MENTIONS":
                H.add_edge(file_node, d, **ed)
    for p in top_papers:
        H.add_node(p, **G.nodes[p])
        for _, v, ed in G.out_edges(p, data=True):
            if v in selected_desc and str(ed.get("type", "")).upper() == "MENTIONS":
                H.add_edge(p, v, **ed)
    return H


def show_pyvis(H: nx.MultiDiGraph, lang_code: str, height_px: int = 600):
    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False)
    for n, data in H.nodes(data=True):
        kind = str(data.get("kind", "")).lower()
        label = data.get("label", "")
        if kind == "descriptor":
            label = get_desc_label(data, lang_code) or label
        title = []
        for k, v in data.items():
            title.append(f"<b>{k}</b>: {str(v)[:200]}")
        tooltip = "<br>".join(title)
        color = "#6baed6" if kind == "file" else ("#31a354" if kind == "paper" else "#636363")
        shape = "box" if kind in {"file", "paper"} else "ellipse"
        net.add_node(n, label=label, title=tooltip, color=color, shape=shape)
    for u, v, d in H.edges(data=True, keys=False):
        et = d.get("type", "")
        lab = f"{et}" if d.get("weight") in (None, "", 0) else f"{et} ({d.get('weight')})"
        net.add_edge(u, v, title=json.dumps(d, ensure_ascii=False)[:500], label=lab, arrows="to")
    html = net.generate_html(notebook=False)
    components.html(html, height=height_px + 40, scrolling=True)


# --------------- NEW helpers for Article selection & timelines ---------------
def doi_or_link(n: Dict) -> str:
    """Return a clickable URL: DOI (preferred), else 'url' attr, else Google Scholar query."""
    doi = str(n.get("doi", "")).strip()
    url = str(n.get("url", "")).strip()
    if doi:
        low = doi.lower()
        if low.startswith("http://") or low.startswith("https://"):
            return doi
        # drop a leading 'doi:' if present
        doi = doi.replace("doi:", "").strip()
        return f"https://doi.org/{doi}"
    if url:
        return url
    title = str(n.get("title", n.get("label", "")))
    q = urllib.parse.quote(title)
    return f"https://scholar.google.com/scholar?q={q}"


def mentions_from_selected_papers(G: nx.MultiDiGraph, paper_ids: List[str], lang_code: str) -> pd.DataFrame:
    """Rows: paper_node, title, date (datetime), period, descriptor_id, descriptor_label, weight."""
    rows = []
    for pid in paper_ids:
        pn = G.nodes[pid]
        title = pn.get("title", pn.get("label", ""))
        journal = pn.get("journal", "")
        date_str = pn.get("date_published", "")
        date_dt = pd.to_datetime(date_str, errors="coerce")
        for _, v, d in G.out_edges(pid, data=True):
            if str(d.get("type", "")).upper() != "MENTIONS":
                continue
            if node_kind(G, v) != "descriptor":
                continue
            w = d.get("weight", 0)
            try:
                w = int(w)
            except Exception:
                try:
                    w = float(w)
                except Exception:
                    w = 0
            label = get_desc_label(G.nodes[v], lang_code)
            rows.append({
                "paper_node": pid,
                "title": title,
                "journal": journal,
                "date_published": date_str,
                "date_dt": date_dt,
                "descriptor_id": v,
                "descriptor_label": label,
                "weight": w,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[~df["date_dt"].isna()].copy()
    return df


def add_period(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return df
    gran = granularity.lower()
    if gran == "day":
        df["period"] = df["date_dt"].dt.to_period("D").dt.to_timestamp()
    elif gran == "year":
        df["period"] = df["date_dt"].dt.to_period("Y").dt.to_timestamp()
    else:  # month default
        df["period"] = df["date_dt"].dt.to_period("M").dt.to_timestamp()
    return df


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Thesaurus Graph Explorer", layout="wide")

st.title("Thesaurus Graph Explorer")

with st.sidebar:
    st.header("Load graph")
    graph_path = st.text_input("GraphML path", value=DEFAULT_GRAPHML, help="Path to your .graphml")
    lang_code = st.selectbox("Descriptor label language", LANG_CHOICES, index=0)
    load_btn = st.button("Load / Reload")

if not graph_path or (load_btn and not os.path.exists(graph_path)):
    st.info("Provide a valid GraphML path in the sidebar and click **Load / Reload**.")
    st.stop()

try:
    G = load_graph(graph_path)
except Exception as e:
    st.error(f"Failed to load graph: {e}")
    st.stop()

# Step 1: choose file nodes
file_nodes = list_nodes_by_kind(G, "file")
file_labels = {n: G.nodes[n].get("label", n) for n in file_nodes}
files_sorted = sorted(file_nodes, key=lambda n: file_labels[n].lower())

st.subheader("Step 1 — Choose which file nodes to include")
included_files = st.multiselect(
    "Included file nodes (searchable):",
    files_sorted,
    format_func=lambda n: f"{file_labels[n]}",
    help="Pick a subset to work with.",
)

if not included_files:
    st.info("Select at least one file node above.")
    st.stop()

# Step 2: active file
st.subheader("Step 2 — Pick one file to explore (acts like clicking its node)")
active_file = st.selectbox(
    "Active file",
    included_files,
    format_func=lambda n: f"{file_labels[n]}",
)

# Step 3: descriptors linked from the active file
desc_neighbors = []
for tgt, ed in outgoing_mentions(G, active_file):
    if node_kind(G, tgt) == "descriptor":
        desc_neighbors.append((tgt, ed))

if not desc_neighbors:
    st.warning("No descriptor links found from this file.")
    st.stop()

desc_rows = []
for d_id, ed in desc_neighbors:
    node = G.nodes[d_id]
    label = get_desc_label(node, lang_code)
    w = ed.get("weight", 0)
    try:
        w = int(w)
    except Exception:
        try:
            w = float(w)
        except Exception:
            w = 0
    terms_json = ed.get("terms_json", "")
    try:
        terms = json.loads(terms_json) if terms_json else {}
    except Exception:
        terms = {}
    top_terms = ", ".join(sorted(terms, key=terms.get, reverse=True)[:5])
    desc_rows.append({
        "descriptor_id": d_id,
        "label": label,
        "weight_from_file": w,
        "top_terms": top_terms
    })

df_desc = pd.DataFrame(desc_rows).sort_values(["weight_from_file", "label"], ascending=[False, True]).reset_index(drop=True)

st.subheader("Step 3 — Select descriptors (deselect to filter)")
with st.expander("Descriptors linked from file (sortable)", expanded=True):
    st.dataframe(df_desc, use_container_width=True, hide_index=True)

selected_labels = st.multiselect(
    "Active descriptors (for linking to papers):",
    df_desc["label"].tolist(),
    default=df_desc["label"].tolist(),
)
selected_desc_ids = set(df_desc[df_desc["label"].isin(selected_labels)]["descriptor_id"].tolist())
if not selected_desc_ids:
    st.warning("No descriptors selected — select at least one to see papers.")
    st.stop()

# Step 4: papers via selected descriptors
st.subheader("Step 4 — Papers connected via selected descriptors")
paper_stats = papers_via_descriptors(G, selected_desc_ids)
df_papers = df_for_papers(G, paper_stats)

st.dataframe(df_papers, use_container_width=True, hide_index=True)
st.download_button(
    "Download results (CSV)",
    df_papers.to_csv(index=False).encode("utf-8"),
    file_name="papers_via_descriptors.csv",
    mime="text/csv",
)

# Optional preview
st.subheader("Optional — Small subgraph preview")
if not HAVE_PYVIS:
    st.info("Install `pyvis` to enable preview: `pip install pyvis`.")
else:
    max_papers = st.slider("Max papers to show in preview", min_value=5, max_value=100, value=25, step=5)
    top_paper_ids = df_papers.head(max_papers)["paper_node"].tolist()
    subG = build_preview_subgraph(G, active_file, list(selected_desc_ids), top_paper_ids)
    show_pyvis(subG, lang_code, height_px=650)

# ----------------------------- NEW: Step 5 & 6 -----------------------------
st.subheader("Step 5 — Choose articles to analyze (timeline)")
if df_papers.empty:
    st.info("No connected papers found to analyze.")
    st.stop()

# Select papers (default: all in the table)
paper_id_list = df_papers["paper_node"].tolist()
paper_display = {
    pid: f"{G.nodes[pid].get('title','')[:80]} ({G.nodes[pid].get('journal','')}, {G.nodes[pid].get('date_published','')})"
    for pid in paper_id_list
}
selected_papers = st.multiselect(
    "Selected articles",
    paper_id_list,
    default=paper_id_list,
    format_func=lambda pid: paper_display.get(pid, pid),
)

if not selected_papers:
    st.warning("Select at least one article above to compute timelines.")
    st.stop()

# Build mentions dataframe from the selected papers (ALL descriptors they mention)
mentions_df = mentions_from_selected_papers(G, selected_papers, lang_code)

# Controls for timeline
st.subheader("Step 6 — Descriptor mentions over time")
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    granularity = st.selectbox("Time granularity", ["Month", "Year", "Day"], index=0)
with col_b:
    topN = st.slider("Top N descriptors (lines)", min_value=5, max_value=30, value=10, step=1)
with col_c:
    agg_mode = st.selectbox("Aggregate", ["Sum of weights", "Count of edges"], index=0)

mentions_df = add_period(mentions_df, granularity)

if mentions_df.empty:
    st.info("No dated articles among your selection; cannot draw timeline.")
else:
    # Overall timeline
    if agg_mode == "Sum of weights":
        overall = mentions_df.groupby("period", as_index=False)["weight"].sum()
        y_col = "weight"
    else:
        overall = mentions_df.groupby("period", as_index=False)["descriptor_id"].count().rename(columns={"descriptor_id":"count"})
        y_col = "count"

    st.markdown("**Overall descriptor mentions**")
    if HAVE_ALTAIR:
        base = alt.Chart(overall).mark_bar().encode(
            x=alt.X("period:T", title="Period"),
            y=alt.Y(f"{y_col}:Q", title=y_col.replace("_", " ").title()),
            tooltip=["period:T", f"{y_col}:Q"]
        )
        st.altair_chart(base, use_container_width=True)
    else:
        st.line_chart(overall.set_index("period")[y_col])

    # Top-N descriptor lines
    totals = mentions_df.groupby("descriptor_label", as_index=False)["weight"].sum()
    top_labels = totals.sort_values("weight", ascending=False).head(topN)["descriptor_label"].tolist()
    top_df = mentions_df[mentions_df["descriptor_label"].isin(top_labels)].copy()

    if not top_df.empty:
        st.markdown("**Top descriptors over time**")
        if agg_mode == "Sum of weights":
            top_series = top_df.groupby(["period", "descriptor_label"], as_index=False)["weight"].sum()
            val_field = "weight"
        else:
            top_series = top_df.groupby(["period", "descriptor_label"], as_index=False)["descriptor_id"].count().rename(columns={"descriptor_id":"count"})
            val_field = "count"

        if HAVE_ALTAIR:
            chart = alt.Chart(top_series).mark_line(point=True).encode(
                x=alt.X("period:T", title="Period"),
                y=alt.Y(f"{val_field}:Q", title=val_field.replace("_", " ").title()),
                color=alt.Color("descriptor_label:N", legend=alt.Legend(title="Descriptor")),
                tooltip=["descriptor_label:N", "period:T", f"{val_field}:Q"]
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            pivot = top_series.pivot(index="period", columns="descriptor_label", values=val_field).fillna(0)
            st.line_chart(pivot)

    # Download CSVs
    st.download_button(
        "Download timeline (overall) CSV",
        overall.to_csv(index=False).encode("utf-8"),
        file_name="descriptor_mentions_overall.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download timeline (top descriptors) CSV",
        (top_series if not top_df.empty else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        file_name="descriptor_mentions_top.csv",
        mime="text/csv",
    )

# Selected papers table with links (DOI/URL/Scholar)
st.subheader("Selected articles (with links)")
paper_rows = []
for pid in selected_papers:
    n = G.nodes[pid]
    paper_rows.append({
        "title": n.get("title", n.get("label", "")),
        "journal": n.get("journal", ""),
        "date_published": n.get("date_published", ""),
        "link": doi_or_link(n),
    })
df_sel = pd.DataFrame(paper_rows)

# Use LinkColumn so links are clickable in the table
st.dataframe(
    df_sel,
    use_container_width=True,
    hide_index=True,
    column_config={
        "link": st.column_config.LinkColumn("Link")
    },
)
st.download_button(
    "Download selected articles (CSV)",
    df_sel.to_csv(index=False).encode("utf-8"),
    file_name="selected_articles.csv",
    mime="text/csv",
)

st.caption("Tip: Change the language dropdown to switch descriptor labels (en/de/fr).")
