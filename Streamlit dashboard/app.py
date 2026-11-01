"""
Link Inspector v2 - Phishing & Malicious URL Classifier (Streamlit)

Run:  streamlit run app.py
Folder must contain:  features.py, phishing_model.pkl   (from train.py)
Recommended:          .streamlit/config.toml            (dark theme, see notes)
Optional:             metrics.json                      (scores shown in sidebar)
"""
import html
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from features import FEATURE_NAMES, extract_features, normalize

BASE = Path(__file__).parent
MODEL_PATH = BASE / "phishing_model.pkl"
METRICS_PATH = BASE / "metrics.json"

st.set_page_config(page_title="Link Inspector", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# STYLES
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@property --n { syntax:'<integer>'; inherits:false; initial-value:0; }

:root{
  --bg:#070B1A; --panel:rgba(255,255,255,.045); --panel-2:rgba(255,255,255,.08);
  --line:rgba(255,255,255,.10); --ink:#E8EEFC; --muted:#8E9BBD;
  --blue:#5B8CFF; --violet:#9B6BFF;
  --safe:#25D985; --safe-d:rgba(37,217,133,.14);
  --risk:#FF3B5C; --risk-d:rgba(255,59,92,.14);
  --warn:#FFB84D;
}

/* ---------- canvas ---------- */
html, body, .stApp { color-scheme: dark; }
.stApp { background:var(--bg); color:var(--ink); font-family:'Manrope',system-ui,sans-serif; }
.stApp p, .stApp li, .stApp label { font-family:'Manrope',system-ui,sans-serif; }
[data-testid="stIconMaterial"]{ font-family:'Material Symbols Rounded','Material Icons' !important; }
.stApp::before{
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(38% 34% at 18% 14%, rgba(91,140,255,.30), transparent 62%),
    radial-gradient(34% 30% at 86% 8%,  rgba(155,107,255,.24), transparent 62%),
    radial-gradient(40% 38% at 72% 92%, rgba(37,217,133,.10), transparent 62%);
  animation:drift 26s ease-in-out infinite alternate;
}
.stApp::after{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
  background-size:46px 46px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 0%,transparent 72%);
          mask-image:radial-gradient(ellipse at 50% 0%,#000 0%,transparent 72%);
}
@keyframes drift{ from{transform:translate3d(-2%,-1%,0) scale(1)} to{transform:translate3d(3%,2%,0) scale(1.1)} }
[data-testid="stAppViewContainer"]{ position:relative; z-index:1; background:transparent; }
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ max-width:1180px; padding-top:2rem; padding-bottom:4rem; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{ background:rgba(8,12,28,.72) !important; backdrop-filter:blur(16px); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div{ background:transparent; }
section[data-testid="stSidebar"] *{ color:#D5DEF5; }
.side-brand{ display:flex; align-items:center; gap:.7rem; }
.side-brand .name{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.2rem; color:#fff; }
.side-h{ font-family:'Sora',sans-serif; font-weight:600; font-size:.9rem; color:#fff; margin:1.3rem 0 .5rem; }
.kv{ display:flex; justify-content:space-between; font-size:.84rem; padding:.34rem 0; border-bottom:1px solid rgba(255,255,255,.07); }
.kv span:first-child{ color:var(--muted); }
.kv span:last-child{ color:#fff; font-weight:600; font-family:'JetBrains Mono',monospace; }
.side-note{ font-size:.79rem; line-height:1.55; color:var(--muted); margin-top:.5rem; }

/* ---------- hero ---------- */
.hero{ display:flex; align-items:center; justify-content:space-between; gap:2rem; margin-bottom:1.1rem; }
.hero h1{
  font-family:'Sora',sans-serif; font-weight:700; font-size:3rem; letter-spacing:-.03em; line-height:1.08; margin:0 0 .7rem;
  background:linear-gradient(90deg,#FFFFFF 0%,#9DB8FF 40%,#C9B6FF 60%,#FFFFFF 100%); background-size:220% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent; -webkit-text-fill-color:transparent;
  animation:shine 9s linear infinite;
}
@keyframes shine{ to{ background-position:-220% center; } }
.hero p{ font-size:1.05rem; color:var(--muted); max-width:58ch; line-height:1.6; margin:0; }
.hero-art{ position:relative; width:150px; height:150px; flex:none; }
.hero-art .orb{ position:absolute; inset:10px; border-radius:50%; background:radial-gradient(circle,rgba(91,140,255,.55),transparent 68%); animation:orb 3.6s ease-in-out infinite; }
.hero-art svg{ position:relative; }
.hero-art .orbit{ transform-origin:60px 60px; animation:spin 14s linear infinite; }
@keyframes orb{ 0%,100%{transform:scale(.85);opacity:.7} 50%{transform:scale(1.15);opacity:1} }
@keyframes spin{ to{ transform:rotate(360deg); } }
.facts{ display:flex; flex-wrap:wrap; gap:.6rem; margin:0 0 1.4rem; }
.fact{ display:flex; align-items:center; gap:.55rem; padding:.45rem .9rem; border:1px solid var(--line); border-radius:999px; background:var(--panel); font-size:.84rem; color:var(--muted); }
.fact b{ color:#fff; font-family:'JetBrains Mono',monospace; font-weight:500; }
.fact i{ width:7px; height:7px; border-radius:50%; background:var(--safe); box-shadow:0 0 10px var(--safe); animation:blink 2.2s ease-in-out infinite; }
@keyframes blink{ 50%{ opacity:.35; } }

/* ---------- inputs (new + legacy Streamlit DOM) ---------- */
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], div[data-baseweb="input"], div[data-baseweb="textarea"]{
  background:rgba(255,255,255,.055) !important; border:1.5px solid var(--line) !important; border-radius:16px !important;
  box-shadow:none; transition:border-color .2s, box-shadow .2s;
}
div[data-baseweb="base-input"]{ background:transparent !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within, div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within{
  border-color:var(--blue) !important; box-shadow:0 0 0 4px rgba(91,140,255,.18), 0 0 30px rgba(91,140,255,.18) !important;
}
.stTextInput input, .stTextArea textarea{ font-family:'JetBrains Mono',monospace !important; font-size:1rem !important; color:#fff !important; -webkit-text-fill-color:#fff !important; caret-color:var(--blue); background:transparent !important; }
.stTextInput input{ padding:.95rem 1.1rem !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder{ color:#6C7899 !important; -webkit-text-fill-color:#6C7899 !important; }
[data-testid="stForm"]{ border:none; padding:0; background:transparent; }
[data-testid="stFileUploaderDropzone"]{ background:rgba(255,255,255,.04) !important; border:1.5px dashed rgba(255,255,255,.22) !important; border-radius:16px !important; }
[data-testid="stFileUploaderDropzone"] *{ color:var(--muted) !important; }

button[kind="primary"], button[kind="primaryFormSubmit"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:linear-gradient(135deg,#4F83FF,#9B6BFF) !important; color:#fff !important; border:none !important; border-radius:16px !important;
  font-weight:700 !important; height:3.35rem; letter-spacing:.01em; box-shadow:0 8px 28px rgba(91,140,255,.35);
  transition:transform .15s ease, box-shadow .2s ease, filter .2s ease;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover, button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  transform:translateY(-2px); box-shadow:0 14px 36px rgba(122,120,255,.5); filter:brightness(1.08);
}
button[kind="primary"]:active, button[kind="primaryFormSubmit"]:active{ transform:translateY(0); }
button[kind="primary"] *, button[kind="primaryFormSubmit"] *, button[data-testid="stBaseButton-primary"] *, button[data-testid="stBaseButton-primaryFormSubmit"] *{ color:#fff !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"]{
  background:var(--panel) !important; color:#CBD6F2 !important; border:1px solid var(--line) !important; border-radius:999px !important;
  font-size:.83rem !important; font-weight:500 !important; min-height:2.2rem; transition:all .18s ease;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover{ border-color:var(--blue) !important; color:#fff !important; background:rgba(91,140,255,.14) !important; transform:translateY(-1px); }
button[kind="secondary"] *, button[data-testid="stBaseButton-secondary"] *{ color:inherit !important; }
.helper{ font-size:.82rem; color:var(--muted); margin:.55rem .2rem 0; }
.try-label{ font-size:.84rem; color:var(--muted); margin:1rem 0 .4rem; }

/* ---------- tabs (new + legacy Streamlit DOM) ---------- */
[role="tablist"]{ gap:.3rem !important; background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:999px !important; padding:.3rem !important; width:fit-content !important; margin-bottom:.8rem; box-shadow:none !important; }
[role="tablist"] [role="tab"]{ border-radius:999px !important; padding:.5rem 1.3rem !important; height:auto !important; background:transparent !important; border:none !important; transition:all .2s; }
[role="tablist"] [role="tab"] p, [role="tablist"] [role="tab"] span{ color:#8E9BBD !important; font-weight:600 !important; margin:0; }
[role="tablist"] [role="tab"]:hover p{ color:#fff !important; }
[role="tablist"] [role="tab"][aria-selected="true"]{ background:linear-gradient(135deg,rgba(91,140,255,.38),rgba(155,107,255,.38)) !important; box-shadow:0 0 22px rgba(91,140,255,.25); }
[role="tablist"] [role="tab"][aria-selected="true"] p, [role="tablist"] [role="tab"][aria-selected="true"] span{ color:#fff !important; }
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none !important; }

/* ---------- scanning sequence ---------- */
.scan{ display:flex; align-items:center; gap:2rem; padding:1.6rem 2rem; margin-top:1.2rem; border:1px solid rgba(91,140,255,.35); border-radius:24px; background:linear-gradient(135deg,rgba(91,140,255,.12),rgba(155,107,255,.06)); }
.radar{ position:relative; width:130px; height:130px; flex:none; border-radius:50%; border:1px solid rgba(91,140,255,.5); overflow:hidden;
  background:repeating-radial-gradient(circle,transparent 0 20px,rgba(91,140,255,.22) 21px 22px), radial-gradient(circle,rgba(91,140,255,.12),transparent 70%); }
.radar .sweep{ position:absolute; inset:0; border-radius:50%; background:conic-gradient(from 0deg,rgba(91,140,255,0) 0deg,rgba(91,140,255,.65) 70deg,rgba(91,140,255,0) 71deg); animation:spin 1.3s linear infinite; }
.radar .blip{ position:absolute; width:8px; height:8px; border-radius:50%; background:#fff; box-shadow:0 0 12px #fff; top:34%; left:62%; animation:blink 1s ease-in-out infinite; }
.scan-body{ flex:1; min-width:0; }
.scan-url{ font-family:'JetBrains Mono',monospace; font-size:.9rem; color:#fff; padding:.5rem .8rem; background:rgba(0,0,0,.28); border-radius:10px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-bottom:.9rem; }
.stp{ display:flex; align-items:center; gap:.7rem; padding:.28rem 0; font-size:.95rem; color:#5F6C8F; transition:color .2s; }
.stp .ic{ width:20px; height:20px; border-radius:50%; border:1.5px solid #3A4568; display:grid; place-items:center; flex:none; }
.stp.done{ color:#C9D4F0; } .stp.done .ic{ background:var(--safe); border-color:var(--safe); }
.stp.done .ic::after{ content:""; width:5px; height:9px; border:solid #04140C; border-width:0 2px 2px 0; transform:rotate(45deg) translate(-1px,-1px); }
.stp.act{ color:#fff; font-weight:600; } .stp.act .ic{ border-color:var(--blue); border-top-color:transparent; animation:spin .7s linear infinite; }

/* ---------- verdict ---------- */
.verdict{ position:relative; display:flex; align-items:center; gap:2.2rem; padding:1.8rem 2.2rem; margin-top:1.2rem; border-radius:26px; border:1.5px solid; overflow:hidden; }
.verdict::after{ content:""; position:absolute; inset:0; background:linear-gradient(110deg,transparent 35%,rgba(255,255,255,.07) 50%,transparent 65%); transform:translateX(-120%); animation:sheen 4.5s ease-in-out 1.4s infinite; pointer-events:none; }
@keyframes sheen{ 0%{transform:translateX(-120%)} 45%,100%{transform:translateX(120%)} }
.verdict.risk{ background:linear-gradient(135deg,rgba(255,59,92,.22),rgba(255,59,92,.05)); border-color:rgba(255,59,92,.55); animation:pulse-risk 2.4s ease-out infinite; }
.verdict.safe{ background:linear-gradient(135deg,rgba(37,217,133,.18),rgba(37,217,133,.04)); border-color:rgba(37,217,133,.5); animation:glow-safe 3.6s ease-in-out infinite alternate; }
@keyframes pulse-risk{ 0%{box-shadow:0 0 0 0 rgba(255,59,92,.38),0 0 50px rgba(255,59,92,.15)} 70%,100%{box-shadow:0 0 0 20px rgba(255,59,92,0),0 0 50px rgba(255,59,92,.15)} }
@keyframes glow-safe{ from{box-shadow:0 0 28px rgba(37,217,133,.10)} to{box-shadow:0 0 56px rgba(37,217,133,.26)} }
.ringbox{ position:relative; width:190px; height:190px; flex:none; }
.ringbox, .ringbox svg{ display:block; overflow:visible; }
.arc{ filter:drop-shadow(0 0 7px currentColor); }
.ring-num{ position:absolute; inset:0; display:grid; place-items:center; text-align:center; }
.ring-num .big{ font-family:'Sora',sans-serif; font-weight:700; font-size:2.9rem; line-height:1; letter-spacing:-.04em; color:#fff; }
.ring-num .big small{ font-size:1.25rem; font-weight:600; margin-left:2px; color:var(--muted); letter-spacing:0; }
.ring-num .cap{ font-size:.72rem; color:var(--muted); margin-top:.35rem; line-height:1.3; }
.cnt{ counter-reset:c var(--n); }
.cnt::after{ content:counter(c); }
.v-main{ flex:1; min-width:0; position:relative; z-index:1; }
.v-title{ display:flex; align-items:center; gap:.7rem; font-family:'Sora',sans-serif; font-weight:700; font-size:2.1rem; letter-spacing:-.02em; line-height:1.15; color:#fff; }
.v-title svg{ flex:none; }
.verdict.risk .v-title{ color:#FFD9DF; } .verdict.safe .v-title{ color:#D2FBE8; }
.v-sub{ font-size:1.02rem; color:#B8C4E2; margin-top:.6rem; line-height:1.55; max-width:56ch; }
.chips{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem; }
.chip{ padding:.32rem .8rem; border-radius:999px; font-size:.8rem; font-weight:600; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); color:#DCE5FA; }
.chip b{ font-family:'JetBrains Mono',monospace; font-weight:500; color:#fff; }
.defang-label{ font-size:.84rem; color:var(--muted); margin:.9rem 0 .25rem; }
[data-testid="stCode"] pre, [data-testid="stCode"] code{ background:rgba(0,0,0,.35) !important; color:#FFB3C0 !important; font-family:'JetBrains Mono',monospace !important; }
[data-testid="stCode"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }

/* ---------- cards ---------- */
.card{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:1.4rem 1.6rem; margin-top:1.2rem; backdrop-filter:blur(14px); transition:border-color .25s, transform .25s; }
.card:hover{ border-color:rgba(255,255,255,.2); }
.card h3{ font-family:'Sora',sans-serif; font-weight:600; font-size:1.12rem; margin:0 0 .3rem; color:#fff; letter-spacing:-.01em; }
.card .hint{ font-size:.88rem; color:var(--muted); margin:0 0 1.2rem; line-height:1.55; }

/* ---------- anatomy ---------- */
.anat{ display:flex; flex-wrap:wrap; align-items:flex-start; gap:.2rem; font-family:'JetBrains Mono',monospace; font-size:1.02rem; }
.seg{ display:flex; flex-direction:column; align-items:flex-start; }
.seg .t{ padding:.4rem .55rem; border-radius:9px; white-space:pre; }
.seg .c{ font-family:'Manrope',sans-serif; font-size:.72rem; color:var(--muted); margin:.35rem 0 0 .2rem; }
.k-scheme .t,.k-port .t,.k-fragment .t{ color:#6C7899; }
.k-subdomain .t{ color:#8FB4FF; background:rgba(91,140,255,.12); }
.k-domain .t{ color:#fff; background:rgba(91,140,255,.26); font-weight:500; box-shadow:inset 0 -3px 0 var(--blue), 0 0 22px rgba(91,140,255,.35); }
.k-suffix .t{ color:#B9CCFF; background:rgba(91,140,255,.12); }
.k-path .t,.k-query .t{ color:#9AA7C7; background:rgba(255,255,255,.05); }
.k-userinfo .t,.k-ip .t{ color:#FFD0D8; background:rgba(255,59,92,.2); font-weight:500; box-shadow:inset 0 -3px 0 var(--risk), 0 0 22px rgba(255,59,92,.35); }

/* ---------- indicators ---------- */
.ind-grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:.75rem; }
.ind{ border:1px solid var(--line); border-radius:14px; padding:.9rem 1rem; background:rgba(255,255,255,.03); transition:border-color .2s, background .2s; }
.ind:hover{ background:rgba(255,255,255,.06); border-color:rgba(255,255,255,.2); }
.ind .top{ display:flex; align-items:center; gap:.5rem; font-weight:600; font-size:.9rem; color:#fff; }
.ind .val{ margin-left:auto; font-family:'JetBrains Mono',monospace; font-weight:500; font-size:.84rem; color:#fff; }
.ind .bar{ height:5px; border-radius:999px; background:rgba(255,255,255,.09); margin:.65rem 0 .5rem; overflow:hidden; }
.ind .bar i{ display:block; height:100%; border-radius:999px; }
.ind .why{ font-size:.79rem; color:var(--muted); line-height:1.45; }
.dot{ width:9px; height:9px; border-radius:50%; flex:none; display:inline-block; }
.dot.ok{ background:var(--safe); box-shadow:0 0 8px var(--safe); } .dot.warn{ background:var(--warn); box-shadow:0 0 8px var(--warn); } .dot.risk{ background:var(--risk); box-shadow:0 0 8px var(--risk); }
.bar i.ok{ background:var(--safe); } .bar i.warn{ background:var(--warn); } .bar i.risk{ background:var(--risk); }
.key{ display:flex; gap:1.2rem; font-size:.8rem; color:var(--muted); margin-top:1rem; }
.key span{ display:flex; align-items:center; gap:.45rem; }

/* ---------- forest ---------- */
.forest{ display:grid; grid-template-columns:repeat(auto-fill,13px); gap:6px; justify-content:center; padding:.3rem 0 .2rem; }
.forest i{ width:13px; height:13px; border-radius:50%; opacity:var(--o,1); }
.forest i.r{ background:var(--risk); box-shadow:0 0 8px rgba(255,59,92,.7); } .forest i.s{ background:var(--safe); box-shadow:0 0 8px rgba(37,217,133,.6); }
.fcount{ display:flex; justify-content:space-between; font-size:.82rem; color:var(--muted); margin-top:.9rem; }
.fcount b{ font-family:'JetBrains Mono',monospace; font-weight:500; }
.fcount .r{ color:#FF8CA0; } .fcount .s{ color:#7BEBB5; }

/* ---------- to-do ---------- */
.todo{ list-style:none; margin:0; padding:0; }
.todo li{ display:flex; gap:.8rem; padding:.7rem 0; border-bottom:1px solid rgba(255,255,255,.07); font-size:.92rem; line-height:1.5; color:#CBD6F2; }
.todo li:last-child{ border-bottom:none; }
.todo .mk{ flex:none; width:22px; height:22px; border-radius:7px; display:grid; place-items:center; font-size:.8rem; font-weight:700; margin-top:1px; }
.todo.risk .mk{ background:var(--risk-d); color:#FF8CA0; } .todo.safe .mk{ background:var(--safe-d); color:#7BEBB5; }

/* ---------- table / pills / tiles ---------- */
.tbl{ width:100%; border-collapse:collapse; font-size:.9rem; }
.tbl th{ text-align:left; font-weight:600; color:var(--muted); font-size:.78rem; padding:.5rem .7rem; border-bottom:1px solid var(--line); }
.tbl td{ padding:.65rem .7rem; border-bottom:1px solid rgba(255,255,255,.06); vertical-align:middle; color:#CBD6F2; }
.tbl tr:hover td{ background:rgba(255,255,255,.035); }
.tbl td.u{ font-family:'JetBrains Mono',monospace; font-size:.82rem; max-width:460px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#fff; }
.pill{ display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.76rem; font-weight:700; }
.pill.risk{ background:var(--risk-d); color:#FF8CA0; border:1px solid rgba(255,59,92,.4); } .pill.safe{ background:var(--safe-d); color:#7BEBB5; border:1px solid rgba(37,217,133,.4); }
.pbar{ display:inline-block; vertical-align:middle; width:90px; height:6px; border-radius:999px; background:rgba(255,255,255,.1); margin-right:.55rem; overflow:hidden; }
.pbar i{ display:block; height:100%; border-radius:999px; }
.stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.9rem; margin:1.2rem 0; }
.stat{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.3rem; backdrop-filter:blur(12px); }
.stat .n{ font-family:'Sora',sans-serif; font-weight:700; font-size:2.1rem; letter-spacing:-.03em; color:#fff; }
.stat .l{ font-size:.82rem; color:var(--muted); margin-top:.15rem; }
.stat.r .n{ color:var(--risk); text-shadow:0 0 24px rgba(255,59,92,.5); } .stat.s .n{ color:var(--safe); text-shadow:0 0 24px rgba(37,217,133,.4); }

/* ---------- about / empty ---------- */
.steps{ counter-reset:s; list-style:none; padding:0; margin:0; }
.steps li{ counter-increment:s; display:flex; gap:1rem; padding:.8rem 0; border-bottom:1px solid rgba(255,255,255,.07); line-height:1.55; color:#CBD6F2; font-size:.93rem; }
.steps li:last-child{ border-bottom:none; }
.steps li::before{ content:counter(s); flex:none; width:1.8rem; height:1.8rem; border-radius:50%; background:linear-gradient(135deg,rgba(91,140,255,.4),rgba(155,107,255,.4)); color:#fff; font-weight:700; font-size:.85rem; display:grid; place-items:center; }
.steps b{ color:#fff; }
.note{ background:rgba(255,184,77,.10); border:1px solid rgba(255,184,77,.35); border-radius:14px; padding:1rem 1.15rem; font-size:.9rem; color:#FFE2B3; line-height:1.6; }
.empty{ display:flex; align-items:center; gap:1.8rem; padding:2rem 2.2rem; margin-top:1.3rem; border:1.5px dashed rgba(255,255,255,.18); border-radius:24px; background:rgba(255,255,255,.025); }
.empty .radar{ width:110px; height:110px; opacity:.85; }
.empty b{ display:block; font-family:'Sora',sans-serif; font-weight:600; font-size:1.2rem; color:#fff; margin-bottom:.35rem; }
.empty span{ color:var(--muted); line-height:1.6; font-size:.95rem; }
.foot{ text-align:center; color:#5F6C8F; font-size:.8rem; margin-top:3rem; }

@media (max-width:760px){
  .hero{ flex-direction:column-reverse; align-items:flex-start; } .hero h1{ font-size:2.1rem; } .hero-art{ width:100px; height:100px; }
  .verdict, .scan, .empty{ flex-direction:column; align-items:flex-start; } .v-title{ font-size:1.6rem; }
}
@media (prefers-reduced-motion:reduce){ *, *::before, *::after{ animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def block(s: str) -> str:
    """Strip indentation and blank lines so Markdown never treats the HTML as code."""
    return "\n".join(l.strip() for l in s.splitlines() if l.strip())


def H(s: str) -> None:
    st.markdown(block(s), unsafe_allow_html=True)


# =============================================================================
# MODEL
# =============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except Exception:
            return None
    return None


if not MODEL_PATH.exists():
    st.error("**Model file not found.** Run `python train.py` first so `phishing_model.pkl` "
             "is created in this folder, then reload this page.")
    st.stop()

model = load_model()
POS = list(model.classes_).index(1)          # predict_proba column for "malicious"
N_TREES = len(getattr(model, "estimators_", [])) or getattr(model, "n_estimators", 0)


def tree_votes(X: pd.DataFrame):
    """Each tree's malicious probability, highest first (for the dot-matrix view)."""
    try:
        arr = X.to_numpy(dtype="float32")
        v = sorted((float(t.predict_proba(arr)[0][POS]) for t in model.estimators_), reverse=True)
        step = max(1, math.ceil(len(v) / 400))
        return v[::step]
    except Exception:
        return []


# =============================================================================
# URL HELPERS
# =============================================================================
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne"}
URL_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)?([^/?#]*)([^?#]*)(\?[^#]*)?(#.*)?$")


def clip(text: str, n: int = 56) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def defang(url: str) -> str:
    """Make a link safe to paste into chats and reports."""
    return re.sub(r"^http", "hxxp", url.strip(), flags=re.I).replace(".", "[.]")


def anatomy_segments(url: str):
    m = URL_RE.match(url.strip())
    if not m:
        return [(clip(url), "path", "URL")]
    scheme, netloc, path, query, frag = m.groups()
    segs = []
    if scheme:
        segs.append((scheme, "scheme", "scheme"))
    if "@" in netloc:
        userinfo, netloc = netloc.rsplit("@", 1)
        segs.append((userinfo + "@", "userinfo", "ignored by browsers"))
    host, port = netloc, ""
    if re.search(r":\d+$", host):
        host, port = host.rsplit(":", 1)
        port = ":" + port
    if IP_RE.match(host):
        segs.append((host, "ip", "raw IP address"))
    elif host:
        labels = host.split(".")
        n = 3 if (len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL) else 2
        if len(labels) < n:
            segs.append((host, "domain", "domain"))
        else:
            sub = labels[:-n]
            if sub:
                segs.append((".".join(sub) + ".", "subdomain", "subdomain"))
            segs.append((labels[-n], "domain", "real domain"))
            segs.append(("." + ".".join(labels[-n + 1:]), "suffix", "ending"))
    if port:
        segs.append((port, "port", "port"))
    if path:
        segs.append((clip(path), "path", "path"))
    if query:
        segs.append((clip(query), "query", "query"))
    if frag:
        segs.append((clip(frag, 24), "fragment", "fragment"))
    return segs


def indicators(vec):
    """Plain-language checks: (name, value, status, explanation, bar ratio)."""
    f = dict(zip(FEATURE_NAMES, vec))

    def num(name, key, warn, risk, why, fmt="{:.0f}"):
        v = f[key]
        st_ = "risk" if v >= risk else "warn" if v >= warn else "ok"
        return (name, fmt.format(v), st_, why, max(0.06, min(v / risk, 1.0)))

    def flag(name, key, why):
        on = bool(f[key])
        return (name, "yes" if on else "no", "risk" if on else "ok", why, 1.0 if on else 0.06)

    return [
        num("URL length", "url_length", 75, 110, "Long URLs can bury the real destination."),
        flag("Raw IP address", "is_ip", "Legitimate sites almost always use a domain name."),
        flag("'@' symbol", "has_at", "Text before '@' is ignored by browsers and can disguise the host."),
        num("Subdomains", "num_subdomains", 2, 3, "Many levels are used to imitate trusted brands."),
        num("Hyphens", "num_hyphens", 3, 6, "Names like secure-login-bank are common in phishing."),
        num("Sensitive keywords", "suspicious_words", 1, 2, "Words such as login, verify, account, password."),
        num("Path depth", "path_depth", 5, 8, "Deep folder paths often point to hosted kits or payloads."),
        num("Digit share", "digit_ratio", 0.25, 0.4, "Many random digits suggest generated addresses.", "{:.0%}"),
        num("Randomness", "entropy", 4.6, 5.0, "High entropy means less readable, more machine-made text.", "{:.2f}"),
    ]


# =============================================================================
# STATE & CALLBACKS
# =============================================================================
EXAMPLES = {
    "Wikipedia article": "https://www.wikipedia.org/wiki/Computer_network",
    "Fake PayPal login": "http://secure-login.paypal-verify-account.top/webscr?cmd=_login&account=update",
    "IP address host": "http://185.220.101.47/bank/confirm/login.php",
    "Hidden by @": "http://paypal.com@free-gift-claim.xyz/verify",
}

for k, v in {"history": [], "result": None, "url_input": "", "input_error": "",
             "pending": None, "scan_id": 0, "focused": False}.items():
    st.session_state.setdefault(k, v)


def run_scan(url: str) -> None:
    url = (url or "").strip()
    st.session_state["input_error"] = ""
    host = normalize(url).split("/")[0] if url else ""
    if not url:
        st.session_state["input_error"] = "Enter a URL to inspect."
    elif len(url) > 4096:
        st.session_state["input_error"] = "That URL is over 4,096 characters. Paste a shorter link."
    elif "." not in host and not IP_RE.match(host):
        st.session_state["input_error"] = "That doesn't look like a web address. Try example.com/login."
    else:
        st.session_state["pending"] = url


def scan_from_input() -> None:
    run_scan(st.session_state["url_input"])


def scan_example(url: str) -> None:
    st.session_state["url_input"] = url
    run_scan(url)


def clear_all() -> None:
    st.session_state.update(history=[], result=None, url_input="", input_error="", pending=None)


# =============================================================================
# SMALL HTML BUILDERS
# =============================================================================
RING_C = 2 * math.pi * 52


def keyframes(sid: int, prob: float) -> str:
    """Per-scan keyframe names, so animations replay on every new scan."""
    return (
        "<style>"
        f"@keyframes ring{sid}{{from{{stroke-dashoffset:{RING_C:.2f}}}to{{stroke-dashoffset:{RING_C * (1 - prob):.2f}}}}}"
        f"@keyframes cnt{sid}{{from{{--n:0}}to{{--n:{round(prob * 100)}}}}}"
        f"@keyframes rv{sid}{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}"
        f"@keyframes pop{sid}{{from{{opacity:0;transform:scale(.2)}}to{{opacity:var(--o,1);transform:scale(1)}}}}"
        f"@keyframes grow{sid}{{from{{width:0}}}}"
        "</style>"
    )


def ring_svg(prob: float, thr: float, kind: str, sid: int) -> str:
    color = "#FF3B5C" if kind == "risk" else "#25D985"
    return (
        f'<svg width="190" height="190" viewBox="0 0 120 120" fill="none">'
        f'<circle cx="60" cy="60" r="52" stroke="rgba(255,255,255,.09)" stroke-width="9"/>'
        f'<circle class="arc" cx="60" cy="60" r="52" stroke="{color}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{RING_C:.2f}" stroke-dashoffset="{RING_C * (1 - prob):.2f}" transform="rotate(-90 60 60)" '
        f'style="color:{color};animation:ring{sid} 1.4s cubic-bezier(.22,.8,.2,1) both"/>'
        f'<g transform="rotate({thr * 360:.1f} 60 60)"><line x1="60" y1="2" x2="60" y2="15" stroke="#fff" '
        f'stroke-width="2.4" stroke-linecap="round"/></g></svg>'
    )


def scan_html(stage: int, url: str) -> str:
    labels = ["Parsing the address", f"Extracting {len(FEATURE_NAMES)} URL features",
              f"Asking {N_TREES} decision trees", "Scoring the result"]
    rows = ""
    for i, lab in enumerate(labels):
        cls = "done" if i < stage else "act" if i == stage else ""
        rows += f'<div class="stp {cls}"><span class="ic"></span>{lab}</div>'
    return block(f"""
    <div class="scan">
      <div class="radar"><div class="sweep"></div><div class="blip"></div></div>
      <div class="scan-body"><div class="scan-url">{html.escape(clip(url, 90))}</div>{rows}</div>
    </div>""")


SHIELD_HERO = """
<svg width="150" height="150" viewBox="0 0 120 120" fill="none">
  <defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7FA6FF"/><stop offset="1" stop-color="#B58CFF"/></linearGradient></defs>
  <g class="orbit"><circle cx="60" cy="60" r="55" stroke="rgba(155,185,255,.35)" stroke-width="1" stroke-dasharray="3 7"/><circle cx="60" cy="5" r="3.5" fill="#9DB8FF"/></g>
  <path d="M60 24l24 9v17c0 15-10 27-24 33-14-6-24-18-24-33V33l24-9z" fill="rgba(91,140,255,.16)" stroke="url(#sg)" stroke-width="2.4" stroke-linejoin="round"/>
  <path d="M50 55l7 7 13-14" stroke="#fff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SHIELD_SMALL = ('<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#9DB8FF" stroke-width="1.8" '
                'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6l7-3z"/>'
                '<path d="M9 12l2 2 4-4"/></svg>')
ICON_WARN = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#FF6B85" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l9.5 17h-19L12 3z"/><path d="M12 10v4.5M12 17.6v.1"/></svg>')
ICON_OK = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#25D985" stroke-width="2.2" '
           'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/></svg>')

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    H(f'<div class="side-brand">{SHIELD_SMALL}<span class="name">Link Inspector</span></div>')
    H('<div class="side-note" style="margin-top:.4rem">Phishing and malicious URL classifier</div>')

    H('<div class="side-h">Decision threshold</div>')
    threshold = st.slider("Flag as malicious above", 0.10, 0.90, 0.50, 0.05, key="threshold",
                          label_visibility="collapsed",
                          help="Lower catches more threats but raises false alarms.")
    H(f'<div class="side-note">Links scoring above <b style="color:#fff">{threshold:.0%}</b> are flagged. '
      "Lower values catch more threats but raise false alarms.</div>")

    H('<div class="side-h">Display</div>')
    animate = st.toggle("Scan animation", value=True, key="anim",
                        help="Plays a short scanning sequence before each result.")

    H('<div class="side-h">Model</div>')
    rows = [("Algorithm", "Random Forest"), ("Trees", str(N_TREES or "-")), ("Features", str(len(FEATURE_NAMES)))]
    metrics = load_metrics()
    if metrics:
        for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score")]:
            if key in metrics:
                rows.append((label, f"{metrics[key] * 100:.1f}%"))
    H("".join(f'<div class="kv"><span>{k}</span><span>{v}</span></div>' for k, v in rows))
    if not metrics:
        H('<div class="side-note">Add metrics.json (see notes) to show test scores here.</div>')

    H('<div class="side-h">Session</div>')
    st.button("Clear results and history", on_click=clear_all, width="stretch")
    H('<div class="side-note">Only the URL text is analysed. The link is never opened, '
      "and nothing is stored after you close the tab.</div>")

# =============================================================================
# HERO
# =============================================================================
H(f"""
<div class="hero">
  <div>
    <h1>Know before<br>you click.</h1>
    <p>Paste any link. A Random Forest trained on hundreds of thousands of labelled URLs scores it
    from the structure of the address alone, without ever visiting the site.</p>
  </div>
  <div class="hero-art"><div class="orb"></div>{SHIELD_HERO}</div>
</div>
<div class="facts">
  <div class="fact"><i></i>Model ready</div>
  <div class="fact"><b>{N_TREES}</b> decision trees</div>
  <div class="fact"><b>{len(FEATURE_NAMES)}</b> URL features</div>
  <div class="fact"><b>0</b> requests sent to the link</div>
</div>
""")

if not st.session_state["focused"]:
    st.session_state["focused"] = True
    _focus_js = ("<script>setTimeout(()=>{const i=window.parent.document.querySelector('input[type=text]');"
                 "if(i)i.focus();},350);</script>")
    try:  # focus the URL box once per session (st.iframe replaces the deprecated components.html)
        if hasattr(st, "iframe"):
            st.iframe(_focus_js, height=1)
        else:
            components.html(_focus_js, height=0)
    except Exception:
        pass

tab_scan, tab_batch, tab_about = st.tabs(["Scan a link", "Scan a list", "How it works"])

# =============================================================================
# TAB 1 - SINGLE SCAN
# =============================================================================
with tab_scan:
    with st.form("scan_form", clear_on_submit=False, border=False):
        c1, c2 = st.columns([5, 1.3], vertical_alignment="center")
        with c1:
            st.text_input("URL", key="url_input", label_visibility="collapsed",
                          placeholder="Paste a link, e.g. https://example.com/account/login")
        with c2:
            st.form_submit_button("Inspect link", type="primary", on_click=scan_from_input, width="stretch")
    H('<div class="helper">Press Enter to inspect. The link is analysed as text and never opened.</div>')

    if st.session_state["input_error"]:
        st.warning(st.session_state["input_error"])

    H('<div class="try-label">Try an example</div>')
    for col, (label, ex_url) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        col.button(label, key=f"ex_{label}", on_click=scan_example, args=(ex_url,), width="stretch")

    # ---- run pipeline for a new scan (with live scanning animation) ----
    pending = st.session_state["pending"]
    if pending:
        st.session_state["pending"] = None
        ph = st.empty()
        pause = 0.38 if animate else 0.0
        if animate:
            ph.markdown(scan_html(0, pending), unsafe_allow_html=True)
        time.sleep(pause)
        if animate:
            ph.markdown(scan_html(1, pending), unsafe_allow_html=True)
        vec = extract_features(pending)
        X = pd.DataFrame([vec], columns=FEATURE_NAMES)
        time.sleep(pause)
        if animate:
            ph.markdown(scan_html(2, pending), unsafe_allow_html=True)
        prob = float(model.predict_proba(X)[0][POS])
        votes = tree_votes(X)
        time.sleep(pause)
        if animate:
            ph.markdown(scan_html(3, pending), unsafe_allow_html=True)
        time.sleep(pause * 0.7)
        ph.empty()

        st.session_state["scan_id"] += 1
        st.session_state["result"] = {"url": pending, "prob": prob, "votes": votes, "vec": vec,
                                      "sid": st.session_state["scan_id"]}
        st.session_state["history"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "url": pending, "prob": prob})
        st.session_state["history"] = st.session_state["history"][:8]
        if prob >= threshold:
            st.toast("Malicious link detected", icon="🚨")
        else:
            st.toast("No threat patterns found", icon="✅")

    res = st.session_state["result"]
    if res is None:
        H("""
        <div class="empty">
          <div class="radar"><div class="sweep"></div></div>
          <div><b>Waiting for a link</b>
          <span>Paste a URL above or pick an example. You will get a verdict, a breakdown of the address,
          the warning signs found, and a view of how every tree in the forest voted.</span></div>
        </div>""")
    else:
        url, prob, votes, sid = res["url"], res["prob"], res["votes"], res["sid"]
        bad = prob >= threshold
        kind = "risk" if bad else "safe"
        title = "Likely phishing or malicious" if bad else "Looks safe"
        sub = ("Do not open this link, and never enter passwords, card numbers or personal details on it."
               if bad else "No strong phishing patterns in the URL structure. Still be careful with links you did not expect.")
        n_flag = sum(1 for v in votes if v >= 0.5)
        tree_chip = f'<span class="chip"><b>{n_flag}</b> of <b>{len(votes)}</b> trees flag it</span>' if votes else ""

        # verdict
        H(f"""
        {keyframes(sid, prob)}
        <div class="verdict {kind}" style="animation-delay:0s">
          <div class="ringbox">{ring_svg(prob, threshold, kind, sid)}
            <div class="ring-num"><div>
              <div class="big"><span class="cnt" style="--n:{round(prob * 100)};animation:cnt{sid} 1.4s cubic-bezier(.22,.8,.2,1) both"></span><small>%</small></div>
              <div class="cap">malicious<br>probability</div></div></div>
          </div>
          <div class="v-main">
            <div class="v-title">{ICON_WARN if bad else ICON_OK}<span>{title}</span></div>
            <div class="v-sub">{sub}</div>
            <div class="chips">
              <span class="chip">Exact score <b>{prob * 100:.1f}%</b></span>
              <span class="chip">Flag threshold <b>{threshold:.0%}</b></span>{tree_chip}
            </div>
          </div>
        </div>""")

        if bad:
            H('<div class="defang-label">Safe to share in chats and reports (defanged, not clickable):</div>')
            st.code(defang(url), language=None)

        # anatomy
        segs = anatomy_segments(url)
        seg_html = "".join(
            f'<div class="seg k-{k}" style="animation:rv{sid} .5s ease both;animation-delay:{250 + i * 90}ms">'
            f'<span class="t">{html.escape(t)}</span><span class="c">{html.escape(c)}</span></div>'
            for i, (t, k, c) in enumerate(segs))
        H(f"""
        <div class="card">
          <h3>Anatomy of the link</h3>
          <p class="hint">Only the highlighted domain decides where you actually land. Everything before it can be made up by the sender.</p>
          <div class="anat">{seg_html}</div>
        </div>""")

        left, right = st.columns([1.35, 1], gap="medium")

        with left:
            inds = indicators(res["vec"])
            flagged = sum(1 for i in inds if i[2] != "ok")
            cards = "".join(
                f'<div class="ind" style="animation:rv{sid} .5s ease both;animation-delay:{300 + i * 70}ms">'
                f'<div class="top"><span class="dot {s}"></span>{html.escape(n)}<span class="val">{html.escape(v)}</span></div>'
                f'<div class="bar"><i class="{s}" style="width:{ratio * 100:.0f}%;animation:grow{sid} .9s ease both;animation-delay:{450 + i * 70}ms"></i></div>'
                f'<div class="why">{html.escape(w)}</div></div>'
                for i, (n, v, s, w, ratio) in enumerate(inds))
            H(f"""
            <div class="card">
              <h3>Warning signs</h3>
              <p class="hint"><b style="color:#fff">{flagged} of {len(inds)}</b> checks raised a flag. These simple rules explain the result;
              the verdict itself comes from the model weighing all {len(FEATURE_NAMES)} features together.</p>
              <div class="ind-grid">{cards}</div>
              <div class="key"><span><i class="dot ok"></i>Normal</span><span><i class="dot warn"></i>Unusual</span><span><i class="dot risk"></i>Suspicious</span></div>
            </div>""")

        with right:
            if votes:
                dots = "".join(
                    f'<i class="{"r" if v >= 0.5 else "s"}" style="--o:{0.4 + 0.6 * abs(v - 0.5) * 2:.2f};'
                    f'animation:pop{sid} .4s ease both;animation-delay:{300 + i * 5}ms"></i>'
                    for i, v in enumerate(votes))
                H(f"""
                <div class="card">
                  <h3>How the forest voted</h3>
                  <p class="hint">Each dot is one decision tree. Red leans malicious, green leans safe; brighter means more certain.</p>
                  <div class="forest">{dots}</div>
                  <div class="fcount"><span class="r"><b>{n_flag}</b> malicious</span><span class="s"><b>{len(votes) - n_flag}</b> safe</span></div>
                </div>""")

            tips = (["Do not click it or open it in a browser.",
                     "Never type a password, card number or one-time code on it.",
                     "Report it to your IT or security team, or to the brand it imitates.",
                     "If you already clicked: change that account's password and turn on two-factor sign-in."]
                    if bad else
                    ["Check the highlighted domain is spelled exactly right.",
                     "Be wary if the link arrived unexpectedly or urges you to act fast.",
                     "For banking or sign-in pages, type the address yourself or use a bookmark.",
                     "A green result lowers risk; it never removes it."])
            mk = "!" if bad else "✓"
            lis = "".join(f'<li><span class="mk">{mk}</span><span>{html.escape(t)}</span></li>' for t in tips)
            H(f"""
            <div class="card">
              <h3>{"What to do now" if bad else "Good habits"}</h3>
              <ul class="todo {kind}">{lis}</ul>
            </div>""")

    # history
    hist = st.session_state["history"]
    if hist:
        trs = ""
        for h in hist:
            b = h["prob"] >= threshold
            pill = f'<span class="pill {"risk" if b else "safe"}">{"Malicious" if b else "Safe"}</span>'
            col = "#FF3B5C" if b else "#25D985"
            trs += (f'<tr><td>{h["time"]}</td><td class="u" title="{html.escape(h["url"])}">{html.escape(clip(h["url"], 70))}</td>'
                    f'<td><span class="pbar"><i style="width:{h["prob"] * 100:.0f}%;background:{col}"></i></span>{h["prob"] * 100:.1f}%</td>'
                    f'<td>{pill}</td></tr>')
        H(f"""
        <div class="card">
          <h3>Recent scans</h3>
          <table class="tbl"><thead><tr><th>Time</th><th>URL</th><th>Score</th><th>Verdict</th></tr></thead><tbody>{trs}</tbody></table>
        </div>""")

# =============================================================================
# TAB 2 - BATCH
# =============================================================================
with tab_batch:
    H('<div class="card" style="margin-top:1rem"><h3>Scan many links at once</h3>'
      '<p class="hint" style="margin:0">Upload a CSV with a column named <b style="color:#fff">url</b>, or paste one link per line. '
      "Up to 50,000 links per run.</p></div>")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    pasted = st.text_area("Or paste links, one per line", height=120,
                          placeholder="https://example.com\nhttp://192.168.1.5/login")

    urls = []
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            col = next((c for c in df_in.columns if str(c).strip().lower() == "url"), df_in.columns[0])
            urls = df_in[col].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Could not read that file: {e}")
    elif pasted.strip():
        urls = [u.strip() for u in pasted.splitlines() if u.strip()]

    if urls:
        if len(urls) > 50000:
            st.warning("Only the first 50,000 links were scanned.")
            urls = urls[:50000]
        with st.spinner(f"Scoring {len(urls):,} links..."):
            Xb = pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)
            probs = model.predict_proba(Xb)[:, POS]
        out = pd.DataFrame({"url": urls, "malicious_probability": probs})
        out["verdict"] = np.where(probs >= threshold, "Malicious", "Safe")
        n_bad = int((out["verdict"] == "Malicious").sum())
        n_all = len(out)
        bsid = 9000 + (abs(hash((n_all, n_bad, round(threshold, 2)))) % 90000)
        H(f"""
        <style>@keyframes bc{bsid}{{from{{--n:0}}to{{--n:{n_all}}}}}@keyframes bb{bsid}{{from{{--n:0}}to{{--n:{n_bad}}}}}
        @keyframes bs{bsid}{{from{{--n:0}}to{{--n:{n_all - n_bad}}}}}</style>
        <div class="stats">
          <div class="stat"><div class="n"><span class="cnt" style="--n:{n_all};animation:bc{bsid} 1s ease-out both"></span></div><div class="l">links scanned</div></div>
          <div class="stat r"><div class="n"><span class="cnt" style="--n:{n_bad};animation:bb{bsid} 1s ease-out both"></span></div><div class="l">flagged malicious</div></div>
          <div class="stat s"><div class="n"><span class="cnt" style="--n:{n_all - n_bad};animation:bs{bsid} 1s ease-out both"></span></div><div class="l">look safe</div></div>
          <div class="stat"><div class="n">{n_bad / n_all:.0%}</div><div class="l">share flagged</div></div>
        </div>""")
        top = out.sort_values("malicious_probability", ascending=False).head(200)
        trs = ""
        for _, r in top.iterrows():
            b = r["verdict"] == "Malicious"
            col = "#FF3B5C" if b else "#25D985"
            trs += (f'<tr><td class="u" title="{html.escape(r["url"])}">{html.escape(clip(r["url"], 80))}</td>'
                    f'<td><span class="pbar"><i style="width:{r["malicious_probability"] * 100:.0f}%;background:{col}"></i></span>'
                    f'{r["malicious_probability"] * 100:.1f}%</td>'
                    f'<td><span class="pill {"risk" if b else "safe"}">{r["verdict"]}</span></td></tr>')
        shown = f"Showing the {len(top)} highest-risk links. " if n_all > len(top) else ""
        H(f"""
        <div class="card" style="margin-top:0">
          <h3>Results, highest risk first</h3>
          <p class="hint">{shown}Download the CSV for every row.</p>
          <div style="max-height:420px;overflow:auto"><table class="tbl"><thead><tr><th>URL</th><th>Malicious probability</th><th>Verdict</th></tr></thead><tbody>{trs}</tbody></table></div>
        </div>""")
        st.download_button("Download full results (CSV)", out.to_csv(index=False).encode(),
                           "link_inspector_results.csv", "text/csv", type="primary")

# =============================================================================
# TAB 3 - HOW IT WORKS
# =============================================================================
with tab_about:
    a, b = st.columns([1.15, 1], gap="medium")
    with a:
        H(f"""
        <div class="card" style="margin-top:1rem">
          <h3>From URL to verdict</h3>
          <p class="hint">The same code runs in training and in this app, so the model always sees data prepared the same way.</p>
          <ol class="steps">
            <li><div><b>Normalise.</b> The scheme (http/https) and a leading "www." are removed. In the training data almost every safe link used both, so keeping them would let the model cheat.</div></li>
            <li><div><b>Extract {len(FEATURE_NAMES)} lexical features.</b> Length, dots, hyphens, digits, '@', IP host, subdomains, path depth, keywords, entropy and more.</div></li>
            <li><div><b>Classify.</b> A Random Forest of {N_TREES} decision trees votes. The average of their votes is the probability you see.</div></li>
            <li><div><b>Decide.</b> A link above the sidebar threshold is shown in red, otherwise in green.</div></li>
          </ol>
        </div>""")
    with b:
        H("""
        <div class="card" style="margin-top:1rem">
          <h3>What this tool cannot do</h3>
          <p class="hint">Know the limits before relying on a score.</p>
          <div class="note"><b>It reads the address, not the website.</b> Brand-new phishing pages on clean-looking
          domains, hijacked legitimate sites and shortened links can slip through. A green result is not a guarantee.
          Real-world defences also use domain age, certificates, page content and threat feeds.</div>
          <ul style="margin:1rem 0 0;padding-left:1.1rem;color:#B8C4E2;line-height:1.8;font-size:.92rem">
            <li>HTTPS is ignored on purpose. A padlock does not prove a site is honest.</li>
            <li>The training set groups phishing, malware and defacement links as "malicious".</li>
            <li>Accuracy on lexical features alone is typically around 90-93%.</li>
          </ul>
        </div>""")

H('<div class="foot">Link Inspector · Network Security project · Lexical analysis with a Random Forest</div>')