"""
TrafficSense Design System
Clean professional light theme — government / operations use.
"""
import streamlit as st

# ── Color tokens ──────────────────────────────────────────────────────────────
C = {
    "bg":       "#f0f2f5",
    "surface":  "#ffffff",
    "surface2": "#f8f9fb",
    "border":   "#dde3ec",
    "border_hi":"#b8c5d6",
    "primary":  "#1a56db",
    "cyan":     "#0891b2",
    "text":     "#111827",
    "muted":    "#4b5563",
    "subtle":   "#9ca3af",
    "success":  "#15803d",
    "warning":  "#b45309",
    "error":    "#b91c1c",
    "critical": "#991b1b",
    "purple":   "#6d28d9",
}

ALERT = {
    "CRITICAL": dict(fg="#991b1b", bg="#fef2f2", border="#fca5a5", badge_bg="#fee2e2"),
    "RED":      dict(fg="#b91c1c", bg="#fff5f5", border="#fca5a5", badge_bg="#fee2e2"),
    "AMBER":    dict(fg="#92400e", bg="#fffbeb", border="#fcd34d", badge_bg="#fef3c7"),
    "GREEN":    dict(fg="#14532d", bg="#f0fdf4", border="#86efac", badge_bg="#dcfce7"),
}

SEV_COLOR = {
    "Critical": "#b91c1c",
    "High":     "#c2410c",
    "Medium":   "#b45309",
    "Low":      "#15803d",
}

PLOTLY = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,249,251,0.6)",
    font=dict(color="#111827", size=12, family="'Inter', -apple-system, sans-serif"),
    colorway=["#1a56db","#b45309","#b91c1c","#15803d","#6d28d9","#0891b2","#ea580c"],
)
AXIS_STYLE   = dict(gridcolor="#e5e7eb", linecolor="#d1d5db",
                    tickfont=dict(color="#374151"), title_font=dict(color="#374151"))
AXIS_NO_GRID = dict(gridcolor="rgba(0,0,0,0)", linecolor="#d1d5db",
                    tickfont=dict(color="#374151"), title_font=dict(color="#374151"))
LEGEND_STYLE = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#374151"))
BASE_MARGIN  = dict(l=50, r=20, t=44, b=50)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ━━━ RESET & BASE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f0f2f5 !important;
    color: #111827 !important;
}

/* Kill the dark header bar */
header[data-testid="stHeader"] {
    background-color: #ffffff !important;
    border-bottom: 1px solid #e5e7eb !important;
    height: 0px !important;
    min-height: 0px !important;
}
header[data-testid="stHeader"] * { display: none !important; }

#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding: 1.75rem 2rem 2rem !important;
    max-width: 1440px !important;
    background: transparent !important;
}

/* ━━━ SCROLLBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f2f5; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ━━━ SIDEBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    box-shadow: 2px 0 6px rgba(0,0,0,.05) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] .block-container {
    padding: .5rem 1rem !important;
    background: transparent !important;
}

a[data-testid="stSidebarNavLink"] {
    border-radius: 7px !important;
    margin: 2px 4px !important;
    padding: 9px 12px !important;
    transition: background .15s ease !important;
    border: 1px solid transparent !important;
    text-decoration: none !important;
}
a[data-testid="stSidebarNavLink"]:hover { background: #f3f4f6 !important; }
a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
}
a[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: #1a56db !important;
    font-weight: 600 !important;
}
a[data-testid="stSidebarNavLink"] span {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ━━━ TYPOGRAPHY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-weight: 700 !important;
    letter-spacing: -.2px !important;
}
p, li { color: #374151 !important; line-height: 1.65 !important; }
strong, b { color: #111827 !important; }
small, .caption { color: #6b7280 !important; font-size: 12px !important; }

/* Streamlit label overrides */
label, .stTextInput label, .stSelectbox label,
.stSlider label, .stRadio label, .stCheckbox label,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ━━━ FORM INPUTS — THE KEY FIX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Streamlit uses BaseWeb which applies its own dark styles via CSS-in-JS.
   We target every layer to guarantee light backgrounds + dark text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Select box — outer container */
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div {
    background-color: #ffffff !important;
    border-color: #d1d5db !important;
    color: #111827 !important;
}
div[data-baseweb="select"] > div {
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}
div[data-baseweb="select"] > div:hover { border-color: #9ca3af !important; }
div[data-baseweb="select"] > div:focus-within {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,.1) !important;
}

/* Select — value text */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div[class*="placeholder"],
div[data-baseweb="select"] div[class*="singleValue"],
div[data-baseweb="select"] input {
    color: #111827 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}
/* Placeholder specifically */
div[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] span,
div[data-baseweb="select"] > div > div > div > div > span { color: #111827 !important; }

/* Dropdown arrow */
div[data-baseweb="select"] svg { fill: #6b7280 !important; }

/* Select dropdown menu — popover portal sits OUTSIDE the stApp so needs its own bg */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {
    background-color: #ffffff !important;
}

div[data-baseweb="menu"],
ul[data-baseweb="menu"],
[data-baseweb="popover"] div[data-baseweb="menu"],
[data-baseweb="popover"] ul[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
    overflow: hidden !important;
}

/* Every list item in the dropdown */
div[data-baseweb="menu"] li,
ul[data-baseweb="menu"] li,
[data-baseweb="popover"] li,
[role="option"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 9px 14px !important;
}
div[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover,
[role="option"]:hover {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
}
div[data-baseweb="menu"] li[aria-selected="true"],
ul[data-baseweb="menu"] li[aria-selected="true"],
[role="option"][aria-selected="true"] {
    background-color: #eff6ff !important;
    color: #1a56db !important;
    font-weight: 600 !important;
}

/* All text inside dropdown items */
[data-baseweb="menu"] li *,
[data-baseweb="menu"] li span,
[role="option"] span,
[role="option"] div {
    color: #111827 !important;
    background-color: transparent !important;
}

/* Text inputs */
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] > div {
    background-color: #ffffff !important;
    border-color: #d1d5db !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 10px 12px !important;
}
div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,.1) !important;
    outline: none !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder { color: #9ca3af !important; }

/* ━━━ RADIO & CHECKBOX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-testid="stRadio"] > div,
div[data-testid="stRadio"] label,
div.stCheckbox label {
    color: #111827 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}
div[data-testid="stRadio"] p { color: #111827 !important; font-size: 14px !important; }

/* Kill the grey oval that BaseWeb draws around the whole label on hover/focus */
div[data-baseweb="radio"],
div[data-baseweb="radio"]:hover,
div[data-baseweb="radio"]:focus-within,
div[data-baseweb="radio"] label,
div[data-baseweb="radio"] > label {
    background-color: transparent !important;
    background: none !important;
    border-radius: 0 !important;
    outline: none !important;
    box-shadow: none !important;
}
/* Also kill it on the inner label/span wrapper BaseWeb uses */
div[data-baseweb="radio"] > div:last-child,
div[data-baseweb="radio"] > div:last-child:hover,
div[data-baseweb="radio"] span {
    background-color: transparent !important;
    background: none !important;
    border-radius: 0 !important;
}

/* Radio circle — unselected: white fill, grey border */
div[data-baseweb="radio"] > div:first-child {
    background-color: #ffffff !important;
    border: 2px solid #d1d5db !important;
    border-radius: 50% !important;
    min-width: 18px !important;
    width: 18px !important;
    height: 18px !important;
    flex-shrink: 0 !important;
}
/* Radio circle — selected: blue */
div[data-baseweb="radio"][data-checked="true"] > div:first-child {
    background-color: #1a56db !important;
    border-color: #1a56db !important;
}
/* Inner white dot */
div[data-baseweb="radio"] > div:first-child > div {
    background-color: #ffffff !important;
}
/* Hover: only tint the circle, not the whole label */
div[data-baseweb="radio"]:hover > div:first-child {
    border-color: #1a56db !important;
}

/* Checkbox */
div[data-baseweb="checkbox"] > div:first-child {
    background-color: #ffffff !important;
    border: 2px solid #d1d5db !important;
    border-radius: 4px !important;
}
div[data-baseweb="checkbox"][data-checked="true"] > div:first-child {
    background-color: #1a56db !important;
    border-color: #1a56db !important;
}
div[data-baseweb="checkbox"],
div[data-baseweb="checkbox"]:hover {
    background: transparent !important;
}

/* ━━━ SLIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div.stSlider [data-testid="stSliderTickBarMin"],
div.stSlider [data-testid="stSliderTickBarMax"],
div.stSlider [data-testid="stThumbValue"],
div.stSlider p { color: #374151 !important; font-size: 13px !important; }

div.stSlider > div > div > div { background: #e5e7eb !important; border-radius: 99px !important; }
div.stSlider > div > div > div > div { background: #1a56db !important; }
div.stSlider > div > div > div > div > div {
    background: #ffffff !important;
    border: 2px solid #1a56db !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.15) !important;
}

/* ━━━ BUTTONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #1a56db !important;
    color: #ffffff !important;
    border: 1px solid #1741b0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    letter-spacing: .01em !important;
    transition: background-color .15s ease, box-shadow .15s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.1) !important;
    width: auto !important;
    cursor: pointer !important;
}
div.stFormSubmitButton > button { width: 100% !important; padding: 12px 24px !important; }
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #1741b0 !important;
    box-shadow: 0 3px 10px rgba(26,86,219,.3) !important;
}
div.stButton > button:active,
div.stFormSubmitButton > button:active { background-color: #1338a0 !important; }

/* ━━━ TABS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-baseweb="tab-list"] {
    background-color: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    border: 1px solid transparent !important;
    transition: all .15s ease !important;
}
button[data-baseweb="tab"]:hover { background: #ffffff !important; color: #374151 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #1a56db !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.1) !important;
    border-color: #e5e7eb !important;
}

/* ━━━ EXPANDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
details[data-testid="stExpander"], div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}
details[data-testid="stExpander"] summary { color: #111827 !important; font-weight: 600 !important; }
details[data-testid="stExpander"] summary p { color: #111827 !important; }

/* ━━━ METRIC CARDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}
div[data-testid="metric-container"] > div > div:first-child {
    color: #6b7280 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"],
div[data-testid="metric-container"] > div > div:nth-child(2) {
    color: #111827 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* ━━━ ALERTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 4px !important; }
div[data-testid="stAlert"] p { color: inherit !important; }

/* ━━━ DATAFRAME ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-testid="stDataFrameContainer"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #ffffff !important;
}

/* ━━━ PROGRESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div.stProgress > div > div { background: #e5e7eb !important; border-radius: 99px !important; }
div.stProgress > div > div > div { background: #1a56db !important; border-radius: 99px !important; }

/* ━━━ SPINNER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-testid="stSpinner"] > div { border-top-color: #1a56db !important; }

/* ━━━ MARKDOWN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
div[data-testid="stMarkdownContainer"] p { color: #374151 !important; }
div[data-testid="stMarkdownContainer"] strong { color: #111827 !important; }
div[data-testid="stMarkdownContainer"] code {
    background: #f1f5f9 !important;
    color: #1a56db !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 13px !important;
}

/* Caption */
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] { color: #9ca3af !important; font-size: 12px !important; }

/* HR */
hr { border-color: #e5e7eb !important; margin: 20px 0 !important; }

/* Number input arrows */
div[data-testid="stNumberInput"] button {
    background: #f9fafb !important;
    border-color: #d1d5db !important;
    color: #6b7280 !important;
}
div[data-testid="stNumberInput"] button:hover { background: #f3f4f6 !important; }

/* Tooltip */
div[data-testid="stTooltipIcon"] svg { fill: #9ca3af !important; }

/* Column gaps */
div[data-testid="column"] { padding: 0 6px !important; }
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


# ━━━ COMPONENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sidebar_brand():
    st.markdown("""
    <div style="padding:18px 12px 16px;">
        <div style="display:flex;align-items:center;gap:11px;">
            <div style="width:38px;height:38px;border-radius:9px;flex-shrink:0;
                        background:#1a56db;display:flex;align-items:center;
                        justify-content:center;">
                <span style="font-size:19px;line-height:1;">🚦</span>
            </div>
            <div>
                <div style="font-size:14px;font-weight:700;color:#111827;letter-spacing:-.2px;">
                    TrafficSense
                </div>
                <div style="font-size:10px;color:#9ca3af;font-weight:500;
                            letter-spacing:.07em;text-transform:uppercase;margin-top:1px;">
                    Event Congestion AI
                </div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:12px;
                    background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:6px;padding:5px 10px;">
            <span style="width:6px;height:6px;border-radius:50%;background:#15803d;
                         display:inline-block;flex-shrink:0;"></span>
            <span style="font-size:11px;color:#15803d;font-weight:600;letter-spacing:.04em;">
                System Online
            </span>
        </div>
    </div>
    <div style="height:1px;background:#e5e7eb;margin:0 0 12px;"></div>
    """, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;
                padding:4px 0 20px;border-bottom:1px solid #e5e7eb;margin-bottom:24px;">
        <div style="width:42px;height:42px;border-radius:10px;display:flex;
                    align-items:center;justify-content:center;flex-shrink:0;
                    background:#eff6ff;border:1px solid #bfdbfe;">
            <span style="font-size:20px;line-height:1;">{icon}</span>
        </div>
        <div>
            <div style="font-size:19px;font-weight:700;color:#111827;
                        letter-spacing:-.2px;line-height:1.2;">{title}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:3px;">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, emoji: str = ""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:22px 0 12px;">
        <span style="font-size:11px;font-weight:700;color:#6b7280;
                     text-transform:uppercase;letter-spacing:.09em;white-space:nowrap;">
            {emoji}&ensp;{title}
        </span>
        <div style="flex:1;height:1px;background:#e5e7eb;"></div>
    </div>
    """, unsafe_allow_html=True)


def alert_html(level: str) -> str:
    a = ALERT.get(level, ALERT["GREEN"])
    labels = {
        "CRITICAL": ("🔴", "Critical Incident"),
        "RED":      ("🟠", "High Alert"),
        "AMBER":    ("🟡", "Elevated Risk"),
        "GREEN":    ("🟢", "Situation Normal"),
    }
    icon, label = labels.get(level, ("⚪", level))
    return f"""
    <div style="background:{a['bg']};border:1px solid {a['border']};
                border-radius:10px;padding:14px 18px;margin:10px 0;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:20px;flex-shrink:0;">{icon}</span>
            <div style="flex:1;">
                <div style="font-size:10px;font-weight:700;color:{a['fg']}aa;
                            letter-spacing:.1em;text-transform:uppercase;margin-bottom:2px;">
                    Alert Level
                </div>
                <div style="font-size:17px;font-weight:700;color:{a['fg']};line-height:1.2;">
                    {label}
                </div>
            </div>
            <div style="background:{a['badge_bg']};border:1px solid {a['border']};
                        border-radius:6px;padding:4px 14px;flex-shrink:0;">
                <span style="font-size:12px;font-weight:700;color:{a['fg']};
                             letter-spacing:.06em;">{level}</span>
            </div>
        </div>
    </div>
    """


def kpi_card(icon: str, label: str, value: str, accent: str | None = None) -> str:
    color = accent or C["primary"]
    return f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                padding:16px 18px;border-top:3px solid {color};
                box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="color:#6b7280;font-size:10px;font-weight:700;
                    text-transform:uppercase;letter-spacing:.09em;margin-bottom:8px;">
            {icon}&ensp;{label}
        </div>
        <div style="color:#111827;font-size:26px;font-weight:700;line-height:1;
                    letter-spacing:-.3px;">{value}</div>
    </div>
    """


def rec_card(icon: str, label: str, value: str, color: str | None = None) -> str:
    color = color or C["primary"]
    return f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                padding:14px 10px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="font-size:22px;line-height:1;margin-bottom:6px;">{icon}</div>
        <div style="color:#6b7280;font-size:9px;font-weight:700;
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">{label}</div>
        <div style="color:{color};font-size:18px;font-weight:700;line-height:1;">{value}</div>
    </div>
    """


def info_card(content: str, accent: str | None = None) -> None:
    color = accent or C["primary"]
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-left:3px solid {color};
                border-radius:10px;padding:14px 18px;font-size:14px;
                color:#374151;line-height:1.75;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        {content}
    </div>
    """, unsafe_allow_html=True)


def weather_widget(weather: dict):
    if not weather:
        return
    icons = {
        "Clear":"☀️","Clouds":"☁️","Rain":"🌧️","Drizzle":"🌦️",
        "Fog":"🌫️","Mist":"🌫️","Haze":"🌫️","Thunderstorm":"⛈️",
        "Snow":"❄️","Smoke":"💨","Dust":"🌪️",
    }
    w_icon  = icons.get(weather.get("condition", ""), "🌤️")
    vis     = weather.get("visibility", 10)
    cond    = weather.get("condition", "")
    vis_color = "#b91c1c" if vis < 2 else "#b45309" if vis < 5 else "#15803d"

    impact = ""
    if "rain" in cond.lower() or "drizzle" in cond.lower():
        impact = ('<span style="background:#fffbeb;border:1px solid #fcd34d;color:#92400e;'
                  'font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;margin-left:8px;">'
                  '⚠ Rain: +20% clearance time</span>')
    elif "fog" in cond.lower() or "mist" in cond.lower() or vis < 3:
        impact = ('<span style="background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;'
                  'font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;margin-left:8px;">'
                  '⚠ Low visibility: secondary incident risk</span>')

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                padding:12px 16px;display:flex;align-items:center;gap:12px;
                margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="font-size:24px;line-height:1;flex-shrink:0;">{w_icon}</div>
        <div style="flex:1;min-width:0;">
            <div style="font-size:10px;font-weight:600;color:#9ca3af;
                        text-transform:uppercase;letter-spacing:.08em;">
                Live Weather · Bengaluru
            </div>
            <div style="color:#111827;font-size:13px;font-weight:600;margin-top:2px;
                        display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
                {cond} · {weather.get('temp',0):.0f}°C {impact}
            </div>
        </div>
        <div style="display:flex;gap:16px;flex-shrink:0;">
            <div style="text-align:center;">
                <div style="color:#9ca3af;font-size:10px;font-weight:600;
                            text-transform:uppercase;margin-bottom:2px;">Visibility</div>
                <div style="color:{vis_color};font-size:13px;font-weight:700;">{vis:.1f} km</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#9ca3af;font-size:10px;font-weight:600;
                            text-transform:uppercase;margin-bottom:2px;">Humidity</div>
                <div style="color:#374151;font-size:13px;font-weight:700;">
                    {weather.get('humidity',0)}%
                </div>
            </div>
            <div style="text-align:center;">
                <div style="color:#9ca3af;font-size:10px;font-weight:600;
                            text-transform:uppercase;margin-bottom:2px;">Wind</div>
                <div style="color:#374151;font-size:13px;font-weight:700;">
                    {weather.get('wind_speed',0):.1f} m/s
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)