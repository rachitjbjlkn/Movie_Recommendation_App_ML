import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import requests

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RachitReel · AI Cinema",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Snow + Purple Glassmorphism CSS + JS ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:            #05000e;
    --bg2:           #0c0020;
    --card:          rgba(255,255,255,0.032);
    --card-hover:    rgba(168,85,247,0.07);
    --border:        rgba(168,85,247,0.16);
    --border-bright: rgba(192,132,252,0.5);
    --accent:        #a855f7;
    --accent2:       #c084fc;
    --accent3:       #7c3aed;
    --hot:           #e879f9;
    --text:          #ede0ff;
    --muted:         #7c6b8f;
    --glow:          rgba(168,85,247,0.18);
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent3); border-radius: 4px; }

/* Main bg */
.main, section[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 18% 8%, #1c0040 0%, #09001c 40%, #05000e 100%) !important;
    min-height: 100vh;
}
.block-container { padding-top: 2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12,0,32,0.98) 0%, rgba(5,0,14,0.98) 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(24px);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 12px !important; }
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #fff, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Hero */
.hero {
    background: linear-gradient(135deg,
        rgba(124,58,237,0.17) 0%,
        rgba(168,85,247,0.06) 55%,
        rgba(192,132,252,0.03) 100%);
    border: 1px solid rgba(168,85,247,0.32);
    border-radius: 22px;
    padding: 42px 50px;
    margin-bottom: 28px;
    backdrop-filter: blur(28px);
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 70px rgba(124,58,237,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero::before {
    content: '';
    position: absolute;
    top: -60%; left: -30%;
    width: 120%; height: 220%;
    background: radial-gradient(ellipse at 40% 40%, rgba(168,85,247,0.1) 0%, transparent 60%);
    animation: hero-pulse 6s ease-in-out infinite;
    pointer-events: none;
}
@keyframes hero-pulse {
    0%,100% { opacity:.4; transform:scale(1); }
    50%      { opacity:1;  transform:scale(1.05); }
}
.hero h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 10%, #e0c3ff 45%, #c084fc 75%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 10px 0 8px;
    line-height: 1.1;
    position: relative; z-index: 1;
}
.hero-badge {
    display: inline-block;
    background: rgba(168,85,247,0.14);
    border: 1px solid rgba(168,85,247,0.38);
    color: var(--accent2);
    font-size: 9px;
    padding: 4px 13px;
    border-radius: 20px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    position: relative; z-index: 1;
}
.hero p {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.18em;
    position: relative; z-index: 1;
    margin-top: 8px;
}

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 16px;
    text-align: center;
    backdrop-filter: blur(16px);
    transition: all .3s ease;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), transparent);
}
.metric-card:hover {
    border-color: var(--border-bright);
    box-shadow: 0 0 30px var(--glow);
    background: var(--card-hover);
    transform: translateY(-3px);
}
.metric-val {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.2rem; font-weight: 700;
    color: var(--accent2); line-height: 1;
}
.metric-lbl {
    font-size: 9px; color: var(--muted);
    letter-spacing: 0.22em; text-transform: uppercase; margin-top: 7px;
}

/* Rec cards */
.rec-card {
    background: rgba(255,255,255,0.024);
    border: 1px solid rgba(168,85,247,0.13);
    border-left: 3px solid var(--accent);
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 11px;
    backdrop-filter: blur(14px);
    transition: all .3s ease;
    position: relative; overflow: hidden;
}
.rec-card:hover {
    border-color: rgba(168,85,247,0.42);
    border-left-color: var(--hot);
    box-shadow: 0 0 38px rgba(168,85,247,0.12);
    transform: translateX(5px);
    background: rgba(168,85,247,0.045);
}
.rec-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 700; font-size: 17px; color: #f4eeff;
}
.rec-meta { font-size: 10px; color: var(--muted); margin-top: 3px; }
.rec-score {
    display: inline-block; font-size: 10px;
    padding: 3px 10px; border-radius: 20px; margin-top: 8px;
}
.score-high { background:rgba(168,85,247,.14); border:1px solid rgba(168,85,247,.4); color:#c084fc; }
.score-mid  { background:rgba(124,58,237,.11); border:1px solid rgba(124,58,237,.35); color:#a78bfa; }
.score-low  { background:rgba(90,60,100,.11);  border:1px solid rgba(90,60,100,.28);  color:var(--muted); }

/* Tags */
.tag {
    display: inline-block;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.28);
    color: #b09dcc; font-size: 9px;
    padding: 3px 8px; border-radius: 6px; margin: 2px;
}

/* Section titles */
.section-title {
    font-size: 9px; font-weight: 500;
    letter-spacing: 0.35em; text-transform: uppercase;
    color: var(--accent); margin: 28px 0 14px;
    display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg, rgba(168,85,247,.3), transparent);
}

/* Real-time TMDB card */
.tmdb-card {
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(168,85,247,0.22);
    border-radius: 18px;
    padding: 24px 28px;
    backdrop-filter: blur(20px);
    margin-top: 14px;
    position: relative; overflow: hidden;
}
.tmdb-card::before {
    content:'◈ LIVE DATA';
    position: absolute; top:13px; right:18px;
    font-size: 8px; letter-spacing:.2em;
    color: var(--accent); opacity:.6;
    animation: blink 2.5s ease-in-out infinite;
}
@keyframes blink {
    0%,100%{ opacity:.6; } 50%{ opacity:.15; }
}
.tmdb-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 16px; font-weight: 700;
    color: var(--accent2); margin-bottom: 14px;
}
.tmdb-overview { font-size: 12px; color: #cbb8e8; line-height: 1.8; }
.tmdb-pill {
    display: inline-block;
    background: rgba(168,85,247,0.12);
    border: 1px solid rgba(168,85,247,0.3);
    color: var(--accent2); font-size: 10px;
    padding: 4px 11px; border-radius: 20px; margin: 3px;
}
.tmdb-poster {
    border-radius: 12px;
    border: 1px solid rgba(168,85,247,0.25);
    box-shadow: 0 0 40px rgba(124,58,237,0.25);
    width: 100%;
}
.provider-badge {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(168,85,247,0.2);
    color: #d0b8ff; font-size: 10px;
    padding: 4px 11px; border-radius: 8px; margin: 3px;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
div[data-testid="stSlider"] > div { color: var(--accent) !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent2) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 12px !important; }
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: #fff !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important;
    padding: 8px 22px !important;
    box-shadow: 0 0 20px rgba(168,85,247,0.3) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 32px rgba(168,85,247,0.5) !important;
    transform: translateY(-1px) !important;
}
</style>

""", unsafe_allow_html=True)

# ── Snow animation via components.html so JS executes reliably ────────────────
components.html("""
<style>
  #snow-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 999999;
  }
</style>
<canvas id="snow-canvas"></canvas>
<script>
(function () {
    const canvas = document.getElementById('snow-canvas');
    const ctx    = canvas.getContext('2d');
    let W, H, flakes = [];

    function resize() {
        W = canvas.width  = window.parent.innerWidth  || window.innerWidth;
        H = canvas.height = window.parent.innerHeight || window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);
    try { window.parent.addEventListener('resize', resize); } catch(e) {}

    function Flake() { this.reset(true); }
    Flake.prototype.reset = function (init) {
        this.x       = Math.random() * W;
        this.y       = init ? Math.random() * H : -10;
        this.r       = Math.random() * 2.8 + 0.8;
        this.speed   = Math.random() * 1.0 + 0.3;
        this.drift   = (Math.random() - 0.5) * 0.6;
        this.opacity = Math.random() * 0.55 + 0.15;
        this.blur    = Math.random() > 0.6;
        const hue = Math.floor(Math.random() * 40) + 260;
        const sat = Math.floor(Math.random() * 30) + 60;
        const lit = Math.floor(Math.random() * 25) + 70;
        this.color = "hsl(" + hue + "," + sat + "%," + lit + "%)";
    };

    for (let i = 0; i < 140; i++) { flakes.push(new Flake()); }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        for (const f of flakes) {
            ctx.save();
            ctx.globalAlpha = f.opacity;
            if (f.blur) ctx.filter = 'blur(1.2px)';
            const g = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.r * 2.2);
            g.addColorStop(0, f.color);
            g.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.beginPath();
            ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
            ctx.fillStyle = g;
            ctx.fill();
            ctx.restore();
            f.y += f.speed;
            f.x += f.drift;
            if (f.y > H + 12 || f.x < -12 || f.x > W + 12) f.reset(false);
        }
        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
""", height=0, scrolling=False)


# ── TMDB real-time helper ─────────────────────────────────────────────────────
TMDB_BASE = "https://api.themoviedb.org/3"
IMG_BASE  = "https://image.tmdb.org/t/p/w500"

def tmdb_search(title, api_key):
    try:
        r = requests.get(f"{TMDB_BASE}/search/movie",
                         params={"api_key": api_key, "query": title}, timeout=5)
        results = r.json().get("results", [])
        return results[0] if results else None
    except:
        return None

def tmdb_details(movie_id, api_key):
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{movie_id}",
                         params={"api_key": api_key}, timeout=5)
        return r.json()
    except:
        return {}

def tmdb_providers(movie_id, api_key):
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{movie_id}/watch/providers",
                         params={"api_key": api_key}, timeout=5)
        data = r.json().get("results", {})
        # try IN first, then US
        region = data.get("IN") or data.get("US") or {}
        flat = region.get("flatrate", [])
        return [p["provider_name"] for p in flat[:6]]
    except:
        return []

def tmdb_credits(movie_id, api_key):
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{movie_id}/credits",
                         params={"api_key": api_key}, timeout=5)
        cast = r.json().get("cast", [])[:5]
        return [c["name"] for c in cast]
    except:
        return []


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_movies():
    movies = pd.DataFrame({
        "id": range(1, 41),
        "title": [
            "Inception", "The Dark Knight", "Interstellar", "The Matrix",
            "Parasite", "Pulp Fiction", "The Godfather", "Fight Club",
            "Forrest Gump", "The Shawshank Redemption", "Goodfellas",
            "Schindler's List", "The Silence of the Lambs", "The Lion King",
            "Titanic", "Avatar", "Avengers: Endgame", "Iron Man",
            "Spider-Man: No Way Home", "Black Panther", "The Social Network",
            "Whiplash", "La La Land", "1917", "Dunkirk",
            "Get Out", "Hereditary", "A Quiet Place", "The Conjuring", "It",
            "Toy Story", "Finding Nemo", "Up", "WALL-E", "Shrek",
            "The Grand Budapest Hotel", "Moonlight", "Birdman",
            "Mad Max: Fury Road", "John Wick"
        ],
        "genre": [
            "Sci-Fi Thriller", "Action Thriller", "Sci-Fi Drama", "Sci-Fi Action",
            "Thriller Drama", "Crime Drama", "Crime Drama", "Drama Thriller",
            "Drama Romance", "Drama", "Crime Drama",
            "Historical Drama", "Thriller Crime", "Animation Family",
            "Romance Drama", "Sci-Fi Action", "Action Superhero", "Action Superhero",
            "Action Superhero", "Action Superhero", "Drama Biography",
            "Drama Music", "Romance Musical", "War Drama", "War Thriller",
            "Horror Thriller", "Horror Drama", "Horror Thriller", "Horror", "Horror",
            "Animation Comedy", "Animation Adventure", "Animation Drama",
            "Animation Sci-Fi", "Animation Comedy",
            "Comedy Drama", "Drama", "Drama Comedy",
            "Action Sci-Fi", "Action Thriller"
        ],
        "director": [
            "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Wachowski Sisters",
            "Bong Joon-ho", "Quentin Tarantino", "Francis Ford Coppola", "David Fincher",
            "Robert Zemeckis", "Frank Darabont", "Martin Scorsese",
            "Steven Spielberg", "Jonathan Demme", "Roger Allers",
            "James Cameron", "James Cameron", "Anthony & Joe Russo", "Jon Favreau",
            "Jon Watts", "Ryan Coogler", "David Fincher",
            "Damien Chazelle", "Damien Chazelle", "Sam Mendes", "Christopher Nolan",
            "Jordan Peele", "Ari Aster", "John Krasinski", "James Wan", "Andy Muschietti",
            "John Lasseter", "Andrew Stanton", "Pete Docter", "Andrew Stanton", "Andrew Adamson",
            "Wes Anderson", "Barry Jenkins", "Alejandro González Iñárritu",
            "George Miller", "Chad Stahelski"
        ],
        "year": [
            2010, 2008, 2014, 1999, 2019, 1994, 1972, 1999,
            1994, 1994, 1990, 1993, 1991, 1994, 1997, 2009, 2019, 2008,
            2021, 2018, 2010, 2014, 2016, 2019, 2017,
            2017, 2018, 2018, 2013, 2017,
            1995, 2003, 2009, 2008, 2001,
            2014, 2016, 2014, 2015, 2014
        ],
        "rating": [
            8.8, 9.0, 8.6, 8.7, 8.5, 8.9, 9.2, 8.8,
            8.8, 9.3, 8.7, 9.0, 8.6, 8.5, 7.9, 7.9, 8.4, 7.9,
            8.3, 7.3, 7.7, 8.5, 8.0, 8.3, 7.9,
            7.7, 7.3, 7.5, 7.5, 6.9,
            8.3, 8.2, 8.2, 8.4, 7.9,
            8.1, 7.4, 7.7, 8.1, 7.4
        ],
        "tags": [
            "dreams reality heist mind-bending", "batman villain chaos crime",
            "space time wormhole love", "simulation robots philosophy",
            "class inequality south-korea suspense", "nonlinear hitmen crime LA",
            "mafia family power Sicily", "anti-consumerism alter-ego dark",
            "disability love Vietnam history", "hope prison friendship",
            "mob gangster NYC real-life", "holocaust jewish war survival",
            "fbi serial-killer cannibalism", "africa pride circle-of-life",
            "ship iceberg love class", "aliens pandora environment CGI",
            "infinity-stones heroes sacrifice", "billionaire suit iron technology",
            "multiverse teenager friendly-neighborhood", "africa king vibranium",
            "facebook startup betrayal", "jazz drums perfectionism obsession",
            "love music jazz dreams LA", "WW1 continuous-shot trenches",
            "WW2 evacuation beach real-time", "racism social-horror sunken-place",
            "grief family demon occult", "monsters silence survival farm",
            "paranormal cases demonology", "clown sewer kids Pennywise",
            "toys friendship cowboy buzz", "fish ocean adventure father-son",
            "adventure balloons old couple love", "robot earth love plant",
            "ogre fairytale comedy layers", "hotel quirky 1930s comedy",
            "identity gay coming-of-age Miami", "actor ego broadway realism",
            "post-apocalyptic cars feminist desert", "hitman assassin dogs NYC"
        ]
    })
    return movies


@st.cache_data
def load_ratings(movies):
    np.random.seed(42)
    n_users = 200
    users = [f"user_{i:03d}" for i in range(1, n_users + 1)]
    ratings_list = []
    for user in users:
        n_rated = np.random.randint(8, 25)
        movie_ids = np.random.choice(movies["id"].values, size=n_rated, replace=False)
        for mid in movie_ids:
            base = movies.loc[movies["id"] == mid, "rating"].values[0]
            noise = np.random.normal(0, 0.7)
            r = float(np.clip(round(base + noise, 1), 1.0, 10.0))
            ratings_list.append({"user_id": user, "movie_id": mid, "rating": r})
    return pd.DataFrame(ratings_list)


@st.cache_resource
def build_content_model(movies):
    m = movies.copy()
    m["soup"] = (
        m["tags"] + " " +
        m["genre"].str.replace(" ", "_") + " " +
        m["director"].str.replace(" ", "_")
    )
    tfidf  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(m["soup"])
    sim    = cosine_similarity(matrix, matrix)
    return sim, m


@st.cache_resource
def build_collab_model(ratings, movies):
    pivot = ratings.pivot_table(index="movie_id", columns="user_id", values="rating").fillna(0)
    pivot = pivot.reindex(movies["id"].values, fill_value=0)
    sim   = cosine_similarity(pivot)
    return sim, pivot


# ── Recommendation functions ──────────────────────────────────────────────────
def content_recommend(title, movies, sim_matrix, n=8):
    idx_s = movies.index[movies["title"] == title]
    if len(idx_s) == 0:
        return pd.DataFrame()
    idx    = idx_s[0]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    result = movies.iloc[[s[0] for s in scores]][["title","genre","director","year","rating"]].copy()
    result["similarity"] = [round(s[1]*100, 1) for s in scores]
    return result.reset_index(drop=True)


def collab_recommend(title, movies, sim_matrix, pivot, n=8):
    title_to_id = dict(zip(movies["title"], movies["id"]))
    if title not in title_to_id:
        return pd.DataFrame()
    mid     = title_to_id[title]
    idx_arr = np.where(movies["id"].values == mid)[0]
    if len(idx_arr) == 0:
        return pd.DataFrame()
    idx    = idx_arr[0]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    rec_ids = movies["id"].values[[s[0] for s in scores]]
    result  = movies[movies["id"].isin(rec_ids)][["title","genre","director","year","rating"]].copy()
    id_to_sim = dict(zip(rec_ids, [round(s[1]*100,1) for s in scores]))
    result["similarity"] = result.apply(
        lambda r: id_to_sim.get(movies.loc[movies["title"]==r["title"],"id"].values[0], 0), axis=1)
    return result.reset_index(drop=True)


def hybrid_recommend(title, movies, content_sim, collab_sim, pivot, n=8, alpha=0.5):
    title_to_id = dict(zip(movies["title"], movies["id"]))
    if title not in title_to_id:
        return pd.DataFrame()
    mid   = title_to_id[title]
    idx_c = movies.index[movies["title"] == title][0]
    idx_f = np.where(movies["id"].values == mid)[0][0]

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    hybrid          = alpha * norm(np.array(content_sim[idx_c])) + (1-alpha) * norm(np.array(collab_sim[idx_f]))
    hybrid[idx_c]   = -1
    top_idx         = np.argsort(hybrid)[::-1][:n]
    result          = movies.iloc[top_idx][["title","genre","director","year","rating"]].copy()
    result["similarity"] = [round(float(hybrid[i])*100, 1) for i in top_idx]
    return result.reset_index(drop=True)


# ── Load data ─────────────────────────────────────────────────────────────────
movies      = load_movies()
ratings     = load_ratings(movies)
content_sim, _ = build_content_model(movies)
collab_sim, pivot = build_collab_model(ratings, movies)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 RachitReel")
    st.markdown("---")

    st.markdown('<div style="font-size:9px;letter-spacing:.3em;color:#a855f7;text-transform:uppercase;margin-bottom:6px;">Algorithm</div>', unsafe_allow_html=True)
    mode = st.radio("", ["🔵 Content-Based", "🟠 Collaborative", "⚡ Hybrid"], index=2, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:9px;letter-spacing:.3em;color:#a855f7;text-transform:uppercase;margin-bottom:6px;">Pick a Movie</div>', unsafe_allow_html=True)
    selected_movie = st.selectbox("", options=sorted(movies["title"].tolist()),
                                  index=movies["title"].tolist().index("Inception"),
                                  label_visibility="collapsed")

    n_recs = st.slider("Recommendations", 3, 12, 6)

    if mode == "⚡ Hybrid":
        alpha = st.slider("Content weight ↔ Collab", 0.0, 1.0, 0.5, 0.1)
    else:
        alpha = 0.5

    st.markdown("---")
    st.markdown('<div style="font-size:9px;letter-spacing:.3em;color:#a855f7;text-transform:uppercase;margin-bottom:6px;">Filter by Genre</div>', unsafe_allow_html=True)
    all_genres   = sorted(set(g for genres in movies["genre"].str.split() for g in genres))
    genre_filter = st.multiselect("", all_genres, default=[], label_visibility="collapsed")

    st.markdown('<div style="font-size:9px;letter-spacing:.3em;color:#a855f7;text-transform:uppercase;margin-bottom:6px;">Min Rating</div>', unsafe_allow_html=True)
    min_rating = st.slider("", 1.0, 10.0, 7.0, 0.5, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:9px;letter-spacing:.3em;color:#a855f7;text-transform:uppercase;margin-bottom:8px;">TMDB API Key</div>', unsafe_allow_html=True)

    # Try loading from Streamlit secrets first (confidential — never shown)
    _secrets_key = st.secrets.get("TMDB_API_KEY", "") if hasattr(st, "secrets") else ""
    if _secrets_key:
        tmdb_key = _secrets_key
        st.markdown('<div style="font-size:10px;color:#4ade80;margin-bottom:8px;">🔒 API key loaded securely</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:10px;color:#7c6b8f;margin-bottom:8px;">Free key at themoviedb.org · stored securely</div>', unsafe_allow_html=True)
        tmdb_key = st.text_input(
            "",
            type="password",
            placeholder="Paste your TMDB key…",
            label_visibility="collapsed",
            help="Your key is never stored or logged — it stays in your browser session only."
        )


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ Rachit's AI Cinema Engine</div>
  <h1>🎬 RachitReel</h1>
  <p>Real-Time · Content-Based · Collaborative Filtering · Hybrid Intelligence</p>
</div>
""", unsafe_allow_html=True)


# ── METRICS ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
algo_label = mode.split(" ", 1)[1]
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [len(movies), ratings["user_id"].nunique(), f"{len(ratings):,}", algo_label],
    ["Movies", "Users", "Ratings", "Algorithm"]
):
    with col:
        font = "1rem;padding-top:10px" if lbl == "Algorithm" else "2.2rem"
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:{font}">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── SELECTED MOVIE ────────────────────────────────────────────────────────────
sel = movies[movies["title"] == selected_movie].iloc[0]
st.markdown('<div class="section-title">// selected movie</div>', unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in sel["tags"].split()[:8])
    st.markdown(f"""
    <div class="rec-card" style="border-left-color:#e879f9;">
      <div class="rec-title">🎬 {sel['title']} ({sel['year']})</div>
      <div class="rec-meta">{sel['genre']} · {sel['director']}</div>
      <span class="rec-score score-high">⭐ {sel['rating']} / 10</span>
      <div style="margin-top:10px">{tags_html}</div>
    </div>
    """, unsafe_allow_html=True)

with right:
    ur  = ratings[ratings["movie_id"] == sel["id"]]
    avg = ur["rating"].mean() if len(ur) else sel["rating"]
    st.metric("User Avg Rating",   f"{avg:.2f} / 10")
    st.metric("Total User Ratings", len(ur))


# ── TMDB REAL-TIME PANEL ──────────────────────────────────────────────────────
if tmdb_key:
    st.markdown('<div class="section-title">// real-time data · TMDB</div>', unsafe_allow_html=True)
    with st.spinner("Fetching live movie data…"):
        hit = tmdb_search(sel["title"], tmdb_key)

    if hit:
        mid      = hit["id"]
        details  = tmdb_details(mid, tmdb_key)
        cast     = tmdb_credits(mid, tmdb_key)
        providers = tmdb_providers(mid, tmdb_key)
        poster   = IMG_BASE + hit["poster_path"] if hit.get("poster_path") else None
        overview = hit.get("overview", "No overview available.")
        tmdb_rating = hit.get("vote_average", "N/A")
        vote_count  = hit.get("vote_count", 0)
        genres_live = [g["name"] for g in details.get("genres", [])]
        runtime     = details.get("runtime", "N/A")
        revenue     = details.get("revenue", 0)
        tagline     = details.get("tagline", "")

        pc, tc = st.columns([1, 3])
        with pc:
            if poster:
                st.markdown(f'<img src="{poster}" class="tmdb-poster">', unsafe_allow_html=True)
        with tc:
            genres_html   = "".join(f'<span class="tmdb-pill">{g}</span>' for g in genres_live)
            cast_html     = "".join(f'<span class="provider-badge">🎭 {c}</span>' for c in cast)
            provider_html = "".join(f'<span class="provider-badge">▶ {p}</span>' for p in providers) if providers else '<span style="color:#7c6b8f;font-size:11px;">Not available in your region</span>'
            rev_str       = f"${revenue/1e6:.0f}M" if revenue else "N/A"

            st.markdown(f"""
            <div class="tmdb-card">
                <div class="tmdb-title">🌐 {sel['title']} — Live Data</div>
                <div style="margin-bottom:12px">{genres_html}</div>
                {f'<div style="font-size:11px;color:#9b8fc4;font-style:italic;margin-bottom:12px;">"{tagline}"</div>' if tagline else ''}
                <div class="tmdb-overview">{overview}</div>
                <div style="margin-top:16px;display:flex;gap:20px;flex-wrap:wrap;">
                    <div><div style="font-size:9px;letter-spacing:.2em;color:#7c6b8f;text-transform:uppercase">TMDB Rating</div>
                         <div style="font-size:1.4rem;font-family:'Cormorant Garamond',serif;color:#c084fc;font-weight:700">⭐ {tmdb_rating}</div>
                         <div style="font-size:9px;color:#7c6b8f">{vote_count:,} votes</div></div>
                    <div><div style="font-size:9px;letter-spacing:.2em;color:#7c6b8f;text-transform:uppercase">Runtime</div>
                         <div style="font-size:1.4rem;font-family:'Cormorant Garamond',serif;color:#c084fc;font-weight:700">{runtime} min</div></div>
                    <div><div style="font-size:9px;letter-spacing:.2em;color:#7c6b8f;text-transform:uppercase">Box Office</div>
                         <div style="font-size:1.4rem;font-family:'Cormorant Garamond',serif;color:#c084fc;font-weight:700">{rev_str}</div></div>
                </div>
                <div style="margin-top:16px">
                    <div style="font-size:9px;letter-spacing:.2em;color:#7c6b8f;text-transform:uppercase;margin-bottom:6px;">Top Cast</div>
                    {cast_html}
                </div>
                <div style="margin-top:14px">
                    <div style="font-size:9px;letter-spacing:.2em;color:#7c6b8f;text-transform:uppercase;margin-bottom:6px;">Where to Watch</div>
                    {provider_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Movie not found on TMDB. Try a different title.")
else:
    st.markdown("""
    <div style="background:rgba(168,85,247,0.06);border:1px dashed rgba(168,85,247,0.25);
    border-radius:14px;padding:18px 22px;margin-top:10px;font-size:12px;color:#7c6b8f;text-align:center;">
        🔑 Paste your free <strong style="color:#a855f7">TMDB API key</strong> in the sidebar to unlock
        live ratings, posters, cast, box office & streaming providers.
        <br><span style="font-size:10px;opacity:.7">Get it free at themoviedb.org → Settings → API</span>
    </div>
    """, unsafe_allow_html=True)


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
if mode == "🔵 Content-Based":
    recs = content_recommend(selected_movie, movies, content_sim, n_recs + 5)
elif mode == "🟠 Collaborative":
    recs = collab_recommend(selected_movie, movies, collab_sim, pivot, n_recs + 5)
else:
    recs = hybrid_recommend(selected_movie, movies, content_sim, collab_sim, pivot, n_recs + 5, alpha)

if genre_filter:
    recs = recs[recs["genre"].apply(lambda g: any(gf in g for gf in genre_filter))]
recs = recs[recs["rating"] >= min_rating].head(n_recs)

st.markdown('<div class="section-title">// recommendations</div>', unsafe_allow_html=True)

if recs.empty:
    st.warning("No recommendations match your filters — try loosening the genre or rating filters.")
else:
    for _, row in recs.iterrows():
        s = row["similarity"]
        score_cls = "score-high" if s >= 60 else "score-mid" if s >= 40 else "score-low"
        st.markdown(f"""
        <div class="rec-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="rec-title">🎬 {row['title']} ({row['year']})</div>
              <div class="rec-meta">{row['genre']} · {row['director']}</div>
            </div>
            <div style="text-align:right;">
              <span class="rec-score {score_cls}">{s}% match</span>
              <div class="rec-meta" style="margin-top:6px;">⭐ {row['rating']}/10</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Analytics", "🔬 How it Works"])

with tab1:
    search  = st.text_input("Search movies", placeholder="e.g. Nolan · thriller · 2019…")
    display = movies.copy()
    if search:
        mask    = display.apply(lambda r: search.lower() in str(r).lower(), axis=1)
        display = display[mask]
    st.dataframe(display[["title","genre","director","year","rating"]].reset_index(drop=True),
                 use_container_width=True, height=300)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ratings Distribution**")
        st.bar_chart(ratings["rating"].round(0).value_counts().sort_index(), color="#a855f7")
    with c2:
        st.markdown("**Top Rated Movies**")
        st.bar_chart(movies.nlargest(10,"rating")[["title","rating"]].set_index("title"), color="#e879f9")
    st.markdown("**Genre Breakdown**")
    st.bar_chart(movies["genre"].str.split().explode().value_counts().head(15), color="#7c3aed")

with tab3:
    st.markdown("""
<div style="display:flex;flex-direction:column;gap:16px;margin-top:8px;">

<div style="background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.22);border-left:3px solid #7c3aed;border-radius:14px;padding:20px 24px;">
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;font-weight:700;color:#c084fc;margin-bottom:8px;">🔵 Content-Based Filtering</div>
  <div style="font-size:12px;color:#cbb8e8;line-height:1.8;">
    Uses <strong style="color:#e0c3ff">TF-IDF vectorization</strong> on movie metadata (tags, genre, director).
    Computes <strong style="color:#e0c3ff">cosine similarity</strong> between movies to find the most similar ones.<br>
    <span style="color:#7c6b8f;font-size:11px;">✦ Best for: finding movies with similar themes and style.</span>
  </div>
</div>

<div style="background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.22);border-left:3px solid #a855f7;border-radius:14px;padding:20px 24px;">
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;font-weight:700;color:#c084fc;margin-bottom:8px;">🟠 Collaborative Filtering (Item-Based)</div>
  <div style="font-size:12px;color:#cbb8e8;line-height:1.8;">
    Builds a <strong style="color:#e0c3ff">user-item rating matrix</strong> and computes <strong style="color:#e0c3ff">item-item cosine similarity</strong>
    based on how users rated movies similarly.<br>
    <span style="color:#7c6b8f;font-size:11px;">✦ Best for: discovering movies loved by users with similar taste.</span>
  </div>
</div>

<div style="background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.22);border-left:3px solid #e879f9;border-radius:14px;padding:20px 24px;">
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;font-weight:700;color:#c084fc;margin-bottom:8px;">⚡ Hybrid</div>
  <div style="font-size:12px;color:#cbb8e8;line-height:1.8;">
    Combines both methods with a weighted average:<br>
    <code style="background:rgba(124,58,237,0.18);padding:3px 8px;border-radius:6px;font-size:11px;color:#d0b8ff;">score = α × content_score + (1−α) × collab_score</code><br><br>
    The <strong style="color:#e0c3ff">α slider</strong> lets you tune how much each method influences the result.<br>
    <span style="color:#7c6b8f;font-size:11px;">✦ Best for: the most balanced and accurate recommendations.</span>
  </div>
</div>

<div style="background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.22);border-left:3px solid #c084fc;border-radius:14px;padding:20px 24px;">
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;font-weight:700;color:#c084fc;margin-bottom:8px;">🌐 Real-Time TMDB Data</div>
  <div style="font-size:12px;color:#cbb8e8;line-height:1.8;">
    Live poster, overview, cast, box office revenue, runtime and streaming providers
    pulled directly from <strong style="color:#e0c3ff">The Movie Database (TMDB)</strong> API on every selection.<br>
    <span style="color:#7c6b8f;font-size:11px;">✦ Your API key is never stored — it lives in your session only.</span>
  </div>
</div>

</div>
""", unsafe_allow_html=True)
