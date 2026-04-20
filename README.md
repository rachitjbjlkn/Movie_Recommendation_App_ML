<div align="center">

# 🎬 RachitReel

### *Your Personal AI Cinema Engine*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![TMDB](https://img.shields.io/badge/TMDB-Live%20API-01D277?style=for-the-badge&logo=themoviedatabase&logoColor=white)](https://themoviedb.org)
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

<br>

> **🌌 A full-featured AI movie recommendation system with real-time TMDB data,
> purple glassmorphism UI, animated snow effects, and three powerful ML algorithms.**

<br>

```
🎞️  40 Movies  ·  👥  200 Users  ·  ⭐  ~3,000 Ratings  ·  🤖  3 ML Algorithms  ·  🌐  Live TMDB Data
```

</div>

---

## 🖼️ Preview

```
┌─────────────────────────────────────────────────────────────────┐
│  ✦ Rachit's AI Cinema Engine                                    │
│                                                                 │
│  🎬 RachitReel                                                  │
│  Real-Time · Content-Based · Collaborative · Hybrid            │
│                                                                 │
│  [40 Movies]  [200 Users]  [3,000 Ratings]  [⚡ Hybrid]        │
└─────────────────────────────────────────────────────────────────┘
```

> 🌨️ *Purple animated snow particles fall across the entire interface*
> 🔮 *Deep space dark theme with glassmorphism cards and glow effects*

---

## ✨ Features

### 🤖 AI Recommendation Algorithms

| Algorithm | Method | Best For |
|-----------|--------|----------|
| 🔵 **Content-Based** | TF-IDF + Cosine Similarity on metadata | Movies with similar themes & style |
| 🟠 **Collaborative** | Item-item CF via user rating matrix | Discoveries loved by similar users |
| ⚡ **Hybrid** | Weighted blend — tunable α slider | Best overall accuracy & balance |

### 🎛️ Smart Filtering
- 🎭 **Filter by Genre** — multi-select across all genres in the dataset
- ⭐ **Minimum Rating** — slider from 1.0 → 10.0 to surface only quality picks
- 🔢 **Recommendation Count** — choose 3 to 12 results per query

### 🌐 Real-Time TMDB Integration
- 🖼️ **Movie Poster** — full HD artwork fetched live
- 📖 **Plot Overview** — official synopsis from TMDB
- ⭐ **Live Ratings** — community vote average + vote count
- 🎭 **Top Cast** — top 5 billed actors
- 💰 **Box Office Revenue** — worldwide gross
- ⏱️ **Runtime** — minutes from TMDB
- 📺 **Where to Watch** — streaming providers for your region (IN / US)

### 🎨 Premium UI / UX
- 🌨️ **Animated Snow** — 160 purple particle flakes via `components.html` (persists across re-renders)
- 🔮 **Glassmorphism Cards** — frosted blur panels with purple glow borders
- 🌑 **Deep Space Theme** — `#05000e` background with radial purple gradients
- ✨ **Hover Animations** — lift, glow, and slide effects on all interactive cards
- 📊 **Analytics Dashboard** — ratings distribution, top-rated chart, genre breakdown

---

## ⚡ Quick Start

### 1️⃣ Prerequisites

```
✅ Python 3.8 or newer
✅ pip package manager
✅ Internet connection (for TMDB live data)
```

### 2️⃣ Clone & Install

```bash
# Clone the repository
git clone https://github.com/yourusername/rachitreel.git
cd rachitreel

# Install all dependencies
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

🌐 Opens automatically at **`http://localhost:8501`**

---

## 🔑 TMDB API Key Setup

The app works fully offline with simulated data. Add a free TMDB key to unlock **live posters, cast, ratings, and streaming providers**.

### 🆓 Get Your Free Key

```
1. Visit  →  https://themoviedb.org
2. Sign Up / Log In
3. Go to  Settings → API → Create API Key (v3 auth)
4. Copy your key
```

### 🔒 Option A — Secrets File *(Recommended — key never visible in UI)*

Create `.streamlit/secrets.toml` in your project root:

```toml
# .streamlit/secrets.toml
[tmdb]
api_key = "your_api_key_here"
```

> ✅ The sidebar will show **"🔒 Key loaded from secrets"** — never displayed in plain text.

### 🖊️ Option B — Sidebar Input *(Session only — not persisted)*

Paste your key directly into the sidebar password field. It lives only in the current session and is never written to disk.

### 🛡️ Security Model

```
🔐 Key sent via Authorization: Bearer header  →  never in URL params or logs
🔐 Secrets file loaded at runtime             →  add to .gitignore!
🔐 Sidebar input is type="password"           →  masked, session-scoped only
```

> ⚠️ **Always add `.streamlit/secrets.toml` to your `.gitignore`** before committing.

---

## 🧠 How the ML Works

### 🔵 Content-Based Filtering

```
Movie Metadata
  (tags + genre + director)
        │
        ▼
  TF-IDF Vectorizer
  (40 movies × N features)
        │
        ▼
  Cosine Similarity Matrix
        (40 × 40)
        │
        ▼
  Top-N most similar movies
```

Builds a **textual fingerprint** for each movie. Movies with overlapping keywords, genres, and directors score higher.

---

### 🟠 Collaborative Filtering *(Item-Based)*

```
User Ratings
  (200 users × 40 movies)
        │
        ▼
  User-Item Pivot Matrix
        │
        ▼
  Item-Item Cosine Similarity
  (based on co-rating patterns)
        │
        ▼
  Top-N co-rated movies
```

Discovers movies that **real users rate similarly** — finds hidden gems that share audience overlap, not just surface-level metadata.

---

### ⚡ Hybrid Mode

```
score = α × content_score + (1−α) × collab_score
```

| α value | Behaviour |
|---------|-----------|
| `1.0` | Pure content-based |
| `0.5` | Equal blend *(default)* |
| `0.0` | Pure collaborative |

The **α slider** in the sidebar lets you tune this live.

---

## 📦 Dependencies

```txt
streamlit>=1.32.0     # UI framework + components.html for snow
pandas>=2.0.0         # Data manipulation
numpy>=1.26.0         # Matrix operations
scikit-learn>=1.4.0   # TF-IDF, cosine similarity
requests              # TMDB API calls (stdlib in Python 3.x)
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
rachitreel/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
│
└── 📁 .streamlit/
    └── 🔒 secrets.toml          # (optional) TMDB API key — DO NOT COMMIT
```

---

## 🗂️ Dataset

The app ships with a fully **self-contained synthetic dataset** — no downloads needed.

| 🎬 Movies | 40 titles across all major genres |
|-----------|----------------------------------|
| 👥 Users | 200 simulated users |
| ⭐ Ratings | ~3,000 entries with realistic noise |
| 🎭 Genres | Action, Drama, Sci-Fi, Horror, Animation, Comedy + more |
| 🎬 Directors | Nolan, Tarantino, Scorsese, Spielberg, Fincher + more |

---

## 🔌 Using Real Data *(MovieLens)*

To replace simulated data with the real MovieLens dataset:

```bash
# Download from GroupLens
# https://grouplens.org/datasets/movielens/
```

```python
# In app.py — replace load_movies() and load_ratings():

import pandas as pd

def load_movies():
    movies = pd.read_csv("movies.csv")          # id, title, genres
    movies["genre"] = movies["genres"].str.replace("|", " ")
    movies["tags"]  = movies["title"]            # add tag data if available
    return movies

def load_ratings():
    return pd.read_csv("ratings.csv")            # userId, movieId, rating
```

---

## 📊 Analytics Dashboard

The **📈 Analytics** tab includes three live charts:

```
📊 Ratings Distribution   →  How users rate movies (1–10 histogram)
🏆 Top Rated Movies       →  Horizontal bar chart of highest-rated titles
🎭 Genre Breakdown        →  Most represented genres in the dataset
```

---

## 🚀 Deployment

### ☁️ Streamlit Cloud *(Recommended)*

```
1. Push your repo to GitHub
2. Go to  →  https://share.streamlit.io
3. Connect repo → set main file to app.py
4. Under "Advanced" → add Secrets:
   [tmdb]
   api_key = "your_key_here"
5. Deploy 🎉
```

### 🐳 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t rachitreel .
docker run -p 8501:8501 rachitreel
```

---

## 🛠️ Troubleshooting

| 🔴 Issue | ✅ Fix |
|----------|--------|
| Snow not visible | Make sure you're on the latest `app.py` — snow uses `components.html` |
| TMDB data not loading | Check API key in sidebar; verify internet access |
| "Invalid API key" error | Re-copy key from TMDB Settings → API (v3 auth key, not v4) |
| Movie not found on TMDB | Minor title mismatch — TMDB search is fuzzy but not perfect |
| Blank recommendations | Loosen genre filter or lower minimum rating |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repo
# 2. Create your feature branch
git checkout -b feature/amazing-feature

# 3. Commit your changes
git commit -m "✨ Add amazing feature"

# 4. Push and open a Pull Request
git push origin feature/amazing-feature
```

---

## 📜 License

```
MIT License — © 2025 Rachit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, and distribute freely.
```

---

<div align="center">

**Built with 💜 by Rachit**

*Powered by Streamlit · scikit-learn · TMDB API*

🎬 *Lights. Camera. Recommend.*

</div>
