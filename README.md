# 🎯 RecoLab — AI Movie Recommendation System

A full-featured movie recommendation app built with **Streamlit** and **scikit-learn**.

## Features
- 🔵 **Content-Based Filtering** — TF-IDF + Cosine Similarity on movie metadata
- 🟠 **Collaborative Filtering** — Item-based CF using user rating matrix
- ⚡ **Hybrid Mode** — Weighted blend of both (tunable with a slider)
- 🎛 Filter by genre and minimum rating
- 📊 Dataset explorer, analytics charts, and algorithm explainer

## Dataset
- 40 movies across all genres
- 200 simulated users with realistic ratings
- ~3,000 ratings total

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.8+ installed

### 2. Install dependencies
```bash
cd recommendation_app
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## How to use
1. **Pick a movie** you like from the sidebar dropdown
2. **Choose an algorithm** (try Hybrid for best results)
3. **Set number of recommendations** with the slider
4. Optionally **filter by genre** or **minimum rating**
5. Hit the recommendations and explore!

---

## Machine Learning Algorithms

### Content-Based Filtering
```
Movie metadata (tags + genre + director)
    → TF-IDF Vectorizer
    → Matrix (40 movies × features)
    → Cosine Similarity Matrix (40×40)
    → Top-N similar movies
```

### Collaborative Filtering (Item-Based)
```
User ratings → User-Item Matrix (40 movies × 200 users)
    → Cosine Similarity between item vectors
    → Top-N most co-rated movies
```

### Hybrid
```
score = α × content_score + (1-α) × collab_score
α is adjustable (default 0.5)
```

---

## Extending the App

To use **real data** (e.g. MovieLens dataset):
1. Download from https://grouplens.org/datasets/movielens/
2. Replace `load_movies()` and `load_ratings()` with CSV readers
3. Map genre/tag columns accordingly

```python
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
```
