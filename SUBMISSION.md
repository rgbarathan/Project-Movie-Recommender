# Assignment 2: Movie Recommender System

**Student:** Raja Gopal Barathan  
**Date:** October 27, 2025

## GitHub Repository

**Repository URL:** https://github.com/rgbarathan/Project-Movie-Recommender

## Project Overview

This project implements a **hybrid movie recommendation system** that combines:
- **Collaborative Filtering** using SVD (Singular Value Decomposition)
- **Content-Based Filtering** using movie genre features and user demographics

The system is built using the **MovieLens 100k dataset** and evaluates recommendations using multiple metrics including precision, recall, NDCG, hit rate, coverage, and intra-list diversity.

## Repository Contents

- **`Movie recommender.py`** - Main recommendation system implementation
- **`README.md`** - Complete setup and usage instructions
- **`requirements.txt`** - All Python dependencies with pinned versions
- **`tests/test_smoke.py`** - Automated test suite
- **`LICENSE`** - MIT License
- **`.gitignore`** - Git configuration

## How to Run

### Prerequisites
- Python 3.9+
- MovieLens 100k dataset (download from https://grouplens.org/datasets/movielens/100k/)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rgbarathan/Project-Movie-Recommender.git
   cd Project-Movie-Recommender
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and extract MovieLens 100k dataset:**
   - Place the `ml-100k/` folder in the project root directory

5. **Run the recommender:**
   ```bash
   python "Movie recommender.py"
   ```

### Example Usage

**Generate recommendations for specific users:**
```bash
python "Movie recommender.py" --users 1 50 100 --topn 10
```

**Adjust hybrid weights:**
```bash
python "Movie recommender.py" --svd-weight 0.8 --content-weight 0.2
```

**Run without evaluation (faster):**
```bash
python "Movie recommender.py" --no-eval
```

**Learn similarity mapping from training data:**
```bash
python "Movie recommender.py" --learn-sim
```

## Key Features

### 1. Hybrid Recommendation Approach
- Combines SVD predictions with content-based similarity scores
- Configurable weights for balancing collaborative and content-based filtering
- Optional learned mapping from cosine similarity to ratings

### 2. Comprehensive Evaluation
- **Rating Prediction:** RMSE and MAE
- **Ranking Metrics:** Precision@K, Recall@K, NDCG@K, Hit Rate@K
- **Diversity Metrics:** Item Coverage, Intra-List Diversity

### 3. Robust Implementation
- Input validation for required data files
- Proper error handling
- Modular code structure with `main()` function
- Automated tests

## System Architecture

1. **Data Loading:** Ratings, movie metadata, and user demographics
2. **Feature Engineering:** Genre vectors and user profile computation
3. **Model Training:** SVD on user-item rating matrix
4. **Content Similarity:** Cosine similarity between user profiles and movie features
5. **Hybrid Scoring:** Weighted combination of SVD and content scores
6. **Evaluation:** Multiple metrics on held-out test set

## Results

The system achieves the following performance on MovieLens 100k (sample output):

**SVD Model:**
- RMSE: 0.9352
- Precision@5: 0.1390
- NDCG@5: 0.1437
- Hit Rate@5: 0.4250

**Hybrid Model:**
- Precision@5: 0.1620 (↑16.5%)
- NDCG@5: 0.1643 (↑14.3%)
- Hit Rate@5: 0.4750 (↑11.8%)

The hybrid approach improves ranking metrics while maintaining diversity.

## Testing

Run the automated test suite:
```bash
python -m pytest -q
```

## Notes for Grading

- All code is properly documented with inline comments
- The repository follows Python best practices (PEP 8)
- Dependencies are pinned for reproducibility
- The `ml-100k/` dataset is excluded from the repository (add it locally to run)
- Complete setup instructions are provided in `README.md`

## Contact

For any questions about this submission, please contact me via Drexel email.

---

**Repository:** https://github.com/rgbarathan/Project-Movie-Recommender  
**Author:** Raja Gopal Barathan  
**License:** MIT
