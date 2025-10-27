Movie Recommender

This script (`Movie recommender.py`) uses the MovieLens 100k dataset (folder `ml-100k/`) to build a hybrid recommender combining SVD (matrix-factorization) and content-based similarity (movie genres + simple user profiles).

Quick setup (macOS / zsh):

1) Create / activate a virtualenv (recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2) Install dependencies:

   pip install -r requirements.txt

3) Run the recommender:

   python "Movie recommender.py"

Flags:
  --svd-weight     Weight for SVD predictions (default 0.7)
  --content-weight Weight for content similarity (default 1 - svd_weight)
  --normalize-sim  Normalize cosine similarity to rating range [1,5]
  --learn-sim      Learn linear mapping from similarity->rating using training data
  --users          List of user IDs to recommend for (default '1 50 100')
  --topn           Number of recommendations per user (default 5)
  --no-eval        Disable evaluation

Notes:
- If you encounter a NumPy / binary extension error, ensure numpy is <2 (the `scikit-surprise` package may require NumPy 1.x builds). The `requirements.txt` pins `numpy<2` to avoid that.
- Dataset files from MovieLens (u.data, u.item, etc.) must be in the `ml-100k/` folder adjacent to the script.
