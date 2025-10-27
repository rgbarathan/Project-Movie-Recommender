
import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse
import sys
from surprise import accuracy
from collections import defaultdict


def ensure_data_files(data_dir):
    required = ['u.data', 'u.item', 'u.user']
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print(f"Required MovieLens files missing in '{data_dir}': {missing}.\nPlease download MovieLens 100k and place files in that folder.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Hybrid recommender mixing SVD and content similarity')
    parser.add_argument('--svd-weight', type=float, default=0.7, help='Weight for SVD prediction (default: 0.7)')
    parser.add_argument('--content-weight', type=float, default=None, help='Weight for content similarity (default: 1 - svd_weight)')
    parser.add_argument('--normalize-sim', action='store_true', help='Normalize cosine similarity to rating scale [1,5] before mixing')
    parser.add_argument('--learn-sim', action='store_true', help='Learn a linear mapping from cosine similarity -> rating using train data')
    parser.add_argument('--users', nargs='+', default=['2', '10', '30'], help='User IDs to generate recommendations for')
    parser.add_argument('--topn', type=int, default=5, help='Top-N recommendations per user')
    parser.add_argument('--data-dir', default='ml-100k', help='Path to MovieLens data directory')
    parser.add_argument('--no-eval', dest='evaluate', action='store_false', help='Disable evaluation (evaluation runs by default)')
    parser.add_argument('--relevance', type=float, default=4.0, help='Rating threshold to mark an item as relevant for ranking metrics')
    args = parser.parse_args()

    data_dir = args.data_dir
    ensure_data_files(data_dir)

    # Step 1: Load ratings
    ratings = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings['user_id'] = ratings['user_id'].astype(str)
    ratings['movie_id'] = ratings['movie_id'].astype(str)

    # Step 2: Load movie genres
    movie_info = pd.read_csv(os.path.join(data_dir, 'u.item'), sep='|', encoding='ISO-8859-1', header=None)
    movie_info = movie_info[[0, 1] + list(range(5, 24))]  # movie_id, title, genre flags
    movie_info.columns = ['movie_id', 'title'] + [f'genre_{i}' for i in range(19)]
    movie_info['movie_id'] = movie_info['movie_id'].astype(str)

    # Step 3: Load user demographics
    user_info = pd.read_csv(os.path.join(data_dir, 'u.user'), sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    user_info['user_id'] = user_info['user_id'].astype(str)
    movie_features = movie_info.drop(columns=['title'])
    movie_feature_matrix = movie_features.set_index('movie_id')
    user_features = user_info[['user_id', 'age', 'gender', 'occupation']]
    user_features = pd.get_dummies(user_features, columns=['gender', 'occupation'])

    # Step 4: Train SVD model
    reader = Reader(rating_scale=(1, 5))
    movie_feature_cols = movie_feature_matrix.columns.tolist()
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd_model = SVD(random_state=42)
    svd_model.fit(trainset)

    # Step 5: Content-based filtering
    user_movie_matrix = ratings.merge(user_features, on='user_id').merge(movie_features, on='movie_id')
    user_profiles = (
        user_movie_matrix
        .groupby('user_id')[movie_feature_cols]
        .mean()
    )

    # Step 6: Hybrid recommendation
    svd_weight = args.svd_weight
    content_weight = args.content_weight if args.content_weight is not None else max(0.0, 1.0 - svd_weight)
    normalize_sim = args.normalize_sim
    learn_sim = args.learn_sim
    sample_users = args.users
    all_movie_ids = set(ratings['movie_id'].unique())
    recommendations = {}

    # If requested, learn a simple linear regression mapping similarity -> rating using trainset
    sim_regressor = None
    if learn_sim:
        sims_train = []
        ratings_train = []
        # iterate over training ratings and collect (similarity, rating) pairs when user profile exists
        for uid_inner, iid_inner, r_val in trainset.all_ratings():
            raw_u = trainset.to_raw_uid(uid_inner)
            raw_i = trainset.to_raw_iid(iid_inner)
            if raw_u in user_profiles.index and raw_i in movie_feature_matrix.index:
                uvec = user_profiles.loc[raw_u].values.reshape(1, -1)
                ivec = movie_feature_matrix.loc[raw_i].values.reshape(1, -1)
                sim = cosine_similarity(uvec, ivec).flatten()[0]
                sims_train.append(sim)
                ratings_train.append(float(r_val))
        if len(sims_train) >= 10:
            sim_regressor = LinearRegression()
            sim_regressor.fit(np.array(sims_train).reshape(-1, 1), np.array(ratings_train))
            print(f"Learned similarity->rating mapping: coef={sim_regressor.coef_[0]:.4f}, intercept={sim_regressor.intercept_:.4f}")
        else:
            print("Not enough training pairs to learn sim->rating mapping; falling back to heuristic scaling if requested.")

    for user_id in sample_users:
        rated_movies = set(ratings[ratings['user_id'] == user_id]['movie_id'])
        unseen_movies = list(all_movie_ids - rated_movies)
        svd_preds = {mid: svd_model.predict(user_id, mid).est for mid in unseen_movies}
        if user_id in user_profiles.index:
            user_vector = user_profiles.loc[user_id].values.reshape(1, -1)
            movie_vectors = movie_feature_matrix.loc[unseen_movies].values
            similarities = cosine_similarity(user_vector, movie_vectors).flatten()
            if normalize_sim and sim_regressor is None:
                # Heuristic mapping: Map cosine similarity in [-1,1] to rating scale [1,5]
                similarities = 3.0 + 2.0 * similarities
                similarities = np.clip(similarities, 1.0, 5.0)
            elif sim_regressor is not None:
                # Use learned mapping to convert similarity -> rating-like value
                similarities = sim_regressor.predict(similarities.reshape(-1, 1))
                similarities = np.clip(similarities, 1.0, 5.0)
        else:
            similarities = np.zeros(len(unseen_movies))

        # Mix SVD predictions and content similarity (ensure weights sum not strictly required)
        hybrid_scores = [
            (mid, svd_weight * svd_preds[mid] + content_weight * sim)
            for mid, sim in zip(unseen_movies, similarities)
        ]
        top_recs = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:args.topn]
        recommendations[user_id] = top_recs

    # Step 7: Display recommendations
    movie_titles = dict(zip(movie_info['movie_id'], movie_info['title']))
    for user, recs in recommendations.items():
        print(f"\nTop {args.topn} hybrid recommendations for User {user}:")
        for movie_id, score in recs:
            title = movie_titles.get(movie_id, "Unknown Title")
            print(f"  {title} (Movie ID: {movie_id}) - Hybrid Score: {score:.2f}")

    if args.evaluate:
        # --- Evaluation: RMSE/MAE on testset ---
        preds = svd_model.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        mae = accuracy.mae(preds, verbose=False)
        print(f"\nSVD evaluation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Build ground-truth (relevant items per user) from testset using relevance threshold
        gt = defaultdict(set)
        for uid, iid, r in testset:
            if float(r) >= args.relevance:
                gt[uid].add(iid)

        users = list(gt.keys())
        if not users:
            print('No users with relevant items in test set at threshold', args.relevance)
        else:
            users_sample = users[:200]

            # prepare seen items from trainset
            seen = defaultdict(set)
            for uid_inner, iid_inner, _ in trainset.all_ratings():
                raw_u = trainset.to_raw_uid(uid_inner)
                raw_i = trainset.to_raw_iid(iid_inner)
                seen[raw_u].add(raw_i)

            # helper metric funcs
            def precision_at_k(ranked, truth, k):
                if k == 0:
                    return 0.0
                return sum(1 for i in ranked[:k] if i in truth) / k

            def recall_at_k(ranked, truth, k):
                if len(truth) == 0:
                    return 0.0
                return sum(1 for i in ranked[:k] if i in truth) / len(truth)

            def dcg_at_k(ranked, truth, k):
                return sum((1 if ranked[i] in truth else 0) / np.log2(i + 2) for i in range(k))

            def ndcg_at_k(ranked, truth, k):
                dcg = dcg_at_k(ranked, truth, k)
                ideal = sum(1 / np.log2(i + 2) for i in range(min(len(truth), k)))
                return dcg / ideal if ideal > 0 else 0.0

            def hit_rate_at_k(ranked, truth, k):
                return 1.0 if any(i in truth for i in ranked[:k]) else 0.0

            def item_coverage(all_recs, n_items):
                recommended = set()
                for recs in all_recs.values():
                    recommended.update([mid for mid, _ in recs])
                return len(recommended) / float(n_items)

            def intra_list_diversity(recs_by_user, item_feature_matrix):
                diversities = []
                for recs in recs_by_user.values():
                    ids = [mid for mid, _ in recs]
                    if len(ids) < 2:
                        diversities.append(0.0)
                        continue
                    feats = item_feature_matrix.loc[ids].values
                    sims = cosine_similarity(feats)
                    n = len(ids)
                    pairs = n * (n - 1) / 2
                    if pairs == 0:
                        diversities.append(0.0)
                        continue
                    s = 0.0
                    for i in range(n):
                        for j in range(i + 1, n):
                            s += (1.0 - sims[i, j])
                    diversities.append(s / pairs)
                return float(np.mean(diversities)) if diversities else 0.0

            # generate top-k for SVD
            svd_topk = {}
            for user in users_sample:
                unseen = list(all_movie_ids - seen.get(user, set()))
                scores = [(mid, svd_model.predict(user, mid).est) for mid in unseen]
                svd_topk[user] = sorted(scores, key=lambda x: x[1], reverse=True)[:args.topn]

            # generate top-k for hybrid
            hybrid_topk = {}
            for user in users_sample:
                seen_u = seen.get(user, set())
                unseen = list(all_movie_ids - seen_u)
                svd_preds = {mid: svd_model.predict(user, mid).est for mid in unseen}
                if user in user_profiles.index:
                    user_vector = user_profiles.loc[user].values.reshape(1, -1)
                    movie_vectors = movie_feature_matrix.loc[unseen].values
                    sims = cosine_similarity(user_vector, movie_vectors).flatten()
                    if normalize_sim and sim_regressor is None:
                        sims = 3.0 + 2.0 * sims
                        sims = np.clip(sims, 1.0, 5.0)
                    elif sim_regressor is not None:
                        sims = sim_regressor.predict(sims.reshape(-1, 1))
                        sims = np.clip(sims, 1.0, 5.0)
                else:
                    sims = np.zeros(len(unseen))
                hybrid_scores = [(mid, svd_weight * svd_preds[mid] + content_weight * sim) for mid, sim in zip(unseen, sims)]
                hybrid_topk[user] = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:args.topn]

            # compute metrics
            def compute_metrics(topk_map):
                precisions = [precision_at_k([mid for mid, _ in topk_map[u]], gt[u], args.topn) for u in users_sample]
                recalls = [recall_at_k([mid for mid, _ in topk_map[u]], gt[u], args.topn) for u in users_sample]
                ndcgs = [ndcg_at_k([mid for mid, _ in topk_map[u]], gt[u], args.topn) for u in users_sample]
                hits = [hit_rate_at_k([mid for mid, _ in topk_map[u]], gt[u], args.topn) for u in users_sample]
                coverage = item_coverage(topk_map, len(all_movie_ids))
                ild = intra_list_diversity(topk_map, movie_feature_matrix)
                return {'precision': np.mean(precisions), 'recall': np.mean(recalls), 'ndcg': np.mean(ndcgs), 'hitrate': np.mean(hits), 'coverage': coverage, 'ild': ild}

            svd_metrics = compute_metrics(svd_topk)
            hybrid_metrics = compute_metrics(hybrid_topk)

            print('\nEvaluation results (averaged over users sample):')
            print('\nModel: SVD')
            for k, v in svd_metrics.items():
                print(f"  {k}: {v:.4f}")
            print('\nModel: Hybrid')
            for k, v in hybrid_metrics.items():
                print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
