
import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import dump as svd_dump
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse
import sys
import random
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
    parser.add_argument('--content-weight', type=float, default=None, help='Weight for content similarity (default: 1 - svd_weight - category_weight)')
    parser.add_argument('--normalize-sim', action='store_true', help='Normalize cosine similarity to rating scale [1,5] before mixing')
    parser.add_argument('--learn-sim', action='store_true', help='Learn a linear mapping from cosine similarity -> rating using train data')
    parser.add_argument('--users', nargs='+', default=None, help='User IDs to generate recommendations for (default: 3 random users)')
    parser.add_argument('--topn', type=int, default=3, help='Top-N recommendations per user')
    parser.add_argument('--data-dir', default='ml-100k', help='Path to MovieLens data directory')
    parser.add_argument('--no-eval', dest='evaluate', action='store_false', help='Disable evaluation (evaluation runs by default)')
    parser.add_argument('--relevance', type=float, default=4.0, help='Rating threshold to mark an item as relevant for ranking metrics')
    # Demographic/category blending
    parser.add_argument('--category-weight', type=float, default=0.2, help='Weight for demographic category similarity (default: 0.2)')
    parser.add_argument('--category-dims', type=str, default='age,gender,occupation', help='Comma-separated list of category dimensions to use: age,gender,occupation (default: all)')
    # Category top-genre boost
    parser.add_argument('--cat-genre-boost', type=float, default=0.0, help='Additional boost for items matching top category genres (default: 0.0 disabled)')
    parser.add_argument('--cat-topk-genres', type=int, default=3, help='K for top category genres used in printing and boosting (default: 3)')
    parser.add_argument('--cat-genre-mode', type=str, default='combined', choices=['combined', 'union'], help='How to select top genres across dimensions: combined (avg profile) or union of per-dim top-K (default: combined)')
    # Model tuning and persistence
    parser.add_argument('--tune-svd', action='store_true', help='Run GridSearchCV to tune SVD hyperparameters')
    parser.add_argument('--save-model', action='store_true', help='Save trained SVD model artifact to disk')
    parser.add_argument('--load-model', action='store_true', help='Load SVD model artifact from disk (skips training if found)')
    parser.add_argument('--model-dir', default='artifacts', help='Directory to save/load model artifacts (default: artifacts)')
    parser.add_argument('--model-file', default='svd_model.dump', help='Filename for the saved SVD model artifact')
    # Recommendation display formatting
    parser.add_argument('--recs-table', action='store_true', help='Display recommendations in a tabular view (default: enabled)')
    parser.add_argument('--no-recs-table', dest='recs_table', action='store_false', help='Disable tabular recommendations view')
    parser.add_argument('--recs-show-components', action='store_true', help='Include SVD/Content/Category/Boost columns in the table')
    parser.add_argument('--recs-max-genres', type=int, default=2, help='Max number of matched genres to display per item (default: 2)')
    parser.add_argument('--recs-show-confidence', action='store_true', help='Show confidence hint for each recommendation (e.g., [####-] 4.2)')
    parser.add_argument('--badges-demographics', action='store_true', help='Print compact demographic badges like [25-34] [M] [programmer] (default: enabled)')
    parser.add_argument('--show-per-dim-top-genres', action='store_true', help='Also print per-dimension (age, gender, occupation) top genre lines')
    parser.add_argument('--show-matched-genres', action='store_true', help='Always compute and display Matched Genres even when boost is disabled (default: enabled)')
    parser.add_argument('--no-show-matched-genres', dest='show_matched_genres', action='store_false', help='Disable Matched Genres display and computation')
    # Top-N control
    parser.add_argument('--only-top3', action='store_true', help='Force Top-N to 3 regardless of other flags')
    # Enable table view by default; can be turned off with --no-recs-table
    parser.set_defaults(recs_table=True)
    # Make concise demographic badges the default
    parser.set_defaults(badges_demographics=True)
    # Enable matched genres by default
    parser.set_defaults(show_matched_genres=True)
    args = parser.parse_args()
    if getattr(args, 'only_top3', False):
        args.topn = 3

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

    # Try to load genre names for nicer printing (optional)
    genre_names = [f'genre_{i}' for i in range(19)]
    genre_path = os.path.join(data_dir, 'u.genre')
    if os.path.exists(genre_path):
        try:
            # file format: name|id, but may include trailing empty line
            gdf = pd.read_csv(genre_path, sep='|', header=None, names=['name', 'id'])
            gdf = gdf.dropna()
            gdf['id'] = gdf['id'].astype(int)
            gdf = gdf.sort_values('id')
            genre_names = gdf['name'].tolist()
        except Exception:
            pass

    # Step 3: Load user demographics
    user_info = pd.read_csv(os.path.join(data_dir, 'u.user'), sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    user_info['user_id'] = user_info['user_id'].astype(str)
    # Derive age groups
    def age_to_group(a):
        try:
            a = int(a)
        except Exception:
            return 'unknown'
        if a < 18:
            return '<18'
        elif a <= 24:
            return '18-24'
        elif a <= 34:
            return '25-34'
        elif a <= 44:
            return '35-44'
        elif a <= 54:
            return '45-54'
        else:
            return '55+'
    user_info['age_group'] = user_info['age'].apply(age_to_group)
    # Quick lookup dict for printing
    user_demo = user_info.set_index('user_id').to_dict(orient='index')
    movie_features = movie_info.drop(columns=['title'])
    movie_feature_matrix = movie_features.set_index('movie_id')
    user_features = user_info[['user_id', 'age', 'gender', 'occupation']]
    user_features = pd.get_dummies(user_features, columns=['gender', 'occupation'])

    # Step 4: Train SVD model
    reader = Reader(rating_scale=(1, 5))
    movie_feature_cols = movie_feature_matrix.columns.tolist()
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Prepare model IO paths
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, args.model_file)

    svd_model = None
    # Try loading model if requested
    if args.load_model and os.path.exists(model_path):
        try:
            _, svd_model = svd_dump.load(model_path)
            print(f"Loaded SVD model from: {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}. Will train a new model.")

    # Train model (with optional tuning) if not loaded
    if svd_model is None:
        if args.tune_svd:
            print("Tuning SVD hyperparameters with GridSearchCV (rmse, 3-fold)...")
            param_grid = {
                'n_factors': [50, 100],
                'n_epochs': [20, 30],
                'lr_all': [0.002, 0.005],
                'reg_all': [0.02, 0.1],
            }
            gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, joblib_verbose=0)
            gs.fit(data)
            best_params = gs.best_params['rmse']
            print(f"Best SVD params (rmse): {best_params}")
            svd_model = SVD(random_state=42, **best_params)
        else:
            svd_model = SVD(random_state=42)

        svd_model.fit(trainset)
        if args.save_model:
            try:
                svd_dump.dump(model_path, algo=svd_model)
                print(f"Saved SVD model to: {model_path}")
            except Exception as e:
                print(f"Failed to save model to {model_path}: {e}")

    # Step 5: Content-based filtering
    user_movie_matrix = ratings.merge(user_features, on='user_id').merge(movie_features, on='movie_id')
    user_profiles = (
        user_movie_matrix
        .groupby('user_id')[movie_feature_cols]
        .mean()
    )

    # Category profiles: average of user_profiles within each demographic group
    # Merge to align profiles with demographics
    profiles_with_demo = user_profiles.merge(user_info[['user_id', 'age_group', 'gender', 'occupation']], left_index=True, right_on='user_id', how='left')
    profiles_with_demo = profiles_with_demo.set_index('user_id')

    cat_profiles = {'age': {}, 'gender': {}, 'occupation': {}}
    if not profiles_with_demo.empty:
        # Age groups
        age_grp = profiles_with_demo.groupby('age_group')[movie_feature_cols].mean()
        cat_profiles['age'] = {k: v.values for k, v in age_grp.iterrows()}
        # Gender
        gen_grp = profiles_with_demo.groupby('gender')[movie_feature_cols].mean()
        cat_profiles['gender'] = {k: v.values for k, v in gen_grp.iterrows()}
        # Occupation
        occ_grp = profiles_with_demo.groupby('occupation')[movie_feature_cols].mean()
        cat_profiles['occupation'] = {k: v.values for k, v in occ_grp.iterrows()}

    # Step 6: Hybrid recommendation
    svd_weight = args.svd_weight
    category_weight = max(0.0, args.category_weight)
    # Determine which category dims to use
    cat_dims = [d.strip().lower() for d in args.category_dims.split(',') if d.strip()]
    valid_dims = {'age', 'gender', 'occupation'}
    cat_dims = [d for d in cat_dims if d in valid_dims]
    # Compute content weight default to keep weights roughly summing to 1
    content_weight = args.content_weight if args.content_weight is not None else max(0.0, 1.0 - svd_weight - category_weight)
    normalize_sim = args.normalize_sim
    learn_sim = args.learn_sim
    # Access attributes with underscores due to argparse conversion
    cat_genre_boost = max(0.0, getattr(args, 'cat_genre_boost', 0.0))
    cat_topk = max(1, getattr(args, 'cat_topk_genres', 3))
    cat_mode = getattr(args, 'cat_genre_mode', 'combined')
    
    # Select users: random if not specified, otherwise use provided list
    if args.users is None:
        all_user_ids = ratings['user_id'].unique().tolist()
        sample_users = [str(uid) for uid in random.sample(all_user_ids, min(3, len(all_user_ids)))]
        print(f"Randomly selected users: {', '.join(sample_users)}")
    else:
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
        # Individual user-to-item similarity
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

        # Category-based similarity (blend of selected demographics)
        cat_sims = np.zeros(len(unseen_movies))
        demo = user_demo.get(user_id)
        # Track top genre indices for boosting and printing
        combined_top_genre_idxs = []
        per_dim_top_idxs = {}
        if demo is not None and len(cat_dims) > 0 and len(unseen_movies) > 0:
            components = []
            for d in cat_dims:
                key = None
                if d == 'age':
                    key = demo.get('age_group')
                elif d == 'gender':
                    key = demo.get('gender')
                elif d == 'occupation':
                    key = demo.get('occupation')
                prof = cat_profiles.get(d, {}).get(key)
                if prof is not None and np.any(np.isfinite(prof)):
                    components.append(prof.reshape(1, -1))
                    # store per-dimension top-K for printing and union mode
                    per_dim_top_idxs[d] = np.argsort(prof)[::-1][:cat_topk].tolist()
            if components:
                # Average the selected category vectors
                cat_vec = np.mean(np.vstack(components), axis=0).reshape(1, -1)
                combined_top_genre_idxs = np.argsort(cat_vec.flatten())[::-1][:cat_topk].tolist()
                movie_vectors = movie_feature_matrix.loc[unseen_movies].values
                cat_sims = cosine_similarity(cat_vec, movie_vectors).flatten()
                if normalize_sim and sim_regressor is None:
                    cat_sims = 3.0 + 2.0 * cat_sims
                    cat_sims = np.clip(cat_sims, 1.0, 5.0)
                elif sim_regressor is not None:
                    cat_sims = sim_regressor.predict(cat_sims.reshape(-1, 1))
                    cat_sims = np.clip(cat_sims, 1.0, 5.0)

        # Optional top-genre selection and boost
        genre_boosts = np.zeros(len(unseen_movies))
        selected_genres = []
        # Determine whether to compute selected genres (for matched display or boost)
        need_selected = getattr(args, 'show_matched_genres', False) or (cat_genre_boost > 0)
        if need_selected and (combined_top_genre_idxs or per_dim_top_idxs):
            # Select genre indices to consider
            if cat_mode == 'union' and per_dim_top_idxs:
                selected_genres = set()
                for li in per_dim_top_idxs.values():
                    selected_genres.update(li)
                selected_genres = list(selected_genres)
            else:
                selected_genres = combined_top_genre_idxs
            # Compute boosts only if boost is enabled
            if cat_genre_boost > 0 and selected_genres:
                item_genres_mat = movie_feature_matrix.loc[unseen_movies].values  # shape (m, 19)
                matches = item_genres_mat[:, selected_genres].sum(axis=1)
                denom = float(max(1, min(cat_topk, len(selected_genres))))
                genre_boosts = (matches / denom) * cat_genre_boost

        # Mix SVD predictions, content similarity, category similarity and genre boost
        hybrid_scores = []
        comp_map = {}
        for mid, sim, csim, gboost in zip(unseen_movies, similarities, cat_sims, genre_boosts):
            svd_c = svd_weight * svd_preds[mid]
            cont_c = content_weight * float(sim)
            cat_c = category_weight * float(csim)
            total = svd_c + cont_c + cat_c + float(gboost)
            hybrid_scores.append((mid, total))
            comp_map[mid] = {
                'svd': svd_c,
                'content': cont_c,
                'category': cat_c,
                'boost': float(gboost),
            }
        top_recs = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:args.topn]
        # Along with top recs, keep components and selected genres for table/matched labels
        recommendations[user_id] = {
            'recs': top_recs,
            'components': comp_map,
            'selected_genre_idxs': selected_genres,
        }

    # Step 7: Display recommendations
    movie_titles = dict(zip(movie_info['movie_id'], movie_info['title']))
    for user, rec_pack in recommendations.items():
        demo = user_demo.get(user, {})
        age = demo.get('age', 'NA')
        age_group = demo.get('age_group', 'NA')
        gender = demo.get('gender', 'NA')
        occupation = demo.get('occupation', 'NA')
        if getattr(args, 'badges_demographics', False):
            # Compact header with embedded badges
            badge_age_group = age_group if age_group != 'NA' else '?'
            badge_gender = gender if gender != 'NA' else '?'
            badge_occ = occupation if occupation != 'NA' else '?'
            print(f"\nTop {args.topn} movie recommendations for User {user} [{badge_age_group}|{badge_gender}|{badge_occ}]:")
        else:
            print(f"\nTop {args.topn} movie recommendations for User {user} (Age: {age}, Age Group: {age_group}, Gender: {gender}, Occupation: {occupation}):")
        recs = rec_pack['recs']
        comp_map = rec_pack['components']
        sel_idxs = rec_pack.get('selected_genre_idxs', []) or []
        # Show top genres from each selected dimension and combined
        components = []
        per_dim_lines = []
        for d in cat_dims:
            key = age_group if d == 'age' else (gender if d == 'gender' else occupation)
            prof = cat_profiles.get(d, {}).get(key)
            if prof is not None:
                components.append(prof)
                idxs = np.argsort(prof)[::-1][:cat_topk]
                names = [genre_names[i] if i < len(genre_names) else f'genre_{i}' for i in idxs]
                label = 'Age-group' if d == 'age' else ('Gender' if d == 'gender' else 'Occupation')
                per_dim_lines.append(f"  {label} top genres: {', '.join(names)}")
        if getattr(args, 'show_per_dim_top_genres', False):
            for line in per_dim_lines:
                print(line)
        if components:
            cg_vec = np.mean(np.vstack(components), axis=0)
            idxs = np.argsort(cg_vec)[::-1][:cat_topk]
            names = [genre_names[i] if i < len(genre_names) else f'genre_{i}' for i in idxs]
            label = 'Combined top genres' if getattr(args, 'badges_demographics', False) else 'Combined category top genres'
            print(f"  {label}: {', '.join(names)}")
        # If tabular view requested, render a compact ASCII table
        if getattr(args, 'recs_table', False):
            # Build rows
            rows = []
            for rank, (movie_id, score) in enumerate(recs, start=1):
                title = movie_titles.get(movie_id, "Unknown Title")
                comps = comp_map.get(movie_id, {})
                matched_names = []
                if sel_idxs:
                    # find which selected genres apply to this movie
                    gvec = movie_feature_matrix.loc[movie_id].values
                    matched_idx = [i for i in sel_idxs if i < len(gvec) and gvec[i] == 1]
                    matched_names = [genre_names[i] if i < len(genre_names) else f'genre_{i}' for i in matched_idx][:max(0, getattr(args, 'recs_max_genres', 2))]
                # Confidence hint: clamp score to [1,5] and render simple 5-slot bar
                conf_col = ''
                if getattr(args, 'recs_show_confidence', False):
                    conf = float(np.clip(score, 1.0, 5.0))
                    filled = int(round((conf - 1.0) / 4.0 * 5))
                    filled = max(0, min(5, filled))
                    bar = '[' + ('#' * filled) + ('-' * (5 - filled)) + f'] {conf:.1f}'
                    conf_col = bar
                if getattr(args, 'recs_show_components', False):
                    base = [str(rank), title, f"{score:.2f}", f"{comps.get('svd', 0.0):.2f}", f"{comps.get('content', 0.0):.2f}", f"{comps.get('category', 0.0):.2f}", f"{comps.get('boost', 0.0):.2f}"]
                    if getattr(args, 'recs_show_confidence', False):
                        base.append(conf_col)
                    base.append(', '.join(matched_names))
                    rows.append(base)
                else:
                    if getattr(args, 'recs_show_confidence', False):
                        rows.append([str(rank), title, f"{score:.2f}", conf_col, ', '.join(matched_names)])
                    else:
                        rows.append([str(rank), title, f"{score:.2f}", ', '.join(matched_names)])

            if getattr(args, 'recs_show_components', False):
                headers = ['#', 'Title', 'Rating', 'SVD', 'Content', 'Category', 'Boost']
                if getattr(args, 'recs_show_confidence', False):
                    headers.append('Conf')
                headers.append('Matched Genres')
            else:
                if getattr(args, 'recs_show_confidence', False):
                    headers = ['#', 'Title', 'Rating', 'Conf', 'Matched Genres']
                else:
                    headers = ['#', 'Title', 'Rating', 'Matched Genres']

            # Compute column widths
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

            def print_row(cols):
                print('  ' + ' | '.join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols)))

            def print_sep():
                print('  ' + '-+-'.join('-' * w for w in col_widths))

            print_row(headers)
            print_sep()
            for row in rows:
                print_row(row)
        else:
            # Fallback simple list view
            for movie_id, score in recs:
                title = movie_titles.get(movie_id, "Unknown Title")
                line = f"  {title} (Movie ID: {movie_id}) - Score: {score:.2f}"
                if getattr(args, 'recs_show_confidence', False):
                    conf = float(np.clip(score, 1.0, 5.0))
                    filled = int(round((conf - 1.0) / 4.0 * 5))
                    filled = max(0, min(5, filled))
                    bar = '[' + ('#' * filled) + ('-' * (5 - filled)) + f'] {conf:.1f}'
                    line += f"  {bar}"
                print(line)

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

            # generate top-k for hybrid (including category component and top-genre boost)
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
                # Category sims
                csims = np.zeros(len(unseen))
                demo_u = user_demo.get(user)
                # prepare top-genre indices for boosting in eval
                combined_idxs_eval = []
                per_dim_top_idxs_eval = {}
                if demo_u is not None and len(cat_dims) > 0 and len(unseen) > 0:
                    parts = []
                    for d in cat_dims:
                        key = None
                        if d == 'age':
                            key = demo_u.get('age_group')
                        elif d == 'gender':
                            key = demo_u.get('gender')
                        elif d == 'occupation':
                            key = demo_u.get('occupation')
                        prof = cat_profiles.get(d, {}).get(key)
                        if prof is not None and np.any(np.isfinite(prof)):
                            parts.append(prof.reshape(1, -1))
                            per_dim_top_idxs_eval[d] = np.argsort(prof)[::-1][:cat_topk].tolist()
                    if parts:
                        cat_vec = np.mean(np.vstack(parts), axis=0).reshape(1, -1)
                        combined_idxs_eval = np.argsort(cat_vec.flatten())[::-1][:cat_topk].tolist()
                        movie_vectors = movie_feature_matrix.loc[unseen].values
                        csims = cosine_similarity(cat_vec, movie_vectors).flatten()
                        if normalize_sim and sim_regressor is None:
                            csims = 3.0 + 2.0 * csims
                            csims = np.clip(csims, 1.0, 5.0)
                        elif sim_regressor is not None:
                            csims = sim_regressor.predict(csims.reshape(-1, 1))
                            csims = np.clip(csims, 1.0, 5.0)
                # genre boost in eval
                genre_boosts_eval = np.zeros(len(unseen))
                if cat_genre_boost > 0 and (combined_idxs_eval or per_dim_top_idxs_eval):
                    if cat_mode == 'union' and per_dim_top_idxs_eval:
                        sel = set()
                        for li in per_dim_top_idxs_eval.values():
                            sel.update(li)
                        sel = list(sel)
                    else:
                        sel = combined_idxs_eval
                    if sel:
                        item_genres_mat = movie_feature_matrix.loc[unseen].values
                        matches = item_genres_mat[:, sel].sum(axis=1)
                        denom = float(max(1, min(cat_topk, len(sel))))
                        genre_boosts_eval = (matches / denom) * cat_genre_boost
                hybrid_scores = [(mid, svd_weight * svd_preds[mid] + content_weight * sim + category_weight * csim + gboost) for mid, sim, csim, gboost in zip(unseen, sims, csims, genre_boosts_eval)]
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

            # Tabular comparison between SVD and Hybrid
            print('\nEvaluation results (averaged over users sample):')
            metrics_order = ['precision', 'recall', 'ndcg', 'hitrate', 'coverage', 'ild']
            headers = ['Metric', 'SVD', 'Hybrid', 'Delta']
            rows = []
            for k in metrics_order:
                s = float(svd_metrics.get(k, float('nan')))
                h = float(hybrid_metrics.get(k, float('nan')))
                d = h - s if (np.isfinite(h) and np.isfinite(s)) else float('nan')
                rows.append([k, f"{s:.4f}", f"{h:.4f}", f"{d:+.4f}" if np.isfinite(d) else 'NA'])

            # Compute column widths
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

            def print_row(cols):
                print('  ' + ' | '.join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols)))

            def print_sep():
                print('  ' + '-+-'.join('-' * w for w in col_widths))

            print()
            print_row(headers)
            print_sep()
            for row in rows:
                print_row(row)


if __name__ == '__main__':
    main()
