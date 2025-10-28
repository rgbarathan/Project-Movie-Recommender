# Quick Reference: Lowering RMSE/MAE

## TL;DR - Best Commands

### 1. Find your best algorithm (run once)
```bash
python "Movie recommender.py" --compare-algos
```
**Result**: KNNBaseline wins with RMSE 0.9170 (2.8% better than SVD)

### 2. Use the best algorithm
```bash
python "Movie recommender.py" --algo knn
```

### 3. Tune for even better results
```bash
python "Movie recommender.py" --algo knn --tune-knn --save-model
```

## Algorithm Cheat Sheet

| Algorithm | RMSE | MAE | When to Use |
|-----------|------|-----|-------------|
| **knn** (KNNBaseline) | **0.9170** ✓✓ | **0.7190** | Best overall, use by default |
| **svdpp** (SVD++) | **0.9231** ✓ | **0.7262** | Good balance of speed/accuracy |
| svd | 0.9434 | 0.7415 | Fast, baseline |
| baseline (BaselineOnly) | 0.9430 | 0.7475 | Very fast, simple biases |
| knnmeans (KNNWithMeans) | 0.9400 | 0.7368 | Alternative to KNN |
| nmf (NMF) | 1.1017 | 0.8394 | Poor on ML-100k, skip |

## New Flags Summary

### Algorithm Selection
- `--algo {svd|svdpp|knn|baseline|knnmeans|nmf}` - Choose CF algorithm
- `--compare-algos` - Benchmark all 6 algorithms

### Tuning
- `--tune-svd` - Tune SVD with expanded grid (n_factors 50-200, etc.)
- `--tune-svdpp` - Tune SVD++ hyperparameters
- `--tune-knn` - Tune KNNBaseline (k, min_k, similarity)
- `--tune-cv 5` - Use 5-fold CV (default: 5)

### Calibration
- `--calibrate-preds` - Apply linear post-hoc calibration to reduce bias

## Typical Workflow

### Discovery Phase
```bash
# Compare algorithms to find best
python "Movie recommender.py" --compare-algos
```

### Optimization Phase
```bash
# Tune the best algorithm
python "Movie recommender.py" --algo knn --tune-knn --save-model
```

### Production Phase
```bash
# Load tuned model for fast recommendations
python "Movie recommender.py" --algo knn --load-model
```

## Expected Improvements

| Starting Point | After Algorithm Switch | After Tuning | After Calibration |
|---------------|----------------------|-------------|------------------|
| SVD 0.9434 | KNN 0.9170 (-2.8%) | ~0.91 (-3-4%) | Variable (~0-1%) |

## Tips

1. **Always start with `--compare-algos`** to see what works best on your data
2. **KNNBaseline** is the winner on ML-100k, but your mileage may vary
3. **Tuning takes time** (5-10 min with full grid), but `--save-model` lets you reuse
4. **Calibration** helps most when predictions are systematically biased
5. **SVD++** is a good compromise if KNN is too slow

## One-Liner for Best Results
```bash
python "Movie recommender.py" --algo knn --tune-knn --save-model
```

Then for future runs:
```bash
python "Movie recommender.py" --algo knn --load-model
```
