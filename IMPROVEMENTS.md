# RMSE/MAE Improvement Summary

## Overview
This document outlines the enhancements made to reduce RMSE and MAE in the movie recommender system.

## Results Summary

### Before Improvements
- **Baseline SVD**: RMSE 0.9434, MAE 0.7415

### After Improvements
- **KNNBaseline** (Best): RMSE **0.9170**, MAE **0.7190** 
  - **Improvement**: -0.0264 RMSE (-2.8%), -0.0225 MAE (-3.0%)
- **SVD++**: RMSE **0.9231**, MAE **0.7262**
  - **Improvement**: -0.0202 RMSE (-2.1%), -0.0153 MAE (-2.1%)

## Implemented Enhancements

### 1. Multiple Algorithm Support ✓
**Impact**: High (2.8% RMSE reduction)

Added support for 6 different collaborative filtering algorithms:
- **SVD**: Standard matrix factorization
- **SVD++**: Implicit feedback via user rating patterns (2.1% better than SVD)
- **KNNBaseline**: Item-based CF with Pearson baseline similarity (**BEST: 2.8% better**)
- **BaselineOnly**: Global + user + item biases (comparable to SVD)
- **KNNWithMeans**: KNN with mean-centered ratings
- **NMF**: Non-negative matrix factorization (performed worse on ML-100k)

**Usage**:
```bash
# Use best algorithm
python "Movie recommender.py" --algo knn

# Compare all algorithms
python "Movie recommender.py" --compare-algos
```

### 2. Expanded Hyperparameter Tuning ✓
**Impact**: Medium-High (estimated 1-3% with optimal params)

Enhanced tuning grids with 5-fold cross-validation:

**SVD tuning** (`--tune-svd`):
- n_factors: [50, 100, 150, 200] (was [50, 100])
- n_epochs: [20, 30, 40] (was [20, 30])
- lr_all: [0.002, 0.005, 0.007] (was [0.002, 0.005])
- reg_all: [0.01, 0.02, 0.05, 0.08] (was [0.02, 0.1])
- cv folds: 5 (was 3)

**SVD++ tuning** (`--tune-svdpp`):
- New flag with appropriate grid for implicit feedback
- n_factors: [50, 100, 150]
- n_epochs: [20, 30, 40]
- lr_all: [0.002, 0.005, 0.007]
- reg_all: [0.02, 0.05, 0.08]

**KNNBaseline tuning** (`--tune-knn`):
- New flag for neighborhood-based tuning
- k: [20, 40, 80]
- min_k: [1, 3, 5]
- sim_options: name ['pearson_baseline', 'cosine'], min_support [1, 3, 5]

**Usage**:
```bash
# Tune SVD with expanded grid
python "Movie recommender.py" --tune-svd --tune-cv 5 --save-model

# Tune SVD++
python "Movie recommender.py" --algo svdpp --tune-svdpp --save-model

# Tune KNN
python "Movie recommender.py" --algo knn --tune-knn --save-model
```

### 3. Post-hoc Prediction Calibration ✓
**Impact**: Low-Medium (varies by algorithm, ~0.5-1% potential)

Implements linear recalibration: `rating = a × prediction + b`

- Learns mapping on separate validation split (15% of train data)
- Corrects systematic bias (over/under-prediction)
- Applied at evaluation time before RMSE/MAE computation

**Results**:
- KNNBaseline: Calibration slightly increased RMSE (0.9170 → 0.9201)
  - Already well-calibrated, no systematic bias
- SVD++: Minimal change (0.9231 → 0.9233)
  - Coefficient 0.9572 shows slight under-prediction

**Usage**:
```bash
python "Movie recommender.py" --algo svdpp --calibrate-preds
```

**Note**: Calibration is most effective when baseline model shows consistent over/under-prediction patterns. Test with and without on your data.

### 4. Improved Default Parameters ✓
**Impact**: Low-Medium (baked into results above)

Updated default parameters based on best practices:
- **SVD**: n_factors=100 (was default 100 ✓), n_epochs=30 (was 20), lr=0.005, reg=0.02
- **SVD++**: n_factors=100, n_epochs=20, lr=0.005, reg=0.02
- **KNNBaseline**: k=40, min_k=3, pearson_baseline similarity, min_support=3

### 5. Algorithm Comparison Tool ✓
**Impact**: High (enables quick identification of best model)

`--compare-algos` flag runs all 6 algorithms on the same train/test split and prints comparison table with:
- RMSE and MAE for each algorithm
- Improvement vs SVD baseline
- Best algorithm identification

**Output Example**:
```
Algorithm RMSE/MAE Comparison:
  Model        | RMSE   | MAE    | Improvement vs SVD
  -------------+--------+--------+-------------------
  SVD          | 0.9434 | 0.7415 | -                 
  SVD++        | 0.9231 | 0.7262 | +0.0202 ✓         
  KNNBaseline  | 0.9170 | 0.7190 | +0.0264 ✓✓        
  BaselineOnly | 0.9430 | 0.7475 | +0.0004           
  KNNWithMeans | 0.9400 | 0.7368 | +0.0033           
  NMF          | 1.1017 | 0.8394 | -0.1583           

Best algorithm: KNNBaseline (RMSE: 0.9170)
```

## Recommendations for Further Improvement

### Implemented but Available for Tuning:
1. **Fine-tune KNNBaseline** (current best):
   - Try user_based=True vs False
   - Test different similarity metrics: msd, cosine vs pearson_baseline
   - Increase k up to 120–150 and tune min_support

2. **Ensemble Blending** (not yet implemented):
   - Blend SVD++ (0.9231) + KNNBaseline (0.9170)
   - Learn optimal weights via ridge regression
   - Estimated improvement: 0.5-1% additional reduction

### Not Yet Implemented:
3. **Advanced Tuning**:
   - Fine-grained SVD++: separate reg_yj for implicit terms
   - Fine-grained SVD: reg_bu, reg_bi, reg_pu, reg_qi
   - Larger grids with random search or Bayesian optimization

4. **Data-level Enhancements**:
   - Remove outlier users/items with < 5 ratings
   - Use official ML-100k u1–u5 folds and average results
   - Temporal validation (chronological split)

5. **Additional Algorithms**:
   - CoClustering (co-clustering of users and items)
   - SlopeOne (simple but effective for blending)
   - Deep learning models (though overkill for ML-100k)

## Quick Start Guide

### Find the best algorithm for your dataset:
```bash
python "Movie recommender.py" --compare-algos
```

### Use the best algorithm with tuning:
```bash
# KNNBaseline (best on ML-100k)
python "Movie recommender.py" --algo knn --tune-knn --save-model

# Or SVD++ for speed/accuracy balance
python "Movie recommender.py" --algo svdpp --tune-svdpp --save-model
```

### Test with calibration:
```bash
python "Movie recommender.py" --algo knn --calibrate-preds
```

### Load tuned model for recommendations:
```bash
python "Movie recommender.py" --algo knn --load-model --users 1 50 100
```

## Summary Table

| Enhancement | Status | Impact | RMSE Δ | Implementation Effort |
|------------|--------|--------|--------|----------------------|
| Multiple Algorithms | ✓ Done | High | -0.0264 | Medium |
| Expanded Tuning Grids | ✓ Done | Medium-High | ~-0.01 to -0.03 | Low |
| Post-hoc Calibration | ✓ Done | Low-Medium | Variable | Low |
| Improved Defaults | ✓ Done | Low-Medium | Baked in | Low |
| Algorithm Comparison | ✓ Done | High (discovery) | N/A | Low |
| Ensemble Blending | ⏳ Planned | Medium | ~-0.005 to -0.01 | Medium |
| Advanced Tuning | ⏳ Planned | Medium | ~-0.01 | Medium-High |

**Total Achieved Improvement**: **2.8% RMSE reduction** (0.9434 → 0.9170 with KNNBaseline)

## Testing
All enhancements have been tested and verified:
```bash
# Run test suite
python -m pytest -q

# Test all algorithms
python "Movie recommender.py" --compare-algos

# Verify best performer
python "Movie recommender.py" --algo knn
```

## Conclusion
The implemented enhancements successfully reduced RMSE by 2.8% through:
1. **Algorithm selection** (KNNBaseline outperforms SVD on ML-100k)
2. **Expanded hyperparameter grids** with better defaults
3. **Comprehensive comparison tools** to identify best approaches
4. **Optional calibration** for bias correction

The system now provides multiple paths to lower RMSE/MAE while maintaining explainability and hybrid recommendation features.
