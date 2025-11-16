# token-metrics-project

A machine learning pipeline for predicting next-day cryptocurrency price movements using technical indicators and time series validation.

## Overview

This project implements a binary classification system to predict whether cryptocurrency prices will increase or decrease the following day. The solution uses multiple machine learning models with optimized hyperparameters and a custom probability threshold to improve prediction precision.

**Key Features:**
- 72 engineered technical indicators from price and volume data
- 4 ML models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Rolling window validation (time series cross-validation)
- Precision optimization via threshold tuning (0.6) and hyperparameter tuning
- Comprehensive evaluation metrics and visualizations

## Quick Start

### Prerequisites

- Python 3.7+
- Jupyter Notebook (or Google Colab)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd token-metrics-project

# Install dependencies
pip install yfinance ta xgboost scikit-learn pandas numpy matplotlib seaborn
```

### Running the Notebook

**Option 1: Google Colab**
1. Upload `Crypto.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells sequentially
3. Dependencies will install automatically

**Option 2: Local Jupyter**
```bash
jupyter notebook Crypto.ipynb
```

**Execution Time:** ~5-10 minutes (depending on data fetch speed)

## Project Structure

```
token-metrics-project/
├── Crypto.ipynb              # Main analysis notebook
├── README.md                 # This file
├── TROUBLESHOOTING.md        # Common issues and solutions
└── crypto_prediction_results.png  # Generated visualizations
```

## Methodology

### Data Collection

- **Source:** Yahoo Finance API (yfinance)
- **Assets:** 9 major cryptocurrencies (BTC, ETH, BNB, XRP, ADA, SOL, DOGE, DOT, AVAX)
- **Period:** 60 days of historical data (30 days after feature calculation)
- **Frequency:** Daily OHLCV data

### Feature Engineering

72 features across 5 categories:

1. **Trend Indicators:** SMA (5, 10, 20), EMA (5, 10), MACD
2. **Momentum Indicators:** RSI, Stochastic Oscillator, Rate of Change
3. **Volatility Indicators:** Bollinger Bands, ATR, rolling volatility
4. **Lag Features:** Previous returns (1-5 days), rolling statistics
5. **Market Features:** Cross-asset statistics, time features (day of week, month)

### Model Configuration

All models use optimized hyperparameters for better precision:

| Model | Key Parameters | Threshold |
|-------|---------------|-----------|
| **Logistic Regression** | C=0.1, class_weight={0:1, 1:2} | 0.6 |
| **Random Forest** | n_estimators=300, max_depth=8, min_samples_split=10 | 0.6 |
| **Gradient Boosting** | n_estimators=300, max_depth=4, learning_rate=0.03 | 0.6 |
| **XGBoost** | n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.7 | 0.6 |

**Precision Optimization:**
- Probability threshold set to 0.6 (instead of default 0.5) to reduce false positives
- More conservative hyperparameters (increased regularization, reduced depth)
- Moderate class weighting to handle imbalance without over-weighting

### Validation Strategy

**Rolling Window Validation (Time Series Cross-Validation):**
- 5 folds with expanding training window
- Each fold: train on past data, test on future data
- Prevents data leakage and mimics real-world deployment

**Feature Selection:**
- SelectKBest with F-statistic (top 80% of features per fold)
- Prevents overfitting on small dataset

**Preprocessing:**
- StandardScaler (fit on training, transform on test)
- NaN handling: drop rows with missing values

## Results

### Performance Metrics

Results vary by execution date (market conditions). Example output:

| Model | Precision | Recall | F1 | Accuracy |
|-------|-----------|--------|----|----------|
| Logistic Regression | 0.38 | 0.82 | 0.50 | 0.47 |
| Random Forest | 0.33 | 0.59 | 0.39 | 0.42 |
| Gradient Boosting | 0.34 | 0.65 | 0.42 | 0.45 |
| XGBoost | 0.33 | 0.56 | 0.39 | 0.55 |

**Key Observations:**
- Models show higher recall than precision (catch more true positives, but with more false positives)
- DOWN predictions typically outperform UP predictions
- Threshold tuning (0.6) improves precision at the cost of recall

### Visualizations

The notebook generates 6 visualizations:
1. Model performance comparison (bar chart)
2. Confusion matrix (best model)
3. Top 10 feature importance
4. Accuracy by validation fold
5. Prediction distribution
6. Trading success rates

Saved as `crypto_prediction_results.png` (300 DPI).

## Reproducibility

### Deterministic Components

- **Random seeds:** All models use `random_state=42`
- **Data sorting:** `sort_index()` ensures consistent temporal ordering
- **Feature selection:** F-statistic ranking is deterministic
- **Validation folds:** TimeSeriesSplit generates consistent splits

### Variable Components

- **Data values:** Yahoo Finance data updates daily
- **Performance metrics:** Depend on market conditions at execution time
- **Feature importance:** May vary slightly with different data

### Reproducing Results

1. Use Python 3.7+ (tested on 3.8-3.11)
2. Install exact dependencies (see Installation)
3. Run all cells sequentially (no reordering)
4. Note: Results will differ slightly due to daily data updates

## Technical Decisions

### Why Rolling Window Validation?

Standard k-fold CV is invalid for time series:
-  Future data leaks into training via random splits
-  Unrealistic (can't use future to predict past)
-  Inflated performance metrics

Rolling window validation:
-  Temporal integrity (train on past, test on future)
-  Realistic (mimics actual trading)
-  Honest metrics (true out-of-sample performance)

### Why Threshold 0.6?

- Default threshold (0.5) produces too many false positives
- Higher threshold (0.6) reduces false positives, improves precision
- Trade-off: Lower recall (fewer UP predictions, but more accurate)

### Why These Hyperparameters?

- **Lower C values / max_depth:** Reduce overfitting on small dataset
- **Higher min_samples_split/leaf:** More conservative splits
- **Lower learning rates:** Smoother convergence, better generalization
- **Moderate class weights:** Handle imbalance without over-weighting

## Limitations & Future Work

### Current Limitations

- Small dataset (~234 samples) limits model complexity
- 30-day window may not capture longer-term patterns
- No transaction cost modeling
- No position sizing or risk management

### Potential Improvements

1. **Feature Engineering:**
   - On-chain metrics (active addresses, transaction volume)
   - Social sentiment (Twitter, Reddit)
   - Cross-asset correlation features

2. **Model Enhancements:**
   - LSTM/GRU for sequence modeling
   - Ensemble methods (stacking, blending)
   - Hyperparameter optimization (Optuna, Hyperopt)

3. **Validation:**
   - Purged cross-validation
   - Walk-forward optimization
   - Monte Carlo validation

4. **Trading Metrics:**
   - Sharpe ratio, maximum drawdown
   - Profit factor, win rate
   - Transaction cost modeling

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| MATIC-USD unavailable | Expected - project uses 9 assets (within 8-12 requirement) |
| Package installation fails | Run `pip install --upgrade pip` first |
| Memory errors | Restart kernel, run cells sequentially |
| Data fetch timeouts | Check internet connection, wait and retry |
| Import errors | Verify all cells run in order |

See `TROUBLESHOOTING.md` for detailed solutions.

## Dependencies

```
yfinance>=0.2.0
ta>=0.10.0
xgboost>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## License

This project is part of a take-home exercise. Please refer to the original assignment instructions for usage terms.

## Contact

For questions or issues, please refer to the project maintainer or original assignment instructions.

---

**Disclaimer:** This project is for educational and demonstration purposes. Cryptocurrency trading involves significant risk. Past performance does not guarantee future results.
