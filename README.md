# AutoML Suite 🤖

Fully automated machine learning: feature engineering → model selection → hyperparameter tuning → ensembling.

## Capabilities

- **Feature Engineering**: 100+ transforms, auto-interactions
- **Model Selection**: XGBoost, LightGBM, CatBoost, NN, linear
- **HPO**: Bayesian optimization, early stopping
- **Ensembling**: Stacking, blending, weighted average

## Results on Kaggle

| Dataset | AutoML Score | Best Single | Improvement |
|---------|-------------|-------------|-------------|
| Titanic | 0.808 | 0.789 | +2.4% |
| House Prices | 0.124 | 0.131 | +5.3% |
| Credit Default | 0.792 | 0.776 | +2.1% |

## Quick Start

```python
from auto_ml import AutoML

automl = AutoML(time_budget=3600)
automl.fit(train_df, target="price")
predictions = automl.predict(test_df)
```

## License

MIT