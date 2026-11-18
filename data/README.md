# Data

The competition data is **not** stored in this repository.

Download `train.csv` from the Kaggle competition (join the competition to accept the rules):
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

Put it here as `data/train.csv`, then:

```bash
python train.py --data data/train.csv       # full pipeline: baselines, LightGBM, test scores, forecast
python predict.py --data data/train.csv     # forecast the next 90 days with models/lgbm_sales_model.txt
```

Columns: `date, store, item, sales` (10 stores x 50 items, daily, 2013-01-01 to 2017-12-31).
