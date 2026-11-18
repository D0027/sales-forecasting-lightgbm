"""
Forecast the next 90 days with the saved model.

The model needs the sales history to build its lag features, so pass the same kind of
train.csv (date, store, item, sales). The forecast starts the day after the last date in the file.

Usage:
    python predict.py --data data/train.csv
    python predict.py --data data/train.csv --model models/lgbm_sales_model.txt --out forecast.csv
"""
import argparse

from src import forecasting as fc


def main() -> None:
    ap = argparse.ArgumentParser(description="Forecast the next 90 days with a saved model.")
    ap.add_argument("--data", required=True, help="path to train.csv (date, store, item, sales)")
    ap.add_argument("--model", default="models/lgbm_sales_model.txt")
    ap.add_argument("--out", default="forecast_next_90_days.csv")
    args = ap.parse_args()

    df = fc.load_train_csv(args.data)
    panel = fc.build_panel(df)
    booster = fc.load_model(args.model)

    forecast = fc.forecast_future(panel, booster)
    forecast.to_csv(args.out, index=False)
    print(f"{panel.n_series} series | history ends {panel.hist_dates[-1].date()} | "
          f"forecast {forecast['date'].min().date()} -> {forecast['date'].max().date()}")
    print(f"Saved {len(forecast)} rows to {args.out}")


if __name__ == "__main__":
    main()
