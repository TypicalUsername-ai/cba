import polars as pl

from cba import cba

if __name__ == "__main__":
    cars = pl.read_csv(
        "./datasets/blood/transfusion.data",
    ).with_columns(pl.nth(4).alias("class"))

    print(cars.describe())
    cba(
        cars.select(pl.exclude("class")).to_numpy(),
        cars["class"].to_numpy(),
        min_support=0.01,
        min_confidence=0.5,
        prune=False,
    )
