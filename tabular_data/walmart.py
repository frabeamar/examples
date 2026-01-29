import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from pandas.plotting import lag_plot

from sklearn.ensemble import RandomForestRegressor

# from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def train_gaussian_process(X_train, y_train):
    # 1. Scale the features (GPs are extremely sensitive to scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 2. Define the Kernel
    # RBF: For the smooth general trend
    # ExpSineSquared: For the periodicity (weekly/yearly seasonality)
    # WhiteKernel: To account for the noise in retail data
    kernel = (
        C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        + ExpSineSquared(1.0, 52.0, periodicity_bounds=(50, 54))
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    )

    # 3. Initialize the GP
    # n_restarts_optimizer helps find the best kernel parameters
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

    # 4. Fit the model
    gp.fit(X_scaled, y_train)

    return gp, scaler


# Usage:
# gp_model, scaler = train_gaussian_process(X_train, y_train)
def run_correlation_analysis(df: pd.DataFrame):
    plt.figure(figsize=(10, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.savefig("correlation.png")


def run_categorical_analysis(df, cat_cols, target_col):
    """Reveals how different categories (like Holidays or Store Types) move the target."""
    num_plots = len(cat_cols)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    if num_plots == 1:
        axes = [axes]  # Handle single plot case

    for i, col in enumerate(cat_cols):
        sns.boxplot(x=col, y=target_col, data=df, ax=axes[i], palette="viridis")
        axes[i].set_title(f"{target_col} by {col}")

    plt.tight_layout()
    plt.savefig("categorical")


def run_numerical_scatter(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        plt.figure(figsize=(8, 5))
        sns.regplot(
            x=col,
            y="Weekly_Sales",
            data=df,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
        )
        plt.title(f"Impact of {col} on Weekly_Sales")
        plt.savefig(f"regplot-{col}.png")


def run_lag_analysis(df: pd.DataFrame, lags=[1, 7]):
    plt.figure(figsize=(15, 1))
    fig, axes = plt.subplots(len(df.columns), len(lags), figsize=(12, 5))
    for j, col in enumerate(df):
        for i, l in enumerate(lags):
            lag_plot(df[col], lag=l, ax=axes[j, i], alpha=0.5)
            axes[j, i].set_title(f"Lag {l} Plot (t vs t+{l})")

            plt.suptitle(f"Autocorrelation Analysis for {col}")
    plt.tight_layout()
    plt.savefig("lag_anal.png")


def run_all(df):
    run_correlation_analysis(df)
    run_numerical_scatter(df)
    run_lag_analysis(df)


import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


# 1. Date Expansion Function
def expand_dates(X):
    X["Date"] = pd.to_datetime(X["Date"]).astype(np.int64) // 10**9
    return X.drop(columns=["month", "Year"])


date_transformer = FunctionTransformer(expand_dates)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                ]
            ),
            [
                "Size",
                "Temperature",
                "Fuel_Price",
                "MarkDown2",
                "MarkDown3",
                "MarkDown4",
                "MarkDown5",
                "Size",
            ]
            + [f"lag_{d}" for d in [1, 4, 12, 52]],
        ),
        ("date", MinMaxScaler(), ["Date", "CPI", "Unemployment", "IsHoliday"]),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            [
                "Store",
                "Dept",
            ]
            + [f"Type_{a}" for a in ["A", "B", "C"]],
        ),
    ],
    remainder="passthrough",
)
# selector = SelectFromModel(RandomForestRegressor(n_estimators=50), threshold="median")
selector = VarianceThreshold(threshold=0.01)
walmart_pipeline = Pipeline(
    steps=[
        ("date_expansion", date_transformer),
        ("preprocessing", preprocessor),
        ("select", selector),
    ]
)


dataset = load_dataset("mayankmishra22/walmart-sales-model")
train = dataset["train"].to_pandas()
train["lag_1"] = train["Weekly_Sales"].shift(1, fill_value=0)  # Previous week
train["lag_4"] = train["Weekly_Sales"].shift(4, fill_value=0)  # Previous week
train["lag_12"] = train["Weekly_Sales"].shift(12, fill_value=0)  # Previous week
train["lag_52"] = train["Weekly_Sales"].shift(52, fill_value=0)  # Same week last year

train["rolling_mean_4"] = (
    train["Weekly_Sales"].shift(1, fill_value=0).rolling(window=4).mean().fillna(0)
)

y = train.pop("Weekly_Sales")
X = train.drop(["MarkDown1"], axis=1)
fitted = walmart_pipeline.fit_transform(X, y)


# Create 5 folds of time-series splits
tscv = TimeSeriesSplit(n_splits=10)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,  # Prevent over-fitting on noise
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
)

for train_index, test_index in tscv.split(fitted):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    lr = MLPRegressor(hidden_layer_sizes=[32, 64, 128, 64, 32, 1])

    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    print(mean_absolute_error(pred, y_test))
    plt.plot(test_index, y_test)
    plt.plot(test_index, pred)
    plt.savefig("prediction")
    plt.close()
    plt.plot(train_index, y_train)
    plt.plot(lr.predict(X_train), pred)
    plt.savefig("prediction_train")
    plt.close()
