import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from pandas.plotting import lag_plot
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def preprocessor(numerical_columns: list[str], categorical_columns: list[str]):
    column_transform = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_columns,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
    )
    return column_transform




def load_walmart_dataset():
    dataset = load_dataset("mayankmishra22/walmart-sales-model")
    train: pd.DataFrame = dataset["train"].to_pandas()

    y = train.pop("Weekly_Sales")
    X = train

    print("Missing variables")
    print(train.isna().any(axis=0))

    train["Date"] = pd.to_datetime(train["Date"]).astype(np.int64) // 10**9
    for lag in [1, 1, 2, 4, 12, 52]:
        X[f"lag_{lag}"] =y.shift(
            lag, fill_value=0
        )  

        X[f"rolling_mean_{lag}"] = (
            y
            .shift(lag, fill_value=0)
            .rolling(window=lag * 4)
            .mean()
            .fillna(0)
        )
    
    numerical_types = train.select_dtypes(include=[np.float64]).columns
    categorical_types = train.select_dtypes(include=[np.int64]).columns
    bools =  train.select_dtypes(include=[bool]).columns
    column_transform = preprocessor(list(numerical_types), list(categorical_types) + list(bools))
    column_transform.set_output(transform="pandas")
    
    X = column_transform.fit_transform(X)
    selector = VarianceThreshold(0.01)
    selector.set_output(transform = "pandas")
    X = selector.fit_transform(X)


    # y = y.diff(periods=1).fillna(0)
    y = y.apply(np.log).clip(0)

    return X, y



def train_time_series(X:pd.DataFrame, y:pd.DataFrame):

    # tscv = TimeSeriesSplit(test_size = int(len(X)*0.2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,  # Prevent over-fitting on noise
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
    )
    # model = MLPRegressor(hidden_layer_sizes=[32, 64, 128, 64, 32, 1], shuffle=False)

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(pred_test)), np.exp(pred_test))
    plt.plot(np.arange(len(pred_test)), np.exp(y_test))
    plt.savefig("prediction")
    plt.close()
    print(mean_absolute_error(np.exp(pred_test), np.exp(y_test)))
    print("r2 score" , r2_score(y_test, pred_test))
    tscv = TimeSeriesSplit(n_splits=5)
   
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)
        print(mean_absolute_error(pred_test, y_test))


X, y = load_walmart_dataset()
train_time_series(X, y)
