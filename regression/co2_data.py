from datetime import datetime
from sklearn.datasets import fetch_openml
import seaborn as sns
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import datetime

import numpy as np
def synthetic_data():
    today = datetime.datetime.now()
    current_month = today.year + today.month / 12
    X_test = np.linspace(start=1958, stop=current_month, num=1_000).reshape(-1, 1)
    return X_test




def kernel():
    long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
    noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
        noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
    )
    irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    seasonal_kernel = (
        2.0**2
        * RBF(length_scale=100.0)
        * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    )

    co2_kernel = (
        long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
    )
    return co2_kernel
    



co2_data = fetch_openml(data_id=41187, as_frame=True)# Mauna Loa CO2
co2 = co2_data["data"].assign(target = co2_data["target"])
co2 = co2.assign(date = pd.to_datetime(co2[["year", "month", "day"]]))
sns.relplot(data = co2, x = "date", y = "target", kind = "line").savefig("co2")

X = co2_data["data"][["year", "month"]].to_numpy()
X = X[:, 0] + X[:, 1] / 12
X = X.reshape(-1, 1)
y = co2_data["target"]
y_mean = y.mean()
gaussian_process = GaussianProcessRegressor(kernel=kernel(), normalize_y=False)
gaussian_process.fit(X, y - y_mean)
X_test = synthetic_data()
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)


mean_y_pred += y_mean
# 2. Plot the mean line
plt.plot(X_test,std_y_pred, label='Mean', color='tab:blue', linewidth=2)

# 3. Create the shaded standard deviation area
plt.fill_between(
    X_test.flatten(), 
    mean_y_pred -  std_y_pred, # Lower bound
    mean_y_pred + std_y_pred, # Upper bound
    color='tab:blue', 
    alpha=0.2,                          # Transparency
)
plt.savefig("co2_estimated")
