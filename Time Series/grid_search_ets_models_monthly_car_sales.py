# Grid search exponential smoothing models for the
# monthly car sales dataset
from numpy import array
from pandas import read_csv
from math import sqrt
from multiprocess import cpu_count
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# One-step Holt Winters Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
    t, d, s, p, b, r = config
    # Define the model
    history = array(history)
    model = ExponentialSmoothing(history,
                                 trend = t,
                                 damped = d,
                                 seasonal = s,
                                 seasonal_periods = p)
    # Fit the model
    model_fit = model.fit(optimized = True,
                          use_boxcox = b,
                          remove_bias = r)
    # Make a one-step forecast
    y_hat = model_fit.predict(len(history), len(history))
    return y_hat[0]

# Calculate errors using RMSE
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# Split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# Walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # Split dataset
    train, test = train_test_split(data, n_test)
    # Seed history with training dataset
    history = [x for x in train]
    # Step over each time step in the test set
    for i in range(len(test)):
        # Fit the model and make a forecast for history
        y_hat = exp_smoothing_forecast(history, cfg)
        # Store the forecast in the list of predictions
        predictions.append(y_hat)
        # Add an actual observation to history for the next loop
        history.append(test[i])
    # Estimate prediction error
    error = measure_rmse(test, predictions)
    return error

# Score a model, return None on failure
def score_model(data, n_test, cfg, debug = False):
    result = None
    # Convert config to a key
    key = str(cfg)
    # Show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # One failure during model validation suggests an unstable config
        try:
            # Never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings('ignore')
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # Check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

# Grid search configs
def grid_search(data, cfg_list, n_test, parallel = True):
    scores = None
    if parallel:
        # Execute configs in parallel
        executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # Remove empty results
    scores = [r for r in scores if r[1] != None]
    # Sort configs by error in ascending order
    scores.sort(key = lambda tup: tup[1])
    return scores

# Create a set of exponential smoothing configs to evaluate
def exp_smoothing_configs(seasonal = [None]):
    models = list()
    # Define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # Create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            models.append(cfg)
    return models

if __name__ == '__main__':
    # Load the dataset
    series = read_csv('monthly-car-sales.csv',
                      header = 0,
                      index_col = 0)
    data = series.values
    # Split the data
    n_test = 12
    # Model configs
    cfg_list = exp_smoothing_configs(seasonal = [0, 6, 12])
    # Grid search
    scores = grid_search(data[:, 0], cfg_list, n_test)
    print('done')
    # List the top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
