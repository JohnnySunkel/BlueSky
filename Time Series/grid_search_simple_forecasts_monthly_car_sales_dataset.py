# Grid search simple forecasts for the
# monthly-car-sales dataset
from math import sqrt
from numpy import mean, median
from pandas import read_csv
from multiprocess import cpu_count
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from sklearn.metrics import mean_squared_error

# One step simple forecast
def simple_forecast(history, config):
    n, offset, avg_type = config
    # Persist value, ignore other config
    if avg_type == 'persist':
        return history[-n]
    # Collect values to average
    values = list()
    if offset == 1:
        values = history[-n:]
    else:
        # Skip bad configs
        if n * offset > len(history):
            raise Exception('Config beyond end of data: %d %d' % (n, offset))
        # Try and collect n values using offset
        for i in range(1, n + 1):
            ix = i * offset
            values.append(history[-ix])
    # Check if we can average
    if len(values) < 2:
        raise Exception('Cannot calculate average')
    # Mean of last n values
    if avg_type == 'mean':
        return mean(values)
    # Median of last n values
    else:
        return median(values)
    
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
        y_hat = simple_forecast(history, cfg)
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

# Create a set of simple configs to evaluate
def simple_configs(max_length, offsets = [1]):
    configs = list()
    for i in range(1, max_length + 1):
        for o in offsets:
            for t in ['persist', 'mean', 'median']:
                cfg = [i, o, t]
                configs.append(cfg)
    return configs

if __name__ == '__main__':
    # Load the data
    series = read_csv('monthly-car-sales.csv',
                      header = 0,
                      index_col = 0)
    data = series.values
    # Split the data
    n_test = 12
    # Model configs
    max_length = len(data) - n_test
    cfg_list = simple_configs(max_length, offsets = [1, 6, 12])
    # Grid search
    scores = grid_search(data, cfg_list, n_test)
    print('done')
    # List the top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
