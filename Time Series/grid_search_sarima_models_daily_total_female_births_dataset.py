# Grid search SARIMA models for the
# daily total female births dataset
from math import sqrt
from pandas import read_csv
from multiprocess import cpu_count
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# One step SARIMA forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # Define the model
    model = SARIMAX(history,
                    order = order,
                    seasonal_order = sorder,
                    trend = trend,
                    enforce_stationarity = False,
                    enforce_invertibility = False)
    # Fit the model
    model_fit = model.fit(disp = False)
    # Make a one step forecast
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
        y_hat = sarima_forecast(history, cfg)
        # Store the forecast in a list of predictions
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

# Create a set of SARIMA configs to try
def sarima_configs(seasonal = [0]):
    models = list()
    # Define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # Create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models

if __name__ == '__main__':
    # Load the dataset
    series = read_csv('daily-total-female-births.csv',
                      header = 0,
                      index_col = 0)
    data = series.values
    # Split the data
    n_test = 165
    # Model configs
    cfg_list = sarima_configs()
    # Grid search
    scores = grid_search(data, cfg_list, n_test)
    print('done')
    # List the top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
