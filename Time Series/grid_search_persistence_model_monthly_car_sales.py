# Persistence forecast for the monthly car sales dataset
from math import sqrt
from numpy import mean, median, std
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# Calculate the RMSE
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted)) 

# Difference the dataset
def difference(data, interval):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]

# Fit a model
def model_fit(train, config):
    return None

# Forecast with a pre-fit model
def model_predict(model, history, config):
    values = list()
    for offset in config:
        values.append(history[-offset])
    return median(values)

# Walk forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # Split the dataset
    train, test = train_test_split(data, n_test)
    # Fit the model
    model = model_fit(train, cfg)
    # Seed history with training dataset
    history = [x for x in train]
    # Step over each time step in the test set
    for i in range(len(test)):
        # Fit model and make a forecast for history
        y_hat = model_predict(model, history, cfg)
        # Store the forecast in a list of predictions
        predictions.append(y_hat)
        # Add actual observations to history for the next loop
        history.append(test[i])
    # Estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error

# Repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats = 30):
    # Fit and evaluate the model n times
    scores = [walk_forward_validation(data, 
                                      n_test, 
                                      config) for _ in range(n_repeats)]
    return scores

# Summarize model performance
def summarize_scores(name, scores):
    # Print a summary
    scores_m, score_std = mean(scores), std(scores)
    print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
    # Box and whisker plot
    plt.boxplot(scores)
    plt.show()
    
series = read_csv('monthly-car-sales.csv',
                  header = 0,
                  index_col = 0)
data = series.values
# Split the dataset
n_test = 12
# Define configuration
config = [12, 24, 36]
# Grid search
scores = repeat_evaluate(data, config, n_test)
# Summarize scores
summarize_scores('persistence', scores)
