# Evaluate a single model
def evaluate_model(train, test, n_input):
    # Fit the model
    model = build_model(train, n_input)
    # History is a list of weekly data
    history = [x for x in train]
    # Walk forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # Predict the week
        y_hat_sequence = forecast(model, history, n_input)
        # Store the predictions
        predictions.append(y_hat_sequence)
        # Get real observation and add to history for
        # predicting the next week
        history.append(test[i, :])
    # Evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# Summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))
    
# Convert history into inputs and outputs
def to_supervised(train, n_input, n_out = 7):
    # Flatten the data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # Step over the entire history one time step at a time
    for _ in range(len(data)):
        # Define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # Ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start: in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end: out_end, 0])
        # Move along one time step
        in_start += 1
    return array(X), array(y)

# Train the model
def build_model(train, n_input):
    # Prepare the data
    train_x, train_y = to_supervised(train, n_input)
    # Define parameters
    verbose = False, 
    epochs = 20
    batch_size = 4
    n_timesteps = train_x.shape[1]
    n_features = train_x.shape[2]
    n_outputs = train_y.shape[1]
    #D Define the model
    model = Sequential()
    model.add(Conv1D(filters = 16,
                     kernel_size = 3,
                     activation = 'relu',
                     input_shape = (n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(n_outputs))
    model.compile(optimizer = 'adam', loss = 'mse')
    # Fit the model
    model.fit(train_x,
              train_y,
              epochs = epochs,
              batch_size = batch_size,
              verbose = verbose)
    return model
