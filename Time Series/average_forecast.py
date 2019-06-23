from numpy import mean, median

# One step average forecast
def average_forecast(history, config):
    n, offset, avg_type = config
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
    # Mean of last n values
    if avg_type is 'mean':
        return mean(values)
    # Median of last n values
    else:
        return median(values)

# Create a synthetic dataset
data = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
print(data)

# Test average forecast
for i in [1, 2, 3]:
    print(average_forecast(data, (i, 3, 'mean')))
