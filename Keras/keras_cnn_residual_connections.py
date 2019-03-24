# The first example is for when the feature-map sizes are the same

from keras import layers

# Assumes the existence of a 4D input tensor 'x'
x = ...

# Applies a transformation to x
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)

# Adds the original x back to the output features
y = layers.add([y, x])


# The second example is for when the feature-map sizes differ

from keras import layers

# Assumes the existence of a 4D input tensor 'x'
x = ...

y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)
y = layers.MaxPooling2D(2, strides = 2)(y)

# Use a 1 x 1 convolution to linearly downsample the original
# x tensor to the same shape as y
residual = layers.Conv2D(128, 1, strides = 2, padding = 'same')(x)

# Add the residual tensor back to the output features
y = layers.add([y, residual])
