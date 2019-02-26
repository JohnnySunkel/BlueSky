from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time


# Path to the target image (image you want to transform)
target_image_path = 'C:/Users/Julie/beach_pic.jpg'

# Path to the style image
style_reference_image_path = 'C:/Users/Julie/starry_night.jpg'

# Dimensions of the generated image
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# Auxiliary functions
def preprocess_image(image_path):
    img = load_img(image_path, target_size = (img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Zero-centering by removing the mean pixel value
    # from ImageNet. This reverses a transformation
    # done by vgg19.preprocess_input.
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # Convert images from 'BGR' to 'RGB'. This is also
    # part of the reversal of vgg19.preprocess_input.
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Load the pre-trained VGG19 network and apply it to
# the three images
target_image = K.constant(
    preprocess_image(target_image_path))
style_reference_image = K.constant(
    preprocess_image(style_reference_image_path))
# Placeholder that will contain the generated image
combination_image = K.placeholder((1, img_height, img_width, 3))

# Combine the three images in a single batch
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis = 0)
    
# Build the VGG19 network with the batch of three
# images as input. The model will be loaded with
# pre-trained ImageNet weights.
model = vgg19.VGG19(input_tensor = input_tensor,
                    weights = 'imagenet',
                    include_top = False)
print('Model loaded.')

# Define the content loss
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# Define the style loss
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# Define the total variation loss
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# Define the final loss to minimize

# Dictionary that maps layer names to activation tensors
outputs_dict = dict([(layer.name, 
                      layer.output) for layer in model.layers])
# Layer used for content loss
content_layer = 'block5_conv2'
# Layers used for style loss
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# Weights in the weighted average of the loss components
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Define the loss by adding all components to a scalar variable
loss = K.variable(0.)
# Adds the content loss
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)

# Adds a style loss component for each target layer
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    s1 = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * s1
    
# Adds the total variation loss
loss += total_variation_weight * total_variation_loss(
    combination_image)


# Set up the gradient descent process

# Gets the gradients of the generated image with regard to the loss
grads = K.gradients(loss, combination_image)[0]

# Function to fetch the values of the current loss and
# the current gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# This class wraps fetch_loss_and_grads in a way that
# lets you retrieve the losses and gradients via two
# separate method calls, which is required by the SciPy
# optimizer you'll use.
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
evaluator = Evaluator()


# Style transfer loop
result_prefix = 'my_result'
iterations = 20

# This is the initial state (the target image)
x = preprocess_image(target_image_path)

# You flatten the image because scipy.optimize.fmin_l_bfgs_b
# can only process flat vectors
x = x.flatten()

# Runs L-BFGS optimization over the pixels of the
# generated image to minimize the neural style loss.
# Note that you have to pass the function that
# computes the loss and the function that computes
# the gradient as two separate arguments.
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime = evaluator.grads,
                                     maxfun = 20)
    print('Current loss value:', min_val)
    # Saves the current generated image
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
