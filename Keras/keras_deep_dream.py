# Load the pretrained Inception V3 model
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import scipy
import matplotlib.pyplot as plt


# You won't be training the model, so this command
# disables all training-specific operations
K.set_learning_phase(0)

# Build the Inception V3 network, without its
# convolutional base. The model will be loaded
# with pretrained ImageNet weights.
model = inception_v3.InceptionV3(weights = 'imagenet',
                                 include_top = False)


# Set up the DeepDream configuration

# Create a dictionary mapping layer names to a coefficient
# quantifying how much the layer's activation contributes
# to the loss you'll seek to maximize. Note that the
# layer names are hardcoded in the built-in Inception V3
# application. You can list all layer names using
# model.summary().
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
    }


# Define the loss to be maximized

# Create a dictionary that maps layer names to layer instances
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# You'll define the loss by adding layer contributions
# to this scalar variable
loss = K.variable(0.)

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    # Retrieve the layer's output
    activation = layer_dict[layer_name].output
    
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # Add the L2 norm of the features of a layer to the
    # loss. You avoid border artifacts by only involving
    # nonborder pixels in the loss.
    loss += coeff * K.sum(K.square(activation[:, 
                                              2: -2, 
                                              2: -2,
                                              :])) / scaling

        
# Set up the gradient ascent process

# This tensor holds the generated image (the dream)
dream = model.input

# Compute the gradients of the dream with regard to the loss
grads = K.gradients(loss, dream)[0]

# Normalize the gradients
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# Set up a Keras function to retrieve the value of the
# loss and gradients, given an input image
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# This function runs gradient ascent for a number of iterations
def gradient_ascent(x, iterations, step, max_loss = None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


# Auxiliary functions
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order = 1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
    
# Utility function to open, resize, and format pictures
# into tensors that Inception V3 can process
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = inception_v3.preprocess_input(img)
    return img

# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        # Undoes preprocessing that was performed by
        # inception_v3.preprocess_input
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Running gradient ascent over different successive scales

# Gradient ascent step size
step = 0.01

# Number of scales at which to run gradient ascent
num_octave = 3

# Size ratio between scales
octave_scale = 1.4

# Number of ascent steps to run at each scale
iterations = 20

# IF the loss grows larger than 10, you'll interrupt the
# gradient ascent process to avoid ugly artifacts
max_loss = 10.

# Fill this with the path to the image you want to use
base_image_path = 'C:/Users/Julie/creative_commons_elephant.jpg'

# Loads the base image into a Numpy array
img = preprocess_image(base_image_path)

# Prepare a list of shape tuples defining the different
# scales at which to run gradient ascent
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
        for dim in original_shape])
    successive_shapes.append(shape)
    
# Reverse the list of shapes so they're in increasing order
successive_shapes = successive_shapes[::-1]

# Resize the Numpy array of the image to the smallest scale
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    # Scales up the dream image
    img = resize_img(img, shape)
    # Run gradient ascent, altering the dream
    img = gradient_ascent(img,
                          iterations = iterations,
                          step = step,
                          max_loss = max_loss)
    # Scales up the smaller version of the original 
    # image (it will be pixelated)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # Computes the high-quality version of the original
    # image at this size
    same_size_original = resize_img(original_img, shape)
    # The difference is the detail that was lost when scaling up
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    # Reinjects lost detail into the dream
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname = 'dream_at_scale_' + str(shape) + '.png')
    
save_img(img, fname = 'final_dream.png')

# Display the dream image
plt.imshow(deprocess_image(np.copy(img)))
plt.show()
