import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.special import logit


# Generate synthetic classification data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
# epsilon is 0.1
x_zeros = np.random.multivariate_normal(
    mean = np.array((-1, -1)), cov = 0.1 * np.eye(2), size = (N // 2,))
y_zeros = np.zeros((N // 2,))
# Ones form a Gaussian centered at (1, 1)
# epsilon = 0.1
x_ones = np.random.multivariate_normal(
    mean = np.array((1, 1)), cov = 0.1 * np.eye(2), size = (N // 2,))
y_ones = np.ones((N // 2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Plot the data
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color = 'blue')
plt.scatter(x_ones[:, 0], x_ones[:, 1], color = 'red')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Toy Logistic Regression Data')
plt.show()


# Generate the tensorflow graph
with tf.name_scope('placeholders'):
    # Note that our datapoints x are 2-dimensional
    x = tf.placeholder(tf.float32, (N, 2))
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope('weights'):
    W = tf.Variable(tf.random_normal((2, 1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope('prediction'):
    y_logit = tf.squeeze(tf.matmul(x, W) + b)
    # The sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # Rounding P(y = 1) will give the correct prediction
    y_pred = tf.round(y_one_prob)
with tf.name_scope('loss'):
    # Compute the cross-entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit,
                                                      labels = y)
    # Sum all contributions
    l = tf.reduce_sum(entropy)
with tf.name_scope('optim'):
    train_op = tf.train.AdamOptimizer(0.01).minimize(l)
with tf.name_scope('summaries'):
    tf.summary.scalar('loss', l)
    merged = tf.summary.merge_all()
    
train_writer = tf.summary.FileWriter('/tmp/logistic-train',
                                     tf.get_default_graph())


# Train the model
n_steps = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_steps):
        feed_dict = {x: x_np, y: y_np}
        _, summary, loss = sess.run([train_op, merged, l],
                                    feed_dict = feed_dict)
        print('step %d, loss: %f' % (i, loss))
        train_writer.add_summary(summary, i)
    
    # Get weights
    w_final, b_final = sess.run([W, b])
    
    # Make predictions
    y_pred_np = sess.run(y_pred, feed_dict = {x: x_np})
    
score = accuracy_score(y_np, y_pred_np)
print('Classification accuracy: %f' % score)


# Plot the results
plt.clf()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Learned model (Classification accuracy: 1.00)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
# Plot zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color = 'blue')
plt.scatter(x_ones[:, 0], x_ones[:, 1], color = 'red')

x_left = -2
y_left = (1.0 / w_final[1]) * (-b_final + logit(0.5) - w_final[0] * x_left)

x_right = 2
y_right = (1.0 / w_final[1]) * (-b_final + logit(0.5) - w_final[0] * x_right)

plt.plot([x_left, x_right], [y_left, y_right], color = 'k')

plt.savefig('logistic_pred.png')
