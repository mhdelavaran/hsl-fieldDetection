import tensorflow as tf
import cfg

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        # Define the layers for your network
        self.conv1 = tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation='relu', padding='same', name='conv1')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(24, (3, 3), activation='relu', padding='same', name='conv3')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.fc1 = tf.keras.layers.Dense(42, activation='relu', name='fc1')
        self.out = tf.keras.layers.Dense(10, name='out')

    def call(self, inputs):
        # Define the forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.out(x)
        return out

if __name__ == '__main__':
    # Create an instance of the Network class
    model = Network()

    # Initialize variables and create a session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Instantiate a writer for TensorBoard
        writer = tf.summary.FileWriter('./graphs', sess.graph)