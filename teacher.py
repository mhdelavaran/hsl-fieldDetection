import pathlib
import numpy as np  # Import numpy
import tensorflow as tf
import cfg
import network
import dataManager
import sys  # Import the sys module

class Teacher:
    def __init__(self, nTry):
        self.nTry = nTry
        self.model = network.Network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.initLR)
        self.num_epochs = cfg.nEpoch

        self.dataSet = dataManager.DataManager()
        self.dataSet.loadDataSet()
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.dataSet.trainSet, self.dataSet.trainSetPrimePointLabel, self.dataSet.trainSetSecondPointUsedLabel))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(cfg.batchSize)

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.valLoss = 0

    def train(self):
        for epoch in range(self.num_epochs):
            for batch in self.train_dataset:
                inputs, labels, used_second_point = batch
                with tf.GradientTape() as tape:
                    predictions = self.model(inputs)
                    loss = self.loss_fn(labels, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Calculate validation loss
            val_inputs = np.array(self.dataSet.valSet) 
            val_labels = np.concatenate(self.dataSet.valSetPrimePointLabel, self.dataSet.valSetSecondPointLabel)
            val_used_second_point = np.array(self.dataSet.valSetSecondPointUsedLabel)
            
            val_predictions = self.model(val_inputs)
            self.valLoss = self.loss_fn(val_labels, val_predictions)
            print(f'Epoch {epoch}, Validation Loss: {self.valLoss:.4f}')

        # Save model weights
        self.model.save_weights('./models/day1/' + str(self.nTry))

if __name__ == '__main__':
    teacher = Teacher(sys.argv[1])
    teacher.train()