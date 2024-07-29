# QCML
## For running on Classical Systems
To implement Hybrid Quantum Neural Networks for Remote Sensing Imagery Classification. 

#### Libraries required such as Qiskit, Pennylane, TensorFlow, and Keras.

```python
pip install qiskit pennylane tensorflow numpy matplotlib
pip install tensorflow_datasets
```
#### Step 2: Prepare the Data
Use a remote sensing dataset. For this example, we will use the EuroSAT dataset available via TensorFlow Datasets.

<img src="https://github.com/abhinandan0y/QCML/blob/main/img/Eurosat.jpg" style="width: 100%;" alt="EuroSAT.jpg">

```python

import tensorflow as tf
import tensorflow_datasets as tfds
import pennylane as qml
import numpy as np
from tensorflow.keras import layers, models

# Load EuroSAT dataset
dataset, info = tfds.load('eurosat/rgb', with_info=True)
#train_data = dataset['train']

train_data, val_data = tfds.load('eurosat/rgb', split=['train[:80%]', 'train[80%:]'], as_supervised=True)

# Preprocess the data
#def preprocess(features):
#    image = tf.image.resize(features['image'], (64, 64)) / 255.0
#    label = features['label']
#    return image, label

def preprocess(image, label):
    image = tf.image.resize(image, (64, 64)) / 255.0
    return image, label

train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
```

#### Step 3: Define the Quantum Layer
#Here, define a simple quantum circuit using Pennylane.

```python

import pennylane as qml

n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[2, 3])
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def quantum_layer(inputs):
    inputs = tf.cast(inputs, dtype=tf.float32)
    outputs = np.array([quantum_circuit(input) for input in inputs])
    outputs = outputs.astype(np.float32)  # Ensure the numpy array is float32
    return tf.convert_to_tensor(outputs, dtype=tf.float32)
```

#### Step 4: Integrate Quantum Layer with Classical Model
#integrate the quantum layer into a Keras model.

```python

from tensorflow.keras import layers, models
import numpy as np

class HybridModel(tf.keras.Model):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(n_qubits, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.numpy_function(quantum_layer, [x], tf.float32)
        x.set_shape((None, n_qubits))  # Ensure shape is set for the output of quantum layer
        x = self.dense2(x)
        return x

model = HybridModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

#### model details
```python
>>> model.summary()
Model: "hybrid_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             multiple                  896       
                                                                 
 max_pooling2d (MaxPooling2  multiple                  0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           multiple                  18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  multiple                  0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           multiple                  0         
                                                                 
 dense (Dense)               multiple                  50180     
                                                                 
 dense_1 (Dense)             multiple                  50        
                                                                 
=================================================================
Total params: 69622 (271.96 KB)
Trainable params: 69622 (271.96 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

#### Save the model
```python
model.save('hybrid_model.h5') #NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. It does not work for subclassed models, because such models are defined via the body of a Python method, which isn't safely serializable. Consider saving to the Tensorflow SavedModel format (by setting save_format="tf") or using `save_weights`.

# Save the model in TensorFlow SavedModel format
model.save('saved_model', save_format='tf')
>>> model.save('QCML_v1_model', save_format='tf')
INFO:tensorflow:Assets written to: QCML_v1_model/assets
INFO:tensorflow:Assets written to: QCML_v1_model/assets

or simply
model.save('saved_model')

# Load the model
loaded_model = tf.keras.models.load_model('saved_model', custom_objects={'quantum_layer': quantum_layer})
```

#### Load the saved model
```python
loaded_model = tf.keras.models.load_model('hybrid_model.h5', custom_objects={'quantum_layer': quantum_layer})

# Evaluate the loaded model to verify it works correctly
val_loss, val_accuracy = loaded_model.evaluate(val_data)

print("Loaded Model Validation Loss:", val_loss)
print("Loaded Model Validation Accuracy:", val_accuracy)
```

#### Tain & Monitor Training Progress:
TensorBoard is used to monitor training progress.

```bash
#Start tensorboard
tensorboard --logdir logs/fit
```
```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Add TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)


# Train the model with TensorBoard callback
model.fit(train_data, epochs=10, callbacks=[tensorboard_callback])

or

# Custom training loop
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Initialize the summary writer
summary_writer = tf.summary.create_file_writer(log_dir)

# Custom training loop with TensorBoard callback
epochs = 10
for epoch in range(epochs):
    print(f'Start of epoch {epoch+1}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y_batch_train, logits)
        
        # Log the loss and accuracy to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=epoch * len(train_data) + step)
            tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch * len(train_data) + step)
        
        if step % 100 == 0:
            print(f'Epoch {epoch+1} Step {step} Loss {loss_value.numpy()} Accuracy {train_acc_metric.result().numpy()}')
    
    train_acc = train_acc_metric.result()
    print(f'Training accuracy over epoch {epoch+1}: {train_acc.numpy()}')
    train_acc_metric.reset_states()

You can then visualize the training metrics using the TensorBoard interface by running tensorboard --logdir=./logs in the terminal and navigating to http://localhost:6006 in your web browser.
```

#### Training model : Custom training loop
```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 10

for epoch in range(epochs):
    print(f'Start of epoch {epoch+1}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y_batch_train, logits)
        if step % 100 == 0:
            print(f'Epoch {epoch+1} Step {step} Loss {loss_value.numpy()} Accuracy {train_acc_metric.result().numpy()}')
    train_acc = train_acc_metric.result()
    print(f'Training accuracy over epoch {epoch+1}: {train_acc.numpy()}')
    train_acc_metric.reset_states()
```

#### Tensorboard Training view
<img src="https://github.com/abhinandan0y/QCML/blob/main/img/TrainAccLoss.png" style="width: 100%;" alt="TrainingResults.png">

#### Step 5: Test/ Evaluate the Model

```python

# Preprocess the validation data
#val_data = val_data.map(preprocess).batch(32)
val_data = val_data.map(lambda image, label: preprocess(image, label)).batch(32)

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_data)

# Print the validation loss and accuracy
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Close the summary writer
summary_writer.close()
```

#### Results:

```bash
169/169 [==============================] - 65s 381ms/step - loss: 2.2502 - accuracy: 0.1667
>>> # Print the validation loss and accuracy
>>> print("Validation Loss:", val_loss)
Validation Loss: 2.250182867050171
>>> print("Validation Accuracy:", val_accuracy)
Validation Accuracy: 0.1666666716337204
```

#### Hybrid quantum-classical model for image classification using EuroSAT dataset. Here are some suggestions for potential Improvements:
```
#Data Augmentation:
    Incorporate data augmentation techniques like rotation, flipping, and scaling to increase the diversity of the training dataset. This can help improve model generalization.

#Complexity of Quantum Circuit: 
    Experiment with different quantum circuit architectures, including varying the number of qubits, layers, and types of gates. More complex quantum circuits may capture richer features from the data.

#Hyperparameter Tuning: 
    Tune hyperparameters such as learning rate, batch size, and the number of epochs to find the optimal configuration for your model.

#Regularization: 
    Apply regularization techniques like dropout or L2 regularization to prevent overfitting and improve model robustness.

#Ensemble Learning: 
    Train multiple hybrid models with different initializations or architectures and combine their predictions to boost performance.

#Quantum Embedding: 
    Explore methods to embed classical data into quantum states more effectively, such as amplitude encoding or quantum feature maps.

#Transfer Learning: 
    Utilize pre-trained classical convolutional neural networks (CNNs) as feature extractors before passing the features to the quantum layer.

#Monitor Training Progress: 
    Visualize training metrics like loss and accuracy over epochs using tools like TensorBoard to identify potential issues or areas for improvement.
```
## For Running on GOOGLE Quantum computer

```bash
#Create environment
python -m venv /mnt/India/bioinfo/tests/qiskit-1.0-venv 
source /mnt/India/bioinfo/tests/qiskit-1.0-venv/bin/activate
#python
pip install Cirq
pip install pennylane tensorflow numpy matplotlib tensorflow_datasets pennylane-qiskit
```
#### Step 2: Prepare the Data

```python

import tensorflow as tf
import tensorflow_datasets as tfds
import pennylane as qml
import cirq
import numpy as np
import itertools
from tensorflow.keras import layers, models

# Load EuroSAT dataset
dataset, info = tfds.load('eurosat/rgb', with_info=True)
train_data, val_data = tfds.load('eurosat/rgb', split=['train[:80%]', 'train[80%:]'], as_supervised=True)

# Preprocess the data
def preprocess(image, label):
    image = tf.image.resize(image, (64, 64)) / 255.0
    return image, label

train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_data.map(preprocess).batch(32)

```
#### Step 3: Define the Quantum Layer
```python
def quantum_layer(inputs):
    inputs = tf.cast(inputs, dtype=tf.float32)
    inputs_np = inputs.numpy()
    outputs = np.array([quantum_circuit(input_np) for input_np in inputs_np])
    outputs = outputs.astype(np.float32)
    return tf.reshape(tf.convert_to_tensor(outputs, dtype=tf.float32), (-1, n_qubits))
```

#### Define the quantum circuit and execute on Google Quantum Computer.
```python
n_qubits = 4
simulator = cirq.Simulator()

def quantum_circuit(inputs):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(float(inputs[i]))(qubit))  # Convert inputs to float
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.CZ(qubits[2], qubits[3]))
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(float(inputs[i]))(qubit))  # Convert inputs to float
    circuit.append(cirq.measure(*qubits, key='m'))
    result = simulator.run(circuit, repetitions=1000)
    counts = result.histogram(key='m')
    probabilities = np.zeros(2**n_qubits)
    for i, state in enumerate(itertools.product([0, 1], repeat=n_qubits)):
        qubit_indices = [i for i, q in enumerate(qubits) if state[i] == 1]
        if len(qubit_indices) > 0:
            state_str = ''.join(str(int(q)) for q in state)
            probabilities[i] = counts[state_str] / 1000
    return probabilities

```
 <img src="https://github.com/abhinandan0y/QCML/blob/main/img/circuit_diagram.png" style="width: 100%;" alt="circuit_diagram.png">

 
#### Step 4: Define the hybrid model
```python
class HybridModel(tf.keras.Model):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(n_qubits, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = quantum_layer(x)
        x = self.dense2(x)
        return x

# Compile the model
model = HybridModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
#### Step 5: Training and Evaluation

```python

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 10

for epoch in range(epochs):
    print(f'Start of epoch {epoch+1}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y_batch_train, logits)
        if step % 100 == 0:
            print(f"Epoch {epoch+1} Step {step} Loss {loss_value.numpy()} Accuracy {train_acc_metric.result().numpy()}")
    train_acc = train_acc_metric.result()
    print(f"Training accuracy over epoch {epoch+1}: {train_acc.numpy()}")
    train_acc_metric.reset_states()

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_data)

# Print the validation loss and accuracy
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

```
### Results on Cirq Google Quantum Computer Simulator
```
Epoch 1 Step 300 Loss 2.3077473640441895 Accuracy 0.11503322422504425

**Training accuracy over epoch 1: 0.11074074357748032**
```
