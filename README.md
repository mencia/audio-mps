# audio-mps
Audio Synthesis with continuous Matrix Product States https://arxiv.org/abs/1911.11879

## Software

* TensorFlow
* Python 3.6.6

## Repository contents

* model.py

It defines our RNN model and contains other methods, to sample from the trained model for example.

* train.py

It performs the training.

* data.py

It creates a synthetic dataset or reads in real data.

* reader.py

When using real data (like Nsynth), it picks out a subset of the data, that contains a desired feature: pitch, instrument ...

* make-small-dataset.py

When using real data (like Nsynth), it imports reader.py to make the dataset.
