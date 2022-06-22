import tensorflow.keras as keras
from os.path import join, abspath, dirname
import pickle

file_dir = dirname(abspath(__file__))

model = keras.models.load_model(join(file_dir, 'ToneNet.hdf5'))
print(model.summary())
with open('ToneNet.pickle', 'wb') as f:
    pickle.dump(model.get_weights(), f)
