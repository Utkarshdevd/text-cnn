from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam
from config import Config
from sklearn.model_selection import StratifiedKFold

print("Loading the configurations...")

config = Config()
conv_layers = config.model.conv_layers
fully_layers = config.model.fully_connected_layers
l0 = config.l0
alphabet_size = config.alphabet_size
embedding_size = config.model.embedding_size
num_of_classes = config.num_of_classes
th = config.model.th
p = config.dropout_p
print("Loaded")

from data_loader import Data

all_data = Data(data_source = config.train_data_source, 
                     alphabet = config.alphabet,
                     l0 = config.l0,
                     batch_size = 0,
                     no_of_classes = config.num_of_classes)

all_data.loadData()
seed = 7
X, Y = all_data.getAllData()
kfold = StratifiedKFold(n_splits=config.kfolds, shuffle=True, random_state=seed)

for train, test in kfold.split(X, Y.reshape(Y.shape[0],)):
    print("Building the model...")
    # building the model

    # Input layer
    inputs = Input(shape=(l0,), name='sent_input', dtype='int64')

    # Embedding layer

    x = Embedding(alphabet_size + 1, embedding_size, input_length=l0)(inputs)

    # Convolution layers
    for cl in conv_layers:
        x = Convolution1D(cl[0], cl[1])(x)
        x = ThresholdedReLU(th)(x)
        if not cl[2] is None:
            x = MaxPooling1D(cl[2])(x)

    x = Flatten()(x)

    #Fully connected layers

    for fl in fully_layers:
        x = Dense(fl)(x)
        x = ThresholdedReLU(th)(x)
        x = Dropout(0.5)(x)

    predictions = Dense(num_of_classes, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print("Built")

    print("Training ...")
    model.fit(X[train], Y[train], epochs=config.training.epochs, verbose=1, batch_size=config.batch_size, validation_data=(X[test], Y[test]))

    print("Done!.")
