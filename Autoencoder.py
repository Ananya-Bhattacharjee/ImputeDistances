import random
import easygui
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1_l2

from collections import defaultdict
def make_reconstruction_loss(n_features):

    def reconstruction_loss(input_and_mask, y_pred):
        X_values = input_and_mask[:, :n_features]
        #X_values.name = "$X_values"

        missing_mask = input_and_mask[:, n_features:]
        #missing_mask.name = "$missing_mask"
        observed_mask = 1 - missing_mask
        #observed_mask.name = "$observed_mask"

        X_values_observed = X_values * observed_mask
        #X_values_observed.name = "$X_values_observed"

        pred_observed = y_pred * observed_mask
        #pred_observed.name = "$y_pred_observed"

        return mse(y_true=X_values_observed, y_pred=pred_observed)
    return reconstruction_loss

def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))

class Autoencoder:

    def __init__(self, data,
                 recurrent_weight=0.78,
                 optimizer="adam",
                 dropout_probability=0.78,
                 hidden_activation="relu",
                 output_activation="sigmoid",
                 init="glorot_normal",
                 l1_penalty=0,
                 l2_penalty=0):
        self.data = data.copy()
        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.init = init
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def _get_hidden_layer_sizes(self):
        n_dims = self.data.shape[1]
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _create_model(self):

        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]

        model = Sequential()

        model.add(Dense(
            first_layer_size,
            input_dim= 2 * n_dims,
            activation=self.hidden_activation,
            W_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
            init=self.init))
        model.add(Dropout(self.dropout_probability))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(
                layer_size,
                activation=self.hidden_activation,
                W_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                init=self.init))
            model.add(Dropout(self.dropout_probability))

        model.add(Dense(
            n_dims,
            activation=self.output_activation,
            W_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
            init=self.init))

        loss_function = make_reconstruction_loss(n_dims)

        model.compile(optimizer=self.optimizer, loss=loss_function)
        return model

    def fill(self, missing_mask):
        self.data[missing_mask] = -1

    def _create_missing_mask(self):
        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)

        return np.isnan(self.data)

    def _train_epoch(self, model, missing_mask, batch_size):
        input_with_mask = np.hstack([self.data, missing_mask])
        n_samples = len(input_with_mask)
        n_batches = int(np.ceil(n_samples / batch_size))
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = input_with_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            model.train_on_batch(batch_data, batch_data)
        return model.predict(input_with_mask)

    def train(self, batch_size=256, train_epochs=100):
        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(train_epochs):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mae = masked_mae(X_true=self.data,
                                    X_pred=X_pred,
                                    mask=observed_mask)
            if epoch % 50 == 0:
                print("observed mae:", observed_mae)

            old_weight = (1.0 - self.recurrent_weight)
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing
        return self.data.copy()


def read_words(filename):

    last = ""
    with open(filename) as inp:
        print(filename)
        while True:
            buf = inp.read(10240)
            if not buf:
                break
            words = (last+buf).split()
            last = words.pop()
            for word in words:
                yield word
        yield last
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



i=0
j=0
numg = -1

filename = easygui.fileopenbox()

for word in read_words(filename):
    if (is_number(word)):
        if (i == 0):
            numg = int(word)
            break
        else:
            print("Wrong format")
            exit()
R = np.zeros(shape=(numg, numg))
print (numg)
for word in read_words(filename):
    if(is_number(word)):
        if(i==0):
            continue
        #print((word))
        R[i-1][j]=(float)(word)
        R[j][i-1]=R[i-1][j]
        j=j+1
    else:
        #print(word.__len__())
        if(word.__len__()>1):
            i=i+1
            j=0
        else:

            R[i-1][j]=np.NAN
            R[j][i-1]=np.NAN
            j=j+1


R_missing=R

df=pd.DataFrame(R)
#print(df)
#df = df.drop(['sroot'], axis=1)
'''prob_missing = 0.1
df_incomplete = df.copy()
ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
    df_incomplete.iat[row, col] = np.nan
    df_incomplete.iat[col, row] = np.nan
'''
df_incomplete = df.copy()

print(df_incomplete)

missing_encoded = pd.get_dummies(df_incomplete)
print(missing_encoded)

imputer = Autoencoder(missing_encoded.values)
complete_encoded = imputer.train(train_epochs=10000, batch_size=numg)

printed = ""
i = 0
j = 0


f = open(filename[:-4] + "CompletedEncoder.dis", "w+")
printed = ""

i = 0
j = 0
missing=0
for word in read_words(filename):
    if (is_number(word)):
        if (i == 0):
            printed = printed + word
            continue
        # print((word))
        R[i - 1][j] = (float)(word)
        printed = printed + " " + word
        j = j + 1
    else:
        # print(word.__len__())
        if (word.__len__() > 1):
            printed = printed + "\n" + word
            i = i + 1
            j = 0
        else:

            R[i - 1][j] = -1
            printed = printed + " " + str(round((complete_encoded[i-1][j]+complete_encoded[j][i-1])/2, 5))
            missing = missing + 1
            j = j + 1

f.write(printed)
f.close

#prob_missing = 0.1
#ix = [(row, col) for row in range(R.shape[0]) for col in range(R.shape[1])]
#for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
#    R_missing[row][col] = np.nan
#    R_missing[col][row] = np.nan

#missing_encoded = pd.get_dummies(R_missing)

#print (R_missing)


'''
for col in df.columns:
    missing_cols = missing_encoded.columns.str.startswith(str(col) + "_")
    missing_encoded.loc[df_incomplete[col].isnull(), missing_cols] = np.nan

imputer = Autoencoder(R_missing)
complete_encoded = imputer.train(train_epochs=300, batch_size=256)

print(R)
'''