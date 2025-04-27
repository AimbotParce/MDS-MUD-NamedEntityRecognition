#! /usr/bin/python3

import sys
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Concatenate
from keras.models import Model
from keras_crf import CRFModel

from codemaps import *
from dataset import *

def build_base_model(codes: Codemaps):
    max_len = codes.maxlen
    n_words = codes.get_n_words()
    n_sufs = codes.get_n_sufs()

    # Inputs
    word_input = Input(shape=(max_len,), dtype=tf.int32, name="word_input")
    suf_input = Input(shape=(max_len,), dtype=tf.int32, name="suf_input")

    # Embeddings
    word_emb = Embedding(input_dim=n_words, output_dim=100)(word_input)
    suf_emb = Embedding(input_dim=n_sufs, output_dim=50)(suf_input)

    # Concatenate embeddings
    merged = Concatenate()([word_emb, suf_emb])

    # BiLSTM layer
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=200, return_sequences=True))(merged)

    # Dense layer before CRF
    dense = Dense(128, activation='relu')(bilstm)

    # Output dense layer with number of labels as units
    n_labels = codes.get_n_labels()
    logits = Dense(n_labels)(dense)

    # Define the base model
    base_model = Model(inputs=[word_input, suf_input], outputs=logits)
    return base_model


if __name__ == "__main__":
    traindir = sys.argv[1]
    validationdir = sys.argv[2]
    modelname = sys.argv[3]

    # Load datasets
    traindata = Dataset(traindir)
    valdata = Dataset(validationdir)

    # Create codemaps (indexes)
    max_len = 150
    suf_len = 5
    codes = Codemaps(traindata, max_len, suf_len)

    # Build base model
    base_model = build_base_model(codes)

    # Build CRF model (num tags = number of labels)
    n_labels = codes.get_n_labels()
    model = CRFModel(base_model, n_labels)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        metrics=['acc']
    )

    model.summary()

    # Prepare data: encode words and labels
    Xt = codes.encode_words(traindata)
    Yt = codes.encode_labels(traindata)
    Xv = codes.encode_words(valdata)
    Yv = codes.encode_labels(valdata)

    # Train model
    model.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=32, epochs=1, verbose=1)

    # Save model and codemaps
    model.save(modelname, save_format="tf")
    codes.save(modelname)
