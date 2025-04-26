#! /usr/bin/python3

import sys
from contextlib import redirect_stdout

import tensorflow as tf
from codemaps import *
from dataset import *
from keras import Input
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Reshape, TimeDistributed, concatenate
from keras.models import Model


def weighted_sparse_categorical_crossentropy(class_weights: dict[int, float]):
    """
    Custom loss function to handle class imbalance.
    """
    max_class = max(class_weights.keys())
    cw_tensor = tf.constant(list(class_weights[k] for k in range(max_class + 1)), dtype=tf.float32)

    def loss(y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        # Convert y_true to int32
        y_true = tf.cast(y_true, tf.int32)
        # Get the class weights for the true labels
        weights = tf.gather(cw_tensor, y_true)
        # Compute the sparse categorical crossentropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # Multiply the loss by the class weights
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

    return loss


def build_network(codes: Codemaps) -> Model:
    # sizes
    n_words = codes.get_n_words()
    n_sufs = codes.get_n_sufs()
    n_labels = codes.get_n_labels()
    n_pos_tags = codes.get_n_pos_tags()
    max_len = codes.maxlen

    # Embed the words themselves
    input_words = Input(shape=(max_len,), name="input_words")
    emb_word = Embedding(
        input_dim=n_words,
        output_dim=100,
        input_length=max_len,
        mask_zero=False,
        name="embed_words",
    )(input_words)
    emb_word = Dropout(0.1)(emb_word)  # Add a dropout layer to prevent overfitting

    # Embed the pos tags
    input_pos = Input(shape=(max_len,), name="input_pos")
    emb_pos = Embedding(
        input_dim=n_pos_tags,
        output_dim=n_pos_tags,
        input_length=max_len,
        mask_zero=False,
        name="embed_pos",
    )(input_pos)

    # Modify the word embeddings to include the pos tag embeddings
    emb_word_pos = concatenate([emb_word, emb_pos])
    # Apply a some dense layers here to mix the embeddings

    emb_word_pos = TimeDistributed(Dense(100, activation="relu"))(emb_word_pos)
    emb_word_pos = Dropout(0.1)(emb_word_pos)  # Add a dropout layer to prevent overfitting
    emb_word_pos = TimeDistributed(Dense(100, activation="relu"))(emb_word_pos)
    emb_word_pos = Dropout(0.1)(emb_word_pos)
    # At this point, emb_word_pos is a 3D tensor of shape (batch_size, max_len, 100) which would be a disambiguated
    # embedding of the word.

    # Embed the suffixes
    input_suffixes = Input(shape=(max_len,), name="input_suffixes")
    emb_suffixes = Embedding(
        input_dim=n_sufs,
        output_dim=50,
        input_length=max_len,
        mask_zero=False,
        name="embed_suffixes",
    )(input_suffixes)
    emb_suffixes = Dropout(0.1)(emb_suffixes)

    # And add the length of the word as a feature
    # This is a 2D tensor of shape (batch_size, max_len)
    input_length = Input(shape=(max_len,), name="input_length")
    # Reshape it to be a 3D tensor of shape (batch_size, max_len, 1)
    res_length = Reshape((max_len, 1))(input_length)

    # Input length will be a single integer at the end... I don't think it'll be that useful
    x = concatenate([emb_word_pos, emb_suffixes, res_length])

    # biLSTM
    x = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))(x)
    # output softmax layer
    x = TimeDistributed(Dense(n_labels, activation="softmax"))(x)

    # build and compile model
    model = Model([input_words, input_suffixes, input_pos, input_length], x)
    return model


if __name__ == "__main__":
    ## --------- MAIN PROGRAM -----------
    ## --
    ## -- Usage:  train.py ../data/Train ../data/Devel  modelname
    ## --

    # directory with files to process
    traindir = sys.argv[1]
    validationdir = sys.argv[2]
    modelname = sys.argv[3]

    # load train and validation data
    traindata = Dataset(traindir)
    valdata = Dataset(validationdir)

    # create indexes from training data
    max_len = 150
    suf_len = 5
    codes = Codemaps(traindata, max_len, suf_len)

    class_weights = {}
    min_class_count = min(codes.class_counts.values())
    print("Class Counts -> Weights:", file=sys.stderr)
    for i, label in enumerate(codes.label_index):
        if label in ["PAD", "UNK"]:
            class_weights[i] = 1.0
        else:
            class_weights[i] = min_class_count / codes.class_counts[label]
        print(f"  {label}: {codes.class_counts.get(label, "NA")} -> {class_weights[i]}", file=sys.stderr)

    # build network
    model = build_network(codes)
    with redirect_stdout(sys.stderr):
        model.summary()

    model.compile(optimizer="adam", loss=weighted_sparse_categorical_crossentropy(class_weights), metrics=["accuracy"])

    # encode datasets
    Xt = codes.encode_words(traindata)
    Yt = codes.encode_labels(traindata)
    Xv = codes.encode_words(valdata)
    Yv = codes.encode_labels(valdata)

    # train model
    with redirect_stdout(sys.stderr):
        model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv, Yv), verbose=1)

    # save model and indexs
    model.save(modelname)
    codes.save(modelname)
