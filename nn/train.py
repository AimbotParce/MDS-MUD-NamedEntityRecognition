#! /usr/bin/python3

import sys
from contextlib import redirect_stdout
from itertools import batched
from typing import Generator, Iterable

import keras.losses
import tensorflow as tf
from codemaps import *
from colorama import Style
from dataset import *
from keras import Input
from keras.layers import Conv1D, Dense, Dropout, Embedding, Reshape, TimeDistributed, concatenate
from keras.models import Model


def weighted_sparse_categorical_crossentropy(class_weights: NDArray[np.float32]):
    """
    Custom loss function to handle class imbalance.
    """
    cw_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Compute the losses for all the predictions
        loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # Shape: (batch_size, max_len)
        # Compute the class weights
        weights = tf.gather(cw_tensor, tf.cast(y_true, tf.int32))
        # Multiply the loss by the class weights
        loss = loss * weights
        return tf.reduce_mean(loss, axis=-1)

    return loss


def build_network(codes: Codemaps) -> Model:
    # sizes
    n_words = codes.get_n_words()
    n_labels = codes.get_n_labels()
    max_len = codes.maxlen

    # Embed the words themselves
    input_subwords = Input(shape=(max_len,), name="input_subwords")
    x = Embedding(input_dim=n_words, output_dim=200, input_length=max_len, mask_zero=False, name="embed_subwords")(
        input_subwords
    )
    x = Dropout(0.1)(x)  # Add a dropout layer to prevent overfitting

    # Operations with the neighbors, to modify embeddings to get contextual information
    x = Conv1D(filters=200, kernel_size=9, padding="same", activation="relu")(x)
    x = Dropout(0.1)(x)  # Add a dropout layer to prevent overfitting
    x = Conv1D(filters=200, kernel_size=7, padding="same", activation="relu")(x)
    x = Dropout(0.1)(x)  # Add a dropout layer to prevent overfitting
    x = Conv1D(filters=200, kernel_size=5, padding="same", activation="relu")(x)
    x = Dropout(0.1)(x)  # Add a dropout layer to prevent overfitting

    # Output layer: Operation with only the word's embedding to convert it to a label
    x = TimeDistributed(Dense(50, activation="relu"))(x)
    x = Dropout(0.1)(x)  # Add a dropout layer to prevent overfitting
    x = TimeDistributed(Dense(n_labels, activation="softmax"), name="output")(x)

    # build and compile model
    model = Model(input_subwords, x)
    return model


def _batch_words(words: Iterable[str], max_len: int, pad_length: int = 0) -> Generator[list[str], None, None]:
    """
    Batch words into lists of lists of strings, each with a maximum length of max_len.
    """
    batch = []
    current_length = 0
    for word in words:
        word_length = len(word) + pad_length
        if current_length + word_length > max_len:
            yield batch
            batch = []
            current_length = 0
        batch.append(word)
        current_length += word_length
    if batch:
        yield batch


def print_sentences(sentences: Iterable[TaggedTokenDict]):
    for sentence in sentences:
        words = [f"{token['form']:<{max(len(token['form']),len(token['tag']))}s}" for token in sentence]
        tags = [f"{token['tag']:<{max(len(token['form']),len(token['tag']))}s}" for token in sentence]
        cols = os.get_terminal_size().columns
        for wl, tl in zip(_batch_words(words, cols, pad_length=1), _batch_words(tags, cols, pad_length=1)):
            print("│".join(wl), file=sys.stderr)
            print("│".join(f"{Style.DIM}{t}{Style.RESET_ALL}" for t in tl), file=sys.stderr)
        print("")


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
    max_len = max(len(sentence) for sentence in traindata.sentences())
    # This isn't really standard, but meh, we'll just use the max length of the training data
    codes = Codemaps(traindata, max_len)

    # build network
    model = build_network(codes)
    with redirect_stdout(sys.stderr):
        model.summary()

    class_weights = np.zeros(codes.get_n_labels(), dtype=np.float32)
    for label, index in codes.label_index.items():
        if label not in ["PAD", "UNK"]:
            class_weights[index] = 1 / np.sqrt(codes.class_counts[label])

    class_weights[codes.label_index["UNK"]] = 0.0
    class_weights = class_weights / np.sum(class_weights)

    print(f"{'Class':<10s}  {'Count':<10s}  Weight", file=sys.stderr)
    for label, index in codes.label_index.items():
        print(
            f"{label:<10s}  {str(codes.class_counts.get(label, "NA")):<10s}  {class_weights[index]:.4f}",
            file=sys.stderr,
        )

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
