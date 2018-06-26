from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle


NUM_EPOCHS = 10
BATCH_SIZE = 100
LATENT_DIM = 292
RANDOM_SEED = 1337

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    return parser.parse_args()

def main():
    np.random.seed(RANDOM_SEED)
    
    data_train, data_test, charset = load_dataset('data/processed.h5')
    print("Charset", charset)
    model = MoleculeVAE()
    model.create(charset, latent_rep_size = 292)


    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)
    checkpointer = ModelCheckpoint(filepath='model.h5',
                                   verbose=1,
                                   save_best_only=True)

    history= model.autoencoder.fit(
        data_train[:1000],
        data_train[:1000],
        shuffle=True,
        nb_epoch=NUM_EPOCHS,
        batch_size=100,
        callbacks=[checkpointer, reduce_lr],
        validation_data=(data_test[:1000], data_test[:1000])
    )
    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == '__main__':
    main()
