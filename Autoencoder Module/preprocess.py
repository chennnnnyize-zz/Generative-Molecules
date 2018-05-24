import argparse
import pandas
import h5py
import numpy as np
from molecules.utils import one_hot_array, one_hot_index

from sklearn.model_selection import train_test_split

MAX_NUM_ROWS = 500000
SMILES_COL_NAME = 'structure'

def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('infile', default = 'data/smiles_50k.h5', type=str, help='Input file name')
    parser.add_argument('outfile', default='data/processed.h5', type=str, help='Output file name')
    parser.add_argument('--length', type=int, metavar='N', default = MAX_NUM_ROWS,
                        help='Maximum number of rows to include (randomly sampled).')
    parser.add_argument('--smiles_column', type=str, default = SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument('--property_column', type=str,
                        help="Name of the column that contains the property values to predict. Default: None")
    return parser.parse_args()

def chunk_iterator(dataset, chunk_size=1000):
    chunk_indices = np.array_split(np.arange(len(dataset)),
                                    len(dataset)/chunk_size)
    for chunk_ixs in chunk_indices:
        chunk = dataset[chunk_ixs]
        yield (chunk_ixs, chunk)
    raise StopIteration

def main():
    #args = get_arguments()
    data = pandas.read_hdf('data/smiles_50k.h5', 'table')
    keys = data[SMILES_COL_NAME].map(len) < 121

    if MAX_NUM_ROWS <= len(keys):
        data = data[keys].sample(n = MAX_NUM_ROWS)
    else:
        data = data[keys]

    structures = data[SMILES_COL_NAME].map(lambda x: list(x.ljust(120)))


    del data

    train_idx, test_idx = map(np.array,
                              train_test_split(structures.index, test_size = 0.20))

    charset = list(reduce(lambda x, y: set(y) | x, structures, set()))

    one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
                                                one_hot_index(row, charset))

    h5f = h5py.File('data/processed.h5', 'w')
    h5f.create_dataset('charset', data = charset)

    def create_chunk_dataset(h5file, dataset_name, dataset, dataset_shape,
                             chunk_size=1000, apply_fn=None):
        new_data = h5file.create_dataset(dataset_name, dataset_shape,
                                         chunks=tuple([chunk_size]+list(dataset_shape[1:])))
        for (chunk_ixs, chunk) in chunk_iterator(dataset):
            if not apply_fn:
                new_data[chunk_ixs, ...] = chunk
            else:
                new_data[chunk_ixs, ...] = apply_fn(chunk)

    create_chunk_dataset(h5f, 'data_train', train_idx,
                         (len(train_idx), 120, len(charset)),
                         apply_fn=lambda ch: np.array(map(one_hot_encoded_fn,
                                                          structures[ch])))
    create_chunk_dataset(h5f, 'data_test', test_idx,
                         (len(test_idx), 120, len(charset)),
                         apply_fn=lambda ch: np.array(map(one_hot_encoded_fn,
                                                          structures[ch])))
    h5f.close()

if __name__ == '__main__':
    main()
