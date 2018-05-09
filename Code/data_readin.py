

import yaml
import time
import os
from data_input import vectorize_data
from hyperparameters import load_params
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_file',
                    help='experiment file', default='exp.json')
parser.add_argument('-d', '--directory',
                    help='exp directory', default=None)
args = vars(parser.parse_args())
if args['directory'] is not None:
    args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

params = load_params(args['exp_file'])
print("All params:", params)
# load data
X_train, X_test = vectorize_data(params)
print(X_train[0])

import matplotlib.pyplot as plt

plt.imshow(X_train[0].reshape(-1,35))
plt.show()