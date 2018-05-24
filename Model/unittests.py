import numpy as np
from cgan_tensorflow import *

import unittest
from unnecessary_math import multiply
import tensorflow as tf

G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)
 
class TestUM(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_shape(self):
        self.assertEqual(np.shape(D_real), np.shape(D_logit_real))
 
    def test_value(self):
        try:
          int(G_sample)
          return True
        except ValueError:
          return False
 
if __name__ == '__main__':
    unittest.main()
