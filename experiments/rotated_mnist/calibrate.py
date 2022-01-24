import argparse
import tensorflow as tf
import tensorflow.keras as tfk

from distribution import GaussianFixedDiagVar
from scod import SCOD
from sketching import Sketch
from experiments.rotated_mnist.train import RotatedMNIST

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="path to load model from")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = tfk.models.load_model(args.model)
    rot_mnist = RotatedMNIST(2, 1024)


    output_dist = GaussianFixedDiagVar()
    sketch = Sketch(model.trainable_variables)
    scod = SCOD(model, output_dist, rot_mnist.train_ds, )
    print(model.summary())
