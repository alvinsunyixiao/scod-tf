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
    parser.add_argument("--repeat", type=int, default=20,
                        help="train dataset repeat count for sampling")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = tfk.models.load_model(args.model)
    rot_mnist = RotatedMNIST(2, 1024)
    train_ds = rot_mnist.train_ds.map(lambda x, y: x).repeat(args.repeat)

    output_dist = GaussianFixedDiagVar()
    scod = SCOD(
        model=model,
        output_dist=output_dist,
        dataset=train_ds,
        num_samples=rot_mnist.x_train_raw.shape[0] * args.repeat,
    )
    print(model.summary())
