import argparse
import tensorflow as tf
import tensorflow.keras as tfk

from distribution import GaussianFixedDiagVar
from scod import SCOD
from experiments.rotated_mnist.train import RotatedMNIST

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="path to load model from")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="path to store output model with uncertainty")
    parser.add_argument("--repeat", type=int, default=5,
                        help="train dataset repeat count for sampling")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = tfk.models.load_model(args.model)

    rot_mnist = RotatedMNIST(2, 128)
    train_ds = rot_mnist.train_ds.map(lambda x, y: x).repeat(args.repeat)

    # calibrate input data distribution
    output_dist = GaussianFixedDiagVar()
    scod_model = SCOD(
        model=model,
        output_dist=output_dist,
        num_samples=rot_mnist.x_train_raw.shape[0] * args.repeat,
        num_eigs=40,
    )
    scod_model.process_dataset(train_ds)

    # calibrate prior scales
    cal_ds = RotatedMNIST(2, 32).test_ds
    cal_ds = cal_ds.concatenate(RotatedMNIST(4, 32).test_ds)
    cal_ds = cal_ds.concatenate(RotatedMNIST(9, 32).test_ds)
    cal_ds = cal_ds.shuffle(cal_ds.cardinality())
    scod_model.calibrate_prior(cal_ds)

    scod_model(model.input)
    scod_model.save(args.output)
