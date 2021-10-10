#!/usr/bin/env python3

from sliceq22.model import SliceQ22Model
from sliceq22.audio import plot_slicq
import os
import json
import random
import argparse
from sliceq22.musdb import get_musdbhq
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sliceq22 trainer")

    parser.add_argument(
        "--musdbhq-root",
        type=str,
        default="~/MUSDB18-HQ",
        help="path to musdbhq",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="./sliceq22-train",
        help="train dir to store weights and checkpoints",
    )
    parser.add_argument(
        "--nsg-config",
        type=str,
        default="./nsg_params.json",
        help="path to nsgconstantq parameter file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="random seed (stdlib and numpy)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="sample rate",
    )
    parser.add_argument(
        "--inference-file",
        type=str,
        default=None,
        help="pass wav file to perform inference"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="pass file to trained model"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    nsg_params = None
    with open(args.nsg_config, 'r') as f:
        nsg_params = json.load(f)

    musdb_dataset = get_musdbhq(args.musdbhq_root)

    s22 = SliceQ22Model(
        args.train_dir,
        nsg_params,
        musdb_dataset,
        sample_rate=args.sample_rate,
        inference=(args.inference_file is not None),
        model_file=args.model_file
    )

    if args.inference_file is None:
        s22.train()
    else:
        x, y_gt, y_pred = s22.inference(args.inference_file)
        plot_slicq(x, y_gt, y_pred)
