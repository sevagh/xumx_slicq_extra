import argparse
import torch
import subprocess
import time
from pathlib import Path
import tqdm
import json
import numpy as np
import random
from git import Repo
import os
import signal
import atexit
import gc
import io
import copy
import sys
import torchaudio
import torchinfo
from contextlib import nullcontext
import sklearn.preprocessing
from torch.utils.tensorboard import SummaryWriter

from slicqt import model, torch_utils
from slicqt import torch_transforms as transforms
from slicqt.loss import LossCriterion

tqdm.monitor_interval = 0


def loop(args, deoverlapnet, encoder, device, criterion, optimizer, train=True):
    # unpack encoder object
    slicqt, islicqt = encoder

    losses = torch_utils.AverageMeter()

    cm = None
    name = ''
    if train:
        deoverlapnet.train()
        cm = nullcontext
        name = 'Train'
    else:
        deoverlapnet.eval()
        cm = torch.no_grad
        name = 'Validation'

    if train:
        batch_size = args.batch_size
        nb_samples = int(args.seq_dur*args.sample_rate)
    else:
        batch_size = args.batch_size_valid
        nb_samples = int(args.seq_dur_valid*args.sample_rate)

    def custom_sampler():
        for _ in range(batch_size):
            yield torch.rand(batch_size, args.nb_channels, nb_samples, dtype=torch.float32, device=device)

    pbar = tqdm.tqdm(custom_sampler(), disable=args.quiet, total=batch_size)
    pbar.set_description(f"{name} batch")

    with cm():
        for x in pbar:
            if train:
                optimizer.zero_grad()

            X = slicqt(x)
            Xmag, Xphase = transforms.complex_2_magphase(X)
            nb_slices = Xmag[0].shape[-2]

            Xmag_ola = slicqt.overlap_add(Xmag)
            ragged_ola_shapes = [X_.shape for X_ in Xmag_ola]

            Xmag_hat = deoverlapnet(Xmag_ola, nb_slices, ragged_ola_shapes)

            with torch.no_grad():
                xhat = islicqt(
                    transforms.magphase_2_complex(Xmag_hat, Xphase),
                    x.shape[-1]
                )

            loss = criterion(
                Xmag_hat,
                Xmag,
                xhat,
                x,
            )

            if train:
                loss.backward()
                optimizer.step()

            losses.update(loss.item(), x.size(1))

    return losses.avg


def main():
    parser = argparse.ArgumentParser(description="sliCQT DeNet Trainer")

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-size-valid", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="maximum number of train epochs (default: 1000)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument('--mcoef-timedomain', type=float, default=1e-2,
                        help='coefficient for mixing: mcoef_timedomain*SNR_Loss + mcoef_freqdomain*MSE_Loss')
    parser.add_argument('--mcoef-freqdomain', type=float, default=1.,
                        help='coefficient for mixing: mcoef_timedomain*SNR_Loss + mcoef_freqdomain*MSE_Loss')

    # Model Parameters
    parser.add_argument(
        "--output",
        type=str,
        default="slicqt-deoverlapnet",
        help="provide output path base folder name",
    )
    parser.add_argument("--model", type=str, help="Path to checkpoint folder")
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=12.0,
        help="Sequence duration in seconds for train random waveforms",
    )
    parser.add_argument(
        "--seq-dur-valid",
        type=float,
        default=60.0,
        help="Sequence duration in seconds for validation random waveforms",
    )

    # sliCQT parameters
    parser.add_argument(
        f"--fscale",
        choices=('bark','mel', 'cqlog', 'vqlog', 'oct'),
        default='bark',
        help="frequency scale for sliCQ-NSGT",
    )
    parser.add_argument(
        f"--fbins",
        type=int,
        default=262,
        help="number of frequency bins for NSGT scale",
    )
    parser.add_argument(
        f"--fmin",
        type=float,
        default=32.9,
        help="min frequency for NSGT scale",
    )
    parser.add_argument(
        f"--gamma",
        type=float,
        default=15.,
        help="gamma (offset) for vqlog",
    )

    parser.add_argument(
        "--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=44100.,
        help="sample rate",
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    if use_cuda:
        print("Configuring NSGT to use GPU")

    try:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except:
        commit = 'n/a'

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    tboard_path = target_path / f"logdir"
    tboard_writer = SummaryWriter(tboard_path)

    # need to globally configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    slicqt_base = transforms.SliCQTBase(
        scale=args.fscale,
        fbins=args.fbins,
        fmin=args.fmin,
        fs=args.sample_rate,
        device=device,
    )

    slicqt, islicqt = transforms.make_filterbanks(
        slicqt_base, sample_rate=args.sample_rate
    )

    slicqt = slicqt.to(device)
    islicqt = islicqt.to(device)
    
    # pack the various encoder/decoders
    encoder = (slicqt, islicqt)

    jagged_slicq = slicqt_base.predict_input_size(args.batch_size, args.nb_channels, args.seq_dur)
    jagged_slicq_mag, jagged_slicq_phase = transforms.complex_2_magphase(jagged_slicq)
    nb_slices = jagged_slicq_mag[0].shape[-2]

    jagged_slicq_ola = slicqt.overlap_add(jagged_slicq_mag)
    ragged_ola_shapes = [X_.shape for X_ in jagged_slicq_ola]

    deoverlapnet = model.DeOverlapNet(
        slicqt,
        jagged_slicq_mag,
    ).to(device)

    if not args.quiet:
        torchinfo.summary(
            deoverlapnet,
            input_data=(slicqt.overlap_add(jagged_slicq_mag), nb_slices, ragged_ola_shapes),
        )

    optimizer = torch.optim.Adam(deoverlapnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = LossCriterion(encoder, args.mcoef_timedomain, args.mcoef_freqdomain)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = torch_utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, "slicqt_deoverlapnet.json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, "slicqt_deoverlapnet.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        deoverlapnet.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    print('Starting Tensorboard')
    tboard_proc = subprocess.Popen(["tensorboard", "--logdir", tboard_path, "--host", "0.0.0.0"])
    tboard_pid = tboard_proc.pid

    def kill_tboard():
        if tboard_pid is None:
            pass
        print('Killing backgrounded Tensorboard process...')
        os.kill(tboard_pid, signal.SIGTERM)

    atexit.register(kill_tboard)

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = loop(args, deoverlapnet, encoder, device, criterion, optimizer, train=True)
        valid_loss = loop(args, deoverlapnet, encoder, device, criterion, None, train=False)

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        torch_utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": deoverlapnet.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            model_name="slicqt_deoverlapnet",
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }

        with open(Path(target_path, "slicqt_deoverlapnet.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if tboard_writer is not None:
            tboard_writer.add_scalar('Loss/train', train_loss, epoch)
            tboard_writer.add_scalar('Loss/valid', valid_loss, epoch)

        if stop:
            print("Apply Early Stopping")
            break

        gc.collect()


if __name__ == "__main__":
    main()
