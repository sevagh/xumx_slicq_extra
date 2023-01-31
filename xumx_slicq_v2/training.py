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
import copy
import sys
import torchaudio
import torchinfo
from contextlib import nullcontext
import sklearn.preprocessing
from torch.utils.tensorboard import SummaryWriter

from xumx_slicq_v2 import data
from xumx_slicq_v2 import model
from xumx_slicq_v2 import utils
from xumx_slicq_v2 import transforms
from xumx_slicq_v2.loss import LossCriterion

tqdm.monitor_interval = 0


def loop(
    args,
    unmix,
    encoder,
    device,
    sampler,
    criterion,
    optimizer,
    amp_cm_cuda,
    amp_cm_cpu,
    train=True,
):
    # unpack encoder object
    nsgt, insgt, cnorm = encoder

    losses = utils.AverageMeter()

    grad_cm = nullcontext
    name = ""
    if train:
        unmix.train()
        name = "Train"
    else:
        unmix.eval()
        name = "Validation"
        grad_cm = torch.no_grad

    pbar = tqdm.tqdm(sampler, disable=args.quiet)

    with grad_cm():
        for track_tensor in pbar:
            pbar.set_description(f"{name} batch")

            # autocast/AMP on forward pass + loss only, _not_ backward pass
            with amp_cm_cuda(), amp_cm_cpu():
                track_tensor_gpu = track_tensor.to(device)

                x = track_tensor_gpu[:, 0, ...]

                # forward call to unmix returns bass, vocals, other, drums
                estimates = unmix(x)

                loss = criterion(
                    estimates,
                    track_tensor_gpu[:, 1:, ...],
                )

            if train:
                loss.backward()
                optimizer.step()

            losses.update(loss.item(), x.size(1))

    return losses.avg


def get_statistics(args, encoder, dataset, time_blocks):
    nsgt, _, cnorm = encoder

    nsgt = copy.deepcopy(nsgt).to("cpu")
    cnorm = copy.deepcopy(cnorm).to("cpu")

    # slicq is a list of tensors so we need a list of scalers
    scalers = [sklearn.preprocessing.StandardScaler() for i in range(time_blocks)]

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.random_chunks = False
    dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x = dataset_scaler[ind][0]

        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = cnorm(nsgt(x[None, ...]))

        for i, X_block in enumerate(X):
            X_block_ola = np.squeeze(
                transforms.overlap_add_slicq(X_block)
                .mean(1, keepdim=False)
                .permute(0, 2, 1),
                axis=0,
            )
            scalers[i].partial_fit(X_block_ola)

    # set inital input scaler values
    std = [
        np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_)) for scaler in scalers
    ]
    return [scaler.mean_ for scaler in scalers], std


def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--root", type=str, help="root path of dataset", default="/MUSDB18-HQ"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="open-unmix",
        help="provide output path base folder name",
    )
    parser.add_argument("--model", type=str, help="Path to checkpoint folder")

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3"
    )
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
    parser.add_argument(
        "--weight-decay", type=float, default=0.00001, help="weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    # Model Parameters
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="skip dataset statistics calculation",
    )
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=1.0,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--fscale",
        choices=("bark", "mel", "cqlog", "vqlog", "oct"),
        default="bark",
        help="frequency scale for sliCQ-NSGT",
    )
    parser.add_argument(
        "--fbins",
        type=int,
        default=262,
        help="number of frequency bins for NSGT scale",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=32.9,
        help="min frequency for NSGT scale",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=4, help="Number of workers for dataloader."
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=-1,
        help="choose which gpu to train on (-1 = 'cuda' in pytorch)",
    )

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend("soundfile")
    use_cuda = torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    if use_cuda:
        print("Configuring NSGT to use GPU")

    dataloader_kwargs = (
        {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}
    )

    try:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except:
        commit = "n/a"

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and args.cuda_device >= 0:
        device = torch.device(args.cuda_device)

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # create output dir if not exist
    target_path = Path("/model")
    target_path.mkdir(parents=True, exist_ok=True)

    # check if it already contains a pytorch model
    model_exists = False
    for file in os.listdir(target_path):
        if file.endswith(".pth") or file.endswith(".chkpnt"):
            model_exists = True
            break

    tboard_path = target_path / f"logdir"
    tboard_writer = SummaryWriter(tboard_path)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        **dataloader_kwargs,
    )

    # need to globally configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = transforms.NSGTBase(
        args.fscale,
        args.fbins,
        args.fmin,
        fs=train_dataset.sample_rate,
        device=device,
    )

    nsgt, insgt = transforms.make_filterbanks(
        nsgt_base, sample_rate=train_dataset.sample_rate
    )
    cnorm = model.ComplexNorm()

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)
    cnorm = cnorm.to(device)

    # pack the 3 pieces of the encoder/decoder
    encoder = (nsgt, insgt, cnorm)

    separator_conf = {
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": 2,
        "seq_dur": args.seq_dur,  # have to do inference in chunks of seq_dur in CNN architecture
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    jagged_slicq, sample_waveform = nsgt_base.predict_input_size(args.batch_size, 2, args.seq_dur)

    jagged_slicq = cnorm(jagged_slicq)
    n_blocks = len(jagged_slicq)

    # data whitening
    if model_exists or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset, n_blocks)

    unmix = model.Unmix(
        jagged_slicq,
        encoder,
        input_means=scaler_mean,
        input_scales=scaler_std,
    ).to(device)

    if not args.quiet:
        torchinfo.summary(unmix, input_data=(sample_waveform,))

    optimizer = torch.optim.Adam(
        unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    criterion = LossCriterion()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if model_exists:
        print("Model exists, resuming training...")

        with open(Path(target_path, "xumx_slicq_v2.json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(target_path, "xumx_slicq_v2.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            args.epochs + 1,
            initial=results["epochs_trained"],
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

    print("Starting Tensorboard")
    tboard_proc = subprocess.Popen(["tensorboard", "--logdir", tboard_path])
    tboard_pid = tboard_proc.pid

    def kill_tboard():
        if tboard_pid is None:
            pass
        print("Killing backgrounded Tensorboard process...")
        os.kill(tboard_pid, signal.SIGTERM)

    atexit.register(kill_tboard)

    ######################
    # PERFORMANCE TUNING #
    ######################
    print("Performance tuning settings")
    print("Enabling cuDNN benchmark...")
    torch.backends.cudnn.benchmark = True

    print("Enabling FP32 (ampere) optimizations for matmul and cudnn")
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision = "medium"

    print(
        "Enabling CUDA Automatic Mixed Precision with bfloat16 for forward pass + loss"
    )
    amp_cm_cuda = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    print(
        "Enabling CPU Automatic Mixed Precision with bfloat16 for forward pass + loss"
    )
    amp_cm_cpu = lambda: torch.autocast("cpu", dtype=torch.bfloat16)

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = loop(
            args,
            unmix,
            encoder,
            device,
            train_sampler,
            criterion,
            optimizer,
            amp_cm_cuda,
            amp_cm_cpu,
            train=True,
        )
        valid_loss = loop(
            args,
            unmix,
            encoder,
            device,
            valid_sampler,
            criterion,
            None,
            amp_cm_cuda,
            amp_cm_cpu,
            train=False,
        )

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
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

        with open(Path(target_path, "xumx_slicq_v2.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if tboard_writer is not None:
            tboard_writer.add_scalar("Loss/train", train_loss, epoch)
            tboard_writer.add_scalar("Loss/valid", valid_loss, epoch)

        if stop:
            print("Apply Early Stopping")
            break

        gc.collect()


if __name__ == "__main__":
    main()
