import argparse
import torch
import subprocess
import shutil
import gzip
import time
from pathlib import Path
import tqdm
import json
import numpy as np
import random
import os
import signal
import gc
import copy
import sys
import torchaudio
import torchinfo
from contextlib import nullcontext
import sklearn.preprocessing
import torch.nn.utils.prune as prune

from .data import MUSDBDataset, custom_collate
import torch.nn as nn

from xumx_slicq_v2 import model
from xumx_slicq_v2 import transforms
from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2.optuna import _SDRLossCriterion
from xumx_slicq_v2.training import _EarlyStopping, _AverageMeter, _ComplexMSELossCriterion

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
    train,
):
    # unpack encoder object
    nsgt, insgt, cnorm = encoder

    losses = _AverageMeter()

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
                track_tensor_gpu = track_tensor.to(device).swapaxes(0, 1)

                x = track_tensor_gpu[0]

                y_targets = track_tensor_gpu[1:]

                Xcomplex = nsgt(x)

                Ycomplex_ests, Ymasks = unmix(Xcomplex, return_masks=True)

                Ycomplex_targets = nsgt(y_targets)

                mse_loss = criterion(
                    Ycomplex_ests,
                    Ycomplex_targets,
                )

                mask_mse_loss = 0.

                ideal_sum_of_masks = [None]*len(Ymasks)

                # sum of all 4 target masks should be exactly 1.0
                for i, Ymask in enumerate(Ymasks):
                    Ymask_sum = torch.sum(Ymask, dim=0, keepdims=False)
                    ideal_mask = torch.ones_like(Ymask_sum)

                    mask_mse_loss += torch.mean((Ymask_sum-ideal_mask)**2)

                mask_mse_loss /= len(Ymasks)

                loss = mse_loss + mask_mse_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.update(loss.item(), x.size(1))

    return losses.avg


def pruning_main():
    parser = argparse.ArgumentParser(description="xumx-sliCQ-V2 Pruning + Fine-tuning")

    # Dataset paramaters; always /MUSDB18-HQ
    parser.add_argument("--samples-per-track", type=int, default=64)

    # Pruning Parameters
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for training",
    )
    parser.add_argument(
        "--batch-size-valid",
        type=int,
        default=1,
        help="batch size for validation",
    )
    # retraining learning rate
    # from: https://towardsdatascience.com/neural-network-pruning-101-af816aaea61
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
        default=2.0,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--fscale",
        choices=("bark", "mel", "cqlog", "vqlog"),
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
        "--fgamma",
        type=float,
        default=15.,
        help="gamma for variable-Q offset",
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

    args = parser.parse_args()

    torchaudio.set_audio_backend("soundfile")
    use_cuda = torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    if use_cuda:
        print("Configuring NSGT to use GPU")

    dataloader_kwargs = (
        {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}
    )

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and args.cuda_device >= 0:
        device = torch.device(args.cuda_device)

    train_dataset, valid_dataset = MUSDBDataset.load_datasets(
        args.seed,
        args.seq_dur,
        samples_per_track=args.samples_per_track,
    )

    # create output dir if not exist
    target_path = Path("/model")
    target_path.mkdir(parents=True, exist_ok=True)

    # check if it already contains a pytorch model
    model_exists = False
    for file in os.listdir(target_path):
        if file.endswith(".pth") or file.endswith(".chkpnt"):
            model_exists = True
            break

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size_valid,
        collate_fn=custom_collate,
        **dataloader_kwargs,
    )

    # need to globally configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = transforms.NSGTBase(
        args.fscale,
        args.fbins,
        args.fmin,
        fgamma=args.fgamma,
        fs=valid_dataset.sample_rate,
        device=device,
    )

    nsgt, insgt = transforms.make_filterbanks(
        nsgt_base, sample_rate=valid_dataset.sample_rate
    )
    cnorm = transforms.ComplexNorm()

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)
    cnorm = cnorm.to(device)

    # pack the 3 pieces of the encoder/decoder
    encoder = (nsgt, insgt, cnorm)

    separator_conf = {
        "sample_rate": valid_dataset.sample_rate,
        "nb_channels": 2,
        "seq_dur": args.seq_dur,  # have to do inference in chunks of seq_dur in CNN architecture
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    jagged_slicq, sample_waveform = nsgt_base.predict_input_size(args.batch_size_valid, 2, args.seq_dur)

    jagged_slicq_cnorm = cnorm(jagged_slicq)
    n_blocks = len(jagged_slicq)

    # always loading a trained module for pruning step
    assert model_exists

    scaler_mean = None
    scaler_std = None

    unmix_orig = model.Unmix(
        jagged_slicq_cnorm,
        input_means=scaler_mean,
        input_scales=scaler_std,
    ).to(device)

    mse_criterion = _ComplexMSELossCriterion()

    es = _EarlyStopping(patience=args.patience)

    optimizer = torch.optim.AdamW(
        unmix_orig.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    # if a model is specified: resume fine-tuning + pruning loop
    print("Model exists, performing pruning...")

    with open(Path(target_path, "xumx_slicq_v2.json"), "r") as stream:
        results = json.load(stream)

    loss_to_beat = results["best_loss"]
    es.best = loss_to_beat

    target_model_path = Path(target_path, f"xumx_slicq_v2.pth")
    state = torch.load(target_model_path, map_location=device)
    unmix_orig.load_state_dict(state, strict=False)

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
        "Enabling CUDA+CPU Automatic Mixed Precision with bfloat16 for forward pass + loss"
    )
    amp_cm_cuda = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    amp_cm_cpu = lambda: torch.autocast("cpu", dtype=torch.bfloat16)

    prunes_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]

    t = tqdm.tqdm(prunes_to_test, disable=args.quiet)
    sub_t = tqdm.trange(1, args.finetune_epochs + 1, disable=args.quiet)

    for prune_proportion in t:
        # create a new unpruned unmix; save last pruning if it was good
        unmix = copy.deepcopy(unmix_orig)

        t.set_description("Pruning Iteration")
        # unmix is blocked per-frequency bin

        prune_list = []

        # target upper frequency blocks to have less impact
        for target in range(4):
            for frequency_bin in range(3*n_blocks//4, n_blocks):
                base = f"sliced_umx.{frequency_bin}.cdaes.{target}"
                print(f"pruning {100*prune_proportion:.1f}% from submodule: {base}")

                # prune each layer
                prune_list.extend([
                    f"{base}.0", # encoder 1 conv
                    f"{base}.1", # encoder 1 batch norm
                    # 2 is the relu
                    f"{base}.3", # encoder 2 conv
                    f"{base}.4", # encoder 2 batch norm
                    # 5 is the relu
                    f"{base}.6", # decoder 1 conv transpose
                    f"{base}.7", # decoder 1 batch norm
                    # 8 is the relu
                    f"{base}.9", # decoder 2 conv transpose
                    # 10 is the sigmoid
                ])

        for prune_name in prune_list:
            module = unmix.get_submodule(prune_name)
            if isinstance(module, nn.BatchNorm2d):
                prune.l1_unstructured(module, name="weight", amount=prune_proportion)
                prune.remove(module, 'weight')
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                prune.ln_structured(module, name="weight", amount=prune_proportion, n=2, dim=0)
                #prune.ln_structured(module, name="weight", amount=prune_proportion, n=2, dim=1)
                prune.remove(module, 'weight')

        # fine tune
        for epoch in sub_t:
            sub_t.set_description("Fine-tune Epoch")

            train_loss = loop(
                args,
                unmix,
                encoder,
                device,
                train_sampler,
                mse_criterion,
                optimizer,
                amp_cm_cuda,
                amp_cm_cpu,
                train=True
            )
            sub_t.set_postfix(train_loss=train_loss)

            # check result
            valid_loss = loop(
                args,
                unmix,
                encoder,
                device,
                valid_sampler,
                mse_criterion,
                None,
                amp_cm_cuda,
                amp_cm_cpu,
                train=False
            )
            sub_t.set_postfix(val_loss=valid_loss)

            es.step(valid_loss)

            if valid_loss == es.best:
                print(f"achieved best loss: {valid_loss}, saving pruned model...")
                pruned_pth_path = os.path.join(target_path, "xumx_slicq_v2_pruned.pth")
                gz_path = os.path.join(target_path, "xumx_slicq_v2_pruned.pth.gz")
                torch.save(unmix.state_dict(), pruned_pth_path)

                # save as gzip; pruning + compression works best...
                with open(pruned_pth_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                # remove pruned pth file
                os.remove(pruned_pth_path)

                # base on best pruned model going forward; otherwise discard failed pruning
                unmix_orig = copy.deepcopy(unmix)

    # do everything i can to avoid crashing
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    pruning_main()
