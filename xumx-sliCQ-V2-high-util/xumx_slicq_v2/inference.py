from pathlib import Path
import torch
import torchaudio
import json
import numpy as np
import os
from xumx_slicq_v2 import utils
from xumx_slicq_v2 import data

import argparse


def separate(
    audio,
    rate=None,
    model_str_or_path="umxhq",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    aggregate_dict=None,
    separator=None,
    device=None,
):
    """
    Open Unmix functional interface

    Separates a torch.Tensor or the content of an audio file.

    If a separator is provided, use it for inference. If not, create one
    and use it afterwards.

    Args:
        audio: audio to process
            torch Tensor: shape (channels, length), and
            `rate` must also be provided.
        rate: int or None: only used if audio is a Tensor. Otherwise,
            inferred from the file.
        model_str_or_path: the pretrained model to use
        targets (str): select the targets for the source to be separated.
            a list including: ['vocals', 'drums', 'bass', 'other'].
            If you don't pick them all, you probably want to
            activate the `residual=True` option.
            Defaults to all available targets per model.
        niter (int): the number of post-processingiterations, defaults to 1
        residual (bool): if True, a "garbage" target is created
        wiener_win_len (int): the number of frames to use when batching
            the post-processing step
        aggregate_dict (str): if provided, must be a string containing a '
            'valid expression for a dictionary, with keys as output '
            'target names, and values a list of targets that are used to '
            'build it. For instance: \'{\"vocals\":[\"vocals\"], '
            '\"accompaniment\":[\"drums\",\"bass\",\"other\"]}\'
        separator: if provided, the model.Separator object that will be used
             to perform separation
        device (str): selects device to be used for inference
    """
    if separator is None:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True,
        )
        separator.freeze()
        if device:
            separator.to(device)

    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = utils.preprocess(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)
    return estimates


def main():
    parser = argparse.ArgumentParser(
        description="UMX Inference",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--indir",
        type=str,
        default="/input",
        help="Directory with wav files to process.",
    )

    parser.add_argument(
        "--model",
        default="/model",
        type=str,
        help="path to mode base directory of pretrained models",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="/output",
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".wav",
        help="Output extension which sets the audio format",
    )

    parser.add_argument(
        "--start", type=float, default=0.0, help="Audio chunk start in seconds"
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend "
        "(`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `soundfile`",
    )

    args = parser.parse_args()

    if args.audio_backend != "stempeg":
        torchaudio.set_audio_backend(args.audio_backend)

    # explicitly use no GPUs for inference
    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using ", device)

    # create separator only once to reduce model loading
    # when using multiple files
    separator = utils.load_separator(
        model_str_or_path=args.model, device=device, pretrained=True
    )

    separator.freeze()
    separator.to(device)

    if args.audio_backend == "stempeg":
        try:
            import stempeg
        except ImportError:
            raise RuntimeError("Please install pip package `stempeg`")

    # loop over the files
    for wav_file in os.listdir(args.indir):
        input_file = os.path.join(args.indir, wav_file)
        if args.audio_backend == "stempeg":
            audio, rate = stempeg.read_stems(
                input_file,
                start=args.start,
                duration=args.duration,
                sample_rate=separator.sample_rate,
                dtype=np.float32,
            )
            audio = torch.tensor(audio, device=device)
        else:
            audio, rate = data.load_audio(
                input_file, start=args.start, dur=args.duration
            )
        estimates = separate(
            audio=audio,
            rate=rate,
            separator=separator,
            device=device,
        )
        if not args.outdir:
            model_path = Path(args.model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + "_" + args.model)
            else:
                outdir = Path(Path(input_file).stem + "_" + model_path.stem)
        else:
            outdir = Path(args.outdir) / Path(input_file).stem
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        if args.audio_backend == "stempeg":
            target_path = str(outdir / Path("target").with_suffix(args.ext))
            # convert torch dict to numpy dict
            estimates_numpy = {}
            for target, estimate in estimates.items():
                estimates_numpy[target] = (
                    torch.squeeze(estimate).detach().cpu().numpy().T
                )

            stempeg.write_stems(
                target_path,
                estimates_numpy,
                sample_rate=separator.sample_rate,
                writer=stempeg.FilesWriter(multiprocess=True, output_sample_rate=rate),
            )
        else:
            for target, estimate in estimates.items():
                target_path = str(outdir / Path(target).with_suffix(args.ext))
                torchaudio.save(
                    target_path,
                    torch.squeeze(estimate).to("cpu"),
                    sample_rate=separator.sample_rate,
                )


if __name__ == "__main__":
    main()
