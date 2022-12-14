from pathlib import Path
import torch
import torchaudio
import json
import numpy as np


from xumx_slicq_22 import utils
from xumx_slicq_22 import predict
from xumx_slicq_22 import data

import argparse


def separate():
    parser = argparse.ArgumentParser(
        description="UMX Inference",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=str, nargs="+", help="List of paths to wav/flac files.")

    parser.add_argument(
        "--model",
        default="umxhq",
        type=str,
        help="path to mode base directory of pretrained models",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        help="provide targets to be processed. \
              If none, all available targets will be computed",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".wav",
        help="Output extension which sets the audio format",
    )

    parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

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
        default="sox_io",
        help="Set torchaudio backend "
        "(`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `sox_io`",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="if provided, must be a string containing a valid expression for "
        "a dictionary, with keys as output target names, and values "
        "a list of targets that are used to build it. For instance: "
        '\'{"vocals":["vocals"], "accompaniment":["drums",'
        '"bass","other"]}\'',
    )

    args = parser.parse_args()

    if args.audio_backend != "stempeg":
        torchaudio.set_audio_backend(args.audio_backend)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using ", device)
    # parsing the output dict
    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)

    # create separator only once to reduce model loading
    # when using multiple files
    separator = utils.load_separator(
        model_str_or_path=args.model,
        targets=args.targets,
        device=device,
    )

    separator.freeze()
    separator.to(device)

    if args.audio_backend == "stempeg":
        try:
            import stempeg
        except ImportError:
            raise RuntimeError("Please install pip package `stempeg`")

    # loop over the files
    for input_file in args.input:
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
            audio, rate = data.load_audio(input_file, start=args.start, dur=args.duration)
        estimates = predict.separate(
            audio=audio,
            rate=rate,
            aggregate_dict=aggregate_dict,
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
                estimates_numpy[target] = torch.squeeze(estimate).detach().cpu().numpy().T

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
