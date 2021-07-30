import gc
import musdb
import os
import tqdm
import argparse
import torch
import museval
import numpy as np
import random
from warnings import warn
try:
    import cupy
except ImportError:
    cupy = None

from openunmix.utils import load_separator as umx_separator
from xumx.test import separate as xumx_separator
from xumx_slicq.utils import load_separator as xumx_slicq_separator

umx_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../pretrained_models/umx')
xumx_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../pretrained_models/x-umx.h5')
xumx_slicq_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vendor/xumx-sliCQ/pretrained-model')


def pretrained_model(track, model, eval_dir=None):
    N = track.audio.shape[0]

    track_audio = torch.unsqueeze(
        torch.tensor(
            track.audio.T,
            dtype=torch.float32,
            device="cpu"
        ),
    dim=0)

    # apply pretrained model forward/inference
    target_estimates = model(track_audio)

    # assign to dict for eval
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():

        # set this as the source estimate
        if name == 'vocals':
            estimates[name] = target_estimate[..., 0]
        elif name == 'bass':
            estimates[name] = target_estimate[..., 1]
        elif name == 'drums':
            estimates[name] = target_estimate[..., 2]
        elif name == 'other':
            estimates[name] = target_estimate[..., 3]

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    gc.collect()

    if cupy:
        # cupy disable fft caching to free blocks
        fft_cache = cupy.fft.config.get_plan_cache()
        fft_cache.set_size(0)

        cupy.get_default_memory_pool().free_all_blocks()

        # cupy reenable fft caching
        fft_cache.set_size(16)
        fft_cache.set_memsize(-1)

    bss_scores = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    return estimates, bss_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained umx-like models'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved',
        default=None,
    )

    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )

    args = parser.parse_args()

    # initiate musdb with test tracks
    mus = musdb.DB(subsets='test', is_wav=True)

    loaded_models = {
            'umx': umx_separator(
                umx_pretrained_path,
                targets=["vocals", "bass", "drums", "other"]
            ),
            'xumx': lambda audio: xumx_separator(
                audio,
                model_path=xumx_pretrained_path
            ),
            'xumx_slicq': xumx_slicq_separator(
                xumx_slicq_pretrained_path
            ),
    }

    for track in tqdm.tqdm(mus.tracks):
        for model in ['umx', 'xumx', 'xumx_slicq']:
            print(f'evaluating track {track.name} with model {model}')
            est_path = os.path.join(args.eval_dir, f'{model}') if args.eval_dir else None
            aud_path = os.path.join(args.audio_dir, f'{model}') if args.audio_dir else None

            est, _ = pretrained_model(
                track,
                loaded_models[model],
                eval_dir=est_path)

            gc.collect()

            if args.audio_dir:
                mus.save_estimates(est, track, aud_path)
