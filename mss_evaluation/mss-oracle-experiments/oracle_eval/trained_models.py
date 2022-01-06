import gc
import musdb
import os
import tqdm
import argparse
import torch
import museval
from museval.metrics import disable_cupy
import numpy as np
import random
import time
import sys
from warnings import warn

from openunmix.utils import load_separator as umx_separator
from xumx_sony.test import load_xumx_model as xumx_separator_sony
from xumx_sony.test import separate as xumx_separate_sony
from xumx_slicq.utils import load_separator as xumx_slicq_separator

umx_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../pretrained_models/umx')
xumx_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../pretrained_models/x-umx.h5')
xumx_slicq_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vendor/xumx-sliCQ/pretrained-model')


def pretrained_model(track, model, eval_dir=None, is_xumx=False, swap_drums_bass=False):
    start = time.time()

    N = track.audio.shape[0]

    if not is_xumx:
        track_audio = torch.unsqueeze(torch.tensor(
            track.audio.T,
            dtype=torch.float32,
            device="cpu"
        ), dim=0)

        # apply pretrained model forward/inference
        target_estimates = torch.squeeze(model(track_audio), dim=0)

        # assign to dict for eval
        estimates = {}
        accompaniment_source = 0

        drums_pos = 2
        bass_pos = 1
        if swap_drums_bass:
            drums_pos = 1
            bass_pos = 2

        for name, source in track.sources.items():

            # set this as the source estimate
            if name == 'vocals':
                estimates[name] = target_estimates[0, ...].detach().cpu().numpy().T
            elif name == 'bass':
                estimates[name] = target_estimates[bass_pos, ...].detach().cpu().numpy().T
            elif name == 'drums':
                estimates[name] = target_estimates[drums_pos, ...].detach().cpu().numpy().T
            elif name == 'other':
                estimates[name] = target_estimates[3, ...].detach().cpu().numpy().T

            # accumulate to the accompaniment if this is not vocals
            if name != 'vocals':
                accompaniment_source += estimates[name]

        estimates['accompaniment'] = accompaniment_source
    else:
        estimates = xumx_separate_sony(
            track.audio,
            model,
            chunk_size=2621440
        )

        estimates['accompaniment'] = estimates['drums'] + estimates['bass'] + estimates['other']

    end = time.time()

    gc.collect()

    print(f'bss evaluation to store in {eval_dir}')
    bss_scores = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    print(bss_scores)

    return estimates, bss_scores, end-start


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
        '--model',
        default="",
        type=str,
        help='model to evaluate ("" == all)'
    )
    parser.add_argument(
        '--track-offset',
        default=0,
        type=int,
        help='track offset'
    )
    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='musdb data split'
    )

    args = parser.parse_args()

    mus = None
    # initiate musdb
    if args.split == 'valid':
        mus = musdb.DB(subsets='train', split='valid', is_wav=True)
    elif args.split == 'test':
        mus = musdb.DB(subsets='test', is_wav=True)
    else:
        raise ValueError(f'musdb18 data split {args.split} unsupported')

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))
    disable_cupy()

    loaded_models = {
            'umx': umx_separator(
                umx_pretrained_path,
                targets=["vocals", "bass", "drums", "other"]
            ),
            'xumx': xumx_separator_sony(
                model_path=xumx_pretrained_path
            ),
            'slicq-wslicq': xumx_slicq_separator(
                xumx_slicq_pretrained_path,
                slicq_wiener=True,
            ),
            'slicq-wstft': xumx_slicq_separator(
                xumx_slicq_pretrained_path,
                slicq_wiener=False,
            ),
    }

    tot = 0.
    pbar = tqdm.tqdm(mus.tracks[args.track_offset:args.track_offset+max_tracks])
    tot_tracks = len(pbar)

    for track in pbar:
        print(f'evaluating track {track.name} with model {args.model}')
        est_path = os.path.join(args.eval_dir, f'{args.model}') if args.eval_dir else None
        aud_path = os.path.join(args.audio_dir, f'{args.model}') if args.audio_dir else None

        est, _, time_taken = pretrained_model(
            track,
            loaded_models[args.model],
            eval_dir=est_path,
            is_xumx=(args.model == 'xumx'),
            swap_drums_bass=('slicq' in args.model)
        )
        print(f'time {time_taken} s for song {track.name}')

        tot += time_taken

        gc.collect()

        if args.audio_dir:
            mus.save_estimates(est, track, aud_path)

    print(f'total time for {tot_tracks} track evaluation: {tot}')
    print(f'time averaged per track: {tot/tot_tracks}')
