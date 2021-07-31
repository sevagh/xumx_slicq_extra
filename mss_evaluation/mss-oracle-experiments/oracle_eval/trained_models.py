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

from openunmix.utils import load_separator as umx_separator
from xumx_sony.test import load_xumx_model as xumx_separator_sony
from xumx_sony.test import separate as xumx_separate_sony
from xumx_slicq.utils import load_separator as xumx_slicq_separator

umx_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../pretrained_models/umx')
xumx_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../pretrained_models/x-umx.h5')
xumx_slicq_pretrained_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vendor/xumx-sliCQ/pretrained-model')


def pretrained_model(track, model, eval_dir=None, is_xumx=False):
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
        for name, source in track.sources.items():

            # set this as the source estimate
            if name == 'vocals':
                estimates[name] = target_estimates[0, ...].detach().cpu().numpy().T
            elif name == 'bass':
                estimates[name] = target_estimates[1, ...].detach().cpu().numpy().T
            elif name == 'drums':
                estimates[name] = target_estimates[2, ...].detach().cpu().numpy().T
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

    gc.collect()

    print(f'bss evaluation to store in {eval_dir}')
    bss_scores = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    print(bss_scores)

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

    args = parser.parse_args()

    # initiate musdb with test tracks
    mus = musdb.DB(subsets='test', is_wav=True)

    loaded_models = {
            'umx': umx_separator(
                umx_pretrained_path,
                targets=["vocals", "bass", "drums", "other"]
            ),
            'xumx': xumx_separator_sony(
                model_path=xumx_pretrained_path
            ),
            'slicq': xumx_slicq_separator(
                xumx_slicq_pretrained_path
            ),
    }

    for track in tqdm.tqdm(mus.tracks[args.track_offset:]):
        for model in (['xumx', 'umx', 'slicq'] if not args.model else [args.model]):
            print(f'evaluating track {track.name} with model {model}')
            est_path = os.path.join(args.eval_dir, f'{model}') if args.eval_dir else None
            aud_path = os.path.join(args.audio_dir, f'{model}') if args.audio_dir else None

            est, _ = pretrained_model(
                track,
                loaded_models[model],
                eval_dir=est_path,
                is_xumx=(model == 'xumx')
            )

            gc.collect()

            if args.audio_dir:
                mus.save_estimates(est, track, aud_path)
