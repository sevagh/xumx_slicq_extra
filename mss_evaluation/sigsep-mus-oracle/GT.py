import musdb
import museval
import argparse
import functools


def GT(track, eval_dir=None):
    """Ground Truth Signals
    """

    # perform separtion
    estimates = {}
    for name, target in track.targets.items():
        # set accompaniment source
        estimates[name] = target.audio

    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

    return estimates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate on Ground Truth Targets'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved'
    )

    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )

    args = parser.parse_args()

    # initiate musdb
    mus = musdb.DB()

    mus.run(
        functools.partial(GT, eval_dir=args.eval_dir),
        estimates_dir=args.audio_dir,
        subsets='test',
        parallel=True,
        cpus=2
    )
