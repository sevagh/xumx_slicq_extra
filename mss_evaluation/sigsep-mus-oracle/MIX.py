import musdb
import museval
import functools
import argparse


def MIX(track, eval_dir=None):
    """Mixture as Estimate
    """

    # perform separtion
    estimates = {}
    for name, target in track.sources.items():
        # set accompaniment source
        estimates[name] = track.audio / len(track.sources)

    estimates['accompaniment'] = estimates['bass'] + \
        estimates['drums'] + estimates['other']

    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

    return estimates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Mixture as Estimate'
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
        functools.partial(
            MIX, eval_dir=args.eval_dir
        ),
        estimates_dir=args.audio_dir,
        subsets='test',
        parallel=True,
        cpus=2
    )
