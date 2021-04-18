import sys
import json
import os
import argparse
import numpy
from boxplot import boxplot


def main(args):
    results_by_oracle = {}

    for d in os.listdir(args.result_dir):
        oracle = d

        # evaluations are always performed on the 'test' subset of musdb18-hq
        # so we expect the results to be nested under './test'

        full_dir = os.path.join(args.result_dir, d, "./test")
        track_jsons = os.listdir(full_dir)

        n_tracks = len(track_jsons)

        results_by_oracle[oracle] = {}

        for track in track_jsons:
            track_name = os.path.splitext(track)[0]
            results_by_oracle[oracle][track_name] = {}

            track_results = None
            with open(os.path.join(full_dir, track)) as tjf:
                track_results = json.load(tjf)

            for target in track_results["targets"]:
                n_frames = len(target["frames"])

                framed_result = numpy.zeros((n_frames, 4))

                for frame_index, frame in enumerate(target["frames"]):
                    framed_result[frame_index][0] = frame["metrics"]["SDR"]
                    framed_result[frame_index][1] = frame["metrics"]["SIR"]
                    framed_result[frame_index][2] = frame["metrics"]["SAR"]
                    framed_result[frame_index][3] = frame["metrics"]["ISR"]

                track_target_sdr = numpy.median(framed_result[:, 0])
                track_target_sir = numpy.median(framed_result[:, 1])
                track_target_sar = numpy.median(framed_result[:, 2])
                track_target_isr = numpy.median(framed_result[:, 3])

                results_by_oracle[oracle][track_name][target["name"]] = {
                    "SDR": track_target_sdr,
                    "SIR": track_target_sir,
                    "SAR": track_target_sar,
                    "ISR": track_target_isr,
                }

    boxplot(results_by_oracle)

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        prog="plot",
        description="generate box plots for oracle mask BSSv4 across MUSDB18 tracks",
    )

    parser.add_argument(
        "result_dir",
        help="json result directory",
        default="",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sys.exit(main(args))
