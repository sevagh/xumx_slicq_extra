#!/usr/bin/env bash

set -xou pipefail

export BATCH_SIZE=2
export TOTAL_BATCHES=8
export REPO_ARG="/home/sevagh/repos/xumx-sliCQ-V2/:/xumx-sliCQ-V2"
export MUSDB_ARG="/home/sevagh/Music/MDX-datasets/MUSDB18-HQ/:/MUSDB18-HQ"
export CAD1D_ARG="/home/sevagh/Music/MDX-datasets/CAD1/cadenza_data/:/CADENZA"
export CAD1R_ARG="/home/sevagh/Music/MDX-datasets/CAD1/cadenza_results/:/exp"

export PODMAN_CMD="podman run --rm -v ${REPO_ARG} -v ${MUSDB_ARG} -v ${CAD1D_ARG} -v ${CAD1R_ARG} xumx-slicq-v2 python -m cadenza.test evaluate.batch_size=${TOTAL_BATCHES}"

# Launch the jobs using GNU Parallel with proper scheduling
# first run 0,1,2,3
# then run 4,5,6,7 opportunistically as the first batch is finishing
seq 0 $((${TOTAL_BATCHES}-1)) | \
        parallel --jobs ${BATCH_SIZE} "\
echo 'Running job {}';
${PODMAN_CMD} evaluate.batch={};
"

# then merge the batches
podman run --rm \
        -v "${REPO_ARG}" -v "${MUSDB_ARG}" -v "${CAD1D_ARG}" -v "${CAD1R_ARG}" \
        xumx-slicq-v2 python -m cadenza.merge_batches_results \
        evaluate.batch_size="${TOTAL_BATCHES}"

echo "now exit the script"
