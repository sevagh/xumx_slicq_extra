# xumx_slicq_extra

Extra scripts, latex files, etc. relating to https://github.com/sevagh/xumx-sliCQ

The subcomponents of xumx_slicq_extra are:
* diagrams.svg: used to create diagrams for https://github.com/sevagh/xumx-sliCQ and https://github.com/sevagh/nsgt
* latex: latex files for reports, presentations, and eventual master's thesis
* misc_scripts: misc scripts for converting Periphery stems into a MUSDB18-compatible dataset (not used in xumx-sliCQ)
* umx_experiments: a scrapyard of many, many variants of umx with the sliCQ that I tried to make work with the aim of submitting them to the [AICrowd Sony ISMIR 2021 Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021)
* mss_evaluation: scripts for oracle (mix-phase, IRM/IBM, etc.) evaluations of the sliCQ, vendored tools from sigsep (musdb, museval, etc.), vendored code for the original Open-Unmix/UMX and Sony X-UMX, pretrained models for umxhq and xumx, and the evaluation and boxplot scripts

The 3-way evaluation to show xumx-sliCQ's performance is reported in the xumx-sliCQ repo, and repeated here:

| Project name | Paper | Repo | Pretrained model |
|--------------|-------|------|------------------|
| Open-Unmix, UMX | [Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document) | https://github.com/sigsep/open-unmix-pytorch | https://zenodo.org/record/3370489 (UMX-HQ) |
| X-UMX, CrossNet-OpenUnmix | [Sawata, Uhlich, Takahashi, Mitsufuji 2020](https://www.ismir2020.net/assets/img/virtual-booth-sonycsl/cUMX_paper.pdf) | https://github.com/sony/ai-research-code/tree/master/x-umx | https://nnabla.org/pretrained-models/ai-research-code/x-umx/x-umx.h5 |
| xumx-sliCQ (this project) | n/a | https://github.com/sevagh/xumx-sliCQ | https://github.com/sevagh/xumx-sliCQ/tree/main/pretrained-model |
