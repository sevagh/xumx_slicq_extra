# umx

My version of the excellent [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) template for source separation.

It contains the ideas I worked on during the AICrowd ISMIR 2021 [Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).

The ideas are presented in reverse chronological order, starting from the most recent submission.

### umx-ep-20210610

This is the first model I submitted. The pretrained models and scripts are in [umx-ep-20210610](./umx-ep-20210610) (with the same name git tag).

The name "-ep-" stands for **enhanced** and **Periphery**, representing the following deviations from classic umx

1. Mixed MUSDB18-HQ and Periphery training data

The extra data consists of 4 studio albums by the metal band [Periphery](http://www.periphery.net/), consisting of 37 songs. The albums (44100kHz stems) are purchasable from their store: [Periphery III: Select Difficulty, Juggernaut Omega, and Juggernaut Alpha](https://store.periphery.net/118281/Periphery-III-Select-Difficulty-Juggernaut-Alpha-Omega-Full-Album-Stem-Download) for $50 USD, and [Hail Stan](https://store.periphery.net/116598/Hail-Stan-Full-Album-Stem-Download) for $30 USD. The stems are made to align with MUSDB using [scripts/periphery2musdb.py](./scripts/periphery2musdb.py) - everything that is not bass, vocals, or drums is summed to produce the "other" target:
```
'Bass.wav' --------------------------> bass.wav
'Rhythm Guitars.wav' ------------\
'Other Guitars.wav' --------------\  
                                   +-> other.wav
'Guitar Solos.wav' ---------------/
'Synths, Strings and Subs.wav'---/
'Vocals.wav' ------------------------> vocals.wav
'Drums.wav' -------------------------> drums.wav
```

Finally, I scrambled the MUSDB18-HQ and Periphery songs to distribute them into 70%/20%/10% train/test/validation sets. No source augmentations were used. The "trackfolder_fix" type of dataset was used (which I modified to support 64 samples per batch, like MUSDB18-HQ).

2. Minor model changes

I made small changes to the UMX architecture:
* GRU instead of LSTM
* Drop the batch normalization layer at the output stage
* Invert the batch norm and activation order to do the activation first
