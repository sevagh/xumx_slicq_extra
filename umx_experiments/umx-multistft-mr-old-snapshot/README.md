# umx-mr

Open-Unmix multi-resolution

## umx-mr-0

Train umx from scratch, establish a baseline

```
sevagh:umx-mr $ for t in drums vocals bass other; do echo "$t: $(jq '.best_loss' umx-mr-0/$t.json)"; done
drums: 0.9338621039475713
vocals: 1.105565301009587
bass: 1.1854630155222756
other: 2.107424723250525
```

## umx-mr-1

Tweaks:
* activation before batchnorm
* remove batchnorm from right before output
* out mean adjusted to 0

## umx-mr-2

GRU instead of LSTM

## umx-mr-4

half + double resolution convolutional cool shit
* parametrized 2x/4x scale factor
* kernel size = scale_factor+1

### evaluation break

eval + boxplot:
1. umx-mr-0
2. umx-mr-1+2+3

## umx-mr-6

periphery dataset

## umx-mr-7 == pick back up on umx-nsgt

# ideas that didn't work too well

* 8192/2048 with default 512 hidden size + 3 layers
* conv instead of linear initial stuff
* sigmoid activation instead of relu at the end
* remove dropout from LSTM, clash with batchnorm

# Hyperparam search for phasemix oracle 

## Single STFT for all targets

best single stft - 8320

```
seed 42:
    best scores
    total:  7.116105713551258       7808
seed 1337:
    best scores
    total:  7.118667022735956       8320
control, 8192:
    bass 6.14
    drums 6.54
    vocals 8.95
    other 6.76
    total sdr: 7.10
```

## Independent STFT per target

Phase-mix oracle and optimal params for the best STFT, using hyperparam search script on the 14 validation tracks of MUSDB18-HQ:

```
seed 42:
    best scores
    bass:   6.311248068797321       14080
    drums:  6.618503770665262       6656
    other:  6.796622276376451       11264
    vocals:         9.005847993076399       6656
    total = 
seed 1337:
    best scores
    bass:   6.326988484095155       15232
    drums:  6.636523858071576       5632
    other:  6.8131178731738204      11648
    vocals:         9.009451935083375       6912
control, 4096:
    bass: 5.55
    drums: 6.56
    vocals: 8.85
    other: 6.51
    total sdr: 6.87
```

Each case considers an overlap `//4`, and candidates were chosen from the range `256,32768` in steps of 128. Each seed instance of the script ran to convergence for 60 iterations as per [according to this](https://stats.stackexchange.com/a/209409/241805).

* Bass: 0.8 improvement of (the theoretical upper limit of) SDR via phase-mixture inversion with a very wide STFT, 15232
* Vocals: 0.15 improvement, decent gain using 6912
* Drums:  0.07, tiny improvement with 5632
* Other: 0.3 improvement with 11648

If these translate directly to the same improvements in the fully trained network, a total SDR gain of 1.3 which is decent.

_however_ does this break the Wiener/EM thing?
