New direction:
    RtDrumSep/FlashLights; C++/Rust/ONNX

    Seq dur = 2048 samples (0.04644 s)
    <2048 sllen (from v1 param search)
        scale=mel, fbins=32, fmin=115.50, fmax=22050.00, sllen=2016, trlen=504
    realtime causal model w/ phasemix sep
    enable sdr mcoef (because tiny)

training args:
    --batch-size=512 --nb-workers=16 --fscale=mel --fbins=32 --fmin=115.5 --sdr-mcoef=10.0 --realtime --seq-dur=0.04644

```
Aggrated Scores (median over frames, median over tracks)
vocals          ==> SDR:   2.116  SIR:   0.824  ISR:   7.016  SAR:   4.863
drums           ==> SDR:   2.710  SIR:   1.160  ISR:   6.257  SAR:   4.149
bass            ==> SDR:   1.399  SIR:   2.590  ISR:   6.054  SAR:   4.587
other           ==> SDR:   1.397  SIR:  -1.425  ISR:   7.734  SAR:   4.334
```

best loss -25.72
now continue training with Periphery
38 tracks; test/train/valid: 70/15/15

26/6/6

then export drum-only w/ onnx, fixed size 2048...
+ onnxruntime-rs
chunk audio in 2048: pass thru xumx-slicq-v2-drum-lite-rt-whatever + btrack
