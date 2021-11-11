if __name__ == '__main__':
    stft_wem_sdr_1 = (2.727 + 2.702 + 4.082 - 0.939)/4.0
    slicqt_ragged_wem_sdr_1 = (2.754 + 2.885 + 4.134 - 0.891)/4.0
    slicqt_zeropad_wem_sdr_1 = (2.753 + 2.864 + 4.140 - 0.893)/4.0
    stft_wem_sdr_2 = (2.368 + 2.669 + 3.892 - 1.620)/4.0
    slicqt_ragged_wem_sdr_2 = (2.481 + 2.832 + 3.957 - 1.505)/4.0

    print(f'{stft_wem_sdr_1=}, time=3m41.839s')
    print(f'{stft_wem_sdr_2=}, time=4m5.906s')
    print(f'{slicqt_ragged_wem_sdr_1=}, time=7m59.104s')
    print(f'{slicqt_ragged_wem_sdr_2=}, time=12m55.510s')
    print(f'{slicqt_zeropad_wem_sdr_1=}, time=4m28.061s')
