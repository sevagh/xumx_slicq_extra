# network inference - 4 target magnitude slicq estimates
Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums = self.xumx_model(Xmag)

nb_slices = X[0].shape[3]
last_dim = 2

X_matrix = torch.zeros((nb_samples, self.nb_channels, self.total_f_bins, nb_slices, self.max_t_bins, last_dim), dtype=X[0].dtype, device=X[0].device)
spectrograms = torch.zeros(X_matrix.shape[:-1] + (nb_sources,), dtype=audio.dtype, device=X_matrix.device)

# create a zero-padded rectangular matrix for the ragged slicqt
freq_start = 0
for i, X_block in enumerate(X):
    nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins, last_dim = X_block.shape

    # assign up to the defined time bins - to the right will be zeros
    X_matrix[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :] = X_block

    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 0] = Ymag_vocals[i]
    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 1] = Ymag_drums[i]
    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 2] = Ymag_bass[i]
    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 3] = Ymag_other[i]

    freq_start += nb_f_bins
