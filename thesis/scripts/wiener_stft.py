# network inference - 4 target magnitude slicq estimates
Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums = self.xumx_model(Xmag)

# initial mix phase + magnitude estimate
Ycomplex_bass = phasemix_sep(X, Ymag_bass)
Ycomplex_vocals = phasemix_sep(X, Ymag_vocals)
Ycomplex_drums = phasemix_sep(X, Ymag_drums)
Ycomplex_other = phasemix_sep(X, Ymag_other)

y_bass = self.insgt(Ycomplex_bass, audio.shape[-1])
y_drums = self.insgt(Ycomplex_drums, audio.shape[-1])
y_other = self.insgt(Ycomplex_other, audio.shape[-1])
y_vocals = self.insgt(Ycomplex_vocals, audio.shape[-1])

# now we switch to the STFT domain for the wiener step
spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

spectrograms[..., 0] = torch.abs(torch.stft(torch.squeeze(y_bass, dim=0)))
spectrograms[..., 1] = torch.abs(torch.stft(torch.squeeze(y_vocals, dim=0)))
spectrograms[..., 2] = torch.abs(torch.stft(torch.squeeze(y_drums, dim=0)))
spectrograms[..., 3] = torch.abs(torch.stft(torch.squeeze(y_other, dim=0)))
