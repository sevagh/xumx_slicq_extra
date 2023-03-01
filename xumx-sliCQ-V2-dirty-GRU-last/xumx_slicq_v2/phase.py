import torch
import norbert


def blockwise_wiener(mix_slicqt, slicqtgrams, wiener_win_len_param: int = 5000):
    mix_slicqt = torch.flatten(mix_slicqt, start_dim=-3, end_dim=-2)
    orig_shape = slicqtgrams.shape
    slicqtgrams = torch.flatten(slicqtgrams, start_dim=-2, end_dim=-1)

    # transposing it as
    # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
    slicqtgrams = slicqtgrams.permute(1, 4, 3, 2, 0)

    # rearranging it into:
    # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
    # into filtering methods
    mix_slicqt = mix_slicqt.permute(0, 3, 2, 1, 4)

    nb_frames = slicqtgrams.shape[1]
    targets_slicqt = torch.zeros(
        *mix_slicqt.shape[:-1] + (4,2,),
        dtype=mix_slicqt.dtype,
        device=mix_slicqt.device,
    )

    pos = 0
    if wiener_win_len_param:
        wiener_win_len = wiener_win_len_param
    else:
        wiener_win_len = nb_frames
    while pos < nb_frames:
        cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
        pos = int(cur_frame[-1]) + 1

        targets_slicqt[:, cur_frame, ...] = torch.view_as_real(norbert.wiener(
            slicqtgrams[:, cur_frame, ...],
            torch.view_as_complex(mix_slicqt[:, cur_frame, ...]),
            1,
            False,
        ))

    # getting to (nb_samples, nb_targets, channel, fft_size, n_frames)
    targets_slicqt = targets_slicqt.permute(4, 0, 3, 2, 1, 5).contiguous()
    targets_slicqt = targets_slicqt.reshape((*orig_shape, 2,))
    return targets_slicqt
