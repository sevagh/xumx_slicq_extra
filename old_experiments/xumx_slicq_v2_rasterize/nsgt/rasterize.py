import torch

'''
input: ragged slice-wise slicqt
        shape: (samples, channels, f_bins, slice, t_bins)
'''
@torch.no_grad()
def rasterize(slicq):
    nb_packed_sampchan = slicq[0].shape[1]
    total_f_bins = sum([slicq_.shape[-2] for slicq_ in slicq])

    n_slices = slicq[0].shape[0]

    # ensure all buckets have same slice length
    assert all([slicq_.shape[0] == n_slices for slicq_ in slicq[1:]])

    max_t_bins = max([slicq_.shape[-1] for slicq_ in slicq])

    interpolated = torch.zeros((n_slices, nb_packed_sampchan, total_f_bins, max_t_bins), dtype=slicq[0].dtype, device=slicq[0].device)

    fbin_ptr = 0
    for i, slicq_ in enumerate(slicq):
        n_slices, nb_packed_sampchan, nb_f_bins, nb_t_bins = slicq_.shape

        if nb_t_bins == max_t_bins:
            # same time width, no interpolation
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, :] = slicq_
        else:
            # repeated interpolation
            interp_factor = max_t_bins//nb_t_bins
            max_assigned = nb_t_bins*interp_factor
            rem = max_t_bins - max_assigned
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, : max_assigned] = torch.repeat_interleave(slicq_, interp_factor, dim=-1)
            interpolated[:, :, fbin_ptr:fbin_ptr+nb_f_bins, max_assigned : ] = torch.unsqueeze(slicq_[..., -1], dim=-1).repeat(1, 1, 1, rem)
        fbin_ptr += nb_f_bins

    return interpolated


@torch.no_grad()
def derasterize(interpolated, ragged_shapes):
    max_t_bins = interpolated.shape[-1]
    full_slicq = []
    fbin_ptr = 0
    for i, bucket_shape in enumerate(ragged_shapes):
        curr_slicq = torch.zeros(bucket_shape, dtype=interpolated.dtype, device=interpolated.device)

        nb_t_bins = bucket_shape[-1]
        freqs = bucket_shape[-2]

        if bucket_shape[-1] == interpolated.shape[-1]:
            # same time width, no interpolation
            curr_slicq = interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :]
        else:
            # inverse of repeated interpolation
            interp_factor = max_t_bins//nb_t_bins
            select = torch.arange(0, max_t_bins,interp_factor, device=interpolated.device)
            curr_slicq = torch.index_select(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], -1, select)

        # crop just in case
        full_slicq.append(curr_slicq[..., : bucket_shape[-1]])

        fbin_ptr += freqs
    return full_slicq
