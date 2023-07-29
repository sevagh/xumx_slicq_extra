class OpenUnmix(nn.Module):
    def __init__(self, jagged_slicq_sample_input, max_bin=None, input_means=None, input_scales=None):
        super(OpenUnmix, self).__init__()
        self.bucketed_unmixes = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            if max_bin is not None and freq_start >= max_bin:
                self.bucketed_unmixes.append(DummyTimeBucket(C_block))
            else:
                self.bucketed_unmixes.append(OpenUnmixTimeBucket(C_block, input_mean=input_mean, input_scale=input_scale))

            # advance global frequency pointer
            freq_idx += C_block.shape[2]

   def forward(self, x) -> Tensor:
        futures = [torch.jit.fork(self.bucketed_unmixes[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        y_est = [torch.jit.wait(future) for future in futures]

        return y_est
