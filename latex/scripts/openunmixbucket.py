class OpenUnmixTimeBucket(nn.Module):
    def __init__(self, slicq_sample_input, input_mean=None, input_scale=None):
        super(OpenUnmixTimeBucket, self).__init__()

        # neural network layers go here #

        self.input_mean = torch.from_numpy(-input_mean).float()
        self.input_scale = torch.from_numpy(1.0 / input_scale).float()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()

        x = overlap_add_slicq(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # apply neural network layers #

        return x*mix # multiplicative skip connection i.e. soft mask
