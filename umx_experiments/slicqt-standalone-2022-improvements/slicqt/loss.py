from slicqt import torch_transforms as transforms
import torch
import auraloss

eps = 1.e-10


def _custom_mse_loss(pred_magnitude, target_magnitude):
    loss = 0
    for i in range(len(target_magnitude)):
        loss += torch.mean((pred_magnitude[i] - target_magnitude[i])**2)
    return loss/len(target_magnitude)


# time-domain and frequency-domain loss inspired by X-UMX
class LossCriterion:
    def __init__(self, encoder, mix_coef_time, mix_coef_freq):
        self.nsgt, self.insgt = encoder
        self.mcoef_time = mix_coef_time
        self.mcoef_freq = mix_coef_freq
        self.mse_loss_criterion = _custom_mse_loss
        self.snr_loss_criterion = auraloss.time.SNRLoss()

    def __call__(
            self,
            pred_magnitude,
            target_magnitude,
            pred_waveform,
            target_waveform
        ):
        snr_loss = 0

        with torch.no_grad():
            # SDR losses with time-domain waveforms
            snr_loss = self.snr_loss_criterion(pred_waveform, target_waveform)

        # MSE loss with frequency-domain magnitude slicq transform
        mse_loss = self.mse_loss_criterion(pred_magnitude, target_magnitude)

        final_loss = self.mcoef_time*snr_loss + self.mcoef_freq*mse_loss

        #print(f'snr_loss: {snr_loss}, mse_loss: {mse_loss}, snr_mix: {self.mcoef*snr_loss}, final_loss: {final_loss}')
        return final_loss
