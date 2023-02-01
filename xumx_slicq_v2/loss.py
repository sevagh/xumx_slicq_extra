from xumx_slicq_v2 import transforms
import torch
import auraloss


def _custom_mse_loss(pred_magnitude, target_magnitude):
    loss = 0
    for i in range(len(target_magnitude)):
        loss += torch.mean((pred_magnitude[i] - target_magnitude[i]) ** 2)
    return loss / len(target_magnitude)


class LossCriterion:
    def __init__(self, mcoef=0.1):
        self.sdr_loss_criterion = auraloss.time.SISDRLoss()
        #self.mse_loss_criterion = _custom_mse_loss
        #self.mcoef = mcoef

    def __call__(
        self,
        pred_waveforms,
        target_waveforms,
        #pred_nsgt_1,
        #pred_nsgt_2,
        #pred_nsgt_3,
        #pred_nsgt_4,
        #target_nsgt_1,
        #target_nsgt_2,
        #target_nsgt_3,
        #target_nsgt_4,
    ):
        sdr_loss = 0

        pred_waveform_1 = pred_waveforms[:, 0, ...]
        pred_waveform_2 = pred_waveforms[:, 1, ...]
        pred_waveform_3 = pred_waveforms[:, 2, ...]
        pred_waveform_4 = pred_waveforms[:, 3, ...]

        target_waveform_1 = target_waveforms[:, 0, ...]
        target_waveform_2 = target_waveforms[:, 1, ...]
        target_waveform_3 = target_waveforms[:, 2, ...]
        target_waveform_4 = target_waveforms[:, 3, ...]

        # 4C1 Combination Losses
        sdr_loss_1 = self.sdr_loss_criterion(pred_waveform_1, target_waveform_1)
        sdr_loss_2 = self.sdr_loss_criterion(pred_waveform_2, target_waveform_2)
        sdr_loss_3 = self.sdr_loss_criterion(pred_waveform_3, target_waveform_3)
        sdr_loss_4 = self.sdr_loss_criterion(pred_waveform_4, target_waveform_4)

        # 4C2 Combination Losses
        sdr_loss_5 = self.sdr_loss_criterion(
            pred_waveform_1 + pred_waveform_2, target_waveform_1 + target_waveform_2
        )
        sdr_loss_6 = self.sdr_loss_criterion(
            pred_waveform_1 + pred_waveform_3, target_waveform_1 + target_waveform_3
        )
        sdr_loss_7 = self.sdr_loss_criterion(
            pred_waveform_1 + pred_waveform_4, target_waveform_1 + target_waveform_4
        )
        sdr_loss_8 = self.sdr_loss_criterion(
            pred_waveform_2 + pred_waveform_3, target_waveform_2 + target_waveform_3
        )
        sdr_loss_9 = self.sdr_loss_criterion(
            pred_waveform_2 + pred_waveform_4, target_waveform_2 + target_waveform_4
        )
        sdr_loss_10 = self.sdr_loss_criterion(
            pred_waveform_3 + pred_waveform_4, target_waveform_3 + target_waveform_4
        )

        # 4C3 Combination Losses
        sdr_loss_11 = self.sdr_loss_criterion(
            pred_waveform_1 + pred_waveform_2 + pred_waveform_3,
            target_waveform_1 + target_waveform_2 + target_waveform_3,
        )
        sdr_loss_12 = self.sdr_loss_criterion(
            pred_waveform_1 + pred_waveform_2 + pred_waveform_4,
            target_waveform_1 + target_waveform_2 + target_waveform_4,
        )
        sdr_loss_13 = self.sdr_loss_criterion(
            pred_waveform_1 + pred_waveform_3 + pred_waveform_4,
            target_waveform_1 + target_waveform_3 + target_waveform_4,
        )
        sdr_loss_14 = self.sdr_loss_criterion(
            pred_waveform_2 + pred_waveform_3 + pred_waveform_4,
            target_waveform_2 + target_waveform_3 + target_waveform_4,
        )

        # All 14 Combination Losses (4C1 + 4C2 + 4C3)
        sdr_loss = (
            sdr_loss_1
            + sdr_loss_2
            + sdr_loss_3
            + sdr_loss_4
            + sdr_loss_5
            + sdr_loss_6
            + sdr_loss_7
            + sdr_loss_8
            + sdr_loss_9
            + sdr_loss_10
            + sdr_loss_11
            + sdr_loss_12
            + sdr_loss_13
            + sdr_loss_14
        ) / 14.0

        return sdr_loss

        #mse_loss = 0.0

        ## 4C1 Combination Losses
        #mse_loss_1 = self.mse_loss_criterion(pred_waveform_1, target_waveform_1)
        #mse_loss_2 = self.mse_loss_criterion(pred_waveform_2, target_waveform_2)
        #mse_loss_3 = self.mse_loss_criterion(pred_waveform_3, target_waveform_3)
        #mse_loss_4 = self.mse_loss_criterion(pred_waveform_4, target_waveform_4)

        ## 4C2 Combination Losses
        #mse_loss_5 = self.mse_loss_criterion(
        #    pred_waveform_1 + pred_waveform_2, target_waveform_1 + target_waveform_2
        #)
        #mse_loss_6 = self.mse_loss_criterion(
        #    pred_waveform_1 + pred_waveform_3, target_waveform_1 + target_waveform_3
        #)
        #mse_loss_7 = self.mse_loss_criterion(
        #    pred_waveform_1 + pred_waveform_4, target_waveform_1 + target_waveform_4
        #)
        #mse_loss_8 = self.mse_loss_criterion(
        #    pred_waveform_2 + pred_waveform_3, target_waveform_2 + target_waveform_3
        #)
        #mse_loss_9 = self.mse_loss_criterion(
        #    pred_waveform_2 + pred_waveform_4, target_waveform_2 + target_waveform_4
        #)
        #mse_loss_10 = self.mse_loss_criterion(
        #    pred_waveform_3 + pred_waveform_4, target_waveform_3 + target_waveform_4
        #)

        ## 4C3 Combination Losses
        #mse_loss_11 = self.mse_loss_criterion(
        #    pred_waveform_1 + pred_waveform_2 + pred_waveform_3,
        #    target_waveform_1 + target_waveform_2 + target_waveform_3,
        #)
        #mse_loss_12 = self.mse_loss_criterion(
        #    pred_waveform_1 + pred_waveform_2 + pred_waveform_4,
        #    target_waveform_1 + target_waveform_2 + target_waveform_4,
        #)
        #mse_loss_13 = self.mse_loss_criterion(
        #    pred_waveform_1 + pred_waveform_3 + pred_waveform_4,
        #    target_waveform_1 + target_waveform_3 + target_waveform_4,
        #)
        #mse_loss_14 = self.mse_loss_criterion(
        #    pred_waveform_2 + pred_waveform_3 + pred_waveform_4,
        #    target_waveform_2 + target_waveform_3 + target_waveform_4,
        #)

        ## All 14 Combination Losses (4C1 + 4C2 + 4C3)
        #mse_loss = (
        #    mse_loss_1
        #    + mse_loss_2
        #    + mse_loss_3
        #    + mse_loss_4
        #    + mse_loss_5
        #    + mse_loss_6
        #    + mse_loss_7
        #    + mse_loss_8
        #    + mse_loss_9
        #    + mse_loss_10
        #    + mse_loss_11
        #    + mse_loss_12
        #    + mse_loss_13
        #    + mse_loss_14
        #) / 14.0

        #return self.mcoef*sdr_loss + mse_loss
