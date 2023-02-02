from xumx_slicq_v2 import transforms
import torch
import auraloss

def inner_mse(target, est):
    return torch.mean((est - target) ** 2)


def total_mse_loss(est_magnitude, target_magnitude):
    loss = 0
    for i in range(len(target_magnitude)):
        mse_loss = 0.

        # 4C1 Combination Losses
        mse_loss_1 = inner_mse(target_magnitude[i][:, 0], est_magnitude[i][0, :])
        mse_loss_2 = inner_mse(target_magnitude[i][:, 1], est_magnitude[i][1, :])
        mse_loss_3 = inner_mse(target_magnitude[i][:, 2], est_magnitude[i][2, :])
        mse_loss_4 = inner_mse(target_magnitude[i][:, 3], est_magnitude[i][3, :])

        # 4C2 Combination Losses
        mse_loss_5 = inner_mse(
            target_magnitude[i][:, 0] + target_magnitude[i][:, 1],
            est_magnitude[i][0, :] + est_magnitude[i][1, :],
        )
        mse_loss_6 = inner_mse(
            target_magnitude[i][:, 0] + target_magnitude[i][:, 2],
            est_magnitude[i][0, :] + est_magnitude[i][2, :],
        )
        mse_loss_7 = inner_mse(
            target_magnitude[i][:, 0] + target_magnitude[i][:, 3],
            est_magnitude[i][0, :] + est_magnitude[i][3, :],
        )
        mse_loss_8 = inner_mse(
            target_magnitude[i][:, 1] + target_magnitude[i][:, 2],
            est_magnitude[i][1, :] + est_magnitude[i][2, :],
        )
        mse_loss_9 = inner_mse(
            target_magnitude[i][:, 1] + target_magnitude[i][:, 3],
            est_magnitude[i][1, :] + est_magnitude[i][3, :],
        )
        mse_loss_10 = inner_mse(
            target_magnitude[i][:, 2] + target_magnitude[i][:, 3],
            est_magnitude[i][2, :] + est_magnitude[i][3, :],
        )

        # 4C3 Combination Losses
        mse_loss_11 = inner_mse(
            target_magnitude[i][:, 0] + target_magnitude[i][:, 1] + target_magnitude[i][:, 2],
            est_magnitude[i][0, :] + est_magnitude[i][1, :] + est_magnitude[i][2, :],
        )
        mse_loss_12 = inner_mse(
            target_magnitude[i][:, 0] + target_magnitude[i][:, 1] + target_magnitude[i][:, 3],
            est_magnitude[i][0, :] + est_magnitude[i][1, :] + est_magnitude[i][3, :],
        )
        mse_loss_13 = inner_mse(
            target_magnitude[i][:, 0] + target_magnitude[i][:, 2] + target_magnitude[i][:, 3],
            est_magnitude[i][0, :] + est_magnitude[i][2, :] + est_magnitude[i][3, :],
        )
        mse_loss_14 = inner_mse(
            target_magnitude[i][:, 1] + target_magnitude[i][:, 2] + target_magnitude[i][:, 3],
            est_magnitude[i][1, :] + est_magnitude[i][2, :] + est_magnitude[i][3, :],
        )

        # All 14 Combination Losses (4C1 + 4C2 + 4C3)
        mse_loss = (
            mse_loss_1
            + mse_loss_2
            + mse_loss_3
            + mse_loss_4
            + mse_loss_5
            + mse_loss_6
            + mse_loss_7
            + mse_loss_8
            + mse_loss_9
            + mse_loss_10
            + mse_loss_11
            + mse_loss_12
            + mse_loss_13
            + mse_loss_14
        ) / 14.0

        loss += mse_loss
    return loss / len(target_magnitude)


class LossCriterion:
    def __init__(self, mcoef=0.1):
        self.sdr_loss_criterion = auraloss.time.SISDRLoss()
        self.mcoef = mcoef

    def __call__(
        self,
        est_waveforms,
        target_waveforms,
        est_mag_nsgts,
        target_mag_nsgts,
    ):
        sdr_loss = 0

        est_waveform_1 = est_waveforms[:, 0, ...]
        est_waveform_2 = est_waveforms[:, 1, ...]
        est_waveform_3 = est_waveforms[:, 2, ...]
        est_waveform_4 = est_waveforms[:, 3, ...]

        target_waveform_1 = target_waveforms[:, 0, ...]
        target_waveform_2 = target_waveforms[:, 1, ...]
        target_waveform_3 = target_waveforms[:, 2, ...]
        target_waveform_4 = target_waveforms[:, 3, ...]

        # 4C1 Combination Losses
        sdr_loss_1 = self.sdr_loss_criterion(est_waveform_1, target_waveform_1)
        sdr_loss_2 = self.sdr_loss_criterion(est_waveform_2, target_waveform_2)
        sdr_loss_3 = self.sdr_loss_criterion(est_waveform_3, target_waveform_3)
        sdr_loss_4 = self.sdr_loss_criterion(est_waveform_4, target_waveform_4)

        # 4C2 Combination Losses
        sdr_loss_5 = self.sdr_loss_criterion(
            est_waveform_1 + est_waveform_2, target_waveform_1 + target_waveform_2
        )
        sdr_loss_6 = self.sdr_loss_criterion(
            est_waveform_1 + est_waveform_3, target_waveform_1 + target_waveform_3
        )
        sdr_loss_7 = self.sdr_loss_criterion(
            est_waveform_1 + est_waveform_4, target_waveform_1 + target_waveform_4
        )
        sdr_loss_8 = self.sdr_loss_criterion(
            est_waveform_2 + est_waveform_3, target_waveform_2 + target_waveform_3
        )
        sdr_loss_9 = self.sdr_loss_criterion(
            est_waveform_2 + est_waveform_4, target_waveform_2 + target_waveform_4
        )
        sdr_loss_10 = self.sdr_loss_criterion(
            est_waveform_3 + est_waveform_4, target_waveform_3 + target_waveform_4
        )

        # 4C3 Combination Losses
        sdr_loss_11 = self.sdr_loss_criterion(
            est_waveform_1 + est_waveform_2 + est_waveform_3,
            target_waveform_1 + target_waveform_2 + target_waveform_3,
        )
        sdr_loss_12 = self.sdr_loss_criterion(
            est_waveform_1 + est_waveform_2 + est_waveform_4,
            target_waveform_1 + target_waveform_2 + target_waveform_4,
        )
        sdr_loss_13 = self.sdr_loss_criterion(
            est_waveform_1 + est_waveform_3 + est_waveform_4,
            target_waveform_1 + target_waveform_3 + target_waveform_4,
        )
        sdr_loss_14 = self.sdr_loss_criterion(
            est_waveform_2 + est_waveform_3 + est_waveform_4,
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

        mse_loss = total_mse_loss(est_mag_nsgts, target_mag_nsgts)
        return self.mcoef*sdr_loss + mse_loss
