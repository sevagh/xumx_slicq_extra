from xumx_slicq_v2 import transforms
import torch
import auraloss


def _inner_mse_loss(pred_block, target_block):
    return torch.mean((pred_block - target_block) ** 2)


def _total_mse_loss(pred_complex, target_complex):
    loss = 0.
    for i, (pred_block, target_block) in enumerate(zip(pred_complex, target_complex)):
        mse_loss = 0.

        # 4C1 Combination Losses
        for j in [0, 1, 2, 3]:
            mse_loss += _inner_mse_loss(pred_block[j], target_block[j])

        # 4C2 Combination Losses
        for (j, k) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            mse_loss += _inner_mse_loss(
                pred_block[j] + pred_block[k],
                target_block[j] + target_block[k],
            )

        # 4C3 Combination Losses
        for (j, k, l) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            mse_loss += _inner_mse_loss(
                pred_block[j] + pred_block[k] + pred_block[l],
                target_block[j] + target_block[k] + target_block[l],
            )

        loss += mse_loss/14.0
    return loss / len(pred_complex)


class LossCriterion:
    def __init__(self, mcoef=0.1):
        self.sdr_loss_criterion = auraloss.time.SDSDRLoss()
        self.mcoef = mcoef

    def __call__(
        self,
        pred_waveforms,
        target_waveforms,
        pred_nsgts,
        target_nsgts,
    ):
        sdr_loss = 0

        pred_waveform_1 = pred_waveforms[0]
        pred_waveform_2 = pred_waveforms[1]
        pred_waveform_3 = pred_waveforms[2]
        pred_waveform_4 = pred_waveforms[3]

        target_waveform_1 = target_waveforms[0]
        target_waveform_2 = target_waveforms[1]
        target_waveform_3 = target_waveforms[2]
        target_waveform_4 = target_waveforms[3]

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

        mse_loss = _total_mse_loss(pred_nsgts, target_nsgts)
        return self.mcoef*sdr_loss + mse_loss
