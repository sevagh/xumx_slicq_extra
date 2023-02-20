import torch
import auraloss
import numpy as np


class MSELossCriterion:
    def __init__(self):
        pass

    def __call__(
        self,
        pred_mag,
        target_mag
    ):
        loss = 0.
        for i, (pred_block, target_block) in enumerate(zip(pred_mag, target_mag)):
            mse_loss = 0.

            # 4C1 Combination Losses
            for j in [0, 1, 2, 3]:
                mse_loss += self._inner_mse_loss(pred_block[j], target_block[j])

            # 4C2 Combination Losses
            for (j, k) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                mse_loss += self._inner_mse_loss(
                    pred_block[j] + pred_block[k],
                    target_block[j] + target_block[k],
                )

            # 4C3 Combination Losses
            for (j, k, l) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                mse_loss += self._inner_mse_loss(
                    pred_block[j] + pred_block[k] + pred_block[l],
                    target_block[j] + target_block[k] + target_block[l],
                )

            loss += mse_loss/14.0
        return loss / len(pred_mag)

    @staticmethod
    def _inner_mse_loss(pred_block, target_block):
        return torch.mean((pred_block - target_block) ** 2)


class ComplexMSELossCriterion:
    def __init__(self):
        pass

    def __call__(
        self,
        pred_complex,
        target_complex,
    ):
        loss = 0.
        for i, (pred_block, target_block) in enumerate(zip(pred_complex, target_complex)):
            mse_loss = 0.

            # 4C1 Combination Losses
            for j in [0, 1, 2, 3]:
                mse_loss += self._inner_mse_loss(pred_block[j], target_block[j])

            # 4C2 Combination Losses
            for (j, k) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                mse_loss += self._inner_mse_loss(
                    pred_block[j] + pred_block[k],
                    target_block[j] + target_block[k],
                )

            # 4C3 Combination Losses
            for (j, k, l) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                mse_loss += self._inner_mse_loss(
                    pred_block[j] + pred_block[k] + pred_block[l],
                    target_block[j] + target_block[k] + target_block[l],
                )

            loss += mse_loss/14.0
        return loss / len(pred_complex)

    @staticmethod
    def _inner_mse_loss(pred_block, target_block):
        assert pred_block.shape[-1] == target_block.shape[-1] == 2
        return torch.mean((pred_block - target_block) ** 2)



class SDRLossCriterion:
    def __init__(self):
        self.sdsdr = auraloss.time.SDSDRLoss()

    def __call__(
        self,
        pred_waveforms,
        target_waveforms,
    ):
        loss = 0.

        # 4C1 Combination Losses
        for i in [0, 1, 2, 3]:
            loss += self.sdsdr(pred_waveforms[i], target_waveforms[i])

        # 4C2 Combination Losses
        for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            loss += self.sdsdr(
                pred_waveforms[i] + pred_waveforms[j],
                target_waveforms[i] + target_waveforms[j],
            )

        # 4C3 Combination Losses
        for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            loss += self.sdsdr(
                pred_waveforms[i] + pred_waveforms[j] + pred_waveforms[k],
                target_waveforms[i] + target_waveforms[j] + target_waveforms[k],
            )

        return loss/14.0
