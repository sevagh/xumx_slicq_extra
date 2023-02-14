import torch


def _inner_mse_loss(pred_block, target_block):
    return torch.mean((pred_block - target_block) ** 2)


def _total_mse_loss(pred_mag, target_mag):
    loss = 0.
    for i, (pred_block, target_block) in enumerate(zip(pred_mag, target_mag)):
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
    return loss / len(pred_mag)


class LossCriterion:
    def __init__(self):
        pass

    def __call__(
        self,
        pred_mag_nsgts,
        target_mag_nsgts,
    ):
        return _total_mse_loss(pred_mag_nsgts, target_mag_nsgts)
