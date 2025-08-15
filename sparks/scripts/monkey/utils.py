import numpy as np
from sklearn.metrics import r2_score

from sparks.utils.losses import bits_per_spike

def get_accuracy(decoder_outputs, test_dataset, test_loss, mode, target_type):
    if mode == 'prediction':
        targets = test_dataset.y_trial_data
        if target_type == 'direction':
            preds = decoder_outputs[..., 100:].cpu().numpy().transpose(0, 2, 1)
            test_acc = np.mean(targets[..., 100:] == preds.argmax(-1, keepdims=True)[..., 0])
        else:
            preds = decoder_outputs[..., 100:].cpu().numpy().reshape([targets.shape[0], targets.shape[1],
                                                                      targets.shape[2] - 100, -1]).transpose(0, 2, 1, 3)
            test_acc = r2_score(targets[..., 100:].transpose(0, 2, 1).reshape(-1, targets.shape[-2]),
                                preds[..., 0].reshape(-1, targets.shape[-2]), multioutput='variance_weighted')
    elif mode == 'spikes_pred':
        targets = test_dataset.x_trial_data
        preds = decoder_outputs.cpu().numpy().reshape([targets.shape[0], targets.shape[1],
                                                       -1, targets.shape[2]])[:, :, 0]
        test_acc = bits_per_spike(preds, targets)
    else:
        test_acc = -test_loss

    return test_acc
