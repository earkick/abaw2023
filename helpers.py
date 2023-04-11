from typing import Tuple, Dict
import torch
from inspect import signature


def extract_model_opts(opts: Dict, model_class) -> Dict:
    """Extract paramerters from opts and return a dictionary of model options.
    Args:
        opts (dict): dictionary of options
        model_class (class): class of model

    Returns:
        dict: dictionary of model options
    """
    sig = signature(model_class)
    params = list(sig.parameters.keys())

    model_params = {}
    for k, v in opts.items():
        if k in params:
            model_params[k] = v

    return model_params


def switch_off_grads(model: torch.nn.Module):
    """Switch off gradient computation for all parameters in a model"""
    for p in model.parameters():
        p.requires_grad = False
    return model


def switch_on_grads(model: torch.nn.Module):
    """Switch on gradient computation for all parameters in a model"""
    for p in model.parameters():
        p.requires_grad = True
    return model


def early_stopping(loss_value: float, epoch: int,
                   best_early_stopping_value: float,
                   early_stop_counter: int,
                   early_stop_minimize: bool = True,
                   early_stop_metric: str = "loss",
                   max_increase_in_loss: float = 0.1,
                   early_stop_patience: int = 10,
                   min_epochs_to_train: int = 10, verbose: bool = True, ) -> Tuple[bool, float, int]:
    """
    Early stopping function. Returns True if early stopping should be applied.
    """
    # check if the current val loss has gone beyond
    diff = loss_value - best_early_stopping_value if early_stop_minimize else best_early_stopping_value - loss_value

    if epoch < min_epochs_to_train:
        if verbose:
            print("Epoch %d is smaller than the minimal epochs to train set to %d." % (epoch, min_epochs_to_train))
            if diff < 0:
                if early_stop_minimize:
                    print("Good news! Monitored value (%s) decreased from %4f to %4f." % (early_stop_metric,
                                                                                          best_early_stopping_value,
                                                                                          loss_value))
                else:
                    print("Good news! Monitored value (%s) increased from %4f to %4f." % (early_stop_metric,
                                                                                          best_early_stopping_value,
                                                                                          loss_value))
            else:
                print("Did not improve. Current value: %.4f. Best value so far: %.4f" % (loss_value,
                                                                                         best_early_stopping_value))

        if early_stop_minimize:
            best_early_stopping_value = min(best_early_stopping_value, loss_value)
        else:
            best_early_stopping_value = max(best_early_stopping_value, loss_value)
        return False, best_early_stopping_value, early_stop_counter

    if diff > max_increase_in_loss:
        return True, best_early_stopping_value, early_stop_counter

    # Control stop by exceeding patience threshold
    early_stop_counter = early_stop_counter + 1 if diff >= 0 else 0
    if verbose:
        if diff >= 0:
            print("Did not improve. Current value: %.4f. Best value so far: %.4f. Early stop counter: %d/%d" % (loss_value,
                                                                                                                best_early_stopping_value,
                                                                                                                early_stop_counter,
                                                                                                                early_stop_patience))
        else:
            print("Value improved from %4f to %4f." % (best_early_stopping_value, loss_value))

    if early_stop_counter >= early_stop_patience:
        if verbose:
            print("No patience any more. Stopping the training...")
        return True, best_early_stopping_value, early_stop_counter

    if early_stop_minimize:
        best_early_stopping_value = min(best_early_stopping_value, loss_value)
    else:
        best_early_stopping_value = max(best_early_stopping_value, loss_value)
    return False, best_early_stopping_value, early_stop_counter