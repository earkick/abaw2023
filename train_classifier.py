import torch
import numpy as np
import os
import wandb
import json
import pandas as pd
import gc
from collections import defaultdict
from os.path import join
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, mean_absolute_error
from typing import Tuple, Union, List, Dict
from utils import make_save_dir
from helpers import early_stopping, extract_model_opts
from model import get_backbone_hidden_dims, AudioEmotionClassifier, make_feature_extractor
from MTL import MTLEqual, MTLDWA


def make_classifier(config: Dict = None, config_json_path: str = None,
                    feature_extractor_checkpoint_path: str = None,
                    classifier_checkpoint_path: str = None,
                    device: str = None):
    assert config or config_json_path, "Provide either the config dict or the path to the config.json"
    if config is None:
        with open(config_json_path, "r") as f:
            config = json.load(f)
    feature_extractor = make_feature_extractor(config)
    if feature_extractor_checkpoint_path is not None:
        feature_extractor.load_state_dict(torch.load(feature_extractor_checkpoint_path))
    model_params = extract_model_opts(config, AudioEmotionClassifier)
    model_params["num_projection_dims"] = len(config["class_names"])
    clf = AudioEmotionClassifier(feature_extractor=feature_extractor, **model_params)
    if classifier_checkpoint_path is not None:
        clf.load_state_dict(torch.load(classifier_checkpoint_path), strict=False)
    if device is not None:
        clf = clf.to(device)
    return clf


class CCATrainer:
    """Classification trainer"""

    def __init__(self,
                 model,
                 class_names: Union[Tuple[str], List[str]],
                 class_weights: torch.Tensor = None,
                 num_epochs: int = 50,
                 device: str = 'cpu',
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 batch_size: int = 32,
                 experiment_dir: str = '.',
                 log_every: int = 20,
                 log_to_wandb: bool = True,
                 checkpoint_every: int = 1,
                 min_epochs_to_train: int = 10,
                 max_increase_in_val_loss: float = 0.3,
                 early_stop_patience: int = 5,
                 early_stop_minimize: bool = True,
                 early_stop_metric: str = "loss",
                 confidence_threshold: float = 0.5,
                 target_threshold: float = 0.25,
                 use_scheduler: bool = False,
                 clip_grad_norm: bool = True,
                 max_grad_norm: float = 1.,
                 gradient_accumulation_steps: int = 1,
                 minority_class_augmentation: bool = False,
                 loss_combination_strategy: str = "equal"):

        # Usual and required parameters for any trainer
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.log_to_wandb = log_to_wandb
        self.checkpoint_every = checkpoint_every
        self.device = torch.device(device)
        self.minority_class_augmentation = minority_class_augmentation

        # optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.use_scheduler = use_scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                  T_0=15,
                                                                                  T_mult=1,
                                                                                  eta_min=1e-7,
                                                                                  last_epoch=-1,
                                                                                  verbose=False)
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm
        self.loss_func = self.soft_multilabel_bce_loss
        self.metric_func = self.compute_metrics
        self.loss_combination_strategy = loss_combination_strategy
        if self.loss_combination_strategy == "equal":
            self.loss_aggregator = MTLEqual(len(class_names))
        elif self.loss_combination_strategy.startswith("dwa"):
            # This is either dwa_default or dwa_trend
            self.loss_aggregator = MTLDWA(len(class_names), algorithm=self.loss_combination_strategy.split("_")[1])

        # Weights
        self.class_names = class_names
        self.num_classes = len(class_names)
        print("Using classes: %s" % ','.join(class_names))
        self.class_weights = class_weights

        # logging directory
        self.log_dir = make_save_dir(experiment_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=60)
        self.best_model_name = None

        # Early stop variables
        # TODO: add other metrics and direction
        self.max_increase_in_val_loss = max_increase_in_val_loss
        self.min_epochs_to_train = min_epochs_to_train
        self.early_stop_patience = early_stop_patience
        self.early_stop_counter = 0
        self.early_stop_minimize = early_stop_minimize
        self.best_early_stopping_value = 100. if self.early_stop_minimize else -100.

        # for binarization
        self.confidence_threshold = confidence_threshold
        self.target_threshold = target_threshold

        self.log_best_values = defaultdict(int)
        self.logged_metrics = ["precision", "recall", "f1bin", "mcc", "mae"]
        self.early_stop_metric = early_stop_metric

    def soft_multilabel_bce_loss(self, yhat, y):
        """
        Soft multilabel BCE loss.
        This is an adaptation of the typical BCE loss for multilabel classification.
        The difference is that the BCE loss is applied to each label independently and
        the loss is summed over all labels. Second, the labels are soft labels, i.e.
        they are not binary but can be in the range [0, 1].
        """
        loss_func = torch.nn.BCEWithLogitsLoss()
        list_losses = []
        for i in range(y.shape[1]):
            if self.class_weights is not None:
                loss = self.class_weights[i] * loss_func(yhat[:, i], y[:, i])
            else:
                loss = loss_func(yhat[:, i], y[:, i])
            list_losses.append(loss)

        total_loss = self.loss_aggregator.aggregate_losses(list_losses)
        return total_loss, list_losses

    def _train_step(self, dataloader, epoch, stage: str = "train"):

        targets = []
        predictions = []
        predicted_scores = []
        N = len(dataloader)
        self.optimizer.zero_grad()

        for i, (y, x) in enumerate(dataloader):

            yhat, loss, _, _ = self.__run_model(y, x)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_scheduler:
                    self.scheduler.step(epoch + i / N)

            prediction_score, prediction = self.get_scores_and_prediction(yhat.detach().cpu())

            predicted_scores.append(prediction_score)
            predictions.append(prediction)
            targets.append(y.detach().cpu().numpy())
            loss = float(loss.detach())

            if i % self.log_every == 0:
                self.metric_func(loss, np.concatenate(targets), np.concatenate(predictions),
                                 np.concatenate(predicted_scores), i, N, epoch, stage)
                gc.collect()
                torch.cuda.empty_cache()

        # last batch
        self.metric_func(loss, np.concatenate(targets), np.concatenate(predictions),
                         np.concatenate(predicted_scores), i, N, epoch, stage + "_end")

    def __run_model(self, y, x):

        x = x.to(self.device)
        y = y.to(self.device)
        y = y.squeeze()
        proj = self.model.project(x)
        yhat = self.model(projection=proj)

        # y = torch.atleast_2d(y)
        # TODO: Confusing...the two case below need to be better detailed.

        if x.shape[0] == 1:
            # This batch has only one single element, we need to unsqueeze y
            y = y.unsqueeze(dim=0)

        # That is the case in which y is only one emotion.
        # We need to expand the dimensions of Y in this case.
        if len(y.shape) == 1 and len(yhat.shape) == 2:
            y = torch.unsqueeze(y, dim=1)

        loss, list_losses_per_task = self.loss_func(yhat, y)
        return yhat, loss, proj, list_losses_per_task

    def _early_stopping(self, loss_value, epoch, verbose=True):

        out = early_stopping(loss_value,
                             epoch,
                             self.best_early_stopping_value,
                             self.early_stop_counter,
                             self.early_stop_minimize,
                             self.early_stop_metric,
                             self.max_increase_in_val_loss,
                             self.early_stop_patience,
                             self.min_epochs_to_train,
                             verbose=verbose)
        early_stop, self.best_early_stopping_value, self.early_stop_counter = out
        return early_stop

    def __is_improvement(self, value):
        diff = value - self.best_early_stopping_value if self.early_stop_minimize else self.best_early_stopping_value - value
        return diff < 0

    def train(self, train_dataloader, val_dataloader):

        # initialize optimizer
        for n in range(1, self.num_epochs + 1):
            self.model = self.model.to(self.device)
            self._train_step(train_dataloader, n)
            self.model.eval()
            val_value = self.evaluate(val_dataloader, epoch=n, stage="val", reload_best_val_model=False)
            self.model.train()

            if self.checkpoint_every == -1 and self.__is_improvement(val_value):
                self.best_model_name = join(self.log_dir, "model_epoch_best.pth")
                torch.save(self.model.cpu().state_dict(), self.best_model_name)

            elif self.checkpoint_every > 0 and n % self.checkpoint_every == 0:
                self.best_model_name = join(self.log_dir, "model_epoch_" + str(n) + ".pth")
                torch.save(self.model.cpu().state_dict(), self.best_model_name)

            if self._early_stopping(val_value, n):
                print("\nFinishing training early!")
                break

        last_model_name = join(self.log_dir, "model_epoch_" + str(n) + ".pth")
        print("Saving the last model at %s" % last_model_name)
        torch.save(self.model.cpu().state_dict(), last_model_name)

        return self.model.eval()

    def train_no_eval(self, dataloader):
        """Simply train the model on a dataset without"""
        self.model.train()
        for n in range(1, self.num_epochs + 1):
            self.model = self.model.to(self.device)
            self._train_step(dataloader, n, "final")

    def evaluate(self, dataloader, epoch: int = -1, reload_best_val_model=False, stage="test"):

        # load best model to run test
        if reload_best_val_model:
            self.model.load_state_dict(torch.load(self.best_model_name))

        total_loss = 0
        list_of_losses = []
        targets = []
        predictions = []
        projections = []  # for umap embedding
        predicted_scores = []
        N = len(dataloader)
        self.model = self.model.to(self.device)

        with torch.no_grad():
            for i, (y, x) in enumerate(dataloader):
                yhat, loss, projection, list_losses_per_task = self.__run_model(y, x)

                prediction_score, prediction = self.get_scores_and_prediction(yhat.detach().cpu())
                # Append predictions, scores and projections
                predicted_scores.append(prediction_score)
                predictions.append(prediction)
                targets.append(y.detach().cpu().numpy())
                projections.append(projection.detach().cpu().numpy())
                # Accumulate loss
                total_loss += float(loss.detach())
                list_of_losses.append([l.detach().cpu().numpy() for l in list_losses_per_task])

        monitored_value = self.metric_func(total_loss / N, np.concatenate(targets),
                                           np.concatenate(predictions),
                                           np.concatenate(predicted_scores), -1, N, epoch, stage)
        # project with umap
        projections = np.concatenate(projections)
        # project every 5 epochs
        if epoch % 5 == 0:
            self.plot_umap_embeddings(projections, np.argmax(np.concatenate(targets), axis=-1), stage)

        gc.collect()
        torch.cuda.empty_cache()
        if stage == "test":
            df_predicted_scores = pd.DataFrame(np.concatenate(predicted_scores),
                                               columns=[c + "_score" for c in self.class_names])
            df_targets = pd.DataFrame(np.concatenate(targets), columns=[c + "_target" for c in self.class_names])
            df_predictions = pd.DataFrame(np.concatenate(predictions),
                                          columns=[c + "_prediction" for c in self.class_names])
            df_results = pd.concat([df_targets, df_predicted_scores, df_predictions], axis=1)
            df_results.to_csv(join(self.log_dir, "final_results.csv"), index=False)

        if stage == "val":
            list_losses_per_task = np.stack(list_of_losses).mean(axis=0)  # Output is (#emotions,)
            self.loss_aggregator.adjust_after_validation(list_losses_per_task, epoch)

        return monitored_value

    def get_scores_and_prediction(self, yhat):
        scores = torch.sigmoid(yhat).detach().cpu().numpy()
        prediction = (scores >= self.confidence_threshold).astype(np.int16)
        return scores, prediction

    def get_binarized_targets(self, y):
        targets = y >= self.target_threshold
        return targets.astype(np.int16)

    def plot_wandb_histogram(self, targets, predictions, stage: str = ""):
        if not self.log_to_wandb:
            return

        target_ids = targets.sum(axis=0)
        pred_ids = predictions.sum(axis=0)  # counts for each class
        data = [[s, t, p] for (s, t, p) in zip(self.class_names, target_ids, pred_ids)]
        table = wandb.Table(data=data, columns=["class_names", "targets", "predictions"])
        wandb.log({stage + " target_distribution": wandb.plot.bar(table,
                                                                  "class_names", "targets",
                                                                  title=stage + " target distribution")})
        wandb.log({stage + " prediction_distribution": wandb.plot.bar(table,
                                                                      "class_names", "predictions",
                                                                      title=stage + " prediction distribution")})

    def compute_metrics(self, loss, targets, predictions, predicted_scores, batch_id, N, epoch, stage):
        returned_value = 0
        binarized_targets = self.get_binarized_targets(targets)

        # individual multilabel confusion matrices losses
        if stage in ["train", "train_end", "val", "test"]:

            results = {}
            for metric in self.logged_metrics:
                results[metric] = np.zeros(len(self.class_names))

            for k in range(len(self.class_names)):
                results["precision"][k] = precision_score(binarized_targets[:, k], predictions[:, k])
                results["recall"][k] = recall_score(binarized_targets[:, k], predictions[:, k])
                results["f1bin"][k] = f1_score(binarized_targets[:, k], predictions[:, k], average="binary")
                results["mcc"][k] = matthews_corrcoef(binarized_targets[:, k], predictions[:, k])
                results["mae"][k] = mean_absolute_error(targets[:, k], predicted_scores[:, k])

            avgs = {}
            for metric in self.logged_metrics:
                avgs[metric] = np.mean(results[metric])

            string = f"Epoch {epoch} batch {batch_id}/{N}, loss {stage} = {loss:.3f}, "
            print(string)

            print("(%s) Avg prec: %.3f; avg rec %.3f; avg f1 bin %.3f; avg MCC: %.3f, avg mae: %.3f" % (stage,
                                                                                                        avgs["precision"],
                                                                                                        avgs["recall"],
                                                                                                        avgs["f1bin"],
                                                                                                        avgs["mcc"],
                                                                                                        avgs["mae"]
                                                                                                        ))
            if self.log_to_wandb:

                if stage in ["train_end", "val", "test"]:
                    self.plot_wandb_histogram(binarized_targets, predictions, stage)

                wandb.log({"%s/avg_precision" % stage: avgs["precision"],
                           "%s/avg_recall" % stage: avgs["recall"],
                           "%s/avg_f1bin" % stage: avgs["f1bin"],
                           "%s/avg_mcc" % stage: avgs["mcc"],
                           "%s/avg_mae" % stage: avgs["mae"],
                           "epoch": epoch,
                           })
                for k in range(len(self.class_names)):
                    wandb.log({"%s/%s_precision" % (stage, self.class_names[k]): results["precision"][k],
                               "%s/%s_recall" % (stage, self.class_names[k]): results["recall"][k],
                               "%s/%s_f1bin" % (stage, self.class_names[k]): results["f1bin"][k],
                               "%s/%s_mcc" % (stage, self.class_names[k]): results["mcc"][k],
                               "%s/%s_mae" % (stage, self.class_names[k]): results["mae"][k],
                               "%s/loss" % stage: loss
                               })

            if stage not in ["train", "test"]:
                for metric in self.logged_metrics:
                    key = "%s/best_avg_%s" % (stage, metric)
                    self.log_best_values[key] = max(self.log_best_values[key], avgs[metric])
                    if self.log_to_wandb:
                        wandb.log({key: self.log_best_values[key]})

            if self.early_stop_metric == "mcc":
                returned_value = avgs["mcc"]
            elif self.early_stop_metric == "f1":
                returned_value = avgs["f1bin"]
            elif self.early_stop_metric == "mae":
                returned_value = avgs["mae"]
            elif self.early_stop_metric == "loss":
                returned_value = loss
            else:
                print("Metric %s not implemented" % self.early_stop_metric)

        return returned_value


def _pad_to_length_1d(audio, pad_length: int = 10 * 16000):
    """
    Pad the audio to the desired length or truncate it
    if it is longer than the desired length.
    """
    duration = len(audio)
    if duration < pad_length:
        audio = torch.concatenate([audio, torch.zeros(pad_length - duration)])
    elif duration > pad_length:
        audio = audio[:pad_length]
    return audio


def collate_pad_batch(batch):
    sizes = [item[1].shape[0] for item in batch]  # X is the dimension 1
    max_size = max(sizes)

    new_x, new_y = [], []
    for item in batch:
        xhat = _pad_to_length_1d(item[1], pad_length=max_size)
        new_x.append(xhat)
        new_y.append(item[0])

    return [torch.stack(new_y), torch.stack(new_x)]


def classification_training(target_network, train_dataset, val_dataset, test_dataset, wandb_run, args) -> torch.nn.Module:
    """
    Train a classification model based on the pre-trained model on the CMUCremaAffWild2 dataset.
    """
    # hidden size of output from backbone
    backbone_hidden = get_backbone_hidden_dims(args.model_type)

    # create new classifier and attach the target network to its feature extractor
    model = AudioEmotionClassifier(None,
                                   feature_extractor=target_network,
                                   freeze_backbone=args.freeze_backbone,
                                   num_layers=args.num_projection_layers,
                                   backbone_hidden=backbone_hidden,
                                   num_projection_dims=len(args.class_names),
                                   nhidden=args.nhidden,
                                   norm_type=args.norm_type)
    model = model.to(args.device)

    # get class_weights
    class_weights = None
    if args.class_weights:
        print("Adding class weights")
        CC = {c: w for (c, w) in zip(args.class_names, train_dataset.class_weights)}
        print(f"Classification class weights are {CC} \n")
        class_weights = train_dataset.class_weights.to(args.device)

    if args.weighted_sampler and class_weights is not None:
        print("Adding weighted sampler....")
        weights = 1. / train_dataset.class_counts
        weights = weights / weights.sum()
        samples_weight = np.array(
            [weights[int(np.argmax(train_dataset.label_matrix[i, :]))] for i in range(len(train_dataset))])
        samples_weight = torch.from_numpy(samples_weight).to(torch.float32)
        # Unfortunately, the current version of WeightedRandomSampler converts samples_weight to double
        # and doubles are not supported by 'mps'
        if args.device == "mps":
            samples_weight = samples_weight.to("cpu")
        else:
            samples_weight = samples_weight.to(args.device)

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    if args.pad_batch:
        collate_fn = collate_pad_batch
    else:
        collate_fn = None

    # setup dataloaders
    train_dataset.generate_labels = True
    val_dataset.generate_labels = True
    test_dataset.generate_labels = True

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
                                  sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                                  drop_last=args.drop_last_train_batch, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # optional set the prediction_head bias to reflect class proportions
    if args.set_prediction_head_bias_to_class_probs:
        print("Setting prediction head bias to class probabilities")
        class_probabilities = train_dataset.class_counts / train_dataset.class_counts.sum()
        model.set_prediction_head_bias(class_probabilities)

    # setup optimizer
    trainer = CCATrainer(model=model,
                         log_to_wandb=args.log_to_wandb,
                         class_names=args.class_names,
                         num_epochs=args.num_epochs,
                         experiment_dir=os.path.join(args.experiment_dir, "classification"),
                         device=args.device,
                         lr=args.lr,
                         target_threshold=args.target_threshold,
                         confidence_threshold=args.confidence_threshold,
                         weight_decay=args.weight_decay,
                         batch_size=args.batch_size,
                         gradient_accumulation_steps=args.gradient_accumulation_steps,
                         log_every=args.log_every,
                         min_epochs_to_train=args.min_epochs_to_train,
                         checkpoint_every=args.checkpoint_every,
                         max_increase_in_val_loss=args.max_increase_in_val_loss,
                         early_stop_patience=args.early_stop_patience,
                         early_stop_minimize=args.early_stop_minimize,
                         early_stop_metric=args.early_stop_metric,
                         class_weights=class_weights,
                         clip_grad_norm=args.clip_grad_norm,
                         use_scheduler=args.use_scheduler,
                         minority_class_augmentation=args.minority_class_augmentation,
                         loss_combination_strategy=args.loss_combination_strategy)

    # over-ride loss function
    if args.log_to_wandb:
        wandb.watch(model, log="gradients", log_freq=50)

    # save config
    with open(os.path.join(trainer.log_dir, 'classification_opts.json'), 'w') as f:
        json.dump(vars(args), f)
    # train
    trainer.train(train_dataloader, val_dataloader)
    # evaluate on test set
    trainer.evaluate(test_dataloader, reload_best_val_model=True)

    if args.retrain_on_all_data:
        print("\nRetraining on val and test data as well for final classifier....")
        trainer.num_epochs = 5
        trainer.train_no_eval(val_dataloader)
        trainer.train_no_eval(test_dataloader)
        model = trainer.model.cpu()

    if args.log_to_wandb:
        trained_classifier_artifact = wandb.Artifact("trained_audio_emotion_classifier", type="model",
                                                     description="Trained audio emotion classifier",
                                                     metadata=vars(args))
        trained_classifier_artifact.add_dir(trainer.log_dir)
        if wandb_run:
            wandb_run.log_artifact(trained_classifier_artifact)

    return model.eval()
