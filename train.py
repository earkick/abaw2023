r"""
This is a training script for the audio only model. The model is trained on the combined
CMU MOSEI, Affwild2, crema-d and ravdess datasets.
"""
import os
import torch
import numpy as np
import wandb
from dataset import CCAAudioEmotionDataset
from model import initialize_backbone
from train_classifier import classification_training
from helpers import make_feature_extractor
from utils import make_save_dir


def main(args):

    print(f"\n#### EXPERIMENT = {args.experiment_name} ####\n")

    if args.drop_labels is not None:
        print("Dropping Labels: ", args.drop_labels)

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_dir = make_save_dir(args.experiment_dir)
    args.experiment_dir = log_dir
    print(f"Experiment dir is {log_dir}")

    # load data
    print("Creating live CCA dataset...")
    train_dataset = CCAAudioEmotionDataset(args.train_data_path,
                                            args.train_label_path,
                                            sample_rate=args.sampling_rate,
                                            min_length_secs=args.min_length_secs,
                                            max_length_secs=args.max_length_secs,
                                            source_dataset=args.train_data_source,
                                            binarize_labels=args.binarize_labels,
                                            drop_rows_when_dropping_labels=args.drop_rows_when_dropping_labels,
                                            label_threshold=args.train_target_threshold,
                                            fill_empty_labels=args.fill_empty_labels,
                                            drop_labels=args.drop_labels,
                                            preprocessing_strategy=args.preprocessing_strategy,
                                            skip_scaler=args.skip_scaler,
                                            reduce_multi_labels=args.reduce_multi_labels,
                                            pad_to_max_length=not args.pad_batch,
                                            discard_audio_after_max_length=True)

    val_dataset = CCAAudioEmotionDataset(args.val_data_path,
                                            args.val_label_path,
                                            sample_rate=args.sampling_rate,
                                            min_length_secs=args.min_length_secs,
                                            max_length_secs=args.max_length_secs,
                                            source_dataset=args.val_data_source,
                                            binarize_labels=args.binarize_labels,
                                            drop_rows_when_dropping_labels=args.drop_rows_when_dropping_labels,
                                            label_threshold=args.val_target_threshold,
                                            fill_empty_labels=args.fill_empty_labels,
                                            drop_labels=args.drop_labels,
                                            preprocessing_strategy=args.preprocessing_strategy,
                                            skip_scaler=args.skip_scaler,
                                            reduce_multi_labels=args.reduce_multi_labels,
                                            pad_to_max_length=not args.pad_batch,
                                            discard_audio_after_max_length=True)

    test_dataset = CCAAudioEmotionDataset(args.test_data_path,
                                            args.test_label_path,
                                            sample_rate=args.sampling_rate,
                                            min_length_secs=args.min_length_secs,
                                            max_length_secs=args.max_length_secs,
                                            source_dataset=args.test_data_source,
                                            binarize_labels=args.binarize_labels,
                                            drop_rows_when_dropping_labels=args.drop_rows_when_dropping_labels,
                                            label_threshold=args.test_target_threshold,
                                            fill_empty_labels=args.fill_empty_labels,
                                            drop_labels=args.drop_labels,
                                            preprocessing_strategy=args.preprocessing_strategy,
                                            skip_scaler=args.skip_scaler,
                                            reduce_multi_labels=args.reduce_multi_labels,
                                            pad_to_max_length=not args.pad_batch,
                                            discard_audio_after_max_length=True)

    # The information about the classes that we are actually using comes from the dataset itself
    args.class_names = train_dataset.labels_in_use

    # wandb logging
    run = None
    if args.log_to_wandb:
        run = wandb.init(dir=args.experiment_dir, project=args.experiment_name,
                         notes=args.experiment_notes, tags=args.experiment_tags,
                         config=vars(args))

    # setup and train model
    # initialize the feature extracter and backbone
    print("Loading pretrained target network to resume training...")
    backbone, backbone_hidden = initialize_backbone(args.model_type)
    args.backbone_hidden = backbone_hidden
    target_network = make_feature_extractor(vars(args), backbone)
    if args.target_net_checkpoint is not None:
        target_network.load_state_dict(torch.load(args.target_net_checkpoint))

    print("\nTraining classification head...")
    final_classifier = classification_training(target_network.to("cpu"), train_dataset,
                                               val_dataset, test_dataset, run, args)
    # save model
    torch.save(final_classifier.state_dict(), os.path.join(args.experiment_dir, "final_classifier.pt"))

    # trace and script model
    if args.generate_jit:
        final_classifier = final_classifier.eval().to("cpu")
        example = train_dataset[0]
        if isinstance(example, tuple):
            example = example[1]
        else:
            example = example
        if example.ndim < 3:
            example = example.unsqueeze(0)  # add a batch dimension

            traced_model = torch.jit.trace(final_classifier, example)
            scripted_model = torch.jit.script(traced_model)
            scripted_model.save(os.path.join(args.experiment_dir, "scripted_classifier.pt"))

    print("## Done! ##")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    HOME = os.getenv("HOME")
    BASE = "/home/ubuntu/CMUCremaAffWild2_audio/"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="CCA_joao")
    parser.add_argument("--experiment_notes", type=str, default="")
    parser.add_argument("--experiment_tags", type=str, nargs="+", default=["cmu-crema-affwild2,byol,classification"])
    parser.add_argument("--experiment_dir", type=str, default=os.path.join(HOME, "ek_experiments/audio/cca_joao"))
    parser.add_argument("--train_data_path", type=str, default=os.path.join(BASE, "train_files"))
    parser.add_argument("--train_label_path", type=str, default=os.path.join(BASE, "train_labels_new_2023-01-12.csv"))
    parser.add_argument("--val_data_path", type=str, default=os.path.join(BASE, "val_files"))
    parser.add_argument("--val_label_path", type=str, default=os.path.join(BASE, "val_labels_new_2023-01-12.csv"))
    parser.add_argument("--test_data_path", type=str, default=os.path.join(BASE, "test_files"))
    parser.add_argument("--test_label_path", type=str, default=os.path.join(BASE, "test_labels_new_2023-01-12.csv"))
    parser.add_argument("--train_data_source", type=str, nargs="+", default=['crema_ravdess', 'cmu_mosei', 'affwild2'])
    parser.add_argument("--val_data_source", type=str, nargs="+", default=['crema_ravdess', 'cmu_mosei', 'affwild2'])
    parser.add_argument("--test_data_source", type=str, nargs="+", default=['crema_ravdess', 'cmu_mosei', 'affwild2'])
    parser.add_argument("--path_to_noise_files", type=str, default=os.path.join(BASE, "open_audio_datasets/noise_train"))
    parser.add_argument("-m", "--model_type", type=str, default="hubert-base")
    parser.add_argument("-r", "--resume_pretraining", action="store_true", help="resumes pretraining")
    parser.add_argument("--feature_extractor_type", type=str, default="mean_pool")
    parser.add_argument("-p", "--do_pretraining", action="store_true")
    parser.add_argument("-c", "--class_weights", action="store_true")
    parser.add_argument("-w", "--weighted_sampler", action="store_true")
    parser.add_argument("-s", "--set_prediction_head_bias_to_class_probs", action="store_true")
    parser.add_argument("--retrain_on_all_data", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--schedulerT0", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--min_length_secs", type=float, default=0.)
    parser.add_argument("--max_length_secs", type=float, default=20.)
    parser.add_argument("--num_projection_dims", type=int, default=32)
    parser.add_argument("--num_projection_layers", type=int, default=2)
    parser.add_argument("--num_transformer_layers", type=int, default=3)
    parser.add_argument("--num_transformer_heads", type=int, default=4)
    parser.add_argument("--num_transformer_dims", type=int, default=128)
    parser.add_argument("--nhidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm_type", type=str, default="batch_norm")
    parser.add_argument("--binarize_labels", action="store_true")
    parser.add_argument("--target_threshold", type=float, default=0.01)
    parser.add_argument("--train_target_threshold", type=float, default=0.01)
    parser.add_argument("--val_target_threshold", type=float, default=0.01)
    parser.add_argument("--test_target_threshold", type=float, default=0.01)
    parser.add_argument("--drop_rows_when_dropping_labels", action="store_true")
    parser.add_argument("--fill_empty_labels", action="store_true", help="after thresholding, some labels maybe all 0, set them to 'other'")
    parser.add_argument("--reduce_multi_labels", action="store_true", help="more than 1 positive labels per sample are squashed to take only first")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--skip_scaler", action="store_true", help="Add standard scaling before padding audio")
    parser.add_argument("--drop_last_train_batch", action="store_true")
    parser.add_argument("--min_epochs_to_train", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=2)
    parser.add_argument("--target_net_checkpoint", type=str, default=None)
    parser.add_argument("--max_increase_in_val_loss", type=float, default=0.2)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_minimize", action="store_true")
    parser.add_argument("--early_stop_metric", type=str, default="f1bin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad_norm", action="store_true")  # TODO: Not implemented
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("-u", "--use_scheduler", action="store_true")
    parser.add_argument("-f", "--freeze_backbone", action="store_true")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--generate_jit", action="store_true")
    parser.add_argument("--drop_labels", type=str, nargs="+", default=None)
    parser.add_argument("--only_freeze", type=str, nargs="+", default=None)  # Options:  "feature_extractor" "feature_projection" "encoder"
    parser.add_argument("--pad_batch", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--preprocessing_strategy", type=str, default="none")
    parser.add_argument("--loss_combination_strategy", type=str, default="equal")

    args = parser.parse_args()
    main(args)
