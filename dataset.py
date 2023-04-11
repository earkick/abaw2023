r""" CMUCremaAffwild2 audio emotion dataset, a.k.a CCA dataset"""
import torch
import os
import pandas as pd
import numpy as np
import librosa
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# import pdb
import warnings
warnings.simplefilter('ignore')


class CCAAudioEmotionDataset(torch.utils.data.Dataset):
    """
    Read labels from a csv file. Load the audio files, segment them,
    pad if necessary and return the data and labels.
    """

    def __init__(self, path2audio: str, path2labels: str,
                 # Ola added source_dataset
                 source_dataset: str,
                 extension: str = ".m4a", sample_rate: int = 16000,
                 generate_labels: bool = True,  # True if labels are needed by Trainer, else False
                 min_length_secs: float = 0,  # The minimal size of an audio, remove if length is smaller than it
                 max_length_secs: float = 10, pad_to_max_length: bool = True, resample_method: str = "soxr_vhq",
                 make_segments: bool = False, preprocessing_strategy: str = "none",
                 binarize_labels: bool = True, label_threshold: float = 0.33, fill_empty_labels: bool = False,
                 skip_scaler: bool = True, reduce_multi_labels: bool = False, drop_labels: List[str] = None,
                 drop_rows_when_dropping_labels: bool = False,  discard_audio_after_max_length: bool = False,
            ):
        self.labels_in_use = ["happy", "sad", "anger", "surprise", "disgust", "fear", "neutral", "other"]


        self.path_to_audio = path2audio
        self.extension = extension
        self.labels = pd.read_csv(path2labels)
        self.sample_rate = sample_rate
        self.generate_labels = generate_labels
        self.max_length_secs = max_length_secs
        self.pad_to_max_length = pad_to_max_length
        self.discard_audio_after_max_length = discard_audio_after_max_length
        self.resample_method = resample_method
        self.max_frame_length = int(self.max_length_secs * self.sample_rate)
        self.make_segments = make_segments
        self.preprocessing_strategy = preprocessing_strategy
        self.skip_scaler = skip_scaler
        self.binarize_labels = binarize_labels
        self.label_threshold = label_threshold
        self.fill_empty_labels = fill_empty_labels
        self.reduce_multi_labels = reduce_multi_labels
        self.source_dataset = source_dataset
        print("Before filtering source:", self.labels.shape)
        self.labels = self.labels[self.labels['source_dataset'].isin(self.source_dataset)]
        print("After filtering source:", self.labels.shape)

        self._drop_files_shorter_than_threshold(min_length_secs)

        self.drop_rows_when_dropping_labels = drop_rows_when_dropping_labels
        self.drop_labels = drop_labels
        if self.drop_labels is not None and len(self.drop_labels) > 0:
            self._drop_labels()

        # Scaler for the audio
        if self.preprocessing_strategy == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.preprocessing_strategy == "standard":
            self.scaler = StandardScaler()
        elif self.preprocessing_strategy == "robust":
            self.scaler = RobustScaler()
        elif self.preprocessing_strategy is None or self.preprocessing_strategy == "none":
            self.skip_scaler = True
        else:
            print("Unknown preprocessing method")
            self.skip_scaler = True

        self.label_matrix = self.get_label_matrix()
        self.class_weights, self.class_counts = self.compute_class_weights()

    def _drop_files_shorter_than_threshold(self, threshold):
        rows_to_drop = self.labels.query("duration < @threshold").index
        self.labels = self.labels.drop(index=rows_to_drop).reset_index(drop=True)

    def _drop_labels(self):

        print(f"Removing labels {self.drop_labels}")
        # remove columns first
        self.labels.drop(columns=self.drop_labels, inplace=True)

        # we also drop these labels from the ones that we are using the dataset:
        for label in self.drop_labels:
            if label in self.labels_in_use:
                self.labels_in_use.remove(label)


        if self.drop_rows_when_dropping_labels:
            # now drop rows where the sum of label values is < self.label_threshold
            rows_to_drop = np.where(self.labels.iloc[:, :-1].sum(axis=1) < self.label_threshold)[0]
            self.labels = self.labels.drop(index=rows_to_drop).reset_index(drop=True)

    def get_label_matrix(self):
        label_matrix = self.labels[self.labels_in_use].to_numpy()

        if self.binarize_labels:
            # binarize labels = only samples with values have 0. or 1. for each label
            # AND, that there is only 1 label per row, not multiple labels

            if self.reduce_multi_labels:
                # it can happen that for a single sample, more than one label has value > 0.0
                # this is only for the CMU Mosei data
                # In order make sure there is only one label here, choose the one with the higher
                # value. If ties, break by choosing first of two (deterministic)
                num_pos = label_matrix.sum(axis=1)
                mult_ids = np.where(num_pos > 1.)[0]
                print(f"Found {len(mult_ids)} samples with more than one label")
                for i in mult_ids:
                    max_i = np.argmax(label_matrix[i, :])
                    label_matrix[i, :] = np.zeros(label_matrix.shape[1])
                    label_matrix[i, max_i] = 1.

            # now threshold all rows
            label_matrix = (label_matrix >= self.label_threshold).astype(float)
            if self.fill_empty_labels:
                num_pos = label_matrix.sum(axis=1)
                zero_ids = num_pos == 0
                print(f"Found {zero_ids.sum()} samples with empty labels after thresholding")
                # take care of samples where none of the labels are = 1
                if label_matrix.shape[1] > 6:
                    # set the "other" or "neutral" class label
                    label_matrix[zero_ids, -1] = 1.  # "other" or "neutral" label is last index
                else:
                    # set it to happy, which is 0th label
                    label_matrix[zero_ids, 0] = 1.

        return label_matrix

    def compute_class_counts(self):
        # this dataset does not have binary labels, only soft labels
        return self.label_matrix.sum(axis=0)

    def compute_class_weights(self):
        class_counts = self.compute_class_counts()
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_counts = torch.from_numpy(class_counts).to(torch.float32)
        class_weights = torch.from_numpy(class_weights).to(torch.float32)
        return class_weights, class_counts

    def __len__(self):
        return self.label_matrix.shape[0]

    def _create_window(self, audio):

        # The number of audio segments that we can extract from this audio
        num_segments = int(np.ceil(audio.shape[0] / self.max_frame_length))
        segments = []

        for seg in range(num_segments - 1):
            segments.append(audio[self.max_frame_length * seg:self.max_frame_length * (seg + 1)])

        last_segment = audio[-self.max_frame_length:]
        last_segment = self._pad_to_max_frame_length(last_segment)
        segments.append(last_segment)

        return np.stack(segments).reshape(-1, self.max_frame_length)

    def _pad_to_max_frame_length(self, audio):
        """
        Pad the audio to the desired length or truncate it
        if it is longer than the desired length.
        """
        duration = len(audio)
        if duration < self.max_frame_length:
            audio = np.concatenate([audio, np.zeros(self.max_frame_length - duration)])
        elif duration > self.max_frame_length:
            audio = audio[:self.max_frame_length]
        return audio

    def load_audio(self, path2audio: str) -> np.ndarray:
        """
        Load the audio file, optionally scale and resample
        and optionally, segment it into segments of frame_length_secs.
        """
        audio, original_sr = librosa.load(path2audio, sr=None)
        # Check if audio needs to be resampled:
        if original_sr != self.sample_rate:
            librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate, res_type=self.resample_method)

        # Scale the audio
        if not self.skip_scaler:
            audio = self.scaler.fit_transform(audio.reshape(-1, 1)).squeeze()

        # Pad the audio to the desired length
        if self.pad_to_max_length:
            audio = self._pad_to_max_frame_length(audio)

        if self.discard_audio_after_max_length and len(audio) > self.max_frame_length:
            audio = audio[:self.max_frame_length]

        if self.make_segments:
            audio = self._create_window(audio)

        return audio

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.labels.id.iloc[idx]
        path2audio = os.path.join(self.path_to_audio, filename + self.extension)
        if self.extension == ".npy":
            audio = np.load(path2audio)  # pre-processed numpy
        else:
            audio = self.load_audio(path2audio)  # np.array
        if self.generate_labels:
            label = self.label_matrix[idx, :]  # one-hot or soft one-hot labels
            return torch.from_numpy(label).float(), torch.from_numpy(audio).float()
        return torch.from_numpy(audio).float()
