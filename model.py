import torch
from typing import Tuple, List, Dict
# From Moody_ml
from helpers import switch_off_grads, extract_model_opts

# hidden size of output from backbone
backbone_dims = {"base": 768,
                 "large": 1024}


def get_backbone_hidden_dims(name: str):
    if "base" in name:
        return backbone_dims["base"]
    if "large" in name or "audeering" in name:
        return backbone_dims["large"]
    raise ValueError("Given model architecture size is not supported!")


def initialize_backbone(model_type: str = "hubert_base") -> Tuple[torch.nn.Module, int]:
    # create online and target networks
    if "wav2vec2" in model_type:
        from transformers import Wav2Vec2Model
        if model_type == "wav2vec2-base":
            backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", return_dict=False, torchscript=True)
        elif model_type == "wav2vec2-large":
            backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53", return_dict=False, torchscript=True)

    elif "hubert" in model_type:
        from transformers import HubertModel
        if model_type == "hubert-base":
            backbone = HubertModel.from_pretrained("facebook/hubert-base-ls960", return_dict=False, torchscript=True)
        elif model_type == "hubert-large":
            backbone = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", return_dict=False, torchscript=True)
        elif model_type == "hubert-xlarge":
            backbone = HubertModel.from_pretrained("facebook/hubert-xlarge-ls960-ft", return_dict=False, torchscript=True)

    elif "data2vec" in model_type:
        from transformers import Wav2Vec2Model
        if model_type == "data2vec-base":
            backbone = Wav2Vec2Model.from_pretrained("facebook/data2vec-audio-base-960h", return_dict=False, torchscript=True)
        if model_type == "data2vec-large":
            backbone = Wav2Vec2Model.from_pretrained("facebook/data2vec-audio-large-960h", return_dict=False, torchscript=True)

    elif "audeering" in model_type:
        from transformers import Wav2Vec2Model
        backbone = Wav2Vec2Model.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                                                 return_dict=False, torchscript=True)
    else:
        raise ValueError("Unknown model type")

    backbone_hidden = get_backbone_hidden_dims(model_type)
    return backbone, backbone_hidden


def initialize_backbone_feature_extracter(model_type: str = "hubert_base") -> Tuple[torch.nn.Module, int]:
    # create online and target networks
    if "wav2vec2" in model_type:
        from transformers import Wav2Vec2FeatureExtractor
        if model_type == "wav2vec2-base":
            backbone_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        elif model_type == "wav2vec2-large":
            backbone_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    elif "hubert" in model_type:
        from transformers import HubertFeatureExtractor
        if model_type == "hubert-base":
            backbone_fe = HubertFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        elif model_type == "hubert-large":
            backbone_fe = HubertFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif model_type == "hubert-xlarge":
            backbone_fe = HubertFeatureExtractor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    elif "audeering" in model_type:
        from transformers import Wav2Vec2FeatureExtractor
        backbone_fe = Wav2Vec2FeatureExtractor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    else:
        raise ValueError("Unknown model type")

    return backbone_fe


class ResBlockBatchNorm(torch.nn.Module):
    """Residual block with MLP type layers and
    Batchnorm
    """

    def __init__(self, ninput: int, nhidden: int = 100):
        super(ResBlockBatchNorm, self).__init__()
        self.norm_input = torch.nn.BatchNorm1d(ninput)
        self.linear_1 = torch.nn.Linear(ninput, nhidden)
        self.linear_2 = torch.nn.Linear(nhidden, ninput)
        self.norm2 = torch.nn.BatchNorm1d(ninput)
        self.act = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.norm_input(x)
        z = self.act(self.linear_1(x))
        z = self.linear_2(z)
        return self.norm2(z) + x


class ResBlockLayerNorm(torch.nn.Module):
    """Residual block with MLP type layers and
    Layernorm
    """

    def __init__(self, ninput: int, nhidden: int = 100):
        super(ResBlockLayerNorm, self).__init__()
        self.norm_input = torch.nn.LayerNorm(ninput)
        self.linear_1 = torch.nn.Linear(ninput, nhidden)
        self.linear_2 = torch.nn.Linear(nhidden, ninput)
        self.norm2 = torch.nn.LayerNorm(ninput)
        self.act = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.norm_input(x)
        z = self.act(self.linear_1(x))
        z = self.linear_2(z)
        return self.norm2(z) + x


class DeepMLPResNet(torch.nn.Module):
    """Multi layer MLP network with residual blocks, so input and
    between block dimensionality remains same till output layer.
    """

    def __init__(self, num_layers: int = 3,
                 ninput: int = 512,
                 nout: int = 20,
                 nhidden: int = 100,
                 dropout: float = 0.2,
                 norm_type="layer_norm"):
        super(DeepMLPResNet, self).__init__()
        res_type = ResBlockBatchNorm if norm_type == "batch_norm" else ResBlockLayerNorm
        assert num_layers >= 1, "Num residual blocks has to be 1 or more"

        self.net = []
        for _ in range(num_layers):
            self.net.append(torch.nn.Sequential(
                res_type(ninput, nhidden),
                # WARNING: Dropout should not be used with BatchNorm!
                # c.f CVPR_2019 Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance
                torch.nn.Dropout(dropout)
            ))
        # attach output layer
        self.net.append(torch.nn.Linear(ninput, nout))
        self.net = torch.nn.ModuleList(self.net)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class AudioEmotionFeatureExtractorMeanPool(torch.nn.Module):
    """This part extracts features only. The second last layer
    of the final classification model. This is the module needed
    for the target network. An initialized classification head
    can be added later
    """

    def __init__(self, pretrained_model=None,
                 freeze_backbone: bool = True,
                 only_freeze: List[str] = None,
                 num_layers: int = 2,
                 backbone_hidden: int = 768,
                 num_projection_dims: int = 128,
                 nhidden: int = 256,
                 norm_type: str = "batchnorm",
                 dropout: float = 0.2):
        super(AudioEmotionFeatureExtractorMeanPool, self).__init__()

        self.only_freeze = only_freeze
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model

            if freeze_backbone:
                self.pretrained_model = self.pretrained_model.eval()
                if self.only_freeze is None:
                    # Freeze the whole network
                    self.pretrained_model = switch_off_grads(self.pretrained_model)
                else:
                    # Only freezes the layers in freeze_named_layers
                    self.pretrained_model = switch_off_partial_grads(self.pretrained_model, layer_names=self.only_freeze)

            self.no_feature_extraction = False
        else:
            self.no_feature_extraction = True

        self.freeze_backbone = freeze_backbone
        self.num_projection_dims = num_projection_dims
        self.pooling = torch.mean
        dp = dropout if norm_type != "batchnorm" else 0.0
        self.projection = DeepMLPResNet(num_layers, backbone_hidden, num_projection_dims,
                                        nhidden, dropout=dp, norm_type=norm_type)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Optional, simple Standard Scaling,
        # call EXPLICITLY before calling forward
        # if needed
        x = (x - x.mean()) / (x.std() + 1e-7)
        # expand the batch dimension and return
        return x.view(1, -1)

    def extract_backbone_features(self, x):
        if self.no_feature_extraction:
            return x
        if self.freeze_backbone and self.only_freeze is None:
            with torch.no_grad():
                return self.pretrained_model(x)[0]
        return self.pretrained_model(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_backbone_features(x)
        x = self.pooling(x, dim=1, keepdim=False)
        x = self.projection(x)
        return x


def switch_off_partial_grads(model: torch.nn.Module, layer_names):
    """Switch off gradient computation for a part of the network specified by layer_names.
    Names used in the networks are: (1) 'feature_extractor', (2) 'feature_projection', (3) 'encoder'

    """
    children = list(model.named_children())
    for name, c in children:
        if name in layer_names:
            for p in c.parameters():
                p.requires_grad = False

    return model


class AudioEmotionFeatureExtractorMeanPoolV2(torch.nn.Module):
    """Meanpool version 2 does not have a projection layer
    """
    def __init__(self, pretrained_model=None,
                 freeze_backbone: bool = True,
                 only_freeze: List[str] = None,
                 backbone_hidden: int = 768,):
        super(AudioEmotionFeatureExtractorMeanPoolV2, self).__init__()

        self.only_freeze = only_freeze
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model

            if freeze_backbone:
                self.pretrained_model = self.pretrained_model.eval()
                if self.only_freeze is None:
                    # Freeze the whole network
                    self.pretrained_model = switch_off_grads(self.pretrained_model)
                else:
                    # Only freezes the layers in freeze_named_layers
                    self.pretrained_model = switch_off_partial_grads(self.pretrained_model, layer_names=self.only_freeze)

            self.no_feature_extraction = False
        else:
            self.no_feature_extraction = True

        self.freeze_backbone = freeze_backbone
        self.num_projection_dims = backbone_hidden
        self.pooling = torch.mean

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Optional, simple Standard Scaling,
        # call EXPLICITLY before calling forward
        # if needed
        x = (x - x.mean()) / (x.std() + 1e-7)
        # expand the batch dimension and return
        return x.view(1, -1)

    def extract_backbone_features(self, x):
        if self.no_feature_extraction:
            return x
        if self.freeze_backbone and self.only_freeze is None:
            with torch.no_grad():
                return self.pretrained_model(x)[0]
        return self.pretrained_model(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_backbone_features(x)
        x = self.pooling(x, dim=1, keepdim=False)
        return x


def switch_off_partial_grads(model: torch.nn.Module, layer_names):
    """Switch off gradient computation for a part of the network specified by layer_names.
    Names used in the networks are: (1) 'feature_extractor', (2) 'feature_projection', (3) 'encoder'

    """

    children = list(model.named_children())
    for name, c in children:
        if name in layer_names:
            for p in c.parameters():
                p.requires_grad = False

    return model


class AudioEmotionFeatureExtractorMeanPoolV3(torch.nn.Module):
    """Meanpool version 2 does not have a projection layer
    """
    def __init__(self, pretrained_model=None,
                 freeze_backbone: bool = True,
                 only_freeze: List[str] = None,
                 backbone_hidden: int = 768,):
        super(AudioEmotionFeatureExtractorMeanPoolV3, self).__init__()

        self.only_freeze = only_freeze
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model

            if freeze_backbone:
                self.pretrained_model = self.pretrained_model.eval()
                if self.only_freeze is None:
                    # Freeze the whole network
                    self.pretrained_model = switch_off_grads(self.pretrained_model)
                else:
                    # Only freezes the layers in freeze_named_layers
                    self.pretrained_model = switch_off_partial_grads(self.pretrained_model, layer_names=self.only_freeze)

            self.no_feature_extraction = False
        else:
            self.no_feature_extraction = True

        self.freeze_backbone = freeze_backbone
        self.num_projection_dims = backbone_hidden
        self.relu = torch.nn.ReLU()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Optional, simple Standard Scaling,
        # call EXPLICITLY before calling forward
        # if needed
        x = (x - x.mean()) / (x.std() + 1e-7)
        # expand the batch dimension and return
        return x.view(1, -1)

    def extract_backbone_features(self, x):
        if self.no_feature_extraction:
            return x
        if self.freeze_backbone and self.only_freeze is None:
            with torch.no_grad():
                return self.pretrained_model(x)[0]
        return self.pretrained_model(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_backbone_features(x)
        x_relu = self.relu(x)
        x, _ = torch.max(x_relu, dim=1)
        return x


def switch_off_partial_grads(model: torch.nn.Module, layer_names):
    """Switch off gradient computation for a part of the network specified by layer_names.
    Names used in the networks are: (1) 'feature_extractor', (2) 'feature_projection', (3) 'encoder'

    """

    children = list(model.named_children())
    for name, c in children:
        if name in layer_names:
            for p in c.parameters():
                p.requires_grad = False

    return model
# Ends here

class AudioEmotionFeatureExtractorTransformer(torch.nn.Module):
    """This part extracts features only. The second last layer
    of the final classification model. This is the module needed
    for the target network. An intialized classification head
    can be added later
    """

    def __init__(self, pretrained_model=None,
                 freeze_backbone: bool = True,
                 only_freeze: List[str] = None,
                 num_layers: int = 2,
                 backbone_hidden: int = 768,
                 num_projection_dims: int = 128,
                 nhidden: int = 256,
                 norm_type: str = "layer_norm",  # options: layer_norm, batchnorm
                 output_length: int = 1,
                 num_transformer_layers: int = 3,
                 num_transformer_heads: int = 4,
                 num_transformer_dims: int = 128,
                 dropout: float = 0.2,
                 stride: int = 2,
                 kernel_size: int = 5,
                 transformer_activation: str = 'relu'):
        super(AudioEmotionFeatureExtractorTransformer, self).__init__()

        self.only_freeze = only_freeze
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model

            if freeze_backbone:
                self.pretrained_model = self.pretrained_model.eval()
                if self.only_freeze is None:
                    # Freeze the whole network
                    self.pretrained_model = switch_off_grads(self.pretrained_model)
                else:
                    # Only freezes the layers in freeze_named_layers
                    self.pretrained_model = switch_off_partial_grads(self.pretrained_model, layer_names=self.only_freeze)
            self.no_feature_extraction = False
        else:
            self.no_feature_extraction = True

        self.freeze_backbone = freeze_backbone
        self.num_projection_dims = num_projection_dims
        self.dim_reducer = torch.nn.Sequential(torch.nn.Linear(backbone_hidden, num_transformer_dims),
                                               torch.nn.LayerNorm(num_transformer_dims),
                                               torch.nn.ReLU(True))
        self.pooling = TimePoolingTransformerConv(output_length,
                                                  num_transformer_layers,
                                                  num_transformer_dims,  # internal data representation dimensionality
                                                  num_transformer_heads,
                                                  nhidden,
                                                  dropout,
                                                  stride,
                                                  kernel_size,
                                                  transformer_activation)
        dp = dropout if norm_type != "batchnorm" else 0.0
        self.projection = DeepMLPResNet(num_layers, num_transformer_dims, num_projection_dims,
                                        nhidden, dropout=dp, norm_type=norm_type)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Optional, simple Standard Scaling,
        # call EXPLICITLY before calling forward
        # if needed
        x = (x - x.mean()) / (x.std() + 1e-7)
        # expand the batch dimension and return
        return x.view(1, -1)

    def extract_backbone_features(self, x):
        if self.no_feature_extraction:
            return x
        if self.freeze_backbone and self.only_freeze is None:
            with torch.no_grad():
                return self.pretrained_model(x)[0]
        return self.pretrained_model(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_backbone_features(x)
        x = self.dim_reducer(x)
        x = self.pooling(x).squeeze()
        x = self.projection(x)
        return x


class AudioEmotionClassifier(torch.nn.Module):
    """The online network contains one more layer as prediction
    network than target network.
    """

    def __init__(self, pretrained_backbone=None,
                 feature_extractor=None,
                 freeze_backbone: bool = True,
                 num_layers: int = 2,
                 backbone_hidden: int = 768,
                 num_projection_dims: int = 128,
                 nhidden: int = 256,
                 norm_type: str = "batchnorm",
                 class_probabilities: torch.Tensor = None):

        super(AudioEmotionClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            assert pretrained_backbone is not None, "If feature extractor is not provided, backbone must be provided!"
            self.feature_extractor = AudioEmotionFeatureExtractorMeanPool(pretrained_backbone,
                                                                          freeze_backbone,
                                                                          num_layers,
                                                                          backbone_hidden,
                                                                          num_projection_dims,
                                                                          nhidden,
                                                                          norm_type)
        self.relu = torch.nn.ReLU(True)
        num_input_to_prediction_head = self.feature_extractor.num_projection_dims
        self.prediction_head = torch.nn.Linear(num_input_to_prediction_head, num_projection_dims)
        # initialize the prediction head with uniform initialization between -0.1 and 0.1
        # torch.nn.init.uniform_(self.prediction_head.weight, -0.1, 0.1)

        # set the bias of the prediction head to reflect the class probabilities
        if class_probabilities is not None:
            self.set_prediction_head_bias(class_probabilities)
        # else:
            # set the bias to be equal to 0
            # self.prediction_head.bias.data = torch.zeros(num_projection_dims)

    def set_prediction_head_bias(self, class_probabilities: torch.Tensor):
        self.prediction_head.bias.data = torch.log(class_probabilities)

    def preprocess(self, x) -> torch.Tensor:
        # preprocessing done by feature_extractor
        return self.feature_extractor.preprocess(x)

    def project(self, x) -> torch.Tensor:
        # just get the feature extractor output
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor = None, projection: torch.Tensor = None) -> torch.Tensor:
        if projection is None:
            x = self.feature_extractor(x)
        else:
            x = projection
        x = self.relu(x)
        x = self.prediction_head(x)
        return x


feature_extractor_registry = {"mean_pool": AudioEmotionFeatureExtractorMeanPool,
                              "mean_pool_v2": AudioEmotionFeatureExtractorMeanPoolV2,
                              "mean_pool_v3": AudioEmotionFeatureExtractorMeanPoolV3,
                              "transformer": AudioEmotionFeatureExtractorTransformer}


def make_feature_extractor(args: Dict, backbone=None):
    feature_extractor_class = feature_extractor_registry[args["feature_extractor_type"]]
    model_params = extract_model_opts(args, feature_extractor_class)
    return feature_extractor_class(backbone, **model_params)