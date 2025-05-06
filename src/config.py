from typing import Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class BertNERConfig(PretrainedConfig):
    model_type = "bertner"
    is_composition = True
    def __init__(self,
            model_name: str = "google-bert/bert-base-uncased",
            encoder_config: Optional[dict | PretrainedConfig] = None,
            pooler: str = 'last',
            freeze_bert: bool = False,
            lstm_layers: Optional[int] = None,
            lstm_hidden_size: Optional[int] = None,
            classifier_lr: Optional[float] = None,
            classifier_dropout: Optional[float] = None,
            no_crf: bool = False,
            weight_O: float = 0.01,
            bias_O: Optional[float] = None,
            **kwargs: str
        ) -> None:
        super().__init__(**kwargs)
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (encoder_config["model_type"] if "model_type" in encoder_config else "google-bert/bert-base-uncased")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        self.encoder_config = encoder_config

        self.model_name = model_name
        self.pooler = pooler
        self.freeze_bert = freeze_bert
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.classifier_lr = classifier_lr
        self.classifier_dropout = classifier_dropout
        self.no_crf = no_crf
        self.weight_O = weight_O
        self.bias_O = bias_O

CONFIG_MAPPING.update({"bertner": BertNERConfig})
