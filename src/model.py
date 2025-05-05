import json
import os
import re
from pathlib import Path
from typing import Optional, Self, Union

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from .config import BertNERConfig
from .modeling.base import TokenModel
from .modeling.model_output import TokenClassifierOutput


class BertNER(nn.Module):
    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None

    def __init__(self, config: BertNERConfig, training: bool = True, encoder_from_pretrained: bool = True):
        super().__init__()
        config.bias_O = config.bias_O if training else None
        self.model = TokenModel(config, encoder_from_pretrained)
        self.config = config

    def forward(self, *args: int, **kwargs: str) -> TokenClassifierOutput:
        """Wrapper function for the model's forward pass."""
        output = self.model(*args, **kwargs)
        return output

    @torch.no_grad()
    def decode(self, *args: int, **kwargs: str) -> TokenClassifierOutput:
        """Wrapper function for the model's decode pass."""
        outputs = self.model(*args, **kwargs)
        outputs.predictions = self.model.decode(outputs.logits, outputs.prediction_mask)
        return outputs

    @property
    def device(self) -> torch.device:
        device = next(self.model.parameters()).device
        return device

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None) -> nn.Embedding:
        self.config.encoder_config.vocab_size = new_num_tokens
        return self.model.encoder.bert.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def prepare_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Prepare state dict in the case of torch.compile
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            key = re.sub(r"_orig_mod\.", "", key)
            new_state_dict[key] = tensor
        return new_state_dict

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[BertNERConfig] = None,
        safe_serialization: bool = False,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            safe_serialization (`bool`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        # model_state_dict = self.prepare_state_dict(self.model.state_dict())
        model_state_dict = self.model.state_dict()
        # save model weights using safetensors
        if safe_serialization:
            save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))
        else:
            torch.save(
                model_state_dict,
                os.path.join(save_directory, "pytorch_model.bin"),
            )

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "config.json")

        return None

    @classmethod
    def from_pretrained(cls, model_id: str, map_location: str = "cpu", strict: bool = False) -> Self:
        """
        Load a pretrained model from a given model ID.

        Args:
            model_id (str): Identifier of the model to load.
            map_location (str): Device to map model to. Defaults to "cpu".
            strict (bool): Enforce strict state_dict loading.

        Returns:
            An instance of the model loaded from the pretrained weights.
        """

        model_dir = Path(model_id)  # / "pytorch_model.bin"
        model_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "pytorch_model.bin")
        config_file = Path(model_dir) / "config.json"

        with open(config_file, "r") as f:
            config_ = json.load(f)
        config = BertNERConfig(**config_)
        bertner = cls(config, encoder_from_pretrained=False)

        if model_file.endswith("safetensors"):
            state_dict = {}
            with safe_open(model_file, framework="pt", device=map_location) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
        bertner.model.load_state_dict(state_dict, strict=strict)
        bertner.model.to(map_location)

        bertner.eval()

        return bertner
