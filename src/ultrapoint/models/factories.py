from abc import ABC, abstractmethod
from typing import Union

import torch
from loguru import logger

from ultrapoint.models.superpoint import SuperPoint
from ultrapoint.models.superpoint_pretrained import SuperPointPretrained


class ModelsFactory(ABC):
    SUPPORTED_MODELS = {}

    @staticmethod
    @abstractmethod
    def create(
        model_name: str, weights_path: str = None, state: dict = None, **model_kwargs
    ) -> torch.nn.Module:
        pass

    @staticmethod
    def load_weights(weights_path: str, device: Union[str, torch.device]) -> dict:
        state = torch.load(weights_path, map_location=device, weights_only=False)
        state = state["model_state_dict"] if "model_state_dict" in state else state
        return state


class SuperPointModelsFactory(ModelsFactory):
    SUPPORTED_MODELS = {
        "SuperPoint": SuperPoint,
        "SuperPointPretrained": SuperPointPretrained,
    }

    @staticmethod
    def create(
        model_name: str,
        weights_path: str = None,
        state: dict = None,
        device: Union[str, torch.device] = "cpu",
        **model_kwargs,
    ) -> torch.nn.Module:
        assert (
            model_name in SuperPointModelsFactory.SUPPORTED_MODELS
        ), f"Model {model_name} is not supported"

        if weights_path is not None:
            state = ModelsFactory.load_weights(weights_path, device)

        model = SuperPointModelsFactory.SUPPORTED_MODELS[model_name](**model_kwargs)

        if state is not None:
            try:
                model.load_state_dict(state)
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")
                raise

        model.to(device)

        return model
