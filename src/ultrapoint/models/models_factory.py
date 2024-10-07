from loguru import logger

from ultrapoint.models.superpoint import SuperPointNet
from ultrapoint.models.superpoint_pretrained import SuperPoint


class ModelsFactory:
    SUPPORTED_MODELS = {
        "SuperPointNet": SuperPointNet,
        "SuperPointPretrained": SuperPoint,
    }

    @staticmethod
    def create(model_name: str, state=None, **kwargs):
        assert (
            model_name in ModelsFactory.SUPPORTED_MODELS
        ), f"Model {model_name} is not supported"

        model = ModelsFactory.SUPPORTED_MODELS[model_name](**kwargs)

        if state is not None:
            try:
                model.load_state_dict(state)
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")
                raise

        return model
