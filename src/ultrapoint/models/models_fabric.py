from loguru import logger

from ultrapoint.models.super_point_net import SuperPointNet


class ModelsFabric:
    SUPPORTED_MODELS = {"SuperPointNet": SuperPointNet}

    @staticmethod
    def create(model_name: str, state=None, **kwargs):
        assert (
            model_name in ModelsFabric.SUPPORTED_MODELS
        ), f"Model {model_name} is not supported"

        model = ModelsFabric.SUPPORTED_MODELS[model_name](**kwargs)

        if state is not None:
            try:
                model.load_state_dict(state)
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")
                raise

        return model
