from ultrapoint.trainers.train_model_subpixel import TrainModelSubpixel
from ultrapoint.trainers.train_model_heatmap import TrainModelHeatmap


class TrainersFactory:
    SUPPORTED_TRAINERS = {
        "TrainModelSubpixel": TrainModelSubpixel,
        "TrainModelHeatmap": TrainModelHeatmap,
    }

    @staticmethod
    def create(config: dict, trainer_name: str, output_directory: str):
        assert (
            trainer_name in TrainersFactory.SUPPORTED_TRAINERS
        ), f"Trainer {trainer_name} not supported"

        trainer_module = TrainersFactory.SUPPORTED_TRAINERS[trainer_name]
        trainer = trainer_module(config, save_path=output_directory)

        return trainer
