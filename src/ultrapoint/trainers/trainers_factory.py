from ultrapoint.trainers.trainer_subpixel import TrainerSubpixel
from ultrapoint.trainers.trainer_heatmap import TrainerHeatmap


class TrainersFactory:
    SUPPORTED_TRAINERS = {
        "TrainerSubpixel": TrainerSubpixel,
        "TrainerHeatmap": TrainerHeatmap,
    }

    @staticmethod
    def create(config: dict, trainer_name: str, output_directory: str):
        assert (
            trainer_name in TrainersFactory.SUPPORTED_TRAINERS
        ), f"Trainer {trainer_name} not supported"

        trainer_module = TrainersFactory.SUPPORTED_TRAINERS[trainer_name]
        trainer = trainer_module(config, save_path=output_directory)

        return trainer
