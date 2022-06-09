from pytorch_lightning import seed_everything, Trainer

from ..data import MNISTDataModule
from ..models import LeNetPLModel


class MTrainer:
    def __init__(self, max_epochs: int = 20) -> None:
        seed_everything(42, workers=True)
        # sets seeds for numpy, torch and python.random.
        self.model = LeNetPLModel()
        self.datamodule = MNISTDataModule()
        self.trainer = Trainer(
            default_root_dir=".",
            detect_anomaly=True,
            max_epochs=max_epochs,
            profiler="pytorch",
            enable_checkpointing=True,
            benchmark=True,
        )

    def train(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)

    def test(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule)
