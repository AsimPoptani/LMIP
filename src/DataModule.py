from MidiDataset import MidiDataset
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class MidiDataModule(LightningDataModule):
    def __init__(
        self,
        csv_file: str,
        path_to_midi: str,
        batch_size: int,
        num_workers: int,
        logger: logging.Logger = logging.getLogger(name="MidiDataModule"),
    ):
        super().__init__()
        self.csv_file = csv_file
        self.path_to_midi = path_to_midi
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.midi_dataset = MidiDataset(self.csv_file, self.path_to_midi)

        # Train dataset if 80% of the data is used for training
        eighty_percent = int(len(self.midi_dataset) * 0.8)
        twenty_percent = len(self.midi_dataset) - eighty_percent
        self.midi_train, self.midi_val = random_split(
            self.midi_dataset, [eighty_percent, twenty_percent]
        )

    def train_dataloader(self):
        return DataLoader(
            self.midi_train, batch_size=1, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.midi_val, batch_size=1, num_workers=self.num_workers, shuffle=False
        )
