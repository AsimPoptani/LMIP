from torch.utils.data import Dataset
import torch, pandas as pd


class MidiDataset(Dataset):
    """This class converts midi files to tensors. It takes in a CSV file generated by the midi_to_csv.py script."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with the midi files and their metadata.
        """
        self.csv_file = csv_file
        self.midi_data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.midi_data)

    def __getitem__(self, idx):
        pass
