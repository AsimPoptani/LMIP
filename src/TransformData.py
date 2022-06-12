from midi_to_csv import FilterMIDI
import os, logging


def transformData(
    logger,
    unprocessed_data_dir="./data/unprocessed",
    processed_data_dir="./data/processed",
    replication=False,
    instruments=[0],
    csv_file="./data/index.csv",
):

    current_path = os.path.dirname(__file__)

    unprocessed_data_dir = os.path.abspath(unprocessed_data_dir)
    processed_data_dir = os.path.abspath(processed_data_dir)
    csv_file = os.path.abspath(csv_file)

    filter_midi = FilterMIDI(
        folder_output=os.path.join(current_path, processed_data_dir),
        csv_file=os.path.join(current_path, csv_file),
        logging=logger,
        instruments=instruments,
        replication=replication,
    )

    filter_midi.filter_and_extract_folder(
        os.path.join(current_path, unprocessed_data_dir)
    )


if __name__ == "__main__":
    # Set up a new logging with log name
    logging.basicConfig(filename="running.log", level=logging.INFO)

    # Create logger
    logger = logging.getLogger(name="TransformData")

    transformData(logger)
