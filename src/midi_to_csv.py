'''
This code will filter and extract sections of the midi files which have a specific instrument(s) and save them in a new folder and put metadata in a csv file.
'''
from fileinput import filename
import os
from wsgiref import headers
import pretty_midi
import logging, glob,datetime
import uuid, pandas as pd
# Import tdqm
from  tqdm import tqdm

# TODO create ENUM class for different instruments


class FilterMIDI():
    """
     Usage:
     # Set up a new logging with log name filter_midi.log 
logging.basicConfig(filename="filter_midi.log",level=logging.INFO)


filter=FilterMIDI(folder_output="extracted/",csv_file="cool.csv",logging=logging.getLogger("FilterMIDI"))
filter.filter_and_extract_folder("maestro-v3.0.0/")
filter.filter_and_extract_folder("example/")
# .extract_midi("example/midi-example.midi")
    """

    def __init__(self,folder_output,csv_file,logging:logging.Logger=logging.getLogger(name="FilterMIDI"), instruments=[0],write_mode="a",replication=False) -> None:

        # folder_output: folder where the filtered midi files will be saved
        self._folder_output = folder_output
        # csv_file: csv file where the metadata will be saved
        self._csv_file = csv_file
        # instruments: list of instruments to filter
        self._instruments = instruments

        # Set up logging
        self._logger = logging

        # Write mode for csv file
        self._write_mode = write_mode

        # Replication
        self._replication = replication

        self._logger.info(f"FilterMIDI initialized at {datetime.datetime.now()}")

        # Check if the folder exists if not create it
        if not os.path.exists(self._folder_output):
            os.makedirs(self._folder_output)
            # Log that the folder was created
            self._logger.warning("Folder created: {}".format(self._folder_output))

        self._csv_file_data=None
        # Check if the csv file exists and read it to a dictionary headers "previous_midi_file_name","new_midi_file_name"
        if os.path.exists(self._csv_file):
            self._csv_file_data = pd.read_csv(self._csv_file,names=["previous_midi_file_name","new_midi_file_name"])
        else:
            # Otherwise create a new csv file
            self._csv_file_data = pd.DataFrame(columns=["previous_midi_file_name","new_midi_file_name"])
            self._csv_file_data.to_csv(self._csv_file,index=False,mode=self._write_mode)
            


    def check_replication_midi(self,midi_file:str) -> bool:
        # Check replication
        if not self._replication:
            if len(midi_file)==0 or not midi_file in self._csv_file_data["previous_midi_file_name"].values:
                return True
        else:
            return True
        self._logger.info("Midi file found and not replicating: {}".format(midi_file))
        return False

    def extract_midi(self,midi_file:str) -> pretty_midi.PrettyMIDI:
        # Return a midi file with only the instruments specified in the list

        # Open the midi file
        midi=pretty_midi.PrettyMIDI(midi_file)

        # Create a new midi file
        new_midi = pretty_midi.PrettyMIDI()
       

        # Get all channels with the instruments specified in the list
        for instrument in midi.instruments:
            # Extract the notes and control name from the instrument
            if instrument.name in self._instruments:
                new_midi.instruments.append(instrument)


        # Extracted midi file
        return new_midi

    def has_instruments(self,midi_file:str) -> bool:
        # Checks if the midi file has the instruments specified in the list
        midi=pretty_midi.PrettyMIDI(midi_file)


        # Return True if the midi file has the instruments specified in the list
        return all(instrument.program in self._instruments for instrument in midi.instruments)

    def filter_and_extract_folder(self,midi_folder) -> None:
        # Filter and extract all the midi files in a folder and save them in a new folder with metadata appended csv file

        # Get all midi files in the folder recursively
        midi_files = glob.glob(midi_folder+"/**/*.midi",recursive=True)

        mid_files=glob.glob(midi_folder+"/**/*.mid",recursive=True)

        midi_files.extend(mid_files)

        
        # Log processing of the folder
        self._logger.info("Processing folder: {} with {} midi files".format(midi_folder,len(midi_files)))
        
        # print(midi_files)
        with open(self._csv_file,self._write_mode) as csv_file:
            midi_file:pretty_midi.PrettyMIDI
            for midi_file in tqdm(midi_files):
                # print(self._logger)
                # Log the midi file
                self._logger.info("Processing midi file: {}".format(midi_file))
                # Get the name of the midi file
                midi_file_name = os.path.basename(midi_file)
                # Check if the midi file has the instruments specified in the list
                if self.check_replication_midi(midi_file) and self.has_instruments(midi_file):
                    # Generate a new midi file name
                    new_midi_file_name = str(uuid.uuid4())+".mid"
                    # Extract the midi file
                    midi = self.extract_midi(midi_file)
                    # Save the midi file
                    midi.write(os.path.join(self._folder_output,new_midi_file_name))
                    # Append the metadata to the csv file (previous_midi_file_name,new_midi_file_name)
                    csv_file.write(f"{midi_file},{new_midi_file_name}\n")
                    

                else:
                    # Log that the midi file was skipped
                    self._logger.info("Midi file skipped: {}".format(midi_file_name))

                

