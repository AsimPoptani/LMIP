import pretty_midi, logging, datetime, torch


class PrettyMIDITensor:
    def __init__(
        self, logger: logging.Logger = logging.getLogger(name="PrettyMIDITensor")
    ):
        self._logger = logger
        self._logger.info(
            "PrettyMIDITensor initialized at {}".format(datetime.datetime.now())
        )

        # Pitch 0-127
        self.pitch_range = range(0, 128)
        # Velocity 0-127
        self.velocity_range = range(0, 128)

    def pretty_midi_to_tensor(
        self, pretty_mid: pretty_midi.PrettyMIDI, filename: str
    ) -> dict:

        # Log that the midi file is being converted
        self._logger.info("Converting midi file to tensor: {}".format(filename))
        # Return a tensor with the midi data
        instrument: pretty_midi.Instrument = pretty_mid.instruments[0]

        notes_sorted = sorted(instrument.notes, key=lambda x: x.start)

        # TODO add control change

        # print(notes_sorted)

        pitch_tensors = []
        velocity_tensors = []
        duration_tensors = []
        step_tensors = []

        note: pretty_midi.Note

        for index, note in enumerate(notes_sorted):
            # Output pitch (0-127), velocity (0-127) , duration (float), step (float)
            # Pitch
            pitch_tensor = torch.tensor(
                self.pitch_range.index(note.pitch), dtype=torch.int
            )
            # Velocity
            velocity_tensor = torch.tensor(
                self.velocity_range.index(note.velocity), dtype=torch.int
            )
            # Duration (float)
            duration_tensor = torch.tensor(note.end - note.start, dtype=torch.float32)
            if not len(notes_sorted) == index + 1:
                # Step (float)
                step_tensor = torch.tensor(
                    notes_sorted[index + 1].start - notes_sorted[index].start,
                    dtype=torch.float32,
                )
            else:
                step_tensor = torch.tensor(0, dtype=torch.float32)

            pitch_tensors.append(pitch_tensor)
            velocity_tensors.append(velocity_tensor)
            duration_tensors.append(duration_tensor)
            step_tensors.append(step_tensor)

        return {
            "pitch_tensors": pitch_tensors,
            "velocity_tensors": velocity_tensors,
            "duration_tensors": duration_tensors,
            "step_tensors": step_tensors,
        }

    def tensor_to_note(
        self,
        pitch_tensor: torch.Tensor,
        velocity_tensor: torch.Tensor,
        duration_tensor: torch.Tensor,
        start_time: float,
    ) -> pretty_midi.Note:
        # Return a note with the pitch, velocity, duration and step
        end = start_time + duration_tensor
        return pretty_midi.Note(
            velocity=self.velocity_range[velocity_tensor],
            pitch=self.pitch_range[pitch_tensor],
            start=start_time,
            end=end,
        )

    def tensor_to_midi(self, tensor: dict) -> pretty_midi.PrettyMIDI:
        # Create a new PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()
        # Create a new Instrument object for a piano instrument
        instrument = pretty_midi.Instrument(program=0)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)

        current_time = 0.0

        # Tensor dict has lists of pitch (0-127), velocity (0-127) , duration (float), step (float)
        """
            "pitch_tensors": pitch_tensors,
            "velocity_tensors": velocity_tensors,
            "duration_tensors": duration_tensors,
            "step_tensors": step_tensors,
        """
        # Extract dictionary
        try:
            pitches = tensor["pitch_tensors"]
            velocities = tensor["velocity_tensors"]
            durations = tensor["duration_tensors"]
            steps = tensor["step_tensors"]
        except Exception as e:
            self._logger.error(
                "One of the parameters passed back is missing please check the model"
            )
            raise e

        notes = []
        for index in range(len(pitches)):
            new_note = self.tensor_to_note(
                pitches[index], velocities[index], durations[index], current_time
            )
            current_time += float(steps[index])
            notes.append(new_note)

        instrument.notes.extend(notes)

        return midi


# import os
# example_data = os.path.join(os.path.__file__,os.path.abspath("./data/processed/0bdd24c1-b87b-4abe-843a-4c9718a12a92.mid"))
# pretty_mid = pretty_midi.PrettyMIDI(example_data)
# pretty_to_tensor=PrettyMIDITensor()
# data=pretty_to_tensor.pretty_midi_to_tensor(pretty_mid, example_data)
# data2=pretty_to_tensor.tensor_to_midi(data)
# print(data2.instruments[0].notes)
# print(sorted(pretty_mid.instruments[0].notes,key= lambda note: note.start))
