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

        # print(notes_sorted)

        pitch_tensors = []
        velocity_tensors = []
        duration_tensors = []
        step_tensors = []

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
        pass


example_data = "../data/processed/0bdd24c1-b87b-4abe-843a-4c9718a12a92.mid"
pretty_mid = pretty_midi.PrettyMIDI(example_data)
print(PrettyMIDITensor().pretty_midi_to_tensor(pretty_mid, example_data)[0])
