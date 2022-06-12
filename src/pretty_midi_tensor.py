import pretty_midi, logging, datetime, torch


class PrettyMIDITensor:
    def __init__(
        self, logger: logging.Logger = logging.getLogger(name="PrettyMIDITensor")
    ):
        self._logger = logger
        self._logger.info(
            "PrettyMIDITensor initialized at {}".format(datetime.datetime.now())
        )

    def pretty_midi_to_tensor(
        self, pretty_midi: pretty_midi.PrettyMIDI
    ) -> torch.Tensor:
        # Return a tensor with the midi data
        pass
