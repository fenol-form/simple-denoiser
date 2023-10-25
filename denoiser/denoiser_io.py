from typing import Tuple
import torch
import soundfile as sf
import sys


def read_audiofile(filename: str) -> Tuple[torch.Tensor, int]:
    try:
        # read signal
        data, sr = sf.read(filename, always_2d=True)
        data = torch.tensor(data)

        # if original signal is not stereo
        if data.shape[1] == 1:
            data = data.repeat((1, 2))
        return data, sr
    except sf.LibsndfileError as err:
        sys.stderr.write("There is no such file or file has unsupported format.\n"
                         "See libsndfile's supported formats.")
        raise err


def save_audiofile(filename: str, audio: torch.Tensor, samplerate: int):
    sf.write(filename, audio, samplerate)