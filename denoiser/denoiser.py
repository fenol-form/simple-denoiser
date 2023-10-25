import torch
import torchaudio
import soundfile as sf


class Denoiser(torch.nn.Module):
    def __init__(self, *args,
                 threshold=1.,
                 red_rate=1.1,
                 n_fft=1024,
                 window_size=1024,
                 hop_size=256,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.red_rate = red_rate
        self.noise_profiles = None

        # window characteristics
        self.__n_fft = n_fft
        self.__window_size = window_size
        self.__hop_size = hop_size
        self.__window = torch.hann_window(self.__window_size)

    def __get_spectrum(self, signal1d: torch.Tensor):
        spectrum = torch.stft(
            signal1d,
            n_fft=self.__n_fft,
            hop_length=self.__hop_size,
            win_length=self.__window_size,
            window=self.__window,

            center=False,

            onesided=True,

            return_complex=True,
        )

        # return spectrum of a signal
        return spectrum

    def fit(self, noise_sample: torch.Tensor):
        assert noise_sample.shape[1] == 2
        first_chan_sp = self.__get_spectrum(noise_sample[:, 0]).abs().mean(axis=1)
        second_chan_sp = self.__get_spectrum(noise_sample[:, 1]).abs().mean(axis=1)
        self.noise_profiles = torch.vstack([first_chan_sp, second_chan_sp])

    def forward(self, input_audio: torch.Tensor) -> torch.Tensor:
        assert input_audio.shape[1] == 2
        output_audio = torch.zeros_like(input_audio)
        for channel in range(2):
            audio_spectrum = self.__get_spectrum(input_audio[:, channel])

            # create audio spectrogramm
            audio_spectrogramm = audio_spectrum.abs()

            # compute spectrogramm of denoised signal
            clean_spectrogramm = (audio_spectrogramm *
                                  (1. - self.noise_profiles[channel][:, None] * self.threshold / audio_spectrogramm) /
                                  self.red_rate)

            # create spectrum of denoised signal
            clean_spectrum = torch.polar(clean_spectrogramm, audio_spectrum.angle())

            # create denoised audio sample
            output_audio[:, channel] = torch.istft(
                clean_spectrum,
                n_fft=self.__n_fft,
                hop_length=self.__hop_size,
                win_length=self.__window_size,
                window=self.__window,

                center=True,

                onesided=True,

                return_complex=False,

                length=input_audio.shape[0]
            )
        return output_audio

