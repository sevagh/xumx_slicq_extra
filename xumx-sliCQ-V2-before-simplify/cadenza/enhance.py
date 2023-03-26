""" Run the dummy enhancement. """
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.io import wavfile
from torchaudio.transforms import Fade
from xumx_slicq_v2.separator import Separator

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
import clarity.utils
from clarity.utils.signal_processing import denormalize_signals, normalize_signal
from .evaluate import make_song_listener_list

# pylint: disable=too-many-locals
# pylint: disable=import-error


logger = logging.getLogger(__name__)


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor | np.ndarray,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    device: torch.device | str = None,
):
    """
    Apply model to a given mixture.
    Use fade, and add segments together in order to add model segment by segment.

    Args:
        model (torch.nn.Module): model to use for separation
        mix (torch.Tensor): mixture to separate, shape (batch, channels, time)
        sample_rate (int): sampling rate of the mixture
        segment (float): segment length in seconds
        overlap (float): overlap between segments, between 0 and 1
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.

    Returns:
        torch.Tensor: estimated sources

    Based on https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html
    """
    device = mix.device if device is None else torch.device(device)
    mix = torch.as_tensor(mix, device=device)

    if mix.ndim == 1:
        # one track and mono audio
        mix = mix.unsqueeze(0)
    elif mix.ndim == 2:
        # one track and stereo audio
        mix = mix.unsqueeze(0)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
            out = out.permute(1, 0, 2, 3)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final.cpu().detach().numpy()


def get_device(device: str) -> tuple:
    """Get the Torch device.

    Args:
        device (str): device type, e.g. "cpu", "gpu0", "gpu1", etc.

    Returns:
        torch.device: torch.device() appropiate to the hardware available.
        str: device type selected, e.g. "cpu", "cuda".
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    if device.startswith("gpu"):
        device_index = int(device.replace("gpu", ""))
        if device_index > torch.cuda.device_count():
            raise ValueError(f"GPU device index {device_index} is not available.")
        return torch.device(f"cuda:{device_index}"), "cuda"

    if device == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unsupported device type: {device}")


def map_to_dict(sources: np.ndarray, sources_list: list[str]) -> dict:
    """Map sources to a dictionary separating audio into left and right channels.

    Args:
       sources (np.ndarray): Signal to be mapped to dictionary.
       sources_list (list): List of strings used to index dictionary.

    Returns:
        Dictionary: A dictionary of separated source audio split into channels.
    """
    audios = dict(zip(sources_list, sources))

    signal_stems = {}
    for source in sources_list:
        audio = audios[source]
        signal_stems[f"left_{source}"] = audio[0]
        signal_stems[f"right_{source}"] = audio[1]

    return signal_stems


# pylint: disable=unused-argument
def decompose_signal(
    model: torch.nn.Module,
    signal: np.ndarray,
    sample_rate: int,
    device: torch.device,
    left_audiogram: np.ndarray,
    right_audiogram: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Decompose signal into 8 stems.

    The left and right audiograms are ignored by the baseline system as it
    is performing personalised decomposition.
    Instead, it performs a standard music decomposition using the
    HDEMUCS model trained on the MUSDB18 dataset.

    Args:
        model (torch.nn.Module): Torch model.
        signal (np.ndarray): Signal to be decomposed.
        sample_rate (int): Sample frequency.
        device (torch.device): Torch device to use for processing.
        left_audiogram (np.ndarray): Left ear audiogram.
        right_audiogram (np.ndarray): Right ear audiogram.

     Returns:
         Dictionary: Indexed by sources with the associated model as values.
    """

    signal, ref = normalize_signal(signal)
    sources = separate_sources(model, signal, sample_rate, device=device)
    # only one element in the batch
    sources = sources[0]
    sources = denormalize_signals(sources, ref)
    signal_stems = map_to_dict(sources, model.sources)
    return signal_stems


def apply_baseline_ha(
    enhancer: NALR,
    compressor: Compressor,
    signal: np.ndarray,
    listener_audiogram: np.ndarray,
    cfs: np.ndarray,
    apply_compressor: bool = False,
) -> np.ndarray:
    """
    Apply NAL-R prescription hearing aid to a signal.

    Args:
        enhancer: A NALR object that enhances the signal.
        compressor: A Compressor object that compresses the signal.
        signal: An ndarray representing the audio signal.
        listener_audiogram: An ndarray representing the listener's audiogram.
        cfs: An ndarray of center frequencies.
        apply_compressor: A boolean indicating whether to include the compressor.

    Returns:
        An ndarray representing the processed signal.
    """
    nalr_fir, _ = enhancer.build(listener_audiogram, cfs)
    proc_signal = enhancer.apply(nalr_fir, signal)
    if apply_compressor:
        proc_signal, _, _ = compressor.process(proc_signal)
    return proc_signal


def process_stems_for_listener(
    stems: dict,
    enhancer: NALR,
    compressor: Compressor,
    audiogram_left: np.ndarray,
    audiogram_right: np.ndarray,
    cfs: np.ndarray,
    apply_compressor: bool = False,
) -> dict:
    """Process the stems from sources.

    Args:
        stems (dict) : Dictionary of stems
        enhancer (NALR) : NAL-R prescription hearing aid
        compressor (Compressor) : Compressor
        audiogram_left (np.ndarray) : Left channel audiogram
        audiogram_right (np.ndarray) : Right channel audiogram
        cfs (np.ndarray) : Center frequencies
        apply_compressor (bool) : Whether to apply the compressor

    Returns:
        processed_sources (dict) : Dictionary of processed stems
    """

    processed_stems = {}

    for stem_str in stems:
        stem_signal = stems[stem_str]

        # Determine the audiogram to use
        audiogram = audiogram_left if stem_str.startswith("l") else audiogram_right

        # Apply NALR prescription to stem_signal
        proc_signal = apply_baseline_ha(
            enhancer, compressor, stem_signal, audiogram, cfs, apply_compressor
        )
        processed_stems[stem_str] = proc_signal
    return processed_stems


@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocal, drums, bass, and other stems.
    Then, the NAL-R prescription procedure is applied to each stem.
    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    Returns 8 stems for each song:
        - left channel vocal, drums, bass, and other stems
        - right channel vocal, drums, bass, and other stems
    """

    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Training stage
    #
    # The baseline is using an off-the-shelf model trained on the MUSDB18 dataset
    # Training listeners and song are not necessary in this case.
    #
    # Training songs and audiograms can be read like this:
    #
    #  with open(config.path.listeners_train_file, "r", encoding="utf-8") as file:
    #        listener_train_audiograms = json.load(file)
    #
    #  with open(config.path.music_train_file, "r", encoding="utf-8") as file:
    #        song_data = json.load(file)
    #  songs_train = pd.DataFrame.from_dict(song_data)
    #
    # train_song_listener_pairs = make_song_listener_list(
    #     songs_train['Track Name'], listener_train_audiograms
    # )

    device, _ = get_device(config.separator.device)
    separation_model = Separator.load(device=device)

    # Processing Validation Set
    # Load listener audiograms and songs
    with open(config.path.listeners_valid_file, encoding="utf-8") as file:
        listener_valid_audiograms = json.load(file)

    with open(config.path.music_valid_file, encoding="utf-8") as file:
        song_data = json.load(file)
    songs_valid = pd.DataFrame.from_dict(song_data)

    valid_song_listener_pairs = make_song_listener_list(
        songs_valid["Track Name"], listener_valid_audiograms
    )
    # Select a batch to process
    valid_song_listener_pairs = valid_song_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    prev_song_name = None
    num_song_list_pair = len(valid_song_listener_pairs)
    for idx, song_listener in enumerate(valid_song_listener_pairs, 1):
        song_name, listener_name = song_listener
        logger.info(
            f"[{idx:03d}/{num_song_list_pair:03d}] "
            f"Processing {song_name} for {listener_name}..."
        )
        # Get the listener's audiogram
        listener_info = listener_valid_audiograms[listener_name]

        # Find the music split directory
        split_directory = (
            "test"
            if songs_valid.loc[songs_valid["Track Name"] == song_name, "Split"].iloc[0]
            == "test"
            else "train"
        )

        critical_frequencies = np.array(listener_info["audiogram_cfs"])
        audiogram_left = np.array(listener_info["audiogram_levels_l"])
        audiogram_right = np.array(listener_info["audiogram_levels_r"])

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        if prev_song_name != song_name:
            # Decompose song only once
            prev_song_name = song_name

            sample_rate, mixture_signal = wavfile.read(
                Path(config.path.music_dir)
                / split_directory
                / song_name
                / "mixture.wav"
            )
            mixture_signal = (mixture_signal / 32768.0).astype(np.float32).T
            assert sample_rate == config.nalr.fs

            stems = decompose_signal(
                separation_model,
                mixture_signal,
                sample_rate,
                device,
                audiogram_left,
                audiogram_right,
            )

        # Baseline applies NALR prescription to each stem instead of using the
        # listener's audiograms in the decomposition. This stem can be skipped
        # if the listener's audiograms are used in the decomposition
        processed_stems = process_stems_for_listener(
            stems,
            enhancer,
            compressor,
            audiogram_left,
            audiogram_right,
            critical_frequencies,
            config.apply_compressor,
        )

        # save processed stems
        n_samples = processed_stems[list(processed_stems.keys())[0]].shape[0]
        out_left, out_right = np.zeros(n_samples), np.zeros(n_samples)
        for stem_str, item in processed_stems.items():
            if stem_str.startswith("l"):
                out_left += item
            else:
                out_right += item

            filename = (
                enhanced_folder
                / f"{listener_info['name']}"
                / f"{song_name}"
                / f"{listener_info['name']}_{song_name}_{stem_str}.wav"
            )
            filename.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(filename, config.nalr.fs, item)

        enhanced = np.stack([out_left, out_right], axis=1)
        filename = (
            enhanced_folder
            / f"{listener_info['name']}"
            / f"{song_name}"
            / f"{listener_info['name']}_{song_name}_remix.wav"
        )

        # Clip and save
        if config.soft_clip:
            enhanced = np.tanh(enhanced)
        n_clipped = np.sum(np.abs(enhanced) > 1.0)
        if n_clipped > 0:
            logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
        np.clip(enhanced, -1.0, 1.0, out=enhanced)
        signal_16 = (32768.0 * enhanced).astype(np.int16)
        wavfile.write(filename, config.nalr.fs, signal_16)


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
