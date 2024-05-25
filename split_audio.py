import shutil
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
from pydub import AudioSegment
from pydub.silence import split_on_silence


def convert_video_to_audio(video_path: Path, output_dir: Path):
    output_path = output_dir / f"{video_path.stem}.wav"
    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ar",
        "44100",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode == 0:
        print(f"Successfully converted \n\t {video_path} \n\t to {output_path}")
    else:
        print(f"Error converting \n\t {video_path}: {result.stderr}")


def batch_convert_videos(video_dir: Path, output_dir: Path, num_processes: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = list(video_dir.rglob("*.mp4"))
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            convert_video_to_audio,
            [(video_path, output_dir) for video_path in video_paths],
        )


@click.command()
@click.argument("video_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--num_processes", type=int, default=1)
def mp4_to_wav(video_dir, output_dir, num_processes):
    """Convert all .mp4 videos in VIDEO_DIR to .wav audio files in OUTPUT_DIR."""
    video_dir, output_dir = Path(video_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_convert_videos(video_dir, output_dir, num_processes)


@click.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--min_duration", default=1.0, type=float, help="Minimum audio duration in seconds"
)
@click.option(
    "--max_duration", default=30.0, type=float, help="Maximum audio duration in seconds"
)
@click.option("--silence_thresh", default=-36, type=int, help="Silence threshold")
@click.option(
    "--min_silence_dur", default=0.3, type=float, help="Minimum silence duration (s)"
)
@click.option(
    "--keep_silence", default=0.5, type=float, help="silence padding duration (s)"
)
@click.option("--seek_step", default=0.02, type=float, help="seek step (s)")
def slice_audio(
    source_dir,
    target_dir,
    min_duration: float,
    max_duration: float,
    silence_thresh: int,
    min_silence_dur: int,
    keep_silence: float,
    seek_step: float,
):
    """Split all audios in SOURCE_DIR into slices and save the slices to TARGET_DIR."""
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    min_dur_ms = min_duration * 1000
    max_dur_ms = max_duration * 1000

    AUDIO_SUFFIX = set([".wav", ".flac", ".mp3"])

    for file in source_dir.iterdir():
        if file.suffix in AUDIO_SUFFIX:
            if "_seg_" in str(file):
                try:
                    seg: AudioSegment = AudioSegment.from_file(file)
                except Exception as e:
                    print("Error: {e}")
                    continue
                base_dir = file.parents[0]
                base_name = file.stem.replace("_seg_", "_")
                seg.export(str(base_dir / (base_name + ".wav")), format="wav")

    tot_dur = 0

    for file in source_dir.iterdir():

        if "_seg_" in str(file):
            continue

        if file.suffix in AUDIO_SUFFIX:
            try:
                audio = AudioSegment.from_file(file)
                if source_dir == target_dir:
                    file.unlink()
            except Exception as e:
                print("Error: {e}")
                continue

            if len(audio) == 0:
                print(f"{file} is corrupted! skipped!")

            segments: list[AudioSegment] = split_on_silence(
                audio,
                min_silence_len=int(min_silence_dur * 1000),
                silence_thresh=int(silence_thresh),
                keep_silence=int(keep_silence * 1000),
                seek_step=int(seek_step * 1000),
            )

            for i, seg in enumerate(segments):
                seg_dur = len(seg)
                if min_dur_ms <= seg_dur <= max_dur_ms:
                    out_file = target_dir / f"{file.stem}_seg_{i}.wav"
                    seg.export(str(out_file), format="wav")
                    tot_dur += seg_dur
                else:
                    print(f"Skipped due to duration: {seg_dur / 1000:.2} s")

    tot_dur /= 1000
    hr, minute, sec = (
        int(tot_dur / 3600),
        (int(tot_dur) % 3600) // 60,
        int(tot_dur) % 60,
    )
    print(f"Got split audios: {hr} hrs:{minute} mins:{sec} secs")

    file_count = len([f for f in target_dir.iterdir() if f.suffix == ".wav"])
    print(f"Split audios from {str(source_dir)}: Done! Generated {file_count} Slices.")


def denoise_audio(audio_path: Path, output_dir: Path):
    output_path = output_dir / f"{audio_path.stem}_denoised.wav"
    command = [
        "demucs",
        "--two-stems=vocals",
        str(audio_path),
        "--out",
        str(output_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        sepd_path = str(output_dir / "htdemucs" / audio_path.stem / "vocals.wav")
        shutil.move(sepd_path, output_path)
        print(f"Successfully denoised \n\t{audio_path} \n\tto {output_path}")
    else:
        print(f"Error denoising {audio_path}: {result.stderr}")


def batch_denoise_audios(audio_dir: Path, output_dir: Path, num_processes: int):
    audio_paths = [
        audio for audio in audio_dir.rglob("*.wav") if "_denoised" not in str(audio)
    ]
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            denoise_audio, [(audio_path, output_dir) for audio_path in audio_paths]
        )
    shutil.rmtree(str(output_dir / "htdemucs"))


@click.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--num_processes",
    default=1,
    type=int,
    help="Number of process for denoising audios",
)
def denoise(source_dir, target_dir, num_processes):
    """Denoise all .wav audios in SOURCE_DIR and save to TARGET_DIR."""
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    batch_denoise_audios(source_dir, target_dir, num_processes)


@click.group()
def cli():
    pass


cli.add_command(slice_audio)
cli.add_command(mp4_to_wav)
cli.add_command(denoise)

if __name__ == "__main__":
    cli()
