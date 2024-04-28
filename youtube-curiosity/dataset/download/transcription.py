import subprocess
import time
import torch
import wave
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pprint import pprint
import os

class Segment:
    def __init__(self, start: float, end: float, text: str, **kwargs):
        self.start = start
        self.end = end
        self.text = text.strip()

    def __str__(self):
        return f"{self.start}\\{self.end}\\{self.text}\n"
    
    @staticmethod
    def from_string(s):
        start, end, text = s.strip().split("\\")
        return Segment(float(start), float(end), text)


def initialize_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v2"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

def download_video(video_id, data_dir_path, verbose=False):
    start_time = time.time()

    try:
        # download video mp4 and audio wav
        p_audio = subprocess.Popen([
            "yt-dlp",
            "-q",
            "-o", f"{data_dir_path}videos/{video_id}/out.wav",
            "-f", "mp4",
            "--force-overwrites",
            "-k",
            "-x", "--audio-format", "wav",
            video_id
        ])

        p_audio.wait(timeout=180)
    except subprocess.TimeoutExpired:
        print(f"{video_id} | failed to download audio")
        if os.path.exists(f"{data_dir_path}videos/{video_id}/out.wav"):
            os.remove(f"{data_dir_path}videos/{video_id}/out.wav")

        return

    # calculate download rate
    download_delta = time.time() - start_time

    if verbose:
        print(f"{video_id} | downloaded video files")

        with wave.open(f"{data_dir_path}videos/{video_id}/out.wav", "r") as wav:
            video_length = wav.getnframes() / wav.getframerate()
            print(f"Downloaded {video_length} seconds of video in {download_delta} seconds [Rate: {video_length / download_delta} sec/s]")


def get_video_transcript(whisper, video_id, data_dir_path, save_to_file=False, verbose=False):
    # transcribe with whisper
    wav_path = f"{data_dir_path}videos/{video_id}/out.wav"
    if not os.path.exists(wav_path):
        return None

    start_time = time.time()
    whisper_response = whisper(wav_path, return_timestamps=True)

    transcription_delta = time.time() - start_time

    if verbose:
        print(f"{video_id} | whisper transcription finished: {whisper_response['text'][:100]}")
        
        with wave.open(wav_path, "r") as wav:
            video_length = wav.getnframes() / wav.getframerate()
            print(f"Transcribed {video_length} seconds of audio in {transcription_delta} seconds [Rate: {video_length / transcription_delta} sec/s]")


    # if whisper failed to generate an ending timestamp, use the video length
    if whisper_response["chunks"][-1]["timestamp"][1] is None:
        with wave.open(wav_path, "r") as wav:
            whisper_response["chunks"][-1]["timestamp"] = (
                whisper_response["chunks"][-1]["timestamp"][0],
                wav.getnframes() / wav.getframerate(),
            )

    whisper_transcript = [
        Segment(
            text=seg["text"],
            start=seg["timestamp"][0],
            end=seg["timestamp"][1]
        ) for seg in whisper_response["chunks"]
    ]

    if save_to_file:
        with open(f"{data_dir_path}videos/{video_id}/transcript.txt", "w") as f:
            for seg in whisper_transcript:
                f.write(str(seg))

    return whisper_transcript


if __name__ == "__main__":
    VIDEO_ID = "5cm25thVkWg"

    whisper = initialize_whisper()

    download_video(VIDEO_ID, data_dir_path="/grogu/user/mhzhou/youtube-curiosity/dataset")
    transcript = get_video_transcript(whisper, VIDEO_ID, data_dir_path="/grogu/user/mhzhou/youtube-curiosity/dataset", save_to_file=True)