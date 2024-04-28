import time
import wave
import os

from dataset.download.transcription import download_video, get_video_transcript, initialize_whisper
from dataset.download.frames import get_video_frames, get_video_frames_no_clip, initialize_clip

# VIDEO_IDS = ['3fiyPrwliv0', '5cm25thVkWg', 'L7NePx9HLG8', 'Z61YcGTJnkY', 'tqFnBcKUdhA', 'IhQC8aOaSuA', '3I1Qap3TbSA', 'rcrgL_RuBKk', 'xcrvQc8r_b4', 'TamEeDdWlmo', 'wVgeynndwGs', '8UsOZZOB4S8', 'QB2mMqQvy7w', 'SAO3OqkIgeM', 'gwJ1BKkZr2U', 'j9ekdg-yF2U', 'n6FOaYbW_5k', '5NSSsKfU4rA', 'nkolvfvYjvE', 'QiYSufsfv_8', 'ju9cASXCE7k', 'RE-2synwBZY', 'DbQH4idwSso', 'gxXMrxDdSus', 'gPbE3KgOEKI', 'Rsw-ErFM2lc', 'u9gR2ypOj-I', 'Z9CWzAEXvQA', 'xLwrRPejFlU', '-1vgx9gJsG4', '5QZVOA7NI98', 'CFedH9gkA4o', 'sLAhqvP1UdA', '2jS3Grb5zQM', '_jb3nT5dMS8', 'JooJu5VIcbM', 'T8ENqS6b51I', '6xftOfsaS5M', 'vvCt5wOUIy4', '-RMUSm28n08', 'b7iuuKtm-uc', '8PALDaYSGqE', 'IW3Rl_zJ9P4', 'H-S-rd_Nd1k', '-qMzc9GNLXs', 'QNtR-X41qh0', '49eH1k5cD68', 'tO-egn5vez8', 'qm7O8Fxgm-w', 'mtF_jtl13oA']

whisper = initialize_whisper()
clip_model, clip_preprocess = initialize_clip()

def prepare_video(video_id, data_dir_path):
    """
    Download video, get transcript, and extract frames from video.
    Returns the number of frames extracted.
    """
    download_video(video_id)

    print(f"{video_id} | starting")
    start_time = time.time()

    download_video(video_id, data_dir_path=data_dir_path)

    transcript = get_video_transcript(
        whisper=whisper,
        video_id=video_id,
        data_dir_path=data_dir_path,
        save_to_file=True,
    )
    n_frames = get_video_frames(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        video_id=video_id,
        transcript=transcript,
        data_dir_path=data_dir_path,
        save_to_file=True,
    )

    end_time = time.time()
    
    if not os.path.exists(f"{data_dir_path}/videos/{video_id}/out.wav"):
        print(f"{video_id} | failed to download audio")
        return 0

    with wave.open(f"{data_dir_path}videos/{video_id}/out.wav", "r") as wav:
        video_length = wav.getnframes() / wav.getframerate()
        rate = video_length / (end_time - start_time)
        print(f"{video_id} | finished in {end_time - start_time} seconds [Rate: {rate} sec/s]")
    
    print(f"[{video_id}] complete")

    return n_frames

def prepare_video_no_clip(video_id, data_dir_path):
    """
    Download video, get transcript, and extract frames from video.
    Returns the number of frames extracted.
    """
    download_video(video_id)

    print(f"{video_id} | starting")
    start_time = time.time()

    download_video(video_id, data_dir_path=data_dir_path)

    transcript = get_video_transcript(
        whisper=whisper,
        video_id=video_id,
        data_dir_path=data_dir_path,
        save_to_file=True,
    )
    n_frames = get_video_frames_no_clip(
        video_id=video_id,
        transcript=transcript,
        data_dir_path=data_dir_path,
        save_to_file=True,
    )

    end_time = time.time()
    
    if not os.path.exists(f"{data_dir_path}videos/{video_id}/out.wav"):
        print(f"{video_id} | failed to download audio")
        return 0

    with wave.open(f"{data_dir_path}videos/{video_id}/out.wav", "r") as wav:
        video_length = wav.getnframes() / wav.getframerate()
        rate = video_length / (end_time - start_time)
        print(f"{video_id} | finished in {end_time - start_time} seconds [Rate: {rate} sec/s]")
    
    print(f"[{video_id}] complete")

    return n_frames
