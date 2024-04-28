import os
import torch
import cv2
import clip
from PIL import Image
import numpy as np

from dataset.download.transcription import Segment

device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_clip():
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    return model, preprocess


def load_transcript(video_id, data_dir_path):
    segments = []

    with open(f"{data_dir_path}videos/{video_id}/transcript.txt", "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                break

            segments.append(Segment.from_string(line))

    return segments

def get_random_frame(video_id, data_dir_path):
    mp4_path = f"{data_dir_path}videos/{video_id}/out.mp4"
    assert os.path.exists(mp4_path)
    cap = cv2.VideoCapture(mp4_path)

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        np.random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 1000 / cap.get(cv2.CAP_PROP_FPS)
    )
    success, img = cap.read()

    return img if success else None

def get_video_frames(clip_model, clip_preprocess, video_id, transcript, data_dir_path, save_to_file=False):
    frames = []

    mp4_path = f"videos/{video_id}/out.mp4"
    if not os.path.exists(mp4_path):
        return 0
    
    cap = cv2.VideoCapture(mp4_path)

    if save_to_file:
        # create frames directory
        os.makedirs(f"{data_dir_path}videos/{video_id}/frames", exist_ok=True)

    # get first frame
    def first_frame(segment):
        cap.set(cv2.CAP_PROP_POS_MSEC, segment.start * 1000)
        success, img = cap.read()

        return img if success else None
    
    # get last frame
    def last_frame(segment):
        cap.set(cv2.CAP_PROP_POS_MSEC, segment.end * 1000)
        success, img = cap.read()

        return img if success else None
    
    def middle_frame(segment):
        cap.set(cv2.CAP_PROP_POS_MSEC, ((segment.start + segment.end) / 2) * 1000)
        success, img = cap.read()

        return img if success else None

    # get frame with highest CLIP score
    def clip_frame(segment):
        templates = [
            'a photo of {}.',
            'a photo of a person {}.',
            'a photo of a person using {}.',
            'a photo of a person doing {}.',
            'a photo of a person during {}.',
            'a photo of a person performing {}.',
            'a photo of a person practicing {}.',
            'a video of {}.',
            'a video of a person {}.',
            'a video of a person using {}.',
            'a video of a person doing {}.',
            'a video of a person during {}.',
            'a video of a person performing {}.',
            'a video of a person practicing {}.',
            'a example of {}.',
            'a example of a person {}.',
            'a example of a person using {}.',
            'a example of a person doing {}.',
            'a example of a person during {}.',
            'a example of a person performing {}.',
            'a example of a person practicing {}.',
            'a demonstration of {}.',
            'a demonstration of a person {}.',
            'a demonstration of a person using {}.',
            'a demonstration of a person doing {}.',
            'a demonstration of a person during {}.',
            'a demonstration of a person performing {}.',
            'a demonstration of a person practicing {}.',
        ]

        best_frame = None

        cap.set(cv2.CAP_PROP_POS_MSEC, segment.start * 1000)

        candidate_frames = []

        prompts = [temp.replace('{}', segment.text) for temp in templates]
        text = clip.tokenize(prompts).to(device)

        while cap.get(cv2.CAP_PROP_POS_MSEC) <= segment.end * 1000:
            success, img = cap.read()

            if success:
                image = clip_preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
                candidate_frames.append(image.squeeze(0))
            else:
                print(f"Unable to load frame at timestamp {cap.get(cv2.CAP_PROP_POS_MSEC)}")

            # increment by 500ms
            curr_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, curr_pos + 500)

            # read to sync up the timestamp. this is a hacky solution
            success, _ = cap.read()
            if not success:     # reached end of video
                break
    
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
            text_features /= text_features.norm(dim=1, keepdim=True)
            average_text_features = text_features.mean(dim=0).unsqueeze(0)

            candidate_frames_features = clip_model.encode_image(torch.stack(candidate_frames))
            candidate_frames_features /= candidate_frames_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * candidate_frames_features @ average_text_features.t()
            logits_per_text = logits_per_image.t()

            probs = logits_per_text.softmax(dim=-1)

            print(probs)

            best_score = torch.max(probs).item()
            best_frame_index = torch.argmax(probs).item()

        # # only return a CLIP-selected frame if the score is above a threshold
        # if best_score < 2 / probs.shape[1]: # threshold is twice the average
        #     print("No frame above threshold")
        #     return None

        cap.set(cv2.CAP_PROP_POS_MSEC, segment.start * 1000 + best_frame_index * 500)
        success, best_frame = cap.read()

        return best_frame

    for i, segment in enumerate(transcript):
        print(f"[BEGIN SEGMENT {i}]: {segment.text}")
        segment_first_frame = first_frame(segment)
        segment_last_frame = last_frame(segment)
        segment_middle_frame = middle_frame(segment)
        segment_clip_frame = clip_frame(segment)

        frames.append({
            'first': segment_first_frame,
            'last': segment_last_frame,
            'middle': segment_middle_frame,
            'clip': segment_clip_frame,
        })

        if segment_first_frame is None:
            print("Unable to load first frame")
        if segment_last_frame is None:
            print("Unable to load last frame")
        if segment_middle_frame is None:
            print("Unable to load middle frame")
        if segment_clip_frame is None:
            print("Unable to load clip frame")

        if save_to_file:
            if segment_first_frame is not None:
                cv2.imwrite(f"{data_dir_path}videos/{video_id}/frames/{i}-first.png", segment_first_frame)
            if segment_last_frame is not None:
                cv2.imwrite(f"{data_dir_path}videos/{video_id}/frames/{i}-last.png", segment_last_frame)
            if segment_middle_frame is not None:
                cv2.imwrite(f"{data_dir_path}videos/{video_id}/frames/{i}-middle.png", segment_middle_frame)
            if segment_clip_frame is not None:
                cv2.imwrite(f"{data_dir_path}videos/{video_id}/frames/{i}-clip.png", segment_clip_frame)

    return len(frames)

def get_video_frames_no_clip(video_id, transcript, data_dir_path, save_to_file=False, verbose=False):
    frames = []

    mp4_path = f"{data_dir_path}videos/{video_id}/out.mp4"
    if not os.path.exists(mp4_path):
        return 0
    
    cap = cv2.VideoCapture(mp4_path)

    if save_to_file:
        # create frames directory
        os.makedirs(f"{data_dir_path}videos/{video_id}/frames", exist_ok=True)
    
    def middle_frame(segment):
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, ((segment.start + segment.end) / 2) * 1000)
            success, img = cap.read()
            return img if success else None
        except:
            return None

    for i, segment in enumerate(transcript):
        if verbose: print(f"[BEGIN SEGMENT {i}]: {segment.text}")
        segment_middle_frame = middle_frame(segment)

        frames.append({
            'middle': segment_middle_frame,
        })

        if segment_middle_frame is None:
            if verbose: print("Unable to load middle frame")

        if save_to_file:
            if segment_middle_frame is not None:
                cv2.imwrite(f"{data_dir_path}videos/{video_id}/frames/{i}-middle.png", segment_middle_frame)

    return len(frames)


if __name__ == "__main__":
    clip_model, clip_preprocess = initialize_clip()

    transcript = load_transcript("5cm25thVkWg", data_dir_path="/grogu/user/mhzhou/youtube-curiosity/dataset")

    get_video_frames(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        transcript=transcript,
        video_id="5cm25thVkWg",
        save_to_file=True,
        data_dir_path="/grogu/user/mhzhou/youtube-curiosity/dataset"
    )
