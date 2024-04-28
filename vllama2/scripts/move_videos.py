"""
Move videos files from <video_id>/out.mp4 to <video_id>.mp4
"""

import os
import shutil

def move_videos ():
    video_dir = '/grogu/user/mhzhou/youtube-curiosity/videos/'
    for video_id in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_id, 'out.mp4')
        new_video_path = os.path.join(video_dir, f'{video_id}.mp4')
        
        # downloading failed
        if not os.path.exists(video_path):
            return None
        
        shutil.move(video_path, new_video_path)

if __name__ == '__main__':
    move_videos()