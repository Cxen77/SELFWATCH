
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pipeline import SelfWatchPipeline
from tools.config_manager import load_config
from tools.video_reader import VideoReader

def run():
    config = load_config('config/default.yaml')
    config['cognitive']['forensic_layer'] = True
    config['cognitive']['for_failure'] = True

    video_path = 'data/test_video.mp4'
    if not os.path.exists(video_path):
        print('No video found')
        return

    reader = VideoReader(video_path)
    pipeline = SelfWatchPipeline(config, reader)
    
    print('Starting headless run...')
    for frame_count, frame in enumerate(reader):
        if frame_count > 385:
            break
        if frame_count < 360:
            continue
            
        pipeline.process_frame(frame)
        
    print('Finished headless run')

if __name__ == '__main__':
    run()
