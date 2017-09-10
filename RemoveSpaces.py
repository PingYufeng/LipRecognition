import re
import os
AUDIO_PATH = "/Users/sameer/Desktop/Python/FaceRecognition/trainVideos/"
for filename in os.listdir(AUDIO_PATH):
    os.rename(os.path.join(AUDIO_PATH, filename), os.path.join(AUDIO_PATH, re.sub(r'\W+', '', filename.split('.')[0]) + '.mp4'))
