import cv2
import numpy as np

def create_dummy_video(filename, frames=16, height=224, width=224, fps=10):
    """Creates a dummy video file with random frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {filename}")
        return

    for _ in range(frames):
        # Generate a random BGR image (OpenCV expects BGR)
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)

    out.release()