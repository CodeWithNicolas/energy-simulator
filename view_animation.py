import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np


def view_animation():
    """Display some sample frames from the animation"""
    try:
        # Open the GIF
        gif = Image.open("mpc_animation.gif")

        print(f"Animation created successfully!")
        print(f"File: mpc_animation.gif")
        print(f"Size: {gif.size}")
        print(f"Number of frames: {gif.n_frames}")
        print(f"Duration per frame: {gif.info.get('duration', 'Unknown')} ms")

        # Display a few key frames
        fig, axes = plt.subplots(2, 2, figsize=(12, 16))
        fig.suptitle("Sample Frames from MPC Animation", fontsize=16)

        frames_to_show = [0, 6, 12, 18]  # Midnight, 6am, noon, 6pm

        for i, frame_idx in enumerate(frames_to_show):
            gif.seek(frame_idx)
            frame = np.array(gif.convert("RGB"))

            row = i // 2
            col = i % 2
            axes[row, col].imshow(frame)
            axes[row, col].set_title(f"Hour {frame_idx}:00")
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error viewing animation: {e}")


if __name__ == "__main__":
    view_animation()
