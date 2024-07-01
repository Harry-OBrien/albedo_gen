import torch
from glob import glob
import os
from tqdm.auto import tqdm

from PIL import Image
import numpy as np

# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, uninvert
from chrislib.data_util import load_image

# import model loading and running the pipeline
from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models

intrinsic_model = load_models('paper_weights')

# # Image list
# EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# rgb_filename_list = glob(os.path.join("C:/Users/Harry/Desktop/input/", "*"))
# rgb_filename_list = [
#     f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
# ]
# rgb_filename_list = sorted(rgb_filename_list)

# img = load_image('C:/Users/Harry/Desktop/source_sequence/Untitled00086617.png')
# torch.hub.download_url_to_file('https://raw.githubusercontent.com/compphoto/Intrinsic/main/figures/avocado.png', 'avo.png')
# img = load_image('./avo.png')

# run the image through the pipeline (use R0 resizing dicussed in the paper)
def run_decomposition(frame, frame_number):
    result = run_pipeline(
        intrinsic_model,
        frame,
        resize_conf=0.0,
        maintain_size=True,
        linear=False,
        device='cuda')

    # convert the inverse shading to regular shading for visualization
    shd = uninvert(result['inv_shading'])
    alb = view(result['albedo']) * 255

    # Convert NumPy array to PIL Image
    image = Image.fromarray(alb.astype('uint8'))

    # Define the path where you want to save the image
    rgb_name_base = "test_video" #os.path.splitext(os.path.basename(rgb_path))[0]
    pred_name_base = f"test_video_{frame_number}_pred"
    npy_save_path = os.path.join("./output", f"{pred_name_base}.png")
    if os.path.exists(npy_save_path):
        logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
    # Save the image
    image.save(npy_save_path)

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    # Create a tqdm progress bar
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            run_decomposition(frame, frame_number)

            frame_number += 1
            pbar.update(1)
    
    # Release the video capture object
    cap.release()
    print("Finished processing video.")

# Example usage
video_path = 'C:/Users/Harry/Downloads/RelightTestKB_2024_0624.mp4'
process_video(video_path)













for rgb_path in tqdm(rgb_filename_list, desc=f"Estimating albedo", leave=True):
    img = load_image(rgb_path)

    
