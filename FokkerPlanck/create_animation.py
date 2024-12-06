import os
import imageio # pip install imageio[ffmpeg]

# Define the path to your PNG files
folder = 'fokker-planck-sim-result'  # Replace with your folder path

# Get a list of all PNG files in the folder
png_files = [f for f in os.listdir(folder) if f.endswith('.png')]

# Sort the files if needed (optional)
png_files.sort()

# Create a list to store the images
images = []

# Load each image and append it to the images list
for file in png_files:
    img_path = os.path.join(folder, file)
    img = imageio.imread(img_path)
    images.append(img)

# Save as MP4 (make sure ffmpeg is available)
imageio.mimsave('animation.mp4', images, fps=5)  # Adjust the FPS as needed
