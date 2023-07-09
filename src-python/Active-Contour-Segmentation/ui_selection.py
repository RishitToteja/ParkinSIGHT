import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
from skimage import io
from matplotlib.widgets import Cursor
import segment
import skimage.data
import skimage.color
from skimage import color
import pandas as pd
import numpy as np
import skimage.filters
import skimage.segmentation
from scipy.interpolate import splprep, splev, BSpline, splrep
import scipy.ndimage
import os.path

# Function to open a file dialog to allow the user to select multiple image files
def open_images():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select images to count",
                                             filetypes=(("image files", "*.jpg;*.png"), ("all files", "*.*")))
    return file_paths

# Function to display the image specified by the file path and asks the user to mark contour points on the image
def display_and_count(file_path, it, df):
    # Display the image
    print(file_path)
    pno = ((file_path.split("/")[-1]).split("-")[-1]).split(".")[0]
    print(pno)
    plt.ion()

# The image is loaded using skimage.io.imread()
    image = io.imread(file_path)

# Assign shape of image as a tuple containing the dimensions (height, width, channels)
    s = image.shape

# Check if it's a colored image (more than 2 dimensions and channel size not equal to 1)
    if(len(s)!=2 and s[2] != 1):
        # If yes, it is converted to grayscale
        image = color.rgb2gray(image)

    image2 = image
    plt.imshow(image)
    plt.title("Mark contour points")
    plt.show()

    # Ask user if they want to mark contour points using a message box

    # Use ginput() to mark the points on the image after it is filtered using a Gaussian filter and displayed again
    image2 = skimage.filters.gaussian(image, 6.0)
    plt.clf()
    plt.imshow(image2)

    points = plt.ginput(n=0, timeout=0)
    print("The user marked", len(points), "points on the image.")
    # Segment the image based on the marked points and return the control points
    cpoints = segment.segment1(image, image2, points, it)
    plt.axvline(x=306, color='b', label='axvline - full height')
    plt.axvline(x=300, color='r', label='axvline - full height')
    # Control points are stored in the DataFrame df along with the patient number and iteration number
    df.loc[it] = [pno]+cpoints
    it+=1

    plt.waitforbuttonpress()
    plt.clf()


# Main function to run the program
if __name__ == "__main__":
    # Get the selected file's paths
    file_paths = open_images()
    it = 1
    print(file_paths)
    df = pd.DataFrame(columns=["Patient_No", "Point1", "Point2", "Point3", "Point4", "Point5", "Point6", "Point7", "Point8", "Point9", "Point10", "Point11", "Point12"])
    for file_path in file_paths:
        display_and_count(file_path, it, df)
        plt.close(it)
        it += 1
    print(df)
    df.to_csv("featureMatrix.csv")
