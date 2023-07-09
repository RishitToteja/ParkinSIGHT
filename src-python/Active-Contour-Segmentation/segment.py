import math
import matplotlib.pyplot as plt
import skimage.data
import skimage.color
import numpy as np
import skimage.filters
import skimage.segmentation
from skimage import io
from scipy.interpolate import splprep, splev, BSpline, splrep
import scipy.ndimage
import helper
from math import sqrt

# Function to calculate the centroid and standard deviation of the x and y values from a list of coordinates as input
def computeCentroidAndStdDev(coords):
    x_values = [x for x, y in coords]
    y_values = [y for x, y in coords]
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    x_std = np.std(x_values)
    y_std = np.std(y_values)
    centroid = (x_mean, y_mean)
    return [(x_std, y_std), centroid]

# Function for image segmentation using the Kass snake algorithm on two contours defined by the given points
def segment1(image, image2, points, it):

    # Append the first point to the end to close the contour
    points.append(points[0])

    # Convert the points to x and y arrays
    x, y = zip(*points)
    # Create the initial contour
    init1 = np.array([x, y]).T

    # Use spline interpolation to create a smooth curve from the points
    tck, u = splprep([x, y], s=0)
    startPointsLeft = np.linspace(0, 1, 400)
    curved_init = splev(startPointsLeft, tck)
    init1 = np.array(curved_init).T
    a = init1
    # Use the Kass snake algorithm to segment the image with given parameters
    snakeContour1 = helper.kassSnake(image2, init1, wLine=0, wEdge=1.0, alpha=0.1, beta=0.1, gamma=0.001,
                                     maxIterations=5, maxPixelMove=None, convergence=0.1)

    # Storing Control points in the List
    contour_pointsLeft = []
    for i in range(1, 7):
        contour_pointsLeft.append((snakeContour1[66 * (i)][0], snakeContour1[66 * (i)][1]))

    # Repeat the process for a second contour
    points = plt.ginput(n=0, timeout=0)
    points.append(points[0])
    x, y = zip(*points)
    init2 = np.array([x, y]).T
    tck, u = splprep([x, y], s=0)
    startPointsRight = np.linspace(0, 1, 400)
    curved_init = splev(startPointsRight, tck)
    init2 = np.array(curved_init).T


    snakeContour2 = helper.kassSnake(image2, init2, wLine=0, wEdge=1.0, alpha=0.1, beta=0.1, gamma=0.001,
                                     maxIterations=5, maxPixelMove=None, convergence=0.1)

    contour_pointsRight = []

    for i in range(1, 7):
        contour_pointsRight.append((snakeContour2[66 * (i)][0], snakeContour2[66 * (i)][1]))

    # Display the original image and the segmented contours
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(init1[:, 0], init1[:, 1], '--r', lw=2)
    plt.plot(snakeContour1[:, 0], snakeContour1[:, 1], '-b', lw=2)
    plt.plot(init2[:, 0], init2[:, 1], '--r', lw=2)
    plt.plot(snakeContour2[:, 0], snakeContour2[:, 1], '-b', lw=2)

    c = np.random.randint(50, 51, 12)
    plt.scatter(*zip(*(contour_pointsLeft+contour_pointsRight)),c=c,marker='X', cmap='summer')
    plt.show()

    # Computing centroid and standard deviation

    statsLeft = computeCentroidAndStdDev(contour_pointsLeft)
    stdDevLeft = statsLeft[0]
    centroidLeft = statsLeft[1]

    statsRight = computeCentroidAndStdDev(contour_pointsRight)
    stdDevRight = statsRight[0]
    centroidRight = statsRight[1]

    plt.axis('off')
    plt.savefig("train_dir/{}.png".format(it))
    plt.axis('on')

# Return a concatenated list of the control points from both contours
    return (contour_pointsRight+contour_pointsLeft);