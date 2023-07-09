import numpy as np
import scipy.linalg
import scipy.ndimage
import skimage
import skimage.filters
import scipy.interpolate
import matplotlib.pyplot as plt

# alpha
# Snake length shape parameter. Higher values makes snake contract faster.

# beta
# Snake smoothness shape parameter. Higher values makes snake smoother.

# w_line
# Controls attraction to brightness. Use negative values to attract toward dark regions.

# w_edge
# Controls attraction to edges. Use negative values to repel snake from edges.

# gamma
# Explicit time stepping parameter.

# Function to implements the Kass snake algorithm for image segmentation
def kassSnake(image, initialContour, edgeImage=None, alpha=0.01, beta=0.1, wLine=0, wEdge=1, gamma=0.01,
              maxPixelMove=None, maxIterations=2500, convergence=0.1):
    maxIterations = int(maxIterations)
    if maxIterations <= 0:
        raise ValueError('maxIterations should be greater than 0.')

    # Set number of previous iterations to consider for convergence checking
    convergenceOrder = 10

    # NORMALIZING PIXEL VALUES - (MIN MAX NORMALIZATION)

    # Convert the input image to floating-point representation
    image = skimage.img_as_float(image)
    # Check if the image has multiple channels (colored image)
    isMultiChannel = (image.ndim == 3)

    # Compute edgeImage from Sobel filter if it's not provided but wEdge (attraction to edges) is non-zero
    if edgeImage is None and wEdge != 0:
        result = scipy.ndimage.sobel(image)
      
        # Normalizes the edge image using min-max normalization
        edgeImage = np.sqrt(scipy.ndimage.sobel(image, axis=0, mode='reflect') ** 2 + scipy.ndimage.sobel(image, axis=1, mode='reflect') ** 2)
        edgeImage = (edgeImage - edgeImage.min()) / (edgeImage.max() - edgeImage.min())

    # If edgeImage is not provided and wEdge is zero, edgeImage is set to zero
    elif edgeImage is None:
        edgeImage = 0

    # If the image is colored, the external energy is calculated as a combination of line (wLine) and edge attraction (wEdge) on each channel
    if isMultiChannel:
        externalEnergy = wLine * np.sum(image, axis=2) + wEdge * np.sum(edgeImage, axis=2)

    # Otherwise, the external energy is calculated directly.
    else:
        # wLine, wEdge -> Hyperparameters

        externalEnergy = (wLine * image) + (wEdge * edgeImage)

        # Temporary variables that hold the indices for energy interpolation
        temp1 = np.arange(externalEnergy.shape[1])
        temp2 = np.arange(externalEnergy.shape[0])

    # Create a RectBivariateSpline object for interpolating the external energy values based on the indices and energy values
    # kx and ky -> Degree of spline interpolation
    # s -> Smoothing factor
    externalEnergyInterpolation = scipy.interpolate.RectBivariateSpline(np.arange(externalEnergy.shape[1]),
                                                                        np.arange(externalEnergy.shape[0]),
                                                                        externalEnergy.T, kx=2, ky=2, s=0)

    x, y = initialContour[:, 0].astype(float), initialContour[:, 1].astype(float)

    # Create empty arrays previousX and previousY to store the previous contour point positions for convergence checking
    previousX = np.empty((convergenceOrder, len(x)))
    previousY = np.empty((convergenceOrder, len(y)))

    # Define variables for matrix operations of the snake algorithm
    n = len(x)
    r = 2 * alpha + 6 * beta
    q = -alpha - 4 * beta
    p = beta

    # Construct the coefficient matrix A used in the linear equation of the snake algorithm
    A = r * np.eye(n) + \
        q * (np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1)) + \
        p * (np.roll(np.eye(n), -2, axis=0) + np.roll(np.eye(n), -2, axis=1))

    # Adding a regularization term to the inverse of the coefficient matrix A
    AInv = scipy.linalg.inv(A + gamma * np.eye(n))

    # Calculate the partial derivatives (fx and fy) of the external energy using the interpolation function
    for i in range(maxIterations):

        fx = externalEnergyInterpolation(x, y, dx=1, grid=False)
        fy = externalEnergyInterpolation(x, y, dy=1, grid=False)

        # Update the xNew and yNew arrays based on the matrix operations of the snake algorithm
        xNew = np.dot(AInv, gamma * x + fx)

        yNew = np.dot(AInv, gamma * y + fy)

        # If maxPixelMove is provided, constrain the movement of contour points within a specified pixel distance
        if maxPixelMove:
            dx = maxPixelMove * np.tanh(xNew - x)
            dy = maxPixelMove * np.tanh(yNew - y)

            x += dx
            y += dy

        # If not, it directly assigns the updated xNew and yNew values to x and y
        else:
            x = xNew
            y = yNew

        # Convergence checking by comparing the distance between the current contour points and the previous contour points stored in previousX and previousY
        j = i % (convergenceOrder + 1)

        if j < convergenceOrder:
            previousX[j, :] = x
            previousY[j, :] = y
        else:
            distance = np.min(np.max(np.abs(previousX - x[None, :]) + np.abs(previousY - y[None, :]), axis=1))

            if distance < convergence:
                break

    print('Finished at', i)

    # Returns the segmented contour points
    return np.array([x, y]).T
