import matplotlib.pyplot as plt
import numpy as np
import cv2
from point_conversion import PI, inv_PI
import random
import math

def compute_camera_matrix(f, dx, dy, alpha=1, beta=1):
    """ Compute the camera matrix """
    A = np.matrix([[f, f*beta, dx],
                  [0, alpha*f, dy],
                  [0, 0, 1]])
    return A
def test1():
    A = compute_camera_matrix(2774.5, 806.8, 622.6, 1, 0)
    print(A)

def compute_projection_matrix(f, dx, dy, alpha, beta, R, t):
    """ Compute the projection matrix """
    A = compute_camera_matrix(f, dx, dy, alpha, beta)
    P = A @ np.hstack((R, t))

    return P
def test2():
    R = cv2.Rodrigues(np.array([0.1, -0.2, -0.1]))[0]
    t = np.array([[0.03], [0.06], [-0.02]])
    f = 350
    dx = 700
    dy = 390
    alpha = 1
    beta = 1
    P = compute_projection_matrix(f, dx, dy, alpha, beta, R, t)
    q = np.array([0.35, 0.17, 1.01, 1]).reshape(4, 1)
    print(PI(P @ q))

def find_SIFT_keypoints_descriptors(im1, im2):
    """ Grayscaling the images im1 & im2, and compute the keypoints and descriptors between them """
    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    return kp1, kp2, des1, des2

def match_SIFT_keypoints(im1, im2):
    kp1, kp2, des1, des2 = find_SIFT_keypoints_descriptors(im1, im2)
    bf = cv2.BFMatcher(crossCheck=True)

    matches = bf.match(des1, des2)
    return matches

def scaleSpaced(im, sigma, n):
    """ Computing the scale space pyramid of an image """
    blurred_imgs = []

    for i in range(n):
        # Compute the Gaussian kernel given the value of sigma
        sigma_i = sigma * 2**i
        kernel_size = int(6 * sigma-i)  # Determine kernel size based on sigma
        kernel = np.empty((kernel_size,kernel_size))

        for x in range(kernel.shape[0]):
            for y in range(kernel.shape[1]):
                kernel[x, y] = (1/(2 * np.pi * sigma_i**2)) * np.exp(-(x**2 + y**2) / (2 * sigma_i**2))

        # Applying the Gaussian kernel to the image 
        blurred_im = cv2.filter2D(im, -1, kernel)
        blurred_imgs.append(blurred_im)

    return blurred_imgs

def differenceOfGaussian(im, sigma, n):
    """ Compute the difference of gaussians for an image given sigma"""
    DoG = []
    im_scales = scaleSpaced(im, sigma, n)  # Obtain the scale space pyramid using the previous function

    for i in range(n - 1):
        dog = im_scales[i] - im_scales[i + 1]
        DoG.append(dog)
    
    return DoG
def test3():
    im = cv2.imread('week8_data/sunflowers.jpg') # Loading the image 
    im = im.astype(float).mean(2)/255 # Converting image to black and white and floating point
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sigma = 2
    n = 4
    #scaleSpace = scaleSpaced(im, sigma, n)
    DoG = differenceOfGaussian(im, sigma, n)

    for d in DoG:
        plt.imshow(d)
        plt.show()

def convert_line_to_homogeneous_coordinates(a, b, c):
    """ Convert a line on the form ax + by = c to homogeneous coordinates """
    return np.array([a, b, -c]).reshape((3,1))

def determine_if_point_on_line(l, p):
    """ Determine if a point is on a line in homogeneous coordinates """
    return l @ p == 0

def determine_intersection_of_lines(l0, l1):
    """ Given to lines in homogeneous coordinates determines their point of intersection """
    return np.cross(l0, l1)

def determine_distance_from_point_to_line(l, p):
    """ Computes the distance from a point to a line (both in homogeneous coordinates)"""
    return abs(l @ p) / np.sqrt(l[0] ** 2 + l[1] ** 2)

def project_points_distortion(f, alpha, beta, dx, dy, R, t, distCoeffs, P):
    """ Project a 3D point to 2D with distortion """
    A_p = np.array([[f, 0, 0],
                    [0, f, 0],
                    [0, 0, 1]])
    
    A_q = np.array([[1, beta, dx],
                    [0, alpha, dy],
                    [0, 0, 1]])
    
    Rt = np.hstack((R, t))

    pd = inv_PI(PI(A_p @ Rt @ P)) # distorted projection coordinates in 2D homogeneous coordinates

    dr = 0 # computing the radial distortion
    r = np.sqrt(pd[0]**2 + pd[1]**2)
    for i, k in enumerate(distCoeffs):
        dr += k * (r ** ((i + 1)*2))
    
    pc = inv_PI(pd[0:2] * (1 + dr)) # corrected projection coordinates in 2D homogeneous coordinates 
    
    q = A_q @ pc # projected point with radial distortion in homogeneous coordinates
    return q
def test4():
    R = np.array([[0.9887, -0.0004, 0.1500],
                  [0.0008, 1.0000, -0.0030],
                  [-0.1500, 0.0031, 0.9887]])
    t = np.array([-2.1811, 0.0399, 0.5072]).reshape((3,1))
    f = 2774.5
    dx = 806.8
    dy = 622.6
    alpha = 1
    beta = 0

    distCoeffs = np.array([-5.1806e-8, 1.4192e-15])

    Q = np.array([-1.3540, 0.5631, 8.8734]).reshape((3,1))

    q = project_points_distortion(f, alpha, beta, dx, dy, R, t, distCoeffs, inv_PI(Q))
    print(q)

def map_points_using_homography(P, H):
    """ Map a set of 2D inhomogeneous points using a homography"""

    P_hom = inv_PI(P)
    return PI(H @ P_hom)

def hest_linear(Q1, Q2):
    """ Estimate a homography given two sets of matching points using the lienar algorithm """
    n = len(Q1[0])

    for i in range(n):
        x1i = Q1[0, i]
        y1i = Q1[1, i]
        x2i = Q2[0, i]
        y2i = Q2[1, i]

        if i == 0:
            B = np.array([[0, -x2i, x2i*y1i, 0, -y2i, y2i*y1i, 0, -1, y1i],
                         [x2i, 0, -x2i*x1i, y2i, 0, -y2i*x1i, 1, 0, -x1i],
                         [-x2i*y1i, x2i*x1i, 0, -y2i*y1i, y2i*x1i, 0, -y1i, x1i, 0]])
        else:
            bi_T = np.array([[0, -x2i, x2i*y1i, 0, -y2i, y2i*y1i, 0, -1, y1i],
                         [x2i, 0, -x2i*x1i, y2i, 0, -y2i*x1i, 1, 0, -x1i],
                         [-x2i*y1i, x2i*x1i, 0, -y2i*y1i, y2i*x1i, 0, -y1i, x1i, 0]])
            
            B = np.vstack((B, bi_T))
        
    _, _, V_T = np.linalg.svd(B,)
    return V_T[-1].reshape((3,3), order='F')

def normalize_points(pts):
    """ Normalize a set of 2D homogeneous points so they have mean 0 and std 1 """

    mean = np.mean(pts, axis=1)
    std = np.std(pts, axis=1)

    T_inv = np.array([[std[0], 0, mean[0]],
                      [0, std[1], mean[1]],
                      [0, 0, 1]])
    
    T = np.linalg.inv(T_inv)
    return T, T @ pts

def test5():
    P = np.array([[1, 0, 2, 2],
                  [1, 3, 3, 4]])
    
    H = np.array([[-2, 0, 1],
                  [1, -2, 0],
                  [0, 0, 3]])
    
    Q = map_points_using_homography(P, H)

    H_est = hest_linear(Q, P)
    scale = H[0,0] / H_est[0,0]
    print(np.round(H_est * (scale)))

    normalize_points(P)
test5()