import matplotlib.pyplot as plt
import numpy as np
import cv2
from point_conversion import PI, inv_PI
import random

def compute_camera_matrix(f, dx, dy, alpha=1, beta=1):
    """ Compute the camera matrix """
    A = np.matrix([[f, f*beta, dx],
                  [0, alpha*f, dy],
                  [0, 0, 1]])
    return A
def test():
    A = compute_camera_matrix(2774.5, 806.8, 622.6, 1, 0)
    print(A)

def compute_projection_matrix(f, dx, dy, alpha, beta, R, t):
    """ Compute the projection matrix """
    A = compute_camera_matrix(f, dx, dy, alpha, beta)
    P = A @ np.hstack((R, t))

    return P
def test():
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


def hest(Q1, Q2):
    """ Estimate a homography given two sets of matching points"""
    n = len(Q1[0])

    B = np.zeros((3*n, 9))

    for i in range(n):
        x1i = Q1[0, i]
        y1i = Q1[1, i]
        x2i = Q2[0, i]
        y2i = Q2[1, i]

        bi_T = np.array([[0, -x2i, x2i*y1i, 0, -y2i, y2i*y1i, 0, -1, y1i],
                         [x2i, 0, -x2i*x1i, y2i, 0, -y2i*x1i, 1, 0, -x1i],
                         [-x2i*y1i, x2i*x1i, 0, -y2i*y1i, y2i*x1i, 0, -y1i, x1i, 0]])

        B[i:(i+3), :] = bi_T
    _, _, V_T = np.linalg.svd(B,)
    return V_T[-1].reshape((3,3), order='F')

def estHomographyRANSAC(Q1, Q2, iterations):
    """ Compute the homography using RANSAC """
    n = len(Q1[0])
    errors = np.empty(iterations)
    homographies = np.empty(iterations)
    inliers_im1 = []
    inliers_im2 = []

    for it in range(iterations):
        # Selecting 4 random points from Q1 and Q2
        i1, i2, i3, i4 = random.sample(range(0, n-1), 4)
        Q1_points = np.hstack((Q1[:, i1].reshape((2, 1)), Q1[:, i2].reshape((2, 1)), Q1[:, i3].reshape((2, 1)), Q1[:, i4].reshape((2, 1))))
        Q2_points = np.hstack((Q2[:, i1].reshape((2, 1)), Q2[:, i2].reshape((2, 1)), Q2[:, i3].reshape((2, 1)), Q2[:, i4].reshape((2, 1))))
        
        # Estimating the homography for the four points
        H = hest(Q1_points, Q2_points)
        print("det H=" + str(np.linalg.det(H)))

        # Computing the error for the estimated points
        distances = np.empty((4))
        for i in range(4):
            print(Q1[:,i].reshape((2,1)))
            p1i = inv_PI(Q1[:, i].reshape((2,1)))
            p2i = inv_PI(Q2[:, i].reshape((2,1)))

            dist = np.linalg.norm(PI(H @ p2i) - PI(p1i))**2 + np.linalg.norm(PI(np.linalg.inv(H) @ p1i) - PI(p2i))**2
            distances[i] = dist

            sigma = 3
            threshold = np.sqrt(3.84 * sigma**2)
            
            if dist < threshold:
                inliers_im1.append(p1i)
                inliers_im2.append(p2i)
        
        error = np.mean(distances)
        
        # Saving the homography and error
        homographies[it] = H
        errors[it] = error

    print(len(inliers_im1))
    # Select the homography with the lowest error
    min_error = np.argmin(errors)
    return homographies[min_error]

def test():
    im1 = cv2.imread('data/im1.jpg')
    im2 = cv2.imread('data/im2.jpg')
    kp1, kp2, des1, des2 = find_SIFT_keypoints_descriptors(im1, im2)
    
    matches = match_SIFT_keypoints(im1, im2)
    n = len(matches)
    
    Q1 = np.empty((2, n))
    Q2 = np.empty((2, n))

    for i, match in enumerate(matches):

        query_idx = match.queryIdx  # Index of the keypoint in the source set
        train_idx = match.trainIdx  # Index of the keypoint in the destination set

        query_point = kp1[query_idx].pt  # Keypoint coordinates in the source set
        train_point = kp2[train_idx].pt  # Keypoint coordinates in the destination set

        Q1[0,i] = query_point[0]
        Q1[1,i] = query_point[1]
    
        Q1[0,i] = train_point[0]
        Q1[1,i] = train_point[1]

    estHomographyRANSAC(Q1, Q2, 200)

    #out = cv2.drawMatches(im1, kp1, im2, kp2, matches[:1], None, flags=2)
    #plt.imshow(out[:,:,::-1]), plt.show()
test()