import numpy as np
import cv2
import matplotlib.pyplot as plt

# Loading the camera matrix and images
K = np.loadtxt('data/K.txt')
im0 = cv2.imread('data/sequence/000001.png')
im1 = cv2.imread('data/sequence/000002.png')
im2 = cv2.imread('data/sequence/000003.png')

def show_ims(pts=None):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    ax[0].imshow(im0)
    ax[0].set_title("Image 0")
    ax[1].imshow(im1)
    ax[1].set_title("Image 1")
    ax[2].imshow(im2)
    ax[2].set_title("Image 2")

    if pts is not None:
        for i, pl in enumerate(pts):
            for p in pl:
                ax[i].plot(p[0], p[1], marker='o', color='red', markersize=2)
    plt.show()
# show_ims()

# Converting the images to grayscale
im0_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

""" Exercise 1 """
# Finding SIFT keypoints and descriptors for all three images (limit 2000 features)
sift = cv2.SIFT_create()
kp0, des0 = sift.detectAndCompute(im0_gray, None)
kp1, des1 = sift.detectAndCompute(im1_gray, None)
kp2, des2 = sift.detectAndCompute(im2_gray, None)

# Limit the number of matches to n
n = 2000
kp0 = kp0[0:n]
kp1 = kp1[0:n]
kp2 = kp2[0:n]

des0 = des0[0:n, :]
des1 = des1[0:n, :]
des2 = des2[0:n, :]

# Converting the features to numpy arrays (2D points)
kp0 = np.array([k.pt for k in kp0])
kp1 = np.array([k.pt for k in kp1])
kp2 = np.array([k.pt for k in kp2])

# Matching the SIFT features between im1-im2 and im2-im3
bf = cv2.BFMatcher()
matches01 = bf.match(des0, des1)
matches12 = bf.match(des1, des2)

# Converting the matches to numppy arrays of the indices
matches01_indices = np.array([(m.queryIdx, m.trainIdx) for m in matches01])
matches12_indices = np.array([(m.queryIdx, m.trainIdx) for m in matches12])

""" Exercise 2 """
# Estimate the essential matrix between im0-im1 with RANSAC
E0, E1 = cv2.findEssentialMat(kp0, kp1, K)

# Decomposing the essential matrix to find the correct pose (R1, t1)
n_inliers, R1, t1, mask = cv2.recoverPose(E0, kp0, kp1)

inliers = []

# Removing the matches that are not inliers from matches01
for i in range(n):
    if mask[i] != 0:
        inliers.append([matches01[i]])

matches01 = np.array(inliers)

""" Exercise 3 """
# Finding the intersection of the matches from im0-im2 hence where the indices point to the same point in the corresponding images
_, idx01, idx12 = np.intersect1d(matches01_indices[:,1], matches12_indices[:,0], return_indices=True)

# Create three lists where the i'th index points to the same 2D point in each image
points0 = kp1
points1 = []
points2 = []

for i in idx01:
    points1.append(kp1[i])

for i in idx12:
    points2.append(kp2[i])

""" Exercise 4 """
# Use the 2D positions in im0 and im1 to triangulate the points in 3D

# Use the 2D positions in im2 to estimate the pose of iamge 2 with RANSAC
# cv2.solvePnPRansac
# distCoeffs = np.zeros(5)
