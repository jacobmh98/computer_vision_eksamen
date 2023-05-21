import matplotlib.pyplot as plt
import numpy as np
import cv2
from point_conversion import PI, inv_PI
import random
import math
from scipy.ndimage import convolve

# Week 1
def compute_camera_matrix(f, dx, dy, alpha=1, beta=1):
    """ Compute the camera matrix """
    A = np.matrix([[f, f*beta, dx],
                  [0, alpha*f, dy],
                  [0, 0, 1]])
    return A
def test_compute_camera_matrix():
    A = compute_camera_matrix(2774.5, 806.8, 622.6, 1, 0)
    print(A)

def compute_projection_matrix(f, dx, dy, alpha, beta, R, t):
    """ Compute the projection matrix """
    A = compute_camera_matrix(f, dx, dy, alpha, beta)
    P = A @ np.hstack((R, t))

    return P
def test_compute_projection_matrix():
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

def convert_line_to_homogeneous_coordinates(a, b, c):
    """ Convert a line on the form ax + by = c to homogeneous coordinates """
    return np.array([a, b, -c]).reshape((3,1))

def test_convert_line_to_homogeneous_coordinates():
    a = 1
    b = 2
    c = 3
    
    hom_line = convert_line_to_homogeneous_coordinates(a, b, c)
    print(hom_line)

def determine_if_point_on_line(l, p):
    """ Determine if a point is on a line in homogeneous coordinates """
    print(l @ p) # Just to see the result
    return l @ p == 0

def test_determine_if_point_on_line():
    l = np.array([1,2,-3])
    p = np.array([1,2,1]) # Point in homogeneous coords

    print(determine_if_point_on_line(l, p))


def determine_intersection_of_lines(l0, l1):
    """ Given to lines in homogeneous coordinates determines their point of intersection """
    return np.cross(l0, l1)

def test_determine_intersection_of_lines():
    l0 = np.array([1,1,-1])
    l1 = np.array([-1,1,-3])

    intersection = determine_intersection_of_lines(l0, l1)
    print(intersection)


def determine_distance_from_point_to_line(l, p):
    """ Computes the distance from a point to a line (both in homogeneous coordinates)"""
    return abs(l @ p) / np.sqrt(l[0] ** 2 + l[1] ** 2)

def test_determine_distance_from_point_to_line():
    l = np.array([(1/math.sqrt(2)), (1/math.sqrt(2)), -1])
    p = np.array([0,0,1])

    distance = determine_distance_from_point_to_line(l, p)
    print(distance)


# Week 2
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

def test_project_points_distortion():
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

def test_map_points_using_homography():
    
    # 2D Points on a plane
    p1 = np.array([[1],[1]])
    p2 = np.array([[0],[3]])
    p3 = np.array([[2],[3]])
    
    # Creating the set of 2D points
    P = np.hstack((p1,p2,p3))
    
    # Homography matrix
    H = np.array([[-2,0,1],
                  [1,-2,0],
                  [0,0,3]])
    
    mapped_points = map_points_using_homography(P, H)
    print(mapped_points)


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

def test_hest_linear():
    P = np.array([[1, 0, 2, 2],
                  [1, 3, 3, 4]])
    
    H = np.array([[-2, 0, 1], [1, -2, 0],[0, 0, 3]])
    
    Q = map_points_using_homography(P, H)
    # Q = np.array([[-0.33, 0.33, -1, -1], [-0.33, -2, -1.33, -2]])

    # Estimated homograhpy
    H_est = hest_linear(Q, P)
    
    # Scale is off, so below it is corrected for scale
    scale = H[0,0] / H_est[0,0]
    print(np.round(H_est * (scale)))


def normalize_points(pts):
    """ Normalize a set of 2D inhomogeneous points so they have mean 0 and std 1 """

    mean = np.mean(pts, axis=1)
    std = np.std(pts, axis=1)

    T_inv = np.array([[std[0], 0, mean[0]],
                      [0, std[1], mean[1]],
                      [0, 0, 1]])
    
    T = np.linalg.inv(T_inv)
    return T, T @ pts

def test_normalize_points():
    # 2D points
    P = np.array([[1, 0, 2, 2],
                  [1, 3, 3, 4]])
    
    T, S = normalize_points(inv_PI(P))
    print(T)
    print(S)


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

    T, S = normalize_points(inv_PI(P))
#test5()


# Week 3
def CrossOp(v):
    """Takes a vector v in 3D, and returns the 3x3 matrix corresponding to
        taking the cross product with that vector"""
    cross_mat = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
    return cross_mat

def test_CrossOp():
    v = np.array([0.2,2,1])
    cv = CrossOp(v)
    print(cv)

def EssentialMatrix1(R,t):
    """Computes the essential matrix between R1 and R2
    This one is when R1 or R2 == Identity matrix and t1 or t2 == [0,0,0]
    
    Returns a 3x3 matrix
    It has 5 degrees of freedom
    
    If we have to sets of points and the intrinsic camera matrix K:
        essential_matrix, _ = cv2.findEssentialMat(points1, points2, K)
    """
    
    E = np.dot(CrossOp(t.ravel()),R)
    return E

def test_EssentialMatrix1():
    """
    K = np.array([[1414.48973, 0.0, 401.393651],
                  [0.0, 1414.27075, 308.091633],
                  [0.0, 0.0, 1.0]])
    Corresponding points in two images
        points1 = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)
        points2 = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)
    """
    
    R = [[0.61141766, -0.76384514, 0.20666167],
         [0.6295392, 0.31131209, -0.71187442],
         [0.47942554, 0.56535421, 0.67121217]]
    
    t = np.array([[0.2],[2],[1]])
    
    E = EssentialMatrix1(R, t)
    print(E)


def EssentialMatrix2(R1, t1, R2, t2):
    """Computes the essential matrix between R1 and R2
    Returns a 3x3 matrix
    It has 5 degrees of freedom
    
    If we have to sets of points and the intrinsic camera matrix K:
        
    Corresponding points in two images
        points1 = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)
        points2 = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)
        
    essential_matrix, _ = cv2.findEssentialMat(points1, points2, K)
    """
    
    E = np.dot(np.dot(CrossOp(t2.ravel()), R2), np.dot(np.transpose(R2), CrossOp(t2.ravel()))) 
    return E

def test_EssentialMatrix2():
    R1 = [[0.61141766, -0.76384514, 0.20666167],
         [0.6295392, 0.31131209, -0.71187442],
         [0.47942554, 0.56535421, 0.67121217]]
    
    t1 = np.array([[0.2],[2],[1]])
    
    R2 = [[0.31141766, 0.6384514, -0.20666167],
         [0.2295392, 0.61131209, -0.21187442],
         [0.47942554, 0.6535421, 0.17121217]]
    
    t2 = np.array([[2],[0.5],[1]])

    
    E = EssentialMatrix2(R1, t1, R2, t2)
    print(E)


def FundmentalMatrix(E, K):
    """ Computes the fundamental matrix using the essential matrix and the
    camera intrinsic matrix K
    
    F is a 3x3 matrix
    
    Corresponding points in two images:
        points1 = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)
        points2 = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)

    Compute the fundamental matrix:
        fundamental_matrix, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    
    """
    F = np.dot(np.linalg.inv(K).T, np.dot(E, np.linalg.inv(K)))
    return F

def test_FundamentalMatrix():
    # E is an essential matrix, can be computed from essential matrix functions
    
    # Camera 1 rotation matrix
    R = [[0.61141766, -0.76384514, 0.20666167],
         [0.6295392, 0.31131209, -0.71187442],
         [0.47942554, 0.56535421, 0.67121217]]
    
    # Camera1 translation vector
    t = np.array([[0.2],[2],[1]])
    
    # Essential matrix (Use essentialmatrix2 function if R2 != I and t2 != [0,0,0])
    E = EssentialMatrix1(R, t) 
    
    # Camera matrix
    K = np.array([[1000,0,300],
                  [0,1000,200],
                  [0,0,1]])

    F = FundmentalMatrix(E, K)
    print(F)


def EpipolarLine(F, p):
    """ Computing the epipolar line l """
    l = np.dot(F,p)

    return l


def test_EpipolarLine():
    """
    It computes the epipolar line l of a point q in a camera 
    """
    
    # Homogeneous 3D point Q
    Q = np.array([[1],[0.5],[4],[1]])
    q = PI(Q) # converting homogeneuos point to inhomogene for projectpoints function

    # Projecting the point to each camera
    #q1 = projectpoints(K, R1, t1, q)
    q1 = np.array([[550],[325]]) # Here hardcoded, but can be found like above
    
    # Computing the fundamental matrix:
    R = [[0.61141766, -0.76384514, 0.20666167],
            [0.6295392, 0.31131209, -0.71187442],
            [0.47942554, 0.56535421, 0.67121217]]
    t = np.array([[0.2],[2],[1]])
    K = np.array([[1000,0,300],
                    [0,1000,200],
                    [0,0,1]])
    E = EssentialMatrix1(R, t)
    F = FundmentalMatrix(E, K)
    
    # Computing the epipolar line
    p1 = inv_PI(q1)

    l = EpipolarLine(F, p1)
    
    # Incase of a different scale of p1. Line is the correct, but can have wrong scale
    # Test with different scales to get correct result (divide multiple choice answer with 
    # number in line to get scale)
    scale = 1
    l = l * scale
    print(l)
    
def isPointOnLine(l, q):
    """ Compute whether a point p is on the line l
    
    if q.T * l = 0 the point is on the line (take numerical inprecission into account (close to 0 is good))
    """
    
    return np.dot(inv_PI(q).T,l)

def test_isPointOnLine():
    
    # Computing the epipolar line:
    # Homogeneous 3D point Q
    #Q = np.array([[1],[0.5],[4],[1]])
    # q = PI(Q) # converting homogeneuos point to inhomogene for projectpoints function

    # Projecting the point to each camera
    #q1 = projectpoints(K, R1, t1, q)
    q1 = np.array([[550],[325]]) # Here hardcoded, but can be found like above
    
    # Below is to get the fundamental matrix
    R = [[0.61141766, -0.76384514, 0.20666167],
            [0.6295392, 0.31131209, -0.71187442],
            [0.47942554, 0.56535421, 0.67121217]]
    t = np.array([[0.2],[2],[1]])
    K = np.array([[1000,0,300],
                    [0,1000,200],
                    [0,0,1]])
    E = EssentialMatrix1(R, t)
    F = FundmentalMatrix(E, K)
    
    # Computing the epipolar line
    p1 = inv_PI(q1)

    l = EpipolarLine(F, p1)
    
    # The point we want to see if it is on the line
    # q2 = projectpoints(K, R2, t2, q)
    q2 = np.array([[582.47256835],[185.98985776]])
    
    # testing if point is on the line
    test_value = isPointOnLine(l, q2)
    print(test_value)


# Week 4
def pest(Q, q):
    """ Estimates the projection matrix given the original and projected 3D points """
    n = len(Q[0, :])

    for i in range(n):
        x_i = q[0, i]
        y_i = q[1, i]
        X_i = Q[0, i]
        Y_i = Q[1, i]
        Z_i = Q[2, i]

        B_i = np.array([[0, -X_i, X_i*y_i, 0, -Y_i, Y_i*y_i, 0, -Z_i, Z_i*y_i, 0, -1, y_i],
                        [X_i, 0, -X_i*x_i, Y_i, 0, -Y_i*x_i, Z_i, 0, -Z_i*x_i, 1, 0, -x_i],
                        [-X_i*y_i, X_i*x_i, 0, -Y_i*y_i, Y_i*x_i, 0, -Z_i*y_i, Z_i*x_i, 0, -y_i, x_i, 0]])
        
    _, _, v_T = np.linalg.svd(B_i)

    P_est = v_T[-1].reshape((3,4), order='F')

    return P_est

def estimateHomographies(Q, qs):
    """ Estimate homographies given a set of 3D points and the projection into the image plane from different views
        Args: 
            Q_omega: array of original un-transformed points
            qs: list of arrays of projected points
    """
    
    homographies = []
    n = len(Q[0])

    for q in qs:
        H = hest_linear(Q, q)
        homographies.append(H)
    return homographies


# Week 5
def compute_residuals(Q,P): # It should work, no promises
    """ Computes the residuals as a vector
    Takes the parameters we want to optimize (Q)
    The residuals are the difference in projection
    P is the list of different projection matrices where the point Q is projected"""
    residuals = []
    Qs = []
    
    for i in range(len(P)):
        qi = PI(P[i] @ inv_PI(Q))
        qih = qi + np.array([[1], [-1]])
        residual_i = PI(P[i] @ inv_PI(Q)) - qih
        residuals.append(residual_i)

    return np.concatenate(residuals)

def test_compute_residuals():
    Q = np.array([[1],[1],[0]])

    R1 = R2 = np.identity(3)
    t1 = np.array([[0],[0],[1]])
    t2 = np.array([[0],[0],[20]])
    K1 = K2 = np.array([[700,0,600],[0,700,400],[0,0,1]])

    # Creating the Rt matrices
    Rt1 = np.append(R1,t1,axis=1)
    Rt2 = np.append(R2,t2,axis=1)
    P1 = K1 @ Rt1
    P2 = K2 @ Rt2
    P = [P1,P2]

    residuals = compute_residuals(Q,P)
    print(residuals)

def triangulate_nonlin(Q,P): # Does not work entirely. GL
    """
    import scipy
    from scipy import optimize
    """
    Qs = []

    def compute_residuals(Q):
        residuals = []
        
        for i in range(len(P)):
            qi = PI(P[i] @ inv_PI(Q))
            Qs.append(inv_PI(qi))
            qih = qi + np.array([[1], [-1]])
            residual_i = PI(P[i] @ inv_PI(Q)) - qih
            residuals.append(residual_i)
    
        return np.concatenate(residuals)
    
    #residuals = compute_residuals(Q)
    #x0 = triangulate1(Qs, P)
    #scipy.optimize.least_squares(residuals, x0, args=(), method='lm')
    
    return 0


# week 6
def gaussian1DKernel(sigma):
    length = int(6 * sigma)  # Length of the Gaussian kernel
    size = length // 2  # Half the length for symmetric kernel
    x = np.linspace(-size, size, length)  # Generate an array of x values

    g = np.exp(-x**2 / (2 * sigma**2))  # Gaussian kernel
    g /= np.sum(g)  # Normalize the Gaussian kernel to sum to 1

    gd = -(x / sigma**2) * g  # Derivative of the Gaussian kernel
    return g, gd

def gaussianSmoothing(im, sigma):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # Convert image to single channel (greyscale)
    im = im.astype(float)  # Convert image to floating point

    g, gd = gaussian1DKernel(sigma)  # Generate 1D Gaussian kernel and its derivative
    # Apply 1D smoothing in the x-direction
    Ix = convolve(im_gray.flatten(), g, mode='nearest')
    Ix = convolve(Ix, gd, mode='nearest' )

    # Apply 1D smoothing in the y-direction
    Iy = convolve(im_gray.flatten(), g, mode='nearest')
    Iy = convolve(Iy, gd, mode='nearest')

    # Apply 2D smoothing
    I = convolve(im_gray, g[:, np.newaxis])
    I = convolve(I, g[np.newaxis, :])

    return I, Ix.reshape((I.shape[0], I.shape[1])), Iy.reshape((I.shape[0], I.shape[1]))

def smoothedHessian(im, sigma, epsilon):
    I, Ix, Iy = gaussianSmoothing(im, sigma)  # Compute smoothed image and derivatives

    g, _ = gaussian1DKernel(sigma)  # Generate 1D Gaussian kernel
    g_epsilon, _ = gaussian1DKernel(epsilon)  # Generate 1D Gaussian kernel with epsilon

    # Compute elements of the smoothed Hessian matrix
    Ix2 = Ix**2
    Iy2 = Iy**2
    IxIy = Ix * Iy

    # Apply Gaussian smoothing with epsilon to the elements of the Hessian matrix
    C = np.empty((300, 300, 4))
    
    C[:, :, 0] = convolve(Ix2.flatten(), g_epsilon, mode='nearest').reshape((I.shape[0], I.shape[1]))
    C[:, :, 1] = convolve(IxIy.flatten(), g_epsilon, mode='nearest').reshape((I.shape[0], I.shape[1]))
    C[:, :, 2] = convolve(IxIy.flatten(), g_epsilon, mode='nearest').reshape((I.shape[0], I.shape[1]))
    C[:, :, 3] = convolve(Iy2.flatten(), g_epsilon, mode='nearest').reshape((I.shape[0], I.shape[1]))
   
    return C

def harrisMeasure(im, sigma, epsilon, k):
    C = smoothedHessian(im, sigma, epsilon)

    a = C[:, :, 0]
    b = C[:, :, 3]
    c = C[:, :, 1]

    r = a * b - c**2 - k * ((a+b)**2)
    return r

def cornerDetector(im, sigma, epsilon, k, tau): # Does not work :(
    r = harrisMeasure(im, sigma, epsilon, k)
    c = []

    for x in range(1, r.shape[0] - 1):
        for y in range(1, r.shape[1] - 1):
           # if #r[x,y] > tau and \
            if r[x, y] > tau and \
                r[x + 1, y] and \
                r[x,y] >= r[x - 1, y] and \
                r[x,y] > r[x, y + 1] and \
                r[x,y] <= r[x, y - 1]:
                c.append([x,y])

    return c

def test_gaussian():
    im = cv2.imread('week06_data/TestIm1.png')
    
    c = cornerDetector(im, 2, 3, 0.06, -1)
    for i, c_i in enumerate(c):
        if i > 1:
            break
        print(c_i)
        plt.plot(c_i, marker="+")
    plt.imshow(im)
    plt.show()
#test_gaussian()


def canny_edge_detection(im1, im2):
    """ Detecting the edges in both im1 and im2 using Canny edge detection """
    # Grayscaling the images
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Computing min and max values in both im1 and im2
    im1_min = np.min(im1)
    im1_max = np.max(im1)
    im2_min = np.min(im2)
    im2_max = np.max(im2)

    # Computing the edges
    im1_edges = cv2.Canny(im1, im1_min, im1_max)
    im2_edges = cv2.Canny(im2, im2_min, im2_max)
    return im1_edges, im2_edges

def test_canny_edge_detection():
    im1 = cv2.imread('week06_data/TestIm1.png')
    im2 = cv2.imread('week06_data/TestIm2.png')

    im1_edges, im2_edges = canny_edge_detection(im1, im2)

    plt.subplot(121),plt.imshow(im1_edges,cmap = 'gray')
    plt.title('Image1 Edges'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(im2_edges,cmap = 'gray')
    plt.title('Image2 Edges'), plt.xticks([]), plt.yticks([])
    plt.show()

#test_canny_edge_detection()


# Week 8
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





