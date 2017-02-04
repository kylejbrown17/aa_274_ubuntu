#!/usr/bin/python

import rospy
import sensor_msgs

import time
import os

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import numpy as np

import pdb

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:
    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, display_flag):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize = (12,5))
            gs = gridspec.GridSpec(1,2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(os.listdir(self.cal_img_path)):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag:
                ax = plt.subplot(gs[0,0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0,1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]

    def undistortImages(self, A, k = np.zeros(2), scale = 0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize = (6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(os.listdir(self.cal_img_path)):
            img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
            img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
            img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

            ax = plt.subplot(gs[0,0])
            ax.imshow(img_dist, cmap='gray')
            ax.axis('off')

            ax = plt.subplot(gs[0,1])
            ax.imshow(img_undist_no_k, cmap='gray')
            ax.axis('off')

            if k[0] != 0:
                ax = plt.subplot(gs[0,2])
                ax.imshow(img_undist_w_k, cmap='gray')
                ax.axis('off')

            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, k = np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize = (8,6))
        plt.clf()

        for p in range(self.n_chessboards):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            if k[0] != 0:
                u_br, v_br = self.transformWorld2PixImageDist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A, k)
                ax.plot(u_br, v_br, 'g+', label='Radial Distortion Calibration')

            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, R, t):
        # Expects R and t to be lists of arrays, just like u_meas, v_meas

        w_board = self.d_square*self.n_corners_x
        h_board = self.d_square*self.n_corners_y
        x_corners = [-w_board/2, w_board/2, w_board/2, -w_board/2]
        y_corners = [-h_board/2, -h_board/2, h_board/2, h_board/2]
        s_cam = 0.02
        d_cam = 0.1
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0,1,2],[0,2,3],[0,3,4],[0,4,1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize = (12,5))
        axim = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p],t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([x_corners[i], y_corners[i], 0, 1]))
                M_tld *= np.sign(M_tld[2])
                M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0],M[1],M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(os.listdir(self.cal_img_path)):

            img = cv2.imread(self.cal_img_path + '/' + file, 0)
            axim.imshow(img, cmap='gray')
            axim.axis('off')

            ax3d.clear()

            for j in range(len(ind_cam)):
                cam = Poly3DCollection(verts_cam[j])
                cam.set_color('green')
                cam.set_alpha(0.2)
                ax3d.add_collection3d(cam)

            for p in range(self.n_chessboards):
                if p == i:
                    boards[p].set_color('blue')
                    boards[p].set_alpha(1.0)
                else:
                    boards[p].set_color('red')
                    boards[p].set_alpha(0.1)

                ax3d.add_collection3d(boards[p])
                ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                plt.show(block=False)

            view_max = 0.2
            ax3d.set_xlim(-view_max,view_max)
            ax3d.set_ylim(-view_max,view_max)
            ax3d.set_zlim(-4*view_max,0)
            ax3d.set_xlabel('X axis')
            ax3d.set_ylabel('Y axis')
            ax3d.set_zlabel('Z axis')

            plt.tight_layout()
            fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

            plt.show(block=False)

            raw_input('<Hit Enter To Continue>')

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = A
        self.c.distortion = np.hstack((k, np.zeros(3))).reshape((5,1))
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:,0][:,0])
            v_meas.append(self.h_pixels - chessboards[0][:,0][:,1])   # Flip Y-axis to traditional direction

        return u_meas, v_meas   # Lists of arrays (one per chessboard)

    def genCornerCoordinates(self, u_meas, v_meas):
        # generate world coordinates (X,Y) for every corner in the chessboard.
        # the ordering must correspond exactly to the order in (u_meas,v_meas)
        #X = [(self.n_corners_x - 1) * self.d_square -self.d_square*float(i%self.n_corners_x) for i in range(len(u_meas))] # from bottom right corner 
        #Y = [self.d_square*float(i/self.n_corners_x) for i in range(len(v_meas))]

        # origin = top left, x increases to the right, y increases down
        X = [self.d_square*float(i%self.n_corners_x) for i in range(len(u_meas[0]))] # from bottom right corner 
        Y = [self.d_square*float(i/self.n_corners_x) for i in range(len(v_meas[0]))]

        return X, Y


    def estimateHomography(self, u_meas, v_meas, X, Y):
        # form matrix L
        M = np.vstack((X, Y, [1.0 for _ in range(len(X))]))
    
        u = np.dot(np.array([u_meas]).T, np.ones([1,3]))
        v = np.dot(np.array([v_meas]).T, np.ones([1,3]))
        uM = u*M.T
        vM = v*M.T
        
        L1 = np.hstack((M.T, np.zeros(M.T.shape), -uM))
        L2 = np.hstack((np.zeros(M.T.shape), M.T, -vM))
        L = np.vstack((L1, L2))
        
        # SVD
        U,S,V  = np.linalg.svd(L,full_matrices=False)
        h = V[-1] # eignevector associated with smallest singular value in S, which is the last vector in V
        H = h.reshape(3,3)
    
        return H


    def getCameraIntrinsics(self, H):
        V = np.zeros([1,6])

        for _,Hmat in H.iteritems():
            h1 = Hmat.T[0]
            h2 = Hmat.T[1]
            h3 = Hmat.T[2]

            v_11 = [h1[0]*h1[0], h1[0]*h1[1]+h1[1]*h1[0], h1[1]*h1[1], h1[2]*h1[0]+h1[0]*h1[2], h1[2]*h1[1]+h1[1]*h1[2], h1[2]*h1[2]]
            v_22 = [h2[0]*h2[0], h2[0]*h2[1]+h2[1]*h2[0], h2[1]*h2[1], h2[2]*h2[0]+h2[0]*h2[2], h2[2]*h2[1]+h2[1]*h2[2], h2[2]*h2[2]]
            v_12 = [h1[0]*h2[0], h1[0]*h2[1]+h1[1]*h2[0], h1[1]*h2[1], h1[2]*h2[0]+h1[0]*h2[2], h1[2]*h2[1]+h1[1]*h2[2], h1[2]*h2[2]]

            if V.shape[0] < 2:
                V = np.vstack((np.array(v_12), np.array(v_11)-np.array(v_22)))
            else:
                V2 = np.vstack((np.array(v_12), np.array(v_11)-np.array(v_22)))
                V = np.vstack((V,V2))

        U,S,V_ = np.linalg.svd(V,full_matrices=False)
        b = V_[-1]

        B = np.array([
            [b[0], b[1], b[3]],
            [b[1], b[2], b[4]],
            [b[3], b[4], b[5]]
        ])

        v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2])/(B[0,0]*B[1,1] - B[0,1]**2)
        lam = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2]))/B[0,0]
        alpha = np.sqrt(lam/B[0,0])
        beta = np.sqrt(lam*B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
        gamma = -B[0,1]* alpha**2 * beta / lam
        u0 = gamma*v0/alpha - B[0,2]*alpha**2 / lam
        
        A = np.array([
            [alpha, gamma, u0],
            [0,     beta,  v0],
            [0,     0,     1]
        ])

        print "#########"
        print A

        #pdb.set_trace()
        
        return A

    def getExtrinsics(self, H, A):

        A_inv = np.linalg.inv(A)
        lam = 1/np.linalg.norm(A_inv.dot(H.T[0]))
        r1 = lam*A_inv.dot(H.T[0])
        r2 = lam*A_inv.dot(H.T[1])
        r3 = np.cross(r1,r2)
        t = lam * A_inv.dot(H.T[2])
        R_g = np.vstack((r1,r2,r3)).T

        U,S,V = np.linalg.svd(R_g,full_matrices=False)
        R = U.dot(V)

        return R, t


    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):
        """
        Note: The transformation functions should only process one chessboard at a time!
        This means X, Y, Z, R, t should be individual arrays
        """

        R_t = np.hstack((R,np.array([t]).T))
        XYZ_W = np.vstack((X,Y,Z,[1 for _ in X])).T # homogeneous world frame coordinates
        XYZ_C = R_t.dot(XYZ_W.T) # camera frame coordinates
        #xyz_c = A.dot(R_t.dot(XYZ_W.T))
        x = XYZ_C[0]/XYZ_C[2]
        y = XYZ_C[1]/XYZ_C[2]

        return x, y

    
    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):
        R_t = np.hstack((R,np.array([t]).T))
        #XYZ_W = np.vstack((X,Y,Z,[1 for _ in X])).T # homogeneous world frame coordinates
        XYZ_W = np.vstack((X,Y,Z,np.ones(X.shape))).T

        xyz_c = A.dot(R_t.dot(XYZ_W.T))
        u = xyz_c[0]/xyz_c[2]
        v = xyz_c[1]/xyz_c[2]

        return u, v

    def transformWorld2NormImageDist(self, X, Y, Z, R, t, k):
        R_t = np.hstack((R,np.array([t]).T))
        XYZ_W = np.vstack((X,Y,Z,np.ones(X.shape))).T # homogeneous world frame coordinates
        XYZ_C = R_t.dot(XYZ_W.T) # camera frame coordinates

        x = XYZ_C[0]/XYZ_C[2]
        y = XYZ_C[1]/XYZ_C[2]

        x_br = x + x*(k[0]*(x**2 + y**2) + k[1]*(x**2 + y**2)**2)
        y_br = y + y*(k[0]*(x**2 + y**2) + k[1]*(x**2 + y**2)**2)

        return x_br, y_br

    def transformWorld2PixImageDist(self, X, Y, Z, R, t, A, k):
        R_t = np.hstack((R,np.array([t]).T))
        #XYZ_W = np.vstack((X,Y,Z,[1 for _ in X])).T # homogeneous world frame coordinates
        XYZ_W = np.vstack((X,Y,Z,np.ones(X.shape))).T

        XYZ_C = R_t.dot(XYZ_W.T) # camera frame coordinates

        # normalize
        XYZ_C[0] = XYZ_C[0]/XYZ_C[2]
        XYZ_C[1] = XYZ_C[1]/XYZ_C[2]
        XYZ_C[2] = XYZ_C[2]/XYZ_C[2]

        x = XYZ_C[0]
        y = XYZ_C[1]

        uv = A.dot(XYZ_C)

        u = uv[0]
        v = uv[1]

        u_0 = A[0,2]
        v_0 = A[1,2]

        u_br = u + (u - u_0)*(k[0]*(x**2 + y**2) + k[1]*(x**2 + y**2)**2)
        v_br = v + (v - v_0)*(k[0]*(x**2 + y**2) + k[1]*(x**2 + y**2)**2)

        return u_br, v_br

    def estimateLensDistortion(self, u_meas, v_meas, X, Y, R, t, A):
        u_0 = A[0,2]
        v_0 = A[1,2]

        Z = np.zeros(X[0].shape)

        D = np.zeros([2*len(u_meas[0]),2])
        d = np.zeros([2*len(u_meas[0]),1])

        for i in range(len(u_meas)):
            x,y = self.transformWorld2NormImageUndist(X[i], Y[i], Z, R[i], t[i])
            u,v = self.transformWorld2PixImageUndist(X[i], Y[i], Z, R[i], t[i], A)

            if i == 0:
                D_1 = np.vstack(((u - u_0) * (x**2 + y**2), (u - u_0) * (x**2 + y**2)**2)).T
                D_2 = np.vstack(((v - v_0) * (x**2 + y**2), (v - v_0) * (x**2 + y**2)**2)).T
                D = np.vstack((D_1,D_2))
                d = np.hstack(([u_meas[i]-u], [v_meas[i]-v])).T

            else:
                D_1 = np.vstack(((u - u_0) * (x**2 + y**2), (u - u_0) * (x**2 + y**2)**2)).T
                D_2 = np.vstack(((v - v_0) * (x**2 + y**2), (v - v_0) * (x**2 + y**2)**2)).T
                # stack with previous matrices
                D = np.vstack((D,D_1,D_2))
                d = np.vstack((d, np.hstack(([u_meas[i]-u], [v_meas[i]-v])).T))

        k = np.linalg.inv(D.T.dot(D)).dot(D.T).dot(d)
        k = k.reshape((2,))
        return k
