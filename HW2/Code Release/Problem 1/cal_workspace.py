#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pdb

from cam_calibrator import CameraCalibrator


def main():
    cc = CameraCalibrator()

    cal_img_path = './webcam_12'  # Location of calibration images
    name = 'webcam'               # Name of the camera
    n_corners = [7, 9]            # Corner grid dimensions
    square_length = 0.0205        # Chessboard square length in meters

    display_flag = False
    cc.loadImages(cal_img_path, name, n_corners, square_length, display_flag)

    u_meas, v_meas = cc.getMeasuredPixImageCoord()

    # Add your code here!

    # 1.1 - generate world frame coordinates
    X,Y = cc.genCornerCoordinates(u_meas,v_meas)

    # 1.2 - estimate homography
    H_dict = {}
    for i in range(len(u_meas)):
        H_dict[i] = cc.estimateHomography(u_meas[i], v_meas[i], X, Y)

    #print H_dict

    # 1.3 - get intrinsic matrix
    A = cc.getCameraIntrinsics(H_dict)

    # 1.4 - get rotations and translations
    R_dict = {}
    t_dict = {}
    for i in range(len(u_meas)):
        R_dict[i], t_dict[i] = cc.getExtrinsics(H_dict[i], A)

    #print R_dict
    #print t_dict

    # reformat X and Y into redundant arrays (how silly)
    X_big = np.repeat(np.array([X]),cc.n_chessboards,axis=0)
    Y_big = np.repeat(np.array([Y]),cc.n_chessboards,axis=0)

    # see how well the world coordinates map back to the image frame (u, v)
#    cc.plotBoardPixImages(u_meas, v_meas, X_big, Y_big, R_dict, t_dict, A)
#    cc.plotBoardLocations(R_dict, t_dict)

    # estimate distortion parameters
    k = cc.estimateLensDistortion(u_meas, v_meas, X_big, Y_big, R_dict, t_dict, A)

    # see how well the world coords map back to the image frame with distortion added
#    cc.plotBoardPixImages(u_meas, v_meas, X_big, Y_big, R_dict, t_dict, A, k)

    # Undistort images
#    cc.undistortImages(A,k)

    # save parameters to yaml file
    cc.writeCalibrationYaml(A,k)
    


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
