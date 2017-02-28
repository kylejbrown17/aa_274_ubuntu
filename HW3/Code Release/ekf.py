import numpy as np
from numpy import sin, cos
import scipy.linalg    # you may find scipy.linalg.block_diag useful
from ExtractLines import ExtractLines, normalize_line_parameters, angle_difference
from maze_sim_parameters import LineExtractionParams, NoiseParams, MapParams
import pdb

class EKF(object):

    def __init__(self, x0, P0, Q):
        self.x = x0    # Gaussian belief mean
        self.P = P0    # Gaussian belief covariance
        self.Q = Q     # Gaussian control noise covariance (corresponding to dt = 1 second)

    # Updates belief state given a discrete control step (Gaussianity preserved by linearizing dynamics)
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def transition_update(self, u, dt):
        g, Gx, Gu = self.transition_model(u, dt)

        self.x = np.reshape(g,[len(self.x),])
        self.P = Gx.dot(self.P).dot(Gx.T) + dt*Gu.dot(self.Q).dot(Gu.T)
        
        
    # Propagates exact (nonlinear) state dynamics; also returns associated Jacobians for EKF linearization
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: (g, Gx, Gu)
    #      g  - result of belief mean self.x propagated according to the system dynamics with control u for dt seconds
    #      Gx - Jacobian of g with respect to the belief mean self.x
    #      Gu - Jacobian of g with respect to the control u
    def transition_model(self, u, dt):
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

        
    
    # Updates belief state according to a given measurement (with associated uncertainty)
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def measurement_update(self, rawZ, rawR):
        z, R, H = self.measurement_model(rawZ, rawR)
        if z is None:    # don't update if measurement is invalid (e.g., no line matches for line-based EKF localization)
            return

        sigma = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(sigma))
        self.x = self.x + np.reshape(K.dot(z),[len(self.x),])
        self.P = self.P - K.dot(sigma).dot(K.T)
        # pdb.set_trace()
        

    # Converts raw measurement into the relevant Gaussian form (e.g., a dimensionality reduction);
    # also returns associated Jacobian for EKF linearization
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: (z, R, H)
    #       z - measurement mean (for simple measurement models this may = rawZ)
    #       R - measurement covariance (for simple measurement models this may = rawR)
    #       H - Jacobian of z with respect to the belief mean self.x
    def measurement_model(self, rawZ, rawR):
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class Localization_EKF(EKF):

    def __init__(self, x0, P0, Q, map_lines, tf_base_to_camera, g):
        self.map_lines = map_lines                    # 2xJ matrix containing (alpha, r) for each of J map lines
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Unicycle dynamics (Turtlebot 2)
    def transition_model(self, u, dt):
        V, om = u
        x, y, Th = self.x        
        #pdb.set_trace()
        if np.abs(u[1]) <= 0.00001: # account for singularity at omega = 0
            g = np.array([x + V*cos(Th)*dt,
                          y + V*sin(Th)*dt,
                          Th + om*dt])
            Gx = np.array([[1, 0, -V*sin(Th)*dt],
                           [0, 1, V*cos(Th)*dt],
                           [0, 0, 1]])
            Gu = np.array([[cos(Th)*dt, 0],
                           [sin(Th)*dt, 0],
                           [0, dt]])
            
        else:
            g = np.array([x + (V/om)*(sin(om*dt + Th) - sin(Th)),
                          y + (V/om)*(-cos(om*dt + Th) + cos(Th)),
                          Th + om*dt])
            # Gx
            dXdX = 1
            dXdY = 0
            dXdTh = (V/om)*(np.cos(om*dt + Th) - np.cos(Th))
            dYdX = 0
            dYdY = 1
            dYdTh = (V/om)*(np.sin(om*dt + Th) - np.sin(Th))
            dThdX = 0
            dThdY = 0
            dThdTh = 1
            
            # Gu
            dXdV = (1/om)*(np.sin(om*dt+Th) - np.sin(Th))
            dXdom = (V/om**2)*(om*np.cos(om*dt+Th)*dt - np.sin(om*dt+Th) + np.sin(Th))
            dYdV = (1/om)*(-np.cos(om*dt+Th) + np.cos(Th))
            dYdom = (V/om**2)*(om*np.sin(om*dt+Th)*dt + np.cos(om*dt+Th) - np.cos(Th))
            dThdV = 0
            dThdom = dt
                
            Gx = np.array([[dXdX, dXdY, dXdTh],
                           [dYdX, dYdY, dYdTh],
                           [dThdX, dThdY, dThdTh]])
            Gu = np.array([[dXdV, dXdom],
                           [dYdV, dYdom],
                           [dThdV, dThdom]])

        
        return g, Gx, Gu

    # Given a single map line m in the world frame, outputs the line parameters in the scanner frame so it can
    # be associated with the lines extracted from the scanner measurements
    # INPUT:  m = (alpha, r)
    #       m - line parameters in the world frame
    # OUTPUT: (h, Hx)
    #       h - line parameters in the scanner (camera) frame
    #      Hx - Jacobian of h with respect to the the belief mean self.x
    def map_line_to_predicted_measurement(self, m):
        alpha, r = m

        x, y, Th = self.x
        xC, yC, ThC = self.tf_base_to_camera
        x = x + xC*np.cos(Th) - yC*np.sin(Th)
        y = y + xC*np.sin(Th) + yC*np.cos(Th)
        
        A = alpha - Th - ThC
        R = r - (x*np.cos(alpha) + y*np.sin(alpha))

        dAdx = 0
        dAdy = 0
        dAdTh = -1
        
        dRdx = -np.cos(alpha)
        dRdy = -np.sin(alpha)
        dRdTh = xC*np.cos(Th) - yC*np.sin(Th)

        h = np.array([[A],
                      [R]])
        try:
            Hx = np.array([[dAdx, dAdy, dAdTh],
                           [dRdx, dRdy, dRdTh]])
        except ValueError:
            pdb.set_trace()
            
        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Given lines extracted from the scanner data, tries to associate to each one the closest map entry
    # measured by Mahalanobis distance
    # INPUT:  (rawZ, rawR)
    #    rawZ - 2xI matrix containing (alpha, r) for each of I lines extracted from the scanner data (in scanner frame)
    #    rawR - list of I 2x2 covariance matrices corresponding to each (alpha, r) column of rawZ
    # OUTPUT: (v_list, R_list, H_list)
    #  v_list - list of at most I innovation vectors (predicted map measurement - scanner measurement)
    #  R_list - list of len(v_list) covariance matrices of the innovation vectors (from scanner uncertainty)
    #  H_list - list of len(v_list) Jacobians of the innovation vectors with respect to the belief mean self.x
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        ##############

        d_list = []
        v_listA = []
        R_listA = []
        H_listA = []

        for i in range(len(rawR)):
            d_list.append(None)
            v_listA.append(None)
            R_listA.append(None)
            H_listA.append(None)
            d_min = np.Inf
            for j in range(self.map_lines.shape[1]):
                m = self.map_lines[:,j]
                h, Hx = self.map_line_to_predicted_measurement(m)
                R = rawR[i]
                z = np.array([rawZ[:,i]]).T
                try:
                    assert(z.shape[0] == 2)
                except AssertionError:
                    print z.shape
                    
                v = z - h
                S = Hx.dot(self.P).dot(Hx.T) + R

                d = (v.T).dot(np.linalg.inv(S)).dot(v)
                if d[0][0] < d_min:
                    d_min = d[0][0]
                    d_list[i] = d_min
                    v_listA[i] = v
                    R_listA[i] = R
                    H_listA[i] = Hx
                    
            v_list = [v_listA[i] for i in range(len(v_listA)) if d_list[i] < self.g]
            R_list = [R_listA[i] for i in range(len(R_listA)) if d_list[i] < self.g]
            H_list = [H_listA[i] for i in range(len(H_listA)) if d_list[i] < self.g]
            
        return v_list, R_list, H_list

    # Assemble one joint measurement, covariance, and Jacobian from the individual values corresponding to each
    # matched line feature
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        z = np.vstack(v_list)
        
        R = np.zeros([2*len(R_list), 2*len(R_list)])
        for i in range(len(R_list)):
            R[2*i:2+2*i,2*i:2+2*i] = R_list[i]        

        H = np.vstack(H_list)
        return z, R, H


class SLAM_EKF(EKF):

    def __init__(self, x0, P0, Q, tf_base_to_camera, g):
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Combined Turtlebot + map dynamics
    # Adapt this method from Localization_EKF.transition_model.
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x[:3]

        #### TODO ####
        # compute g, Gx, Gu (some shape hints below)
        # g = np.copy(self.x)
        # Gx = np.eye(self.x.size)
        # Gu = np.zeros((self.x.size, 2))
        ##############

        return g, Gx, Gu

    # Combined Turtlebot + map measurement model
    # Adapt this method from Localization_EKF.measurement_model.
    #
    # The ingredients for this model should look very similar to those for Localization_EKF.
    # In particular, essentially the only thing that needs to change is the computation
    # of Hx in map_line_to_predicted_measurement and how that method is called in
    # associate_measurements (i.e., instead of getting world-frame line parameters from
    # self.map_lines, you must extract them from the state self.x)
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H (should be identical to Localization_EKF.measurement_model above)
        ##############


        
        return z, R, H

    # Adapt this method from Localization_EKF.map_line_to_predicted_measurement.
    #
    # Note that instead of the actual parameters m = (alpha, r) we pass in the map line index j
    # so that we know which components of the Jacobian to fill in.
    def map_line_to_predicted_measurement(self, j):
        alpha, r = self.x[3+2*j, 3+2*j+2]    # j is zero-indexed! (yeah yeah I know this doesn't match the pset writeup)

        #### TODO ####
        # compute h, Hx (you may find the skeleton for computing Hx below useful)

        Hx = np.zeros((2,self.x.size))
        Hx[:,:3] = FILLMEIN
        # First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
        if j > 1:
            Hx[0, 3+2*j] = FILLMEIN
            Hx[1, 3+2*j] = FILLMEIN
            Hx[0, 3+2*j+1] = FILLMEIN
            Hx[1, 3+2*j+1] = FILLMEIN

        ##############
        
        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Adapt this method from Localization_EKF.associate_measurements.
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        ##############

        return v_list, R_list, H_list
