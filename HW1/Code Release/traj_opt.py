import numpy as np
import math
from scipy import linalg
import scikits.bvp_solver
import matplotlib.pyplot as plt


def q1_ode_fun(tau, y):

    #Code in the BVP ODEs
    # y = [x, y, th, p1, p2, p3, r]
    X = y[0]
    Y = y[1]
    p1 = y[3]
    p2 = y[4]
    p3 = y[5]
    r = y[6]

    V = -0.5*(y[3] * np.cos(y[2]) + y[4] * np.sin(y[2]))
    om = -0.5*y[5]

    x_dot = y[6]*V*np.cos(y[2])
    y_dot = y[6]*V*np.sin(y[2])
    TH_dot = y[6]*om
    p1_dot = 0
    p2_dot = 0
    p3_dot = y[6]*y[3]*V*np.sin(y[2]) - y[6]*y[4]*V*np.cos(y[2])
    r_dot = 0

    return np.hstack((x_dot,y_dot,TH_dot,p1_dot,p2_dot,p3_dot,r_dot))


def q1_bc_fun(ya, yb):

    #lambda
    # lambda_test = 0.245
    lambda_test = 0.250

    #goal pose
    x_g = 5
    y_g = 5
    th_g = -np.pi/2.0
    xf = [x_g, y_g, th_g]

    #initial pose
    x0 = [0, 0, -np.pi/2.0]

    #Code boundary condition residuals
    # y = [x, y, th, p1, p2, p3, r]
    Vb = -0.5*(yb[3] * np.cos(yb[2]) + yb[4] * np.sin(yb[2]))
    omb = -0.5*yb[5]

    # Left BCs:
    LeftBC = np.array([ya[0], ya[1], ya[2]+np.pi/2.0])

    # Free time constraint
    H_f = lambda_test + Vb**2 + omb**2 + yb[3]*Vb*np.cos(yb[2]) + yb[4]*Vb*np.sin(yb[2]) + yb[5]*omb

    RightBC = np.array([yb[0]-5, yb[1]-5, yb[2]+np.pi/2.0, H_f])
    
    return (LeftBC, RightBC)

#Define solver state: y = [x, y, th, ...? ]
# y = [x, y, th, p1, p2, p3, r]

problem = scikits.bvp_solver.ProblemDefinition(num_ODE=7, #Number of ODes
                                                num_parameters = 0, #Number of parameters                                                                                                                 
                                                num_left_boundary_conditions = 3, #Number of left BCs                                                                                                     
                                                boundary_points = (0,1), #Boundary points of independent coordinate                                                                                       
                                                function = q1_ode_fun, #ODE function                                                                                                                        
                                                boundary_conditions = q1_bc_fun) #BC function  
# y = [x, y, th, p1, p2, p3, r]

# soln = scikits.bvp_solver.solve(problem, solution_guess = (2.5,2.5,-np.pi/2+.05,1,1,1,1)) #
# soln = scikits.bvp_solver.solve(problem, solution_guess = (2.0,2.5,1.1,1,1,1,1)) #weird squiggle
# soln = scikits.bvp_solver.solve(problem, solution_guess = (2.5,2.0,-.9,-1,-1,1,7)) # NEAR OPTIMAL!!!
soln = scikits.bvp_solver.solve(problem, solution_guess = (2.5,2.0,-.9,-1,-1,1,7)) # NEAR OPTIMAL!!!

dt = 0.005

# Test if solution flips
y_0 = soln(0)
flip = 0
if y_0[-1] < 0:
    t_f = -y_0[-1]
    flip = 1
else:
    t_f = y_0[-1]

t = np.arange(0,t_f,dt)
y = soln(t/t_f)
if flip:
    y[3:7,:] = -y[3:7,:]
y = y.T # solution arranged column-wise

V = -0.5*(y[:,3] * np.cos(y[:,2]) + y[:,4] * np.sin(y[:,2]))
om = -0.5*y[:,5]

V = np.array([V]).T # Convert to 1D column matrices
om = np.array([om]).T
X = np.array([y[:,0]]).T
Y = np.array([y[:,1]]).T
TH = np.array([y[:,2]]).T
p1 = np.array([y[:,3]]).T
p2 = np.array([y[:,4]]).T
p3 = np.array([y[:,5]]).T

#Save Data
# data = (X,Y,TH,V,om)
# data =  np.hstack([X,Y,TH,p1,p2,p3,V,om])
data =  np.hstack([X,Y,TH,V,om])

np.save('traj_opt_data',data)

# Plots
plt.figure()
plt.plot(y[:,0], y[:,1],'k-',linewidth=2)
plt.quiver(y[1:-1:200,0],y[1:-1:200,1],np.cos(y[1:-1:200,2]),np.sin(y[1:-1:200,2]))
plt.grid('on')
plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
plt.xlabel('X'); plt.ylabel('Y')

plt.figure()
plt.plot(t, V,linewidth=2)
plt.plot(t, om,linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'],loc='center left', bbox_to_anchor=(1,0.5))

plt.show()
