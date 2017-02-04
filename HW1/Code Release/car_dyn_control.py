import numpy as np
from scipy import linalg

def car_dyn(x, t, ctrl, noise):

    u_0 = ctrl[0] + noise[0]
    u_1 = ctrl[1] + noise[1]

    dxdt = [u_0*np.cos(x[2]), u_0*np.sin(x[2]), u_1]

    return dxdt

def ctrl_traj(x,y,th,dyn_state,ctrl_prev,x_d,y_d,xd_d,yd_d,xdd_d,ydd_d,x_g,y_g,th_g):
    #(x,y,th): current state
    #dyn_state: compensator internal dynamic state
    #ctrl_prev: previous control input (V,om)
    #(xd_d, yd_d): desired Velocity
    #(xdd_d, ydd_d): desired acceleration
    #(x_g,y_g,th_g): desired final state

    # Timestep
    dt = 0.005

    # Gains
    kpx = 7.0
    kpy = 7.0
    kdx = 9.0
    kdy = 9.0

    #Code trajectory controller (Switch to pose controller once "close" enough)
    PROX_LIMIT = 0.1

    #Define control inputs (V,om) - without saturation constraints
    th = wrapToPi(th)
    xDot = ctrl_prev[0]*np.cos(th) # current x velocity
    yDot = ctrl_prev[0]*np.sin(th) # current y velocit

    u1 = xdd_d + kpx*(x_d - x) + kdx*(xd_d - xDot)
    u2 = ydd_d + kpy*(y_d - y) + kdy*(yd_d - yDot)


    V = max(0.1, dyn_state) # ( eta )
    om = (u2*np.cos(th) - u1*np.sin(th))/V

    #Define accel = dV/dt
    accel = u1*np.cos(th) + u2*np.sin(th) # etaDot

    #Integrate dynamic state with anti-windup
    if (V > 0.5 or om > 1.0) and (accel > 0.0): #integration will only make things worse
        if (V > 0.5): dyn_state_up = 0.5 #cap-off integrator at max
        else: dyn_state_up = dyn_state #or just freeze integration
    else:
        dyn_state_up = dyn_state + accel*dt # dyn_state is the velocity

    if np.sqrt((x - x_g)**2 + (y - y_g)**2) <= PROX_LIMIT:
        ctrl_feedback = ctrl_pose(x,y,th,x_g,y_g,th_g)
        return np.array([ctrl_feedback[0],ctrl_feedback[1],dyn_state_up]) 

    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    # DYNAMIC COMPENSATOR

    return np.array([V, om, dyn_state_up])

def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi

def ctrl_pose (x,y,th,x_g,y_g,th_g):
    #(x,y,th): current state
    #(x_g,y_g,th-g): desired final state

    #Code pose controller
    #...FILL...#
    Dx = x_g - x
    Dy = y_g - y
    rho = np.sqrt(Dx**2 + Dy**2)
    alpha = wrapToPi(np.arctan2(Dy,Dx) - th)
    delta = wrapToPi(th + alpha - th_g)

    #Define control inputs (V,om) - without saturation constraints
    k1 = 0.8 / np.pi
    k2 = 1.25 / np.pi
    k3 = 1.0

    V = k1 * rho * np.cos(alpha)
    om = k2 * alpha + k1 * (np.sin(alpha) * np.cos(alpha) / alpha) * ( alpha + k3 * delta)

    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
