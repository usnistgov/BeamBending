import numpy as np
import matplotlib.pyplot as plt
import types # To check for integer/float or lambda function
from scipy.integrate import solve_ivp
from scipy import integrate

#%% Stress and energy functions


# Calculate minimal radius by tensile strength and load
def tensile_radius(m, g, n, sigma, S_f):
    r = np.sqrt((m * g) / (np.pi * sigma / S_f)) / 1000 / np.sqrt(n)
    return r

# def elastic_energy_torsion(value, T, s):
#     r = value[1]
#     E = value[2]
#     nu = value[3]
#     M = T
#     I_P = np.pi * r**4 / 2 # Polar area moment of inertia used for torsion
#     G = E / (2 * (1 + nu)) # Shear modulus in Pa
#     dV_el = 1/(2 * G * I_P) * M**2
#     V_el = integrate.cumtrapz(dV_el, s, initial=0)
#     V_el_end = V_el[-1]
#     K_el = V_el_end * 2 / phi**2 # Nm/rad
#     return V_el, V_el_end, K_el
# 
# def elastic_energy_bending(dictionary, M, theta, z, x):
#     s_prev = 0 # Last absolute s value from previous iteration
#     for key, value in dictionary.items():
#         s_1 = s_prev  # First item of the first list
#         s_2 = s_prev + value[0]
#         s = s_lin(s_1, s_2)
#         s_span = (s[0], s[-1]) # Span vector for initial and last s value
#         if is_lambda_function(value[1]):  # Variable X-section
#             func = value[1]
#             if key.endswith("_sym"): # Reverse the arrays of symmetric elements after calculation
#                 r = func(s - s_span[0])
#             else: 
#                 r = func(s - s_span[-1])
#         elif is_float_or_int(value[1]):  # Constant X-section
#             r = value[1]
#         else:
#             print("Error")
#             return None
#     E = value[2]
#     I_B = np.pi * r**4 / 4 # Polar area moment of inertia used for bending
#     dV_el = 1/(2 * E * I_B) * M**2
#     V_el = integrate.cumtrapz(dV_el, s, initial=0)
#     V_el_end = V_el[-1]
#     K_el = V_el_end * 2 / theta**2 # Nm/rad
#     s_prev = s_2
#     return V_el, V_el_end, K_el

#%% Other functions

def derivative(x, y):
    #dy = np.diff(y, 1)
    #dx = np.diff(x, 1)
    yfirst = (y[1:]-y[0:-1])/(x[1:]-x[0:-1])
    #yfirst = dy / dx
    xfirst = 0.5 * (x[:-1] + x[1:])
    return xfirst, yfirst

def s_lin(s_start, s_end, ds = 1001): # s is the variable along the neutral line
    return np.linspace(s_start, s_end, ds)

def is_float_or_int(value):
    return isinstance(value, (float, int))

def is_lambda_function(value):
    return isinstance(value, types.LambdaType)

def iterate_segments(dictionary):
    # Initialize arrays
    s_arr = np.array([])
    r_arr = np.array([])
    E_arr = np.array([])
    nu_arr = np.array([])
    rho_arr = np.array([])
    alpha_arr = np.array([])
    beta_arr = np.array([])
    i = 0
    s_prev = 0 # Last absolute s value from previous iteration
    
    # Iterate through the segments_dict dictionary
    for key, value in dictionary.items():
        if i == 0:  # First element has to have s_start @ 0
            s_1 = 0
            s_2 = s_prev + value[0]
            s = s_lin(s_1, s_2)
            s_arr = np.concatenate((s_arr, s))  # Concatenate s to s_arr
        if i > 0:   # Ensure there are at least two segments for comparison
            s_1 = s_prev  # First item of the first list
            s_2 = s_prev + value[0]
            s = s_lin(s_1, s_2)[1:] # [1:] so that not the same value is in the final array twice --> avoids problem in sampling s_arr 
            s_arr = np.concatenate((s_arr, s))  # Concatenate s to s_arr
        
        if is_lambda_function(value[1]):  # Variable X-section
            func = value[1]
            if key.endswith("_sym"): # Reverse the arrays of symmetric elements after calculation
                _s_end = s[0]
            else: 
                _s_end = s[-1]         
            _s = s - _s_end
            _r_arr = np.array([])
            for j in _s: # Iterate through array to evaluate lambda function at individual values and not a whole array
                _r = func(j)
                _r_arr = np.concatenate((_r_arr, [_r]))  # Concatenate r to r_arr
        elif is_float_or_int(value[1]):  # Constant X-section
            _r_arr = np.full(len(s), value[1])
        else:
            print("Error")
            return None
        r_arr = np.concatenate((r_arr, _r_arr))  # Concatenate r to r_arr
        E_arr = np.concatenate((E_arr, np.full(len(s), value[2])))
        nu_arr = np.concatenate((nu_arr, np.full(len(s), value[3])))
        rho_arr = np.concatenate((rho_arr, np.full(len(s), value[4])))
        alpha_arr = np.concatenate((alpha_arr, np.full(len(s), value[5])))
        beta_arr = np.concatenate((beta_arr, np.full(len(s), value[6])))
        i += 1
        s_prev = s_2
        
    return s_arr, r_arr, E_arr, nu_arr, rho_arr, alpha_arr, beta_arr, i


#%% Segments: We assume a base segment, some sort of connection to the fiber (catalysis bonding or laser welding), a fiber (or n fibers) which build up the torsion spring -- the fibers can have separate segments
# The deformation problem is totally symmetric so that half the pendulum only needs to be modeled in some instances ...
# Common stress in fibers can be established by birefringence analysis in fused silica??

# LIGO usually does accurate modeling based on FEA. I think we can do just as accurate with this script and be much simpler and faster with results and parametric studies
# Including all the bonding or welding loss that other analytical models do not have. We should write a paper about this and do it with Glasgow on actual LIGO Q data for individual modes.

l_conn_base = 60e-9
R_conn_base = 1.5e-3
R_neck = 1e-3
l_neck = 6.35e-3
r_neck = 0.02e-3

# Segments have to have prefix 'seg_' to work with the following code, the order in which the segments are written matters for the following
# seg_base         = [6.35e-3, 12.7e-3, 72e9, 0.17, 2200, 1, 1] # Each segment has the same setup: l, r, E, nu, rho, alpha, beta
# seg_connect_base = [l_conn_base, R_conn_base, 7.9e9, 0.17, 2200, 1, 1] # Catalysis bond
# seg_neck         = [l_neck, lambda s: (R_neck - np.sqrt(R_neck**2 - (R_neck**2 / (l_neck + l_neck * 0.0001)**2) * (s)**2) + r_neck), 72e9, 0.17, 2200, 1, 1]
# seg_connect_neck = [2e-3, r_neck, 72e9, 0.17, 2200, 1, 1] # Laser weld
# seg_fiber_1      = [2e-3, 0.02e-3, 72e9, 0.17, 2200, 1, 1] # Can be thermoelastic compensation segment
seg_fiber_2 = [5e-3, 0.01e-3, 72e9, 0.17, 2200, 1, 1] # Actual torsion segment

# # Symmetric design, almost except for bob, symmetric parts need suffix "_sym"
# seg_fiber_2_sym = seg_fiber_2
# seg_fiber_1_sym = seg_fiber_1
# seg_connect_neck_sym = seg_connect_neck
# seg_neck_sym = [l_neck, lambda s: (R_neck - np.sqrt(R_neck**2 - (R_neck**2 / (l_neck + l_neck * 0.0001)**2) * (s)**2) + r_neck), 72e9, 0.17, 2200, 1, 1]
# seg_connect_base_sym = seg_connect_base
# seg_bob_sym = [6.35e-3, 50e-3, 72e9, 0.17, 2200, 1, 1] # Bob

# Get local variables
local_vars = locals()
# Filter out lists with names starting with "seg_"
segments_dict = {key: value for key, value in local_vars.items() if isinstance(value, list) and key.startswith("seg_")}
print(segments_dict)

s_arr, r_arr, E_arr, nu_arr, rho_arr, alpha_arr, beta_arr, no_of_segments = iterate_segments(segments_dict)
# print(r_arr)

#%% Plotting stuff

# To plot all sections of the fiber so that each section has an equal relative resolution, a relative plot has to be done

s_abs = np.linspace(0, no_of_segments, len(s_arr))

fig, ax = plt.subplots(1)

ax.plot(s_arr, r_arr, 'b')
ax.set_ylabel('radius')
ax.set_xlabel('s /m')
# ax.plot(r_arr * 1e3, -s_abs, 'b')

# ax.plot(np.diff(s_arr), np.diff(r_arr))

# ax.plot(np.diff(s_arr))
# ax.set_ylim(0, 1e-8)

#%% Start the differential equations

def torsion_equations(s, y, key, value, s_span, T_0): # Dictionary is dictionary of segments
    if is_lambda_function(value[1]):  # Variable X-section
        func = value[1]
        if key.endswith("_sym"): # Reverse the arrays of symmetric elements after calculation
            r = func(s - s_span[0])
        else: 
            r = func(s - s_span[-1])
    elif is_float_or_int(value[1]):  # Constant X-section
        r = value[1]
    else:
        print("Error")
        return None
    I_P = np.pi * r**4 / 2 # Polar moment of area used for torsion of fiber
    E = value[2]
    nu = value[3]
    G = E / (2 * (1 + nu)) # Shear modulus in Pa
    
    # Derivative of phi with respect to s (phi_prime)
    phi_prime = T_0 / (G * I_P)
    
    # Derivative of torque with respect to s (T_prime)
    # T_prime = 0  # Torque is assumed to be constant
    
    return [phi_prime]  #,T_prime]

def solving_torsion(func, s, y_0, dictionary, T_0): # s here is unnecessary
    sol = None  # Initialize the solution object
    i = 0
    s_prev = 0 # Last absolute s value from previous iteration
    y = y_0 # Initial condition for the first segment
    I_P_list = []
    G_list = []
    s_list = []
    r_list = []
    for key, value in dictionary.items():
        s_1 = s_prev  # First item of the first list
        s_2 = s_prev + value[0]
        s = s_lin(s_1, s_2)
        s_span = (s[0], s[-1]) # Span vector for initial and last s value
        sol_temp = solve_ivp(func, s_span, y, t_eval=s, method='RK45', args=(key, value, s_span, T_0), atol=1e-18, rtol=1e-13) # solve initial value problem for phi
        E = value[2]
        nu = value[3]
        _G = E / (2 * (1 + nu))
        G = np.full(len(s), _G)
        if is_lambda_function(value[1]):  # Variable X-section
            fun = value[1]
            if key.endswith("_sym"): # Reverse the arrays of symmetric elements after calculation
                r = fun(s - s_span[0])
            else: 
                r = fun(s - s_span[-1])
        elif is_float_or_int(value[1]):  # Constant X-section
            r = np.full(len(s), value[1])
        else:
            print("Error")
            return None
        I_P = np.pi * r**4 / 2 # Polar area moment of inertia used for torsion
        if sol is None:
            sol = sol_temp
            G_list = G
            I_P_list = I_P
            s_list = s
            r_list = r
        else:
            sol.y = np.hstack((sol.y, sol_temp.y))  # Horizontally stack the solution arrays
            sol.t = np.hstack((sol.t, sol_temp.t))
            G_list = np.hstack((G_list, G))
            I_P_list = np.hstack((I_P_list, I_P))
            s_list = np.hstack((s_list, s))
            r_list = np.hstack((r_list, r))
        i += 1
        # print(i, s_span, y[0], sol_temp.y[0, 0])
        y = sol.y[:, -1]
        s_prev = s_2
    return sol.y, sol.t, G_list, I_P_list, s_list, r_list

# def newton_raphson_torsion(phi_end, func, s, dictionary, initial_guess = 1e-9, max_iterations=100, tolerance=1e-18):
#     """
#     Implementation of the Newton-Raphson method for finding the roots of a function.

#     Parameters:
#         func (callable): The function for which to find the root.
#         initial_guess (float): Initial guess for the root.
#         tolerance (float): Tolerance for convergence.
#         max_iterations (int): Maximum number of iterations allowed.

#     Returns:
#         float: Approximation of the root.
#     """
#     T_0 = initial_guess

#     for iteration in range(max_iterations):
#         y_0 = [0] # Initial condition for phi
#         sol_y, sol_t, G_list, I_P_list, s_list, r_list = solving_torsion(func, s, y_0, dictionary, T_0) # First guessed value for phi
#         phi = sol_y[0, :]
#         _phi = phi[-1] - phi_end # Necessary to subtract the end condition from the end value to find the root using Newton Raphson.
#         sol_s = sol_t
#         # Calculate the function value and its derivative at current guess
#         fx = _phi # In rad

#         sol_y, sol_t, G_list, I_P_list, s_list, r_list = solving_torsion(func, s, y_0, dictionary, T_0 + tolerance) # Second guessed value for phi to extract derivative
#         phi = sol_y[0, :]
#         _phi = phi[-1] - phi_end # Access the solution array from the result object
#         # Calculate the function value and its derivative at current guess
#         fx_tol = _phi # In rad
        
#         derivative = (fx_tol - fx) / tolerance # In rad / Nm

#         # Update the guess using Newton's method
#         T_0_next = T_0 - fx / derivative # In rad / ( rad / Nm ), i.e. in Nm
#         print('>>', iteration, T_0, phi[-1], T_0_next - T_0)

#         # Check for convergence
#         if abs(T_0_next - T_0) < tolerance:
#             dV_el = 1/(2 * G_list * I_P_list) * T_0**2
#             V_el = integrate.cumtrapz(dV_el, s_list, initial=0)  
#             return phi, T_0, sol_s, V_el, r_list
        
#         T_0 = T_0_next

#     raise ValueError("Newton-Raphson method did not converge within the maximum number of iterations.")

def shoot_torsion(phi_end, func, s, dictionary, low=-1, high=1, max_iters=100, tol=1e-9):
    i = 0
    while i <= max_iters:
        i += 1
        T_0 = np.mean([low, high])  # Calculate the mean of low and high and create a 1-dimensional array
        # y_0 = np.array([0, T_0]) # Boundary conditions at s = 0
        y_0 = [0]

        sol_y, sol_t, G_list, I_P_list, s_list, r_list = solving_torsion(func, s, y_0, dictionary, T_0)
        phi = sol_y[0, :] # Access the solution array from the result object
        # T = sol_y[1, :]
        sol_s = sol_t

        err = phi[-1] - phi_end
        abs_err = np.abs(err)

        if abs_err <= tol:
            break
        if err < 0:
            low = np.mean([low, high])
        else:
            high = np.mean([low, high])
    dV_el = 1/(2 * G_list * I_P_list) * T_0**2
    V_el = integrate.cumtrapz(dV_el, s_list, initial=0)   
    if i >= max_iters:
        print("Shooting Method FAILED")
    return phi, T_0, sol_s, V_el, r_list

import numpy as np
from scipy.integrate import solve_ivp

def bending_equations(s, y, key, value, s_span, tensile_load, shear_load):
    """
    Calculates one ds element at location s.
    
    s: length along flexure axis (array)
    y: y[0] moment at this element
       y[1] tangent of ds
    
    value: geometry and mechanical parameters as a function of s
    tensile_load: Fz (suspended mass)
    shear-load: FX(s=end) = -Fx(s=0)
    
    return:
        M_prime     : dM/ds (s) same length as s
        theta_prime : dtheta/ds (s)
    """
    
    if is_lambda_function(value[1]):  # Variable X-section
        func = value[1]
        if key.endswith("_sym"): # Reverse the arrays of symmetric elements after calculation
            r = func(s - s_span[0])
        else: 
            r = func(s - s_span[-1])
    elif is_float_or_int(value[1]):  # Constant X-section
        r = value[1]
    else:
        print("Error")
        return None
    
    I_B = np.pi * r**4 / 4  # Moment of area used for bending of fiber
    E = value[2]
    Fx = tensile_load
    Fy = shear_load
    
    # M_prime = -Fx * np.sin(y[1]) + Fy * np.cos(y[1])  # Bending moment
    M_prime = Fx * np.sin(y[1]) - Fy * np.cos(y[1])  # Bending moment
    theta_prime = y[0] / (E * I_B)
    
    return [M_prime, theta_prime]

def bending_jacobian(s, y, key, value, s_span, tensile_load, shear_load):
    """
    Computes the Jacobian matrix of the bending equations.
    
    s: length along flexure axis
    y: array containing the dependent variables (M_prime, theta_prime)
    key: key identifier
    value: geometry and mechanical parameters as a function of s
    s_span: span vector for initial and last s value
    tensile_load: Fz (suspended mass)
    shear-load: FX(s=end) = -Fx(s=0)
    
    return:
        jacobian: the Jacobian matrix
    """
    # Extracting parameters
    if is_lambda_function(value[1]):  
        func = value[1]
        if key.endswith("_sym"):
            r = func(s - s_span[0])
        else: 
            r = func(s - s_span[-1])
    elif is_float_or_int(value[1]):  
        r = value[1]
    else:
        print("Error")
        return None
    
    I_B = np.pi * r**4 / 4  # Moment of area used for bending of fiber
    E = value[2]
    Fx = tensile_load
    Fy = shear_load
    
    # Computing the elements of the Jacobian matrix
    dM_dM = 0
    dM_dtheta = Fx * np.cos(y[1]) + Fy * np.sin(y[1])
    dtheta_dM = 1 / (E * I_B)
    dtheta_dtheta = 0
    
    # Assembling the Jacobian matrix
    jacobian = np.array([[dM_dM, dM_dtheta], [dtheta_dM, dtheta_dtheta]])
    
    return jacobian

def solving_bending(func, s, y_0, dictionary, tensile_load, shear_load):
    sol = None  # Initialize the solution object
    i = 0
    s_prev = 0 # Last absolute s value from previous iteration
    y = y_0 # Initial condition for the first segment
    I_B_list = []
    E_list = []
    s_list = []
    r_list = []
    
    for key, value in dictionary.items():
        s_1 = s_prev  # First item of the first list
        s_2 = s_prev + value[0]
        s = s_lin(s_1, s_2)
        s_span = (s[0], s[-1]) # Span vector for initial and last s value
        first_step = (max(s) - min(s)) / len(s) /100
        max_step = (max(s) - min(s)) / len(s) /100
        sol_temp = solve_ivp(func, s_span, y, t_eval=s, method='RK45',\
                     args=(key, value, s_span, tensile_load, shear_load,),\
                     jac=bending_jacobian, atol=1e-6, rtol=1e-3,\
                      first_step=first_step, max_step=max_step)
        
        E = np.full(len(s), value[2])
        if is_lambda_function(value[1]):  # Variable X-section
            fun = value[1]
            if key.endswith("_sym"): # Reverse the arrays of symmetric elements after calculation
                r = fun(s - s_span[0])
            else: 
                r = fun(s - s_span[-1])
        elif is_float_or_int(value[1]):  # Constant X-section
            r = np.full(len(s), value[1])
        else:
            print("Error")
            return None
        I_B = np.pi * r**4 / 4 # Polar area moment of inertia used for bending

        if sol is None:
            sol = sol_temp
            E_list = E
            I_B_list = I_B
            s_list = s
            r_list = r
        else:
            sol.y = np.hstack((sol.y, sol_temp.y))  # Horizontally stack the solution arrays
            sol.t = np.hstack((sol.t, sol_temp.t))
            E_list = np.hstack((E_list, E))
            I_B_list = np.hstack((I_B_list, I_B))
            s_list = np.hstack((s_list, s))
            r_list = np.hstack((r_list, r))

        i += 1
        y = sol.y[:, -1]
        s_prev = s_2
    return sol.y, sol.t, E_list, I_B_list, s_list, r_list


def newton_raphson_bending(x_end, func, s, dictionary, tensile_load, initial_guess = 0, max_iterations=100, tolerance=1e-18, Tolerance = 1e-13):
    """
    Implementation of the Newton-Raphson method for finding the roots of a function.

    Parameters:
        func (callable): The function for which to find the root.
        initial_guess (float): Initial guess for the root.
        tolerance (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations allowed.

    Returns:
        float: Approximation of the root.
    """
    
    shear_load = tensile_load # tensile_load/100000000 #tensile_load * x_end / 1e-3 / 100 # Assuming a 100 mm long fiber for the start and safety factor of 100
    
    for Iteration in range(max_iterations):
        
        M_0 = initial_guess
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_ylabel('M/Nm')
        ax[1].set_ylabel('theta/rad')
        ax[1].set_xlabel('s/m')
        for iteration in range(max_iterations):
            y_0 = np.array([M_0, 0]) # Boundary conditions at s = 0
            sol_y, sol_t, E_list, I_B_list, s_list, r_list = \
                solving_bending(func, s, y_0, dictionary, tensile_load, shear_load)
            M = sol_y[0, :]
            theta = sol_y[1, :]
            # z = sol_y[2, :]
            # x = sol_y[3, :]
            sol_s = sol_t
            _theta = theta[-1] # We are trying to null that angle
            # Calculate the function value and its derivative at current guess
            fx = _theta
            if iteration >= 0:
                ax[0].plot(sol_s, M)
                ax[1].plot(sol_s, theta)
    
            y_0 = np.array([M_0 + tolerance, 0]) # Boundary conditions at s = 0
            sol_y, sol_t, E_list, I_B_list, s_list, r_list = \
                solving_bending(func, s, y_0, dictionary, tensile_load, shear_load)
            theta = sol_y[1, :]
            _theta = theta[-1] # We are trying to null that angle
            # Calculate the function value and its derivative at current guess
            fx_tol = _theta
            
            derivative = (fx_tol - fx) / (tolerance)
            # Update the guess using Newton's method
            M_0_next = M_0 - fx / derivative
            print('>', 'M_0:', M_0, 'fx:', fx, 'fx_tol:', fx_tol, 'derivative:', derivative, 'iteration:', iteration, 'shear_load:', shear_load, 'tensile_load:', tensile_load, M_0 / (72e9 * np.pi * 0.01e-3**4 / 4))
    
            # Check for convergence
            if abs(M_0_next - M_0) < tolerance:
                print('done:', M_0_next - M_0)
                break
            
            M_0 = M_0_next
        
        x = integrate.cumtrapz(np.sin(theta), sol_s, initial=0)  
        z = integrate.cumtrapz(np.cos(theta), sol_s, initial=0)  
        _x = x[-1] - x_end # We are trying to make that displacement equal to x_end thus need to subtract x_end to find the roots
        Fxx = _x
            
        M_0 = initial_guess
        for iteration in range(max_iterations):
            y_0 = np.array([M_0, 0]) # Boundary conditions at s = 0
            sol_y, sol_t, E_list, I_B_list, s_list, r_list = \
                solving_bending(func, s, y_0, dictionary, tensile_load, shear_load + tolerance)
            M = sol_y[0, :]
            theta = sol_y[1, :]
            # z = sol_y[2, :]
            # x = sol_y[3, :]
            sol_s = sol_t
            _theta = theta[-1] # We are trying to null that angle
            # Calculate the function value and its derivative at current guess
            fx = _theta
    
    
            y_0 = np.array([M_0 + tolerance, 0]) # Boundary conditions at s = 0
            sol_y, sol_t, E_list, I_B_list, s_list, r_list = \
                solving_bending(func, s, y_0, dictionary, tensile_load, shear_load + tolerance)
            theta = sol_y[1, :]
            _theta = theta[-1] # We are trying to null that angle
            # Calculate the function value and its derivative at current guess
            fx_tol = _theta
            
            derivative = (fx_tol - fx) / tolerance
    
            # Update the guess using Newton's method
            M_0_next = M_0 - fx / derivative
            print('>>', 'M_0:', M_0, 'fx:', fx, 'fx_tol:', fx_tol, 'derivative:', derivative, 'iteration:', iteration, 'shear_load:', shear_load, 'tensile_load:', tensile_load)
    
            # Check for convergence
            if abs(M_0_next - M_0) < tolerance:
                break
            
            M_0 = M_0_next
        
        x = integrate.cumtrapz(np.sin(theta), sol_s, initial=0)  
        _x = x[-1] - x_end # We are trying to make that displacement equal to x_end thus need to subtract x_end to find the roots
        Fx_tol = _x
        
        Derivative = (Fx_tol - Fxx) / Tolerance

        # Update the guess using Newton's method
        shear_load_next = shear_load - Fxx / Derivative
        print('>>>', shear_load, Fxx, Iteration)

        # Check for convergence
        if abs(shear_load_next - shear_load) < Tolerance:
            dV_el = 1/(2 * E_list * I_B_list) * M**2
            V_el = integrate.cumtrapz(dV_el, s_list, initial=0)  
            return M, theta, z, x, sol_s, V_el, E_list, r_list
        
        shear_load = shear_load_next

    raise ValueError("Newton-Raphson method did not converge within the maximum number of iterations.")

# def shoot_bending(x_end, func, s, dictionary, tensile_load, Low=-1e-10, High=1e-10,\
#                   lowS=-1e-6, highS=1e-6, max_iters=100, tol=1e-6):
#     j = 0
#     while j <= max_iters:
#         j += 1
#         shear_load = np.mean([Low, High])
#         i = 0
#         low = lowS
#         high = highS
#         while i <= max_iters:
#             i += 1
#             M_0 = np.mean([low, high])  # Calculate the mean of low and high and create a 1-dimensional array
#             y_0 = np.array([M_0, 0]) # Boundary conditions at s = 0
    
#             sol_y, sol_t, E_list, I_B_list, s_list, r_list = \
#                 solving_bending(func, s, y_0, dictionary, tensile_load, shear_load)

#             M = sol_y[0, :]
#             theta = sol_y[1, :]
#             # z = sol_y[2, :]
#             # x = sol_y[3, :]
#             sol_s = sol_t
#             err_inner = theta[-1] # We want theta[-1] = 0 for parallelogram boundary conditions
#             abs_err_inner = np.abs(err_inner)
            
#             print('>>', theta[-1], M_0, low, high, i, j, shear_load)
#             # print(err_inner, M_0)
            
#             # print(abs_err, T[-1], phi[-1])
            
#             if abs_err_inner > 0.05 and i > 1:
#                 low, high = low * 0.1, high * 0.1
#             else:
#                 if abs_err_inner <= 1e-6: # Maybe make this a relative comparison??
#                     break
#                 if err_inner < 0:
#                     low = np.mean([low, high])
#                 else:
#                     high = np.mean([low, high])
#                 if i >= max_iters:
#                     print("Shooting Method FAILED due to inner loop")
#                     break
#         x = integrate.cumtrapz(np.sin(theta), sol_s, initial=0)  
#         err_outer = x[-1] - x_end
#         abs_err_outer = np.abs(err_outer)
#         print('>', x_end, x[-1], err_inner, shear_load, M_0, i, j, low, high, Low, High, max(abs(theta)))
        
#         # print(abs_err, T[-1], phi[-1])

#         if abs_err_outer <= tol: # Maybe make this a relative comparison??
#             break
#         if err_outer < 0:
#             Low = np.mean([Low, High])
#         else:
#             High = np.mean([Low, High])
#         if j >= max_iters:
#             print("Shooting Method FAILED due to outer loop")
#             break
        
#     z = integrate.cumtrapz(np.cos(theta), sol_s, initial=0)  
#     dV_el = 1/(2 * E_list * I_B_list) * M**2
#     V_el = integrate.cumtrapz(dV_el, s_list, initial=0)  

#     return M, theta, z, x, sol_s, V_el, E_list, r_list


#%% Loss functions

def thermoelastic_loss(omega, E, sigma0, alpha=0.55e-6, beta=1.52e-4, C=740, rho=2200, T=293, bending=True):
    losses = []
    for E_val, sigma0_val in zip(E, sigma0):
        if bending:
            loss = (E_val * T) / (rho * C) * (alpha - sigma0_val * beta / E_val)**2 * (omega * T) / (1 + (omega * T)**2)
        else:
            loss = 0
        losses.append(loss)
    return losses

def tensile_stress(m, r, n = 1, g = 9.81): # n is the number of fibers. Here is 1 for now.
    return [m * g / (np.pi * radius**2 * n) for radius in r]

def bulk_loss(f, exp = 0.77, C2 = 1.18e-11): # C2 for Suprasil 2 (assumed to be accurate for Suprasil 312, too) empirically determined by Penn S D et al 2006 Frequency and surface dependence of the mechanical loss in fused silica Phys. Lett. A 352 3–6
    loss = C2 * f**exp
    return loss

def surface_loss(r, h_phi_s = 4e-12): # h_phi_s for fused silica laser polished approx. 4e-12 m from Heptonstall. Note that this is different for the weld as shown in Cumming 2020
    loss = 8.53 * h_phi_s / (2 * r)
    return loss

def total_loss_bending(thermoelastic_loss, bulk_loss, surface_loss, V_el):
    bulk_loss_list = np.full(len(V_el), bulk_loss) # Bulk loss is only one value and has to be made a list with len(V_el) many values for the integration
    V_el_diff = [0] + np.diff(V_el) # Elastic energy in each element of s, V_el is energy integral along s
    weighted_bulk_loss_list = [x / V_el[-1] for x in [x * y for x, y in zip(bulk_loss_list, V_el_diff)]] # Summation according to Cumming 2020 or many other Glasgow papers.
    weighted_surface_loss_list = [x / V_el[-1] for x in [x * y for x, y in zip(surface_loss, V_el_diff)]]
    weighted_thermoelastic_loss_list = [x / V_el[-1] for x in [x * y for x, y in zip(thermoelastic_loss, V_el_diff)]]
    loss_list = [x + y + z for x, y, z in zip(weighted_bulk_loss_list, weighted_surface_loss_list, weighted_thermoelastic_loss_list)]

    loss = sum(loss_list) # Sum of all loss elements in the list along the fiber IS SUMMING CORRECT OR DO I NEED INTEGRAL???
    return loss, loss_list, weighted_bulk_loss_list, weighted_surface_loss_list, weighted_thermoelastic_loss_list

def total_loss_torsion(bulk_loss, surface_loss, V_el):
    bulk_loss_list = np.full(len(V_el), bulk_loss) # Bulk loss is only one value and has to be made a list with len(V_el) many values for the integration
    V_el_diff = [0] + np.diff(V_el) # Elastic energy in each element of s, V_el is energy integral along s
    weighted_bulk_loss_list = [x / V_el[-1] for x in [x * y for x, y in zip(bulk_loss_list, V_el_diff)]] # Summation according to Cumming 2020 or many other Glasgow papers.
    weighted_surface_loss_list = [x / V_el[-1] for x in [x * y for x, y in zip(surface_loss, V_el_diff)]]
    loss_list = [x + y for x, y in zip(weighted_bulk_loss_list, weighted_surface_loss_list)]

    loss = sum(loss_list) # Sum of all loss elements in the list along the fiber
    return loss, loss_list, weighted_bulk_loss_list, weighted_surface_loss_list

#%% Use the shooting function to solve for phi

_phi = 1 * np.pi / 180 # Torsion angle at end of the pendulum in rad
print(_phi)

f = 0.01 # pendulum frequency in Hertz
omega = 2 * np.pi * f

# phi, T, _, V_el, r = shoot_torsion(phi_end = _phi, func = torsion_equations, s = s_arr, dictionary = segments_dict)
# print(phi)
# torsion_bulk_loss = bulk_loss(f = f)
# torsion_surface_loss = surface_loss(r = r)
# torsion_total_loss, torsion_total_loss_list, weighted_bulk_loss, weighted_surface_loss  = total_loss_torsion(bulk_loss = torsion_bulk_loss, surface_loss = torsion_surface_loss, V_el = V_el)

# fig, ax = plt.subplots(1)

# ax.plot(weighted_bulk_loss, label = 'weighted_bulk_loss')
# ax.plot(weighted_surface_loss, label = 'weighted_surface_loss')
# ax.plot(torsion_total_loss_list, label = 'torsion_total_loss')
# ax.legend()
# V_el_torsion = V_el

# print(_phi - phi[-1], torsion_total_loss)

r_n = 1e-3 # Radius in meter of fiber separation in diluted system
x_end = r_n * _phi # Scaling law for horizontal displacement of fiber end at given torsion angle phi due to r_n for dilution
print(x_end)
m = 0.01 # Mass per fiber, not full mass of the bob
g = 9.81
tensile_load = m * g
print(tensile_load)

M, theta, z, x, _, V_el, E, r = newton_raphson_bending(x_end = x_end,\
                                              func = bending_equations,\
                                            s = s_arr,\
                                        dictionary = segments_dict,\
                                            tensile_load = tensile_load)
print('M', M[0], M[-1])

#%% Tryout torsion

fig, ax = plt.subplots(4)

ax[0].plot(_,theta,'r-')
ax[1].plot(_,M,'b-')
ax[2].plot(_,x,'b-')
ax[3].plot(z,x,'b-')

print(type(r))

 #%% Tryout bending

sigma = tensile_stress(m = m, r = r)
bending_thermoelastic_loss = thermoelastic_loss(omega = omega, E = E, sigma0 = sigma)
bending_bulk_loss = bulk_loss(f = f)
bending_surface_loss = surface_loss(r = r)
bending_total_loss, bending_total_loss_list, weighted_bulk_loss, weighted_surface_loss, weighted_thermoelastic_loss  = total_loss_bending(thermoelastic_loss = bending_thermoelastic_loss, bulk_loss = bending_bulk_loss, surface_loss = bending_surface_loss, V_el = V_el)

print(phi, np.max(theta), np.max(V_el))
# fig, ax = plt.subplots(5)

# ax[0].plot(x)
# ax[1].plot(z)
# ax[2].plot(theta)
# ax[3].plot(M)
# ax[4].plot(V_el)

fig, ax = plt.subplots(1)

ax.plot(weighted_thermoelastic_loss, label = 'weighted_thermoelastic_loss')
ax.plot(weighted_bulk_loss, label = 'weighted_bulk_loss')
ax.plot(weighted_surface_loss, label = 'weighted_surface_loss')
ax.plot(bending_total_loss_list, label = 'bending_total_loss')
ax.legend()

print(len(V_el), max(bending_thermoelastic_loss), bending_total_loss, "Bending energy total:", V_el[-1])
print(torsion_bulk_loss, torsion_total_loss, sum(torsion_surface_loss), "Torsion energy total:", V_el_torsion[-1])

##################  I NEED TO NORMALIZE ENERGIES TO THE TOTAL ENERGY IN THE SYSTEM NOT JUST THE ENERGY FROM TORSION AND BENDING, IT IS WRONG AS IT IS 05092024

#%% Energies



# #%% Tryout

# fig, ax = plt.subplots(1)
# x, y = derivative(_, theta)
# # ax.plot(np.diff(_), np.diff(phi))
# ax.plot(x, y, '.')
# # ax.plot(np.diff(_)
# # Filter the data to keep only points where y is 0
# x_filtered = x[y == 0]
# y_filtered = y[y == 0]
# # Plot the filtered points
# ax.plot(x_filtered, y_filtered, 'o', label='y = 0')
# ax.set_xlabel('s/m')
# ax.set_ylabel('dphi/rad')


# fig, ax = plt.subplots(1)
# # x, y = derivative(_, phi)
# # ax.plot(x, y)
# ax.plot(_, theta,'b.')
# # ax.plot(_, np.log10(-phi),'r.')
# ax.set_xlabel('s/m')
# ax.set_ylabel('phi/rad')




# #%% Tryout

# i=0
# s_prev=0
# allx=[]
# allz=[]
# alltheta=[]
# allM=[]
# alls=[]
# for key, value in segments_dict.items():
#     s_1 = s_prev  # First item of the first list
#     s_2 = s_prev + value[0]
#     s = s_lin(s_1, s_2)
#     s_span = (s[0], s[-1]) # Span vector for initial and last s value
#     for ss in s:
#         _x, _z, _theta, _M =  bending_equations(ss, [M[0], 0, 0, 0], key, value, s, tensile_load, shear_load) # s, y, key, value, s_span, tensile_load, shear_load
#         allx.append(_x)
#         allz.append(_z)
#         alltheta.append(_theta)
#         allM.append(_M)
#         alls = np.r_[alls,ss]
#     i=i+1
#     s_prev = s_2
# print(len(s))
# fig, ax = plt.subplots(1)
# ax.plot(alls,alltheta)
# ax.set_xlabel('s/m')
# ax.set_ylabel('dphi/rad')

















































































































