N = 1000
dt = 0.01
m = 1.4
g = 9.81
I = 0.02
Lmax = 0.32

body_l = 0.38
body_h = 0.03

maxiter = 5
eps_f = 1e-10

phi_r = 1000.0
phi_l = 10.0
phi_th = 100.0
phi_k = 100.0
L_r = phi_r / 10.0
L_l = phi_l
L_th = phi_th
L_k = phi_k
L_p = 100.0
foot_air_tol = 0.1
psi = 0.1
eps_contact = 1e-3

dim_x = 14

dim_x_fqp = 10
dim_dyn_fqp = 6
dim_fric_fqp = 6
dim_kin_fqp = 8

dim_x_cqp = 8
dim_dyn_cqp = 3
dim_loc_cqp = 4
dim_kin_cqp = 8

osqp_settings = {}
osqp_settings["verbose"] = False
osqp_settings["eps_abs"] = 1e-7
osqp_settings["eps_rel"] = 1e-7
osqp_settings["eps_prim_inf"] = 1e-6
osqp_settings["eps_dual_inf"] = 1e-6
osqp_settings["polish"] = True
osqp_settings["scaled_termination"] = True
osqp_settings["adaptive_rho"] = True
osqp_settings["check_termination"] = 50
