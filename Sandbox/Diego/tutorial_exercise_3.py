# prior
v_p = 3
sigma_p = 1

# observation and its noise (standard deviation)
u = 2
sigma_u = 1

# initialise model nodes
phi = v_p
eps_p = 0
eps_u = 0

#define max number of iterations
max_iter = 5

def g(x):
    return x**2

def dg_dx(x):
    return 2*x

for i in range(0, max_iter):
    phi_new = eps_u*dg_dx(phi) - eps_p
    eps_p_new = phi - v_p - eps_p*sigma_p
    eps_u_new = u - g(phi) - sigma_u*eps_u