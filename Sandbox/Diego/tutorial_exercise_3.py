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
max_iter = 134

# define g and its derivative
def g(x):
    return x**2

def dg_dx(x):
    return 2*x

# run the simulation
for i in range(0, max_iter):
    # compute the new values into temporary variables
    phi_new = eps_u*dg_dx(phi) - eps_p
    eps_p_new = phi - v_p - eps_p*sigma_p
    eps_u_new = u - g(phi) - sigma_u*eps_u
    
    # update the actual variables
    phi = phi + 0.1 * phi_new
    eps_p = eps_p + 0.1 * eps_p_new
    eps_u = eps_u + 0.1 * eps_u_new    

    print('--------' + str(i) + '-------')
    print(phi)
    print(eps_p)
    print(eps_u)