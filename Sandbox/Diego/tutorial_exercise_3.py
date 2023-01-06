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

#define hyperparameters
max_iter = 300
lr = 0.12

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
    phi = phi + lr * phi_new
    eps_p = eps_p + lr * eps_p_new
    eps_u = eps_u + lr * eps_u_new    

    print('-------- iteration ' + str(i) + ' -------')
    print('phi: ' + str(phi))
    print('eps_p: ' + str(eps_p))
    print('eps_u: ' + str(eps_u))
    print('eps_p_new: ' + str(eps_p_new))
    print('eps_u_new: ' + str(eps_u_new))