import numpy as np

def PowerMethod(B, epsilon):
    u_steps = []
    eig_steps = [0]
    break_condition = False
    step = 0 
    u_new = []

    while not break_condition:
        if step == 0:
            u_current = np.random.rand(B.shape[0])
        else:
            u_current = u_new
        u_steps.append(u_current) 
        eig_steps.append(np.dot(u_current.T,B).dot(u_current))
        B_u = np.dot(B,u_current)
        u_new = B_u/np.linalg.norm(B_u)
        step = step + 1 
        break_condition = abs(eig_steps[step]-eig_steps[step-1]) < epsilon

    u1 = u_steps[-1]
    h1 = eig_steps[-1]

    return u1, h1


def PowerMethod2(B, epsilon):
    
    u1, h1 = PowerMethod(B, epsilon)
    C = np.array(B - h1*np.matmul(np.matrix(u1).T, np.matrix(u1)))
    phi2, h2 = PowerMethod(C, epsilon)
    u2 = (h2-h1)*phi2+h1*(np.dot(u1,phi2))*u1
    u2 = u2/np.linalg.norm(u2)

    return u2, h2

print('ya')