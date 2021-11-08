from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

defect = 1  # 1 for positive, -1 for negative

##### Defining the constants
Gamma = 0.1
eta = 1
alpha = -1/Gamma
zeta = np.sqrt(eta/Gamma)


def u_boundary(x, on_boundary):
    return on_boundary


def p_boundary(x):
    return x[0] > 1.0 - DOLFIN_EPS


##### this will be helpfull when saving data
defect_type = ''
if defect == 1:
    defect_type = 'positive'
elif defect == -1:
    defect_type = 'negative'

###### this one is convinient if we want to plot the nematic director
def atan_2(y, x):
    return np.angle(x + 1j * y)


small_N_list = [200, 225, 250, 275, 300]
N_50_list = [400, 425, 450, 475, 500]

N_list = [0, 0, 0, 0, 0]
r_list = [1, 10]

out_list = np.zeros((55, 7))  # r, N ,number of cells, hmax, ux, uy, omega
counter = 0

for r in [1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50]:
    if r < 20:
        N_list[:] = small_N_list[:]
    else:
        N_list[:] = N_50_list[:]
    for N in N_list:
        print("r = ", r, " N = ", N)
        # making domain
        rectangle = Rectangle(Point(-r / 2, -r / 2),
                              Point(r / 2, r / 2))  # nb: rectangular domain is very unstable
        circel = Circle(Point(0.0, 0.0), r)
        mesh = generate_mesh(circel, N)
##### creating a file to utput the data, this is only done for some parameters
        velocity_stiring = 'Data_velocity/U_' + defect_type + str(r)+ '.xdmf'
        vorticity_string = 'Data_vorticity/omega_' + defect_type + str(r)+ '.xdmf'

        xdmffile_u = XDMFFile(velocity_stiring)
        xdmffile_omega = XDMFFile(vorticity_string)

###### Defining our function spaces
        p_u = 2
        p_p = 1
        V = VectorElement("Lagrange", mesh.ufl_cell(), p_u)
        Q = FiniteElement("Lagrange", mesh.ufl_cell(), p_p)
        print(mesh.num_cells())
        TH = V * Q

        W = FunctionSpace(mesh, TH)

        P1 = VectorFunctionSpace(mesh, "Lagrange", p_u)
        P2 = FunctionSpace(mesh, "Lagrange", p_p)

##### Deffining test and trial functions
        u, p = TrialFunctions(W)

        v, q = TestFunctions(W)

        omega = TrialFunction(P2)
        psi = TestFunction(P2)

##### Boundary conditions, note that with the non-slip bc the pressure is only determined up to a constant.
        u_analytical = Constant([0.0, 0.0])

        bc_u = DirichletBC(W.sub(0), u_analytical, u_boundary)

########### nematic orderparameter, Using this is more stable than calculating the force
        Qp = Expression((('alpha*x[0]/sqrt(x[0]*x[0] + x[1]*x[1])', 'defect*alpha*x[1]/sqrt(x[0]*x[0] + x[1]*x[1])')
                         ,
                         ('defect*alpha*x[1]/sqrt(x[0]*x[0] + x[1]*x[1])', '-alpha*x[0]/sqrt(x[0]*x[0] + x[1]*x[1])')),
                        alpha=alpha, defect=defect, degree=6)

        bc = [bc_u]
##### Setting up the weak form of the equations
        a = zeta**2 * inner(grad(u), grad(v)) * dx + dot(u, v) * dx + div(u) * q * dx - div(v) * p * dx  #
        L = -inner(Qp, grad(v)) * dx

        UP = Function(W)
        A, b = assemble_system(a, L, bc)

        solve(A, UP.vector(), b, 'mumps')

        U, P = UP.split()
###### Calculating the vorticity
        a1 = omega * psi * dx
        L1 = U[1].dx(0) * psi * dx - U[0].dx(1) * psi * dx
        omega = Function(P2)
        solve(a1 == L1, omega)

        point = Point((0.0, 0.0))
        u_num = U(point)
        omega_num = omega(point)

##### Saving some data
        out_list[counter, 0] = r
        out_list[counter, 1] = N
        out_list[counter, 2] = mesh.num_cells()
        out_list[counter, 3] = mesh.hmax()
        out_list[counter, 4] = u_num[0]
        out_list[counter, 5] = u_num[1]
        out_list[counter, 6] = omega_num  # obs: omega is multivalued at zero
        # print(mesh.hmin())
        counter += 1

        xdmffile_u.write(U)
        xdmffile_omega.write(omega)




np.savetxt('output.txt', out_list)
