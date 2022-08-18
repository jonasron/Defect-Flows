from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np



defect = 1/2  # The topological charge of the defect, can have any half-integer or integer value

Gamma = 1
eta = 1
alpha = 0
alpha_x = 1 # coefficient of the gradient in the x-direction
alpha_y = 0 # coefficient of the gradient in the y-direction
zeta = np.sqrt(eta / Gamma)

phi = -np.pi  # The background nematic orientation, theta_0 in article


def u_boundary(x, on_boundary):
    return on_boundary


def p_boundary(x):
    return x[0] > 1.0 - DOLFIN_EPS


#### this will be helpfulle when saving data
defect_type = 'Gradient_alpha_at_some_angle_'
if np.sign(defect) == 1:
    defect_type += 'positive charge'
elif np.sign(defect) == -1:
    defect_type += 'negative charge'
defect_type += str(np.abs(defect)) +'phi_' +str(phi)


small_N_list = [64]
large_N_list = [256]

N_list = [0]
r_list = [10]


for r in r_list:
    if r < 10:
        N_list[:] = small_N_list[:]
    else:
        N_list[:] = large_N_list[:]
    for N in  N_list:
        print("r = ", r, " N = ", N)
        # making domain
        circel = Circle(Point(0.0, 0.0), r)
        mesh = generate_mesh(circel, N)
        n = FacetNormal(mesh)

        velocity_stiring = 'Data_velocity/U_' + defect_type +'size_'+ str(r)+ '.xdmf'
        vorticity_string = 'Data_vorticity/omega_' + defect_type +'size_'+ str(r)+ '.xdmf'

        xdmffile_u = XDMFFile(velocity_stiring)
        xdmffile_omega = XDMFFile(vorticity_string)


        p_u = 2
        p_p = 1
        V = VectorElement("Lagrange", mesh.ufl_cell(), p_u)
        Q = FiniteElement("Lagrange", mesh.ufl_cell(), p_p)
        print(mesh.num_cells())
        TH = V * Q

        W = FunctionSpace(mesh, TH)

        P1 = VectorFunctionSpace(mesh, "Lagrange", p_u)
        P2 = FunctionSpace(mesh, "Lagrange", p_p)

        u, p = TrialFunctions(W)

        v, q = TestFunctions(W)

        omega = TrialFunction(P2)
        psi = TestFunction(P2)

        u_analytical = Constant([0.0, 0.0])

        bc_u = DirichletBC(W.sub(0), u_analytical, u_boundary)
        #bc_p = DirichletBC(W.sub(1), Constant(0), u_boundary)

        # nematic director
        Qp = Expression((('(alpha+alpha_x*x[0]+alpha_y*x[1])*cos(2*defect*atan2(x[1],x[0]) +2*phi)', '(alpha+alpha_x*x[0]+alpha_y*x[1])*sin(2*defect*atan2(x[1],x[0])+2*phi )')
                         ,
                         ('(alpha+alpha_x*x[0]+alpha_y*x[1])*sin(2*defect*atan2(x[1],x[0])+2*phi)', '-(alpha+alpha_x*x[0]+alpha_y*x[1])*cos(2*defect*atan2(x[1],x[0])+2*phi)')),
                        alpha=alpha,alpha_x=alpha_x,alpha_y=alpha_y, defect=defect,phi=phi, degree=6)
        F = Expression(('alpha*(x[0] *cos(2*phi) - x[1]*sin(2*phi) )/(x[0]*x[0] +x[1]*x[1])',
                        'alpha*(x[1] *cos(2*phi) + x[0]*sin(2*phi) )/(x[0]*x[0] +x[1]*x[1])'),
                       alpha=alpha, defect=defect, phi=phi, degree=6 )


        a = zeta ** 2 * inner(grad(u), grad(v)) * dx  + div(u) * q * dx - div(v) * p * dx  +dot(u, v) * dx
        L = -inner(Qp, grad(v)) * dx

        UP = Function(W)
        A, b = assemble_system(a, L,[bc_u])

        solve(A, UP.vector(), b, 'mumps')

        U, P = UP.split()

        a1 = omega * psi * dx
        L1 = U[1].dx(0) * psi * dx - U[0].dx(1) * psi * dx
        omega = Function(P2)
        solve(a1 == L1, omega)

        xdmffile_u.write(U)
        xdmffile_omega.write(omega)





