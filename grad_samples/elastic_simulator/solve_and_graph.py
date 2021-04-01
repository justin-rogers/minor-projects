import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib as mpl
from dolfin import *


def elastic_solve(lam, mu, b, g=9.81):
    """Solves the given elasticity problem for the 3 (4) parameters"""
    assert (lam > 0 and mu > 0 and b > 0), "Need all constants to be positive"
    x_len = 1.
    y_len = 0.25
    mesh = RectangleMesh(Point(0, 0), Point(x_len, y_len), 20, 20)

    # Define the essential boundary condition, u=0 on Gamma_2
    def boundary_gamma2(x, on_boundary):
        """specifies {0,1}*[0,1/4]"""
        eps = 10**-14
        return on_boundary and (x[0] < eps or x[0] > 1 - eps)

    # Define function space on mesh (piecewise bilinear)
    V = VectorFunctionSpace(mesh, 'P', 1)
    ebc = DirichletBC(V, Constant((0., 0.)), boundary_gamma2)

    # Efficiently handling the lower boundary condition requires a little
    # bit of finesse: I'm using some power tools designed to handle
    # combinations of multiple boundary conditions.
    # There's probably an easier method.
    # The goal here: defining our ds measure, which we integrate load against
    # See page 96 of "Solving PDEs in Python, Vol I" for more details
    class LowerBoundary(SubDomain):
        """Defines the region [0,1]*{0}"""
        def inside(self, x, on_boundary):
            return on_boundary and (x[1] < 10**-14)

    lower_edge = LowerBoundary()
    # 'size_t' means it takes nonnegative int values, a C++ convention.
    lower_edge_marker = MeshFunction('size_t', mesh, 1)
    lower_edge.mark(lower_edge_marker, 0)
    ds = Measure('ds', domain=mesh, subdomain_data=lower_edge_marker)

    # Now ds(0) is a measure over the bottom edge, define the problem.

    load = Expression(("0", "b * exp(-pow((x[0] - 0.5), 2) * 100)"),
                      degree=1,
                      b=b)

    def deform(u):
        """deformation tensor epsilon"""
        return (grad(u) + grad(u).T) / 2

    f = Constant((0, -g))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (lam * (div(u) * div(v)) + mu * inner(deform(u), deform(v))
         ) * dx  # inner() is frobenius inner product
    L = dot(f, v) * dx + dot(load, v) * ds(0)
    # Now we solve.
    u = Function(V)
    solve(a == L, u, ebc)
    # file_sol = File("elastic_v0_sol.pvd")
    # file_sol << u
    # Can uncomment and save to VTK format, but I'm not a fan of it.
    return u


def trigraph(u, mu, lam, b):
    """Returns a figure with various information"""
    mesh = u.function_space().mesh()
    xy = mesh.coordinates()
    tri_mesh = Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    data = u.vector().get_local()  # np array
    # next few lines are taken from the plotting module, workaround import bug
    w0 = u.compute_vertex_values(mesh)
    nv = mesh.num_vertices()
    X = [mesh.coordinates()[:, i] for i in range(2)]
    U = [w0[i * nv:(i + 1) * nv] for i in range(2)]

    u_norms = np.sqrt(U[0]**2 + U[1]**2)
    mag_norm = mpl.colors.Normalize(vmin=0, vmax=max(u_norms))
    x_norm = mpl.colors.Normalize(vmin=min(U[0]), vmax=max(U[0]))
    y_norm = mpl.colors.Normalize(vmin=min(U[1]), vmax=max(U[1]))

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Visualizing u for mu={:.1e}, lam={:.1e}, b={:.1e}'.format(
        mu, lam, b))
    ax1a = fig.add_subplot(321, projection='3d')
    ax1a.set_title('norm(u)')
    ax1a.plot_trisurf(tri_mesh, u_norms, cmap='cool')

    ax1b = fig.add_subplot(323, projection='3d')
    ax1b.set_title('u_1')
    ax1b.plot_trisurf(tri_mesh, U[0], cmap='coolwarm')

    ax1c = fig.add_subplot(325, projection='3d')
    ax1c.set_title('u_2')
    ax1c.plot_trisurf(tri_mesh, U[1], cmap='PiYG')

    ax2a = fig.add_subplot(322)
    # Plot the displacement arrows
    ax2a.quiver(X[0],
                X[1],
                U[0],
                U[1],
                u_norms,
                scale_units="x",
                cmap='cool',
                norm=mag_norm)

    ax2b = fig.add_subplot(324)
    # x values
    ax2b.quiver(X[0],
                X[1],
                U[0],
                0 * U[1],
                U[0],
                scale_units="x",
                cmap='coolwarm',
                norm=x_norm)


    ax2c = fig.add_subplot(326)
    # y values
    ax2c.quiver(X[0],
                X[1],
                0 * U[0],
                U[1],
                U[1],
                scale_units="x",
                cmap='PiYG',
                norm=y_norm,
                label="u_2")

    all_axes = [ax1a, ax1b, ax1c, ax2a, ax2b, ax2c]
    for ax in all_axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    return fig

# A simple test case
mu, lam, b = 1, 1, 1
u = elastic_solve(mu, lam, b)
fig = trigraph(u, mu, lam, b)
plt.show()
