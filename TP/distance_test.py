# %%
import cyipopt
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
from scipy.spatial import HalfspaceIntersection, ConvexHull, Delaunay
from scipy.optimize import linprog
from cyipopt import Problem
from numba import njit
# To display in notebook
from IPython.display import HTML, display
import webbrowser
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

################## POLYGON RELATED FUNCTIONS ##################

def get_polytope_vertices_opt(A, b, tol=1e-6):
    n_dim = A.shape[1]  # Dimension of the polytope (e.g., 2 for 2D)
    vertices = []

    # Generate direction vectors (all combinations of ±1 in each dimension)
    directions = []
    for signs in combinations([-1, 1] * n_dim, n_dim):
        directions.append(np.array(signs))
    directions = np.unique(directions, axis=0)  # Remove duplicates

    # Solve LP for each direction
    for c in directions:
        res = linprog(
            c=-c,  # Maximize c^T x (linprog minimizes, so we negate)
            A_ub=A,
            b_ub=b,
            bounds=(None, None),  # No bounds beyond Ax ≤ b
            method="highs",  # Uses the HiGHS solver
        )
        if res.success:
            vertex = np.round(res.x, int(-np.log10(tol)))
            if not any(np.allclose(vertex, v, atol=tol) for v in vertices):
                vertices.append(vertex)
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    vertices = reconstructed_vertices[hull.vertices]

    return np.array(vertices)

def is_point_inside_polygon(point, A, b, tol=1e-6):
    res = linprog(
            c=[0.0, 0.0],  # dummy objective
            A_ub=A,
            b_ub=b,
            bounds=[(point[0], point[0]), (point[1], point[1])],
            method="highs"
        )
    return res.success and res.status == 0

def polygons_intersect(A1, b1, A2, b2):
    A_combined = np.vstack([A1, A2])
    b_combined = np.vstack([b1.reshape(-1, 1), b2.reshape(-1, 1)])

    res = linprog(
        c=[0.0, 0.0],
        A_ub=A_combined,
        b_ub=b_combined,
        bounds=(None, None),
        method="highs"
    )
    return res.success and res.status == 0
def find_strictly_feasible_point(A, b):
    """
    Solve an LP to find a strictly feasible point x such that A x < b
    """
    m, n = A.shape
    # Objective: maximize δ (slack)
    c = np.zeros(n + 1)
    c[-1] = -1  # Maximize δ ⇒ minimize -δ

    # Constraints: A x + δ ||A_i|| ≤ b_i
    norms = np.linalg.norm(A, axis=1)
    A_lp = np.hstack((A, norms[:, None]))
    bounds = [(None, None)] * n + [(0, None)]  # δ ≥ 0

    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
    if res.success:
        return res.x[:-1]  # Return x (ignore δ)
    else:
        raise ValueError("Could not find a strictly feasible point.")

def generate_random_polygon(
    max_vertices=20, radius_lim=(1e-1, 1.0), bbox=(-5, -5, 5, 5), seed=None
):
    rng = np.random.default_rng(seed)
    num_vertices = rng.integers(3, max_vertices + 1).item()
    radius = rng.uniform(radius_lim[0], radius_lim[1])
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    vertices = np.array([radius * np.cos(angles), radius * np.sin(angles)]).T
    # Calculate safe translation boundaries
    xmin, ymin, xmax, ymax = bbox
    offset = rng.uniform(
        low=[xmin + radius, ymin + radius],
        high=[xmax - radius, ymax - radius]
    )
    vertices += offset
    hull = ConvexHull(vertices)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    area = hull.volume
    return A, b, vertices, area

def generate_random_polygon_set(
    n_polygons=4,
    intersect_polygons=False,
    q0=None,
    qd=None,
    max_vertices=20,
    radius_lim=(1e-1, 1.0),
    bbox=(-5, -5, 5, 5),
    min_area=None,
    max_attempts=1000,
    seed=None
):
    rng = np.random.default_rng(seed)
    polygons = []
    attempts = 0

    if min_area is None:
        min_area = np.pi/2 * radius_lim[0] ** 2

    while len(polygons) < n_polygons and attempts < max_attempts:
        A, b, vertices, area = generate_random_polygon(
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            bbox=bbox,
            seed=rng.integers(1e9)
        )
        attempts += 1

        if area < min_area:
            continue
 
        # Check q0, qd are outside
        if q0 is not None and is_point_inside_polygon(q0.ravel(), A, b):
            continue
        if qd is not None and is_point_inside_polygon(qd.ravel(), A, b):
            continue

        # Check intersections with previous polygons
        if not intersect_polygons:
            if any(polygons_intersect(A, b, Ap, bp) for (Ap, bp, _, _, _) in polygons):
                continue
        # Get center of polygon and radius of circumscribed circle
        center = np.mean(vertices, axis=0)
        radius = 1.001*np.max(np.linalg.norm(vertices - center, axis=1)).item()

        # Passed all checks
        polygons.append((A, b, vertices, center, radius))

    if attempts == max_attempts:
        raise RuntimeError("Too many attempts to generate non-overlapping polygons")

    return polygons

################## PLOTTING RELATED FUNCTIONS ##################
def add_polygon(fig, A, b):
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    ordered_vertices = reconstructed_vertices[hull.vertices]

    x = np.append(ordered_vertices[:, 0], ordered_vertices[0, 0])
    y = np.append(ordered_vertices[:, 1], ordered_vertices[0, 1])

    fig.add_trace(go.Scatter(
        x=x, y=y, fill="toself",
        fillcolor="rgba(163, 159, 158, 0.2)",
        line=dict(color="rgba(163, 159, 158, 1)"),
    ))
    # vertices = get_polytope_vertices_opt(A, b)
    # hull = ConvexHull(vertices)
    # # Add polygon
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.append(vertices[hull.vertices, 0], vertices[hull.vertices[0], 0]),
    #         y=np.append(vertices[hull.vertices, 1], vertices[hull.vertices[0], 1]),
    #         fill="toself",
    #         fillcolor="rgba(163, 159, 158, 0.2)",
    #         line=dict(color="rgba(163, 159, 158, 1)"),
    #     )
    # )

def add_level_sets(
        fig,
        constraints,
        pc_list,
        R_list,
        eps=1e-3,
        r=1e-1,
        h=1e-1,
        eta=1.0,
        kind='both',
        bulge=True,
        bbox=(-5, -5, 5, 5),
        n_points=100,
        n_countours=50,
        ignore=[],
        test_=False):
    x_min, y_min, x_max, y_max = bbox
    p1 = np.linspace(x_min, x_max, n_points)
    p2 = np.linspace(y_min, y_max, n_points)
    P1, P2 = np.meshgrid(p1, p2)
    P = np.vstack([P1.ravel(), P2.ravel()]).T
    distances = []

    for j, pi_ in enumerate(P):
        # print(j)
        pi = pi_.copy().reshape(-1, 1)
        pi_dists = []
        test_dists = []
        for i, (A_, b_) in enumerate(constraints):
            if ignore:
                keep_mask = np.ones(A_.shape[0], dtype=bool)
                keep_mask[ignore[i]] = False
                A = A_[keep_mask]
                b = b_[keep_mask]
                d_test, *_ = e_s_hat(
                    pi,
                    A,
                    b,
                    phi,
                    kind=kind,
                    bulge=bulge,
                    pc=pc_list[i],
                    R=R_list[i],
                    eps=eps,
                    r=r,
                    h=h,
                    eta=eta,
                )
                test_dists.append(d_test)
            # else:
            if True:
                A = A_
                b = b_
                d_, *_ = e_s_hat(
                    pi,
                    A,
                    b,
                    phi,
                    kind=kind,
                    bulge=bulge,
                    pc=pc_list[i],
                    R=R_list[i],
                    eps=eps,
                    r=r,
                    h=h,
                    eta=eta,
                )
            pi_dists.append(d_)
        if test_:
            dist_wo_const = -(
                (1/len(test_dists)) * np.sum(
                    [abs(z) ** (-1 / r) for z in test_dists]
                )) ** (-r)
            # dist_w_const = -((1 / len(pi_dists)) * np.sum(
            #     np.abs(pi_dists) ** (-1 / r)
            # ) ** (-r))
            r_ = r/10
            dist_w_const = (-r_)*np.log(np.sum(np.exp(-np.array(pi_dists)/(r_))))
            # dist_w_const = np.min(pi_dists)
            e_s = np.minimum(np.min(pi_dists), dist_wo_const) # THIS WORKS
            # e_s = np.minimum(dist_w_const, dist_wo_const)
            # e_s = -r_ * np.log(np.exp(-dist_wo_const/r_) + np.exp(-dist_w_const/r_))
            # # e_s = -np.maximum(-e_s, np.sum(-pi_dists))
            # e_s = -((0.5 * ((-e_s)**(1/r_) + (-np.sum(pi_dists))**(1/r_))
            #         ) ** (r_))
            # e_s = -(0.5 *
            #          (np.sum(-pi_dists)**(1/r_) + np.sum(-test_dists)**(1/r_))
            #        ) ** (r_)
            # e_s = (0.5 *
            #        (-dist_wo_const)**(1/r) + (-np.min(pi_dists))**(1/r)
            #        ) ** (1/r)
        else:
            e_s = np.min(pi_dists)
        distances.append(e_s)

    distances = np.array(distances).reshape(P1.shape)
    contour = go.Contour(
        x=p1,
        y=p2,
        z=distances,
        colorscale="RdBu",
        ncontours=n_countours,
        # contours=dict(
        #     start=0,
        #     end=np.max(distances),
        #     size=0.1
        # ),
        name="Level Sets",
    )
    fig.add_trace(contour)
    
def create_planning_plot(
        constraints,
        pc_list, R_list,
        q0,
        qd,
        path,
        init_path,
        bbox=(-5, -5, 5, 5),
        n_points=100,
        n_points_contour=None,
        n_countours=50,
        kind_countor=None,
        eps=1e-3,
        r=1e-1,
        h=1e-1,
        eta=1.0,
        bulge=True,
        plot_cicles=False,
        ignore=[],
        test_=False):
    
    fig = go.Figure()
    q0_ = q0.reshape(-1, 1)
    qd_ = qd.reshape(-1, 1)
    path = path.copy().reshape(-1, n_points)
    init_path = init_path.copy().reshape(-1, n_points)
    
    if n_points_contour is None:
        n_points_contour = n_points
    if kind_countor is None:
        kind_countor = 'both'

    for A, b in constraints:
        add_polygon(fig, A, b)

    fig.add_trace(
        go.Scatter(
            x=[q0_[0, 0].item()],
            y=[q0_[1, 0].item()],
            mode="markers",
            marker=dict(color="green", size=10, symbol="x"),
            name="q0",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[qd_[0, 0].item()],
            y=[qd_[1, 0].item()],
            mode="markers",
            marker=dict(color="blue", size=10, symbol="star"),
            name="qd",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=init_path[0, :],
            y=init_path[1, :],
            mode="lines",
            line=dict(color="cyan", width=2, dash="dash"),
            name="Initial Path",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=path[0, :],
            y=path[1, :],
            mode="markers+lines",
            line=dict(color="black", width=2),
            name="Deformed Path",
        )
    )

    if plot_cicles:
        # Plot auxiliary circle around polygons
        for pc, R in zip(pc_list, R_list):
            fig.add_trace(
                go.Scatter(
                    x=pc[0] + R * np.cos(np.linspace(0, 2 * np.pi, 100)),
                    y=pc[1] + R * np.sin(np.linspace(0, 2 * np.pi, 100)),
                    mode="lines",
                    line=dict(color="orange", width=1, dash="dot"),
                    name="Auxiliary Circle",
                )
            )

    add_level_sets(
        fig,
        constraints,
        pc_list,
        R_list,
        eps=eps,
        r=r,
        h=h,
        eta=eta,
        kind=kind_countor,
        bulge=bulge,
        bbox=bbox,
        n_points=n_points_contour,
        n_countours=n_countours,
        ignore=ignore,
        test_=test_,
    )
    # Update layout
    fig.update_layout(
        xaxis_title="x", yaxis_title="y", showlegend=False, width=800, height=800,
        xaxis_range=[bbox[0], bbox[2]], yaxis_range=[bbox[1], bbox[3]],
        margin=dict(t=0, l=10, r=10, b=10)
    )

    return fig

def add_polygon_plt(ax, A, b, alpha=0.5, color=(0.64, 0.62, 0.61)):
    """Add polygon to matplotlib axes"""
    vertices = get_polytope_vertices_opt(A, b)
    # print(vertices)
    hull = ConvexHull(vertices)
    poly_vertices = vertices[hull.vertices]
    ax.fill(
        poly_vertices[:, 0],
        poly_vertices[:, 1],
        color=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1,
    )

def animate_deformation_matplotlib(
    path_list,
    init_path,
    obstacles,
    q0,
    qd,
    p1,
    p2,
    distances,
    frame_delay=200,  # ms between frames
):
    # Create static figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    n_iters = len(path_list)
    # Precompute all paths
    paths = [init_path.copy()]
    paths.extend(path_list)

    # Setup static elements
    # Create RdBu-like colormap
    cmap = LinearSegmentedColormap.from_list("RdBu", ["#2166ac", "#f7f7f7", "#b2182b"])
    contour = ax.contourf(p1, p2, distances, levels=20, cmap=cmap, alpha=1.0)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Distance")

    # Add polygons
    for A, b in obstacles:
        add_polygon_plt(ax, A, b)

    ax.plot(q0[0, 0], q0[1, 0], "gx", markersize=10, label="q0")
    ax.plot(qd[0, 0], qd[1, 0], "b*", markersize=10, label="qd")

    # Initialize path line
    (path_line,) = ax.plot([], [], "ok-", linewidth=2)

    # Add legend and set limits
    ax.legend()
    ax.set_xlim(np.min(p1), np.max(p1))
    ax.set_ylim(np.min(p2), np.max(p2))
    ax.set_aspect("equal", "box")
    ax.set_title("Path Deformation Animation")
    ax.grid(True)

    # Animation update function
    def update(frame):
        path = paths[frame]
        path_line.set_data(path[0, :], path[1, :])
        ax.set_title(f"Deformation Step: {frame}/{n_iters}")
        return (path_line,)

    # Create animation
    ani = FuncAnimation(
        ax.figure, update, frames=n_iters, interval=frame_delay, blit=True
    )

    return ani
################## DISTANCE RELATED FUNCTIONS ##################

# def phi(s, h=0.1, *args, **kwargs):
#     if s < 0:
#         val = 0
#         grad = 0
#     else:
#         val = (s**3) / (2 * (s + h))
#         grad = (3 * s**2) / (2 * (s + h)) - (s**3) / (2 * (s + h)**2)

#     return val, grad


@njit
def phi(s, h=0.5, r=0.1): 
    """Smooth approximation of the distance function
    h*log(cosh(x)), derivative is h*tanh(x), second derivative is h*sech(x)**2
    """
    if s < 0:
        val = 0.0
        grad = 0.0
        hess = 0.0
    else:
        val = np.log(np.cosh(s / h)) * h
        grad = np.tanh(s / h)
        hess = ((1 / np.cosh(s / h)) ** 2) / h

    return val, grad, hess


@njit
def inner_distance(f, p, A, b, r=0.1, h=0.5):
    """Uses short-circuit algorithm"""
    N, m = A.shape
    in_dists, v_grad = 0.0, np.zeros((1, m))
    v_hessian = np.zeros((m, m))  # This nonzero only for i=j
    for i, ai_ in enumerate(A):
        ai = ai_.copy().reshape(-1, 1)
        s = (b[i] - ai.T @ p).item()
        f_val, f_grad, f_hess = f(s, h)
        if f_val > 1e-6:
            # in_dists.append(f_val ** (-1 / r))
            in_dists += (f_val ** (-1 / r))
            v_grad += (-1 / r) * (f_val ** ((-1 / r) - 1)) * f_grad * (-ai.T)
            v_hessian += (-1 / r) * (
                (-1/r - 1) * (f_val ** (-1 / r - 2)) * (f_grad ** 2)
                + (f_val ** (-1/r - 1) * f_hess)
            ) * (ai @ ai.T)
        else:
            # in_dists.append(np.inf)
            # in_dists.append((1e-6) ** (-1 / r))
            in_dists += 1e-6 ** (-1 / r)

    # S = np.sum(in_dists)
    S = in_dists
    in_dist = (1 / N * S) ** (-r)
    # -r/N * (S ** (-r - 1)) * 1/N [* d(in_dist)/dS]
    first_chain = -r * (in_dist / S)  # * 1 / N
    in_grad = first_chain * v_grad.reshape(1, -1)
    in_hessian = first_chain * v_hessian + (
        # (r(r+1)) * (S/N)**(-r-2) * 1/N * 1/N [d(v)/dS ** 2]
        (r * (r + 1)) * (in_dist / (S**2)) * (v_grad.T @ v_grad)
    )
    return -in_dist, -in_grad, -in_hessian


@njit
def outter_distance(f, p, A, b, r=0.1, h=0.5):
    N, m = A.shape
    out_dist, out_grad = 0.0, np.zeros((1, m))
    out_hessian = np.zeros((m, m))

    for i, ai_ in enumerate(A):
        ai = ai_.copy().reshape(-1, 1)
        s = (ai.T @ p - b[i]).item()
        f_val, f_grad, _ = f(s, r=r, h=h)
        # out_dist.append(f_val)
        out_dist += f_val
        out_grad += f_grad * ai.T

    out_dist = (1 / N) * out_dist #np.sum(out_dist)
    out_grad = (1 / N) * out_grad.reshape(1, -1)
    return out_dist, out_grad, out_hessian


@njit
def bulging(dist, grad_dist, hess_dist, p, pc, R, eps=1e-3, out=True):
    p_ = np.asarray(p).reshape(-1, 1)
    m = p_.shape[0]
    pc_ = np.asarray(pc).reshape(-1, 1)
    grad = np.asarray(grad_dist).reshape(1, -1)
    hess = np.asarray(hess_dist).reshape((m, m))
    rho = 0.5 * (((p_ - pc_).T @ (p_ - pc_)) - R**2).item()
    grad_rho = (p_ - pc_).T
    hess_rho = np.eye(m)
    if not out:
        rho = -rho
        grad_rho = -grad_rho
        hess_rho = -hess_rho
    beta = (eps**2 * rho**2) + (1 - 2 * eps) * (dist**2)
    # Add 1e-12 for numerical stability
    sqrt_term = np.sqrt(beta + 1e-12)
    bulge_dist = eps * rho + sqrt_term
    grad_beta = (
        2 * (eps**2) * rho * grad_rho + 2 * (1 - 2 * eps) * dist * grad
    )
    hess_beta = (
            2 * (eps ** 2) * ((grad_rho.T @ grad_rho) + rho * hess_rho)
            + 2 * (1 - 2 * eps) * ((grad.T @ grad) + dist * hess)
    )

    bulge_grad = eps * grad_rho + 0.5 * (1 / sqrt_term) * grad_beta
    bghess_t1 = eps * hess_rho
    bghess_t2 = (1 / (2 * sqrt_term)) * hess_beta
    bghess_t3 = -(1 / (4 * beta ** 1.5)) * (grad_beta.T @ grad_beta)

    bulge_hess = bghess_t1 + bghess_t2 + bghess_t3
    return bulge_dist, bulge_grad, bulge_hess


@njit
def e_s_hat(
    p,
    A,
    b,
    f,
    kind="both",
    bulge=False,
    R=None,
    pc=None,
    eps=1e-3,
    r=0.1,
    h=0.5,
    eta=1.0,
):
    out_dist, out_grad, out_hess = outter_distance(f, p, A, b, r=r, h=h)
    in_dist, in_grad, in_hess = inner_distance(f, p, A, b, r=r, h=h)
    if bulge:
        # in_dist, in_grad, in_hess = bulging(in_dist, in_grad, in_hess, p, pc, R, eps=eps, out=False)
        # in_dist, in_grad, in_hess = -in_dist, -in_grad, -in_hess
        out_dist, out_grad, out_hess = bulging(out_dist, out_grad, out_hess, p, pc, R, eps=eps)

    in_dist *= eta
    in_grad *= eta

    if kind == "out":
        return out_dist, out_grad, out_hess
    elif kind == "in":
        return in_dist, in_grad, in_hess
    else:
        return out_dist + in_dist, out_grad + in_grad, out_hess + in_hess
    
################## SOLVE RELATED FUNCTIONS ##################
@njit
def deform_path(
    init_path,
    obstacles,
    R_list=[],
    pc_list=[],
    kind='in',
    h=0.1,
    r=0.8,
    eps=5e-2,
    bulge=False,
    min_path=False,
    k=1.0,
    zeta=1.0,
    alpha=1.0,
    eta=10.0,
):
    path = init_path.copy()
    dists, grads = [], []
    for j, p_ in enumerate(init_path.T):
        p = p_.copy().reshape(-1, 1)
        dist, grad = 0.0, np.zeros((1, path.shape[0]))
        for i, obstacle in enumerate(obstacles):
            pc = pc_list[i]
            R = R_list[i]
            A, b = obstacle
            dist_, grad_, _ = e_s_hat(
                p, A, b, phi, kind=kind, bulge=bulge, R=R, pc=pc, eps=eps, r=r, h=h, eta=eta
            )
            if kind == 'both':
                # if dist_ < dist:
                #     if dist < 0:
                #         dist += dist_
                #         grad += grad_
                #     else:
                #         dist = dist_
                #         grad = grad_
                # else:
                # print("saturating")
                dist += -1/alpha * np.log(np.exp(-alpha * dist_)/2 + 0.5)
                expterm = (1.0 / (np.exp(alpha * dist_) + 1.0))
                # print(dist_, expterm, -1/alpha * np.log(np.exp(-alpha * dist_)/2 + 0.5))
                grad += expterm * grad_.reshape(1, -1)
            else:
                dist += dist_
                grad += grad_.reshape(1, -1)
        dists.append(dist)
        # grads.append(grad.T / (np.linalg.norm(grad) + 1e-8))
        grads.append(grad.T)

    min_idx = np.argmin(np.asarray(dists))
    min_dist, min_grad = dists[min_idx], grads[min_idx]

    for j in range(path.shape[1] - 2):
        k = j + 1 # Do not change initial and final point
        # const_obs = [b - A @ path[:, j].reshape(-1, 1) for A, b in obstacles]
        # err = np.max(const_obs)
        # coeff = np.abs(dists[j])  # np.sign(dists[j]) * np.sqrt(np.abs(dists[j]))
        coeff = 1.0
        if dists[k] > 0:
            coeff = np.sqrt(np.abs(dists[k]))
        else:
            coeff = np.sqrt(np.abs(dists[k]))
        path[:, k] += grads[k] * coeff

    if min_path:
        for j, point in enumerate(path.T[1:-1]):
            k = j + 1
            prev_grad = path[:, k - 1].ravel() - point.ravel()
            next_grad = path[:, k + 1].ravel() - point.ravel()
            path[:, k] = point + zeta*(prev_grad + next_grad)

    return path, dists, grads, 0.0

# GARBAGE IDEA FOR LEVEL SETS INTERPOLATION
# def minminmin(x_fixed, x_vary, y_fixed, y_vary, A_list, b_list, func, *args, **kwargs):
#     f_verticals = []
#     f_horizontals = []
#
#     for A, b in zip(A_list, b_list):
#         ith_vertical, ith_horizontal = [], []
#
#         for x in x_vary:
#             p_hori = np.array([x, y_fixed])
#             ith_horizontal.append(func(p_hori, A, b, *args, **kwargs))
#         f_horizontals.append(np.min(ith_horizontal).item())
#
#         for y in y_vary:
#             p_vert = np.array([x_fixed, y])
#             ith_vertical.append(func(p_vert, A, b, *args, **kwargs))
#         f_verticals.append(np.min(ith_vertical).item())
#
#     return np.minimum(np.min(f_verticals), np.min(f_horizontals))
#
class OptimalPathProblem:
    def __init__(
        self,
        obstacles,
        q0,
        qd,
        f,
        phi,
        kappa,
        delta,
        n_points,
        bulge,
        R_list,
        pc_list,
        eps,
        min_path,
        r,
        h,
        kind='in'
    ):
        self.N = n_points
        self.d = q0.shape[0]
        self.f = f
        self.phi = phi
        self.q0 = q0
        self.qd = qd
        self.R_list = R_list
        self.pc_list = pc_list
        self.bulge = bulge
        self.kappa = kappa
        self.obstacles = obstacles
        self.delta = delta
        self.eps = eps
        self.min_path = min_path
        self.r = r
        self.h = h
        self.kind = kind
        self.n = self.N * self.d  # Total number of variables
        self.m = 2 * self.d + (self.N - 1)

    def objective(self, x):
        path = x.copy().reshape(self.N, self.d)
        dists = 0.0
        for p in path:
            dist = 0
            for i, obs in enumerate(self.obstacles):
                A, b = obs
                d_, *_ = e_s_hat(
                    p.reshape(-1, 1),
                    A,
                    b,
                    self.phi,
                    kind=self.kind,
                    bulge=self.bulge,
                    R=self.R_list[i],
                    pc=self.pc_list[i],
                    eps=self.eps,
                    r=self.r,
                    h=self.h,
                )
                dist += d_
            dists += dist
        # val = np.sum(dists)
        val = dists
        curvature = 0.5 * sum(
            np.linalg.norm(path[i + 2] - 2 * path[i + 1] + path[i]) ** 2
            for i in range(self.N - 2)
        )
        # Negative val because min problem
        return -val + self.kappa * curvature

    def gradient(self, x):
        path = x.copy().reshape(self.N, self.d)
        jac = np.zeros((self.N, self.d))
        for j, p in enumerate(path):
            grad = np.zeros((1, self.d))
            for i, obs in enumerate(self.obstacles):
                A, b = obs
                _, grad_, _ = e_s_hat(
                    p.reshape(-1, 1),
                    A,
                    b,
                    self.phi,
                    kind=self.kind,
                    bulge=self.bulge,
                    R=self.R_list[i],
                    pc=self.pc_list[i],
                    eps=self.eps,
                    r=self.r,
                    h=self.h,
                )
                grad += grad_
            jac[j, :] = grad.ravel()
        grad = -jac.reshape((self.N, self.d))

        for i in range(self.N - 2):
            v_aux = path[i] - 2 * path[i + 1] + path[i + 2]
            grad[i] += self.kappa * v_aux
            grad[i + 1] += -2 * self.kappa * v_aux
            grad[i + 2] += self.kappa * v_aux

        return grad.flatten()

    def constraints(self, x):
        path = x.copy().reshape(self.N, self.d)
        constraints_ = []
        constraints_.extend((path[0] - self.q0.ravel()).tolist())  # p_1 = q0
        constraints_.extend((path[-1] - self.qd.ravel()).tolist())# p_N = qd
        # ||p_{i+1} - p_i||<=delta
        for i in range(self.N - 1):
            constraints_.append(np.linalg.norm(path[i + 1] - path[i]).item())
        return np.array(constraints_).ravel()

    def jacobian(self, x):
        path = x.copy().reshape(self.N, self.d)
        jac = np.zeros((self.m, self.n))
        row = 0

        # First 2d rows are equality consts
        for i in range(self.d):
            jac[row, i] = 1.0
            row += 1

        for i in range(self.d):
            jac[row, i - self.d] = 1.0
            row += 1

        for i in range(self.N - 1):
            idx_i = i * self.d
            idx_ip1 = (i + 1) * self.d

            diff = path[i + 1] - path[i]
            norm = np.linalg.norm(diff)

            if norm == 0:
                grad_i = np.zeros(self.d)
            else:
                grad_i = -diff / norm
            grad_ip1 = -grad_i
            jac[row, idx_i : idx_i + self.d] = grad_i
            jac[row, idx_ip1 : idx_ip1 + self.d] = grad_ip1
            row += 1

        return jac.flatten()

    def jacobianstructure(self):
        # Full dense Jacobian: just return all indices
        row_indices = []
        col_indices = []
        for row in range(self.m):
            for col in range(self.n):
                row_indices.append(row)
                col_indices.append(col)
        return np.array(row_indices), np.array(col_indices)

    def hessian(self, x, lagrange, obj_factor):
        path = x.copy().reshape(self.N, self.d)
        hess = np.zeros((self.n, self.n)) # N*d x N*d

        for j, p in enumerate(path):
            p_ = p.reshape(-1, 1)
            hess_block = np.zeros((self.d, self.d))
            for i, obs in enumerate(self.obstacles):
                A, b = obs
                _, _, hess_ = e_s_hat(
                    p_,
                    A,
                    b,
                    self.phi,
                    kind=self.kind,
                    bulge=self.bulge,
                    R=self.R_list[i],
                    pc=self.pc_list[i],
                    eps=self.eps,
                    r=self.r,
                    h=self.h,
                )
                hess_block += hess_
            start = j * self.d
            hess[start: start + self.d, start:start + self.d] += -obj_factor * hess_block

        for i in range(self.N - 2):
            hess_options = [
                            (i, i, self.kappa),
                            (i, i + 1, -2 * self.kappa),
                            (i, i + 2, self.kappa),
                            (i + 1, i, -2 * self.kappa),
                            (i + 1, i + 1, 4 * self.kappa),
                            (i + 1, i + 2, -2 * self.kappa),
                            (i + 2, i, self.kappa),
                            (i + 2, i + 1, -2 * self.kappa),
                            (i + 2, i + 2, self.kappa),
                         ]
            for (j, k, val) in hess_options:
                row, col = j * self.d, k * self.d
                hess[row:row + self.d, col:col + self.d] += obj_factor * val * np.eye(self.d)

        # Constraints Hessian:
        # 2*self.d linear equality constraints
        # self.N - 1 constraints ||p_{i+1} - p_{i}|| <= delta
        for i in range(self.N - 1):
            v = (path[i + 1] - path[i]).reshape(-1, 1)
            norm_v = np.linalg.norm(v) + 1e-8
            hess_base = (1 / (norm_v ** 3)) * (
                            ((norm_v ** 2) * np.eye(self.d)) - (v @ v.T)
                        )
            lambda_i = lagrange[2*self.d + i]
            hess_options = [
                (i, i, hess_base),
                (i, i+1, -hess_base),
                (i+1, i, -hess_base),
                (i+1, i+1, hess_base)
            ]
            for (j, k, val) in hess_options:
                row, col = j * self.d, k * self.d
                block = val * lambda_i
                hess[row:row + self.d, col:col + self.d] += block

        row, col = self.hessianstructure()
        if np.linalg.norm(hess - hess.T) != 0:
            raise ValueError(f"hessian is not symmetric")
        return hess[row, col]

    def hessianstructure(self):
        i_lower, j_lower = np.tril_indices(self.n)
        return i_lower, j_lower
#%%
"""RANDOM MAP"""
# max_poly = 10, max_vert = 15, seed = 100, max_iter = 100
max_polygons = 10
max_vertices = 15
bounding_box = (-20, -20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 6)
q0 = np.array([-3, -12]).reshape(-1, 1)
q0 = np.array([-8.5, -16]).reshape(-1, 1)
# qd = np.array([15, 15]).reshape(-1, 1)
qd = np.array([11, 15]).reshape(-1, 1)
n_points = 100
h = 1.1
r = 0.8
eta = 4.0
eps = 5e-2
alpha = np.log(2)/0.2
bulge = True
min_path = True
k = 5e-1
max_iters = 500
seed = 1001 #1001, 69 cool, 42 NICE post mods, 100 is cool

obstacles = generate_random_polygon_set(
    n_polygons=max_polygons,
    intersect_polygons=False,
    q0=q0,
    qd=qd,
    max_vertices=max_vertices,
    radius_lim=radius_limits,
    bbox=bounding_box,
    seed=seed
)
a_list, b_list, vertices_list, pc_list, R_list = list(zip(*obstacles))
constraints = [(a, b) for a, b in zip(a_list, b_list)]

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
dists = [-100]
iter_ = 0
path_hist = []

# while any dists is negative and iters < max_iters:
while np.any(np.array(dists) < 2.0):
# while iter_ < max_iters:
    
    if iter_ > max_iters:
        print(f"reached max iterations: {max_iters}")
        break

    path, dists, grads, num_grads = deform_path(
        path,
        constraints,
        R_list=R_list,
        pc_list=pc_list,
        h=h,
        r=r,
        eps=eps,
        kind='both',
        bulge=bulge,
        min_path=min_path,
        k=k,
        zeta=0.5,
        alpha=alpha,
        eta=eta,
    )
    path_hist.append(path)

    if iter_ % 10 == 0:
        print(f"iteration {iter_}: min dist = {np.min(dists)}")

    iter_ += 1

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
# print(dists)
fig = create_planning_plot(
    constraints,
    pc_list,
    R_list,
    q0,
    qd,
    path,
    init_path,
    bbox=bounding_box,
    n_points=n_points,
    n_points_contour=200,
    n_countours=40,
    eps=eps,
    r=r,
    h=h,
    eta=eta,
    bulge=bulge,
    plot_cicles=False
)
fig.show()
# fig.write_image("./TP/media/sim1.pdf")
# %%
""" PLOTLY ANIMATION 1ST"""
x_min, y_min, x_max, y_max = bounding_box
p1 = np.linspace(x_min, x_max, 100)
p2 = np.linspace(y_min, y_max, 100)
P1, P2 = np.meshgrid(p1, p2)
P = np.vstack([P1.ravel(), P2.ravel()]).T
distances = []

for j, pi_ in enumerate(P):
    # print(j)
    pi = pi_.copy().reshape(-1, 1)
    pi_dists = []
    test_dists = []
    for i, (A_, b_) in enumerate(constraints):
        A = A_
        b = b_
        d_, *_ = e_s_hat(
            pi,
            A,
            b,
            phi,
            kind='both',
            bulge=bulge,
            pc=pc_list[i],
            R=R_list[i],
            eps=eps,
            r=r,
            h=h,
            eta=eta,
        )
        pi_dists.append(d_)
    e_s = np.min(pi_dists)
    distances.append(e_s)

distances = np.array(distances).reshape(P1.shape)


def show_animation(animation, filename="anim.html"):
    path = Path(filename).absolute()
    animation.save(path, writer="html")
    webbrowser.open(f"file://{path}")

animation = animate_deformation_matplotlib(
    path_hist,
    init_path,
    constraints,
    q0,
    qd,
    p1,
    p2,
    distances,
    frame_delay=(5/len(path_hist))*1000,
)
show_animation(animation, filename="anim3.html")
# %%
"""RANDOM MAP IPOPT"""
kappa = 1
delta = 3 * np.linalg.norm(init_path[:, 0] - init_path[:, -1])

for A, b in constraints:
    print(f"A: {A}, b: {b}")
problem = OptimalPathProblem(
    constraints,
    q0,
    qd,
    e_s_hat,
    phi,
    kappa,
    delta,
    n_points,
    bulge,
    R_list,
    pc_list,
    eps,
    min_path,
    r,
    h,
    kind='in'
)

# Bounds: no bounds on x
x_L = np.full(problem.n, -np.inf)
x_U = np.full(problem.n, np.inf)

# Constraint bounds
c_L = np.zeros(problem.m)
c_U = np.zeros(problem.m)
# First 2*d are equalities
c_U[: 2 * 2] = 0.0
# Last N-1 are upper bounds only (||p_{i+1} - p_i|| ≤ δ)
c_U[2 * 2 :] = delta
# Minimum distance between points is the initial one
# c_L[2*2:] = np.linalg.norm(init_path[:, 2] - init_path[:, 1])

nlp = Problem(
    n=problem.n, m=problem.m, problem_obj=problem, lb=x_L, ub=x_U, cl=c_L, cu=c_U
)
# nlp.add_option('derivative_test', 'first-order')  # Checks gradient (first derivatives)
# nlp.add_option('derivative_test_print_all', 'yes')  # Prints detailed results
# nlp.add_option('print_level', 12)
options = {
    # Force first-order method (no second derivatives)
    # "hessian_approximation": "exact",
    # "derivative_test": "second-order",
    # "derivative_test_print_all": "yes",
    "hessian_approximation": "limited-memory",
    # 'gradient_approximation': 'finite-difference-values',
    # 'jacobian_approximation': 'finite-difference-values',
    # Configure L-BFGS parameters
    # 'limited_memory_update_type': 'bfgs',  # Standard BFGS update
    # 'limited_memory_max_history': 10,      # History size (10-50 is typical)
    # Disable second-order features
    # 'mehrotra_algorithm': 'no',           # Disable second-order correction
    # 'fast_step_computation': 'no',         # Disable advanced step calc
    'alpha_for_y': 'primal',  # More aggressive step sizing
    'recalc_y': 'no',  # Reduces computational overhead
    "mu_strategy": "adaptive",
    # 'linear_solver': 'mumps',
    # Adjust convergence criteria for first-order method
    "tol": 1e-1,  # Relax tolerance (default 1e-8)
    "max_iter": 100,  # Increase iteration limit
    "acceptable_iter": 10,  # Stop after 10 "good enough" iters
    # Output control
    "print_level": 5,
    "print_frequency_iter": 10,
}
for key in options.keys():
    nlp.add_option(key, options[key])
# nlp.add_option('max_iter', 100)
print(init_path.flatten().shape)

irow, jcol = problem.hessianstructure()
# print(f"Hessian size: {problem.n} x {problem.n}")
# print(f"Number of structure entries: {len(irow)}")
# time.sleep(5)
x_opt, info = nlp.solve(init_path.T.flatten())

opt_path = x_opt.copy().reshape(n_points, 2).T
fig = create_planning_plot(
    constraints,
    pc_list,
    R_list,
    q0,
    qd,
    opt_path,
    init_path,
    bbox=bounding_box,
    n_points=n_points,
    n_points_contour=40,
    eps=eps,
    r=r,
    h=h,
    bulge=bulge,
    plot_cicles=True
)
fig.show()

#%%
""" TEST NONCONVEX POLYGONS """
# L shape
n_points = 50
max_iters = 100
h = 0.1
r = 0.8
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
max_iters = 200
bounding_box = (-6, -6, 6, 6)

seed = 42 # 100 is cool
# Bottom horizontal
A1 = np.array([
    [1.0, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
])
b1 = np.array([2.0, 0, 0, 1])
# b1 = np.array([2.0, 2.0, 0, 1])
# Vertical
A2 = np.array([
    [1.0, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
])
b2 = np.array([1.0, 0, 3, -1])
# Top horizontal
A3 = np.array([
    [1.0, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
])
b3 = np.array([2.0, 0, -3, 4])
A4 = np.array([
    [-1.0, 0],
    [-1, 1],
    [1, 1],
    [1, -1],
    [-2, -1]
])
b4 = np.array([-2.0, 2, 8, 2, -7])
A5 = np.array([
    [1.0, 0],
    [-1, -1],
    [-1, 1]
])
b5 = np.array([0.0, -1, 3])

constraints = [(A1, b1), (A2, b2), (A3, b3), (A4, b4)]#(A4, b4), (A5, b5)]
R_list, pc_list = [], []
ignore = [[3], [1, 2, 3], [0, 2], [0], ]#[0]]
# ignore = [[3], [2, 3], [2]]

q0 = np.array([-5.0, -5]).reshape(-1, 1)
qd = np.array([5, 5.0]).reshape(-1, 1)

for A, b in constraints:
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    vertices = hs.intersections
    center = np.mean(vertices, axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1)).item()
    pc_list.append(center)
    R_list.append(radius)

# constraints = [(a, b) for a, b in zip(a_list, b_list)]

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
dists = [-100]
iter_ = 0

# while any dists is negative and iters < max_iters:
while np.any(np.array(dists) < -1e-8):
    if iter_ > max_iters:
        print(f"reached max iterations: {max_iters}")
        break

    path, dists, grads, num_grads = deform_path(
        path,
        constraints,
        R_list=R_list,
        pc_list=pc_list,
        h=h,
        r=r,
        eps=eps,
        bulge=bulge,
        min_path=min_path,
        k=k,
    )

    if iter_ % 10 == 0:
        print(f"iteration {iter_}: min dist = {np.min(dists)}")

    iter_ += 1

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")

fig = create_planning_plot(
    constraints,
    pc_list,
    R_list,
    q0,
    qd,
    path*0,
    init_path*0,
    bbox=bounding_box,
    n_points=n_points,
    n_points_contour=100,
    kind_countor='in',
    eps=eps,
    r=r,
    h=h,
    bulge=False,
    plot_cicles=False,
    ignore=ignore,
    test_=True
)
fig.show()


#%%
"""PREDEFINED MAP"""
A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b1 = np.array([0.5, 0.5, 1, 0.5])
# b1 = np.array([0.5, 0.5, 1, -0.5])

A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b2 = np.array([2, -1, -0.5, 1.5])

n_points = 50
p1 = np.linspace(-1.5, 2.5, n_points)
p2 = np.linspace(-2, 2, n_points)
P1, P2 = np.meshgrid(p1, p2)
P = np.vstack([P1.ravel(), P2.ravel()]).T
distances = []
d1_list, d2_list = [], []

q0 = np.array([-1.15, 1.6]).reshape(-1, 1)
qd = np.array([2.3, -1.75]).reshape(-1, 1)
qd = np.array([1.7, -1.75]).reshape(-1, 1)

# q0 = np.array([-1.1, 0.45]).reshape(-1, 1)
# qd = np.array([1.1, 0.45]).reshape(-1, 1)

obstacles = [(A1, b1), (A2, b2)]
pc_list = [[0, 0.25], [1.5, -1]]
R_list = [0.91, 0.71]
# obstacles = [(A1, b1)]

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
neg_grad = []
dists = []
grads = []
idx = 0
start_idx, end_idx = None, None

h = 0.1
r = 0.8
eps = 5e-2
bulge = True
min_path = True
k = 5e-2
max_iters = 100


for _ in range(max_iters):
    path, dists, grads, num_grads = deform_path(
        path,
        obstacles,
        R_list=R_list,
        pc_list=pc_list,
        h=h,
        r=r,
        eps=eps,
        bulge=bulge,
        min_path=min_path,
        k=k,
    )

# for j, p in enumerate(init_path.T):
#     p = p.copy().reshape(-1, 1)
#     for A, b in obstacles:
#         dist, grad = e_s_hat(p, A, b, phi, kind='both', r=0.1, h=0.1)
#         if collision and (dist > 0):
#             print(dist, j)
#             collision = False
#             end_idx = j
#         if dist < 0:
#             # print(dist, j)
#             collision = True
#             if start_idx is None:
#                 start_idx = j
#             grads.append(grad.T / np.linalg.norm(grad))
#             # p -= np.abs(dist) * 1 *grad.T / np.linalg.norm(grad)
#             p -= (grad.T / np.linalg.norm(grad)) * np.abs(dist)
#         neg_grad.append(np.linalg.norm(grad))
#         dists.append(dist)
#     path[:, j] = p.ravel()

# min_idx = np.argmin(dists[start_idx:end_idx])
# min_dist, min_grad = dists[start_idx + min_idx], grads[min_idx]
# path[:, start_idx : end_idx] -= np.abs(min_dist) * 5 * min_grad



def create_plot(
    obstacles, q0, qd, R_list, pc_list, xgrid, ygrid, zgrid, path, init_path
):
    """Avoids nvim plotting multiple figures (due to iron.nvim)"""
    fig = go.Figure()

    for A, b in obstacles:
        add_polygon(fig, A, b)

    fig.add_trace(
        go.Scatter(
            x=[q0[0, 0].item()],
            y=[q0[1, 0].item()],
            mode="markers",
            marker=dict(color="green", size=10, symbol="x"),
            name="q0",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[qd[0, 0].item()],
            y=[qd[1, 0].item()],
            mode="markers",
            marker=dict(color="blue", size=10, symbol="star"),
            name="qd",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=init_path[0, :],
            y=init_path[1, :],
            mode="lines",
            line=dict(color="cyan", width=2, dash="dash"),
            name="Initial Path",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=path[0, :],
            y=path[1, :],
            mode="markers+lines",
            line=dict(color="black", width=2),
            name="Deformed Path",
        )
    )

    # Plot auxiliary circle around polygons
    for pc, R in zip(pc_list, R_list):
        fig.add_trace(
            go.Scatter(
                x=pc[0] + R * np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=pc[1] + R * np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode="lines",
                line=dict(color="orange", width=1, dash="dot"),
                name="Auxiliary Circle",
            )
        )

    # Add level sets
    contour = go.Contour(
        x=xgrid,
        y=ygrid,
        z=zgrid,
        colorscale="RdBu",
        ncontours=50,
        # contours=dict(
        #     start=0,
        #     end=np.max(distances),
        #     size=0.1
        # ),
        name="Level Sets",
    )
    fig.add_trace(contour)
    # Update layout
    fig.update_layout(
        xaxis_title="x1", yaxis_title="x2", showlegend=False, width=800, height=600
    )
    return fig


fig = create_plot(
    obstacles, q0, qd, R_list, pc_list, p1, p2, distances, path, init_path
)
fig.show()

# %%
"""TEST GRADIENT NUMERICALLY"""
import plotly.express as px

delta_x = 1e-4  # Small perturbation for numerical gradient approximation
num_grads, dists, hessians = [], [], []
num_hessians = []
grads = []
kind = "in"
bulge = True
# def e_s_hat(p, A, b, f, r=0.1, *args, **kwargs):
#     """Uses short-circuit algorithm"""
#     N, m = A.shape
#     in_dists, in_grad = [], np.zeros((1, m))
#
#     for i, ai in enumerate(A):
#         ai = ai.reshape(-1, 1)
#         s = (b[i] - ai.T @ p).item()
#         f_val, f_grad = f(s, *args, **kwargs)
#         if f_val != 0:
#             in_dists.append(f_val ** (-1 / r))
#             in_grad += (-1 / r) * (f_val ** ((-1 / r) - 1)) * f_grad * (-ai.T)
#         else:
#             in_dists.append(np.inf)
#
#     S = np.sum(in_dists)
#     in_dist = (1 / N * S) ** (-r)
#     first_chain = -r * (in_dist / S)  # -r/N * (S ** (-r - 1))
#     in_grad = first_chain * in_grad.reshape(1, -1)
#     return in_dist, in_grad
#

for p in init_path.T:
    dist, grad, _ = e_s_hat(
        p,
        A1,
        b1,
        phi,
        kind=kind,
        bulge=bulge,
        R=R_list[0],
        pc=pc_list[0],
        eps=eps,
        r=r,
        h=h,
    )
    dists.append(dist)
    pdx = np.array([delta_x, 0])
    pdy = np.array([0, delta_x])
    grad_curr = []
    for diff in [pdx, pdy]:
        daux1, *_ = e_s_hat(
            p - diff,
            A1,
            b1,
            phi,
            kind=kind,
            bulge=bulge,
            R=R_list[0],
            pc=pc_list[0],
            eps=eps,
            r=r,
            h=h,
        )
        daux2, *_ = e_s_hat(
            p + diff,
            A1,
            b1,
            phi,
            kind=kind,
            bulge=bulge,
            R=R_list[0],
            pc=pc_list[0],
            eps=eps,
            r=r,
            h=h,
        )
        grad_curr.append((daux2 - daux1) / (2 * delta_x))
    grads.append(grad.T)
    num_grads.append(np.array(grad_curr).reshape(-1, 1))
idx = np.argmin(dists)
print(grads[idx], num_grads[idx])
test = [np.linalg.norm(g1 - g2) for g1, g2 in zip(grads, num_grads)]
px.line(test)


delta = 1e-3  # try smaller values like 1e-5 or 1e-6 if needed
for _ in range(10):
    v = np.random.randn(2, 1)
    v /= np.linalg.norm(v)  # unit direction

    for p_ in path.T:
        p = np.array(p_).reshape(-1, 1)

        # Evaluate at p, p + delta v, p - delta v
        f0, *_ = e_s_hat(p, A1, b1, phi, kind=kind, bulge=bulge,
                         R=R_list[0], pc=pc_list[0], eps=eps, r=r, h=h)
        f_plus, *_ = e_s_hat(p + delta * v, A1, b1, phi, kind=kind, bulge=bulge,
                             R=R_list[0], pc=pc_list[0], eps=eps, r=r, h=h)
        f_minus, *_ = e_s_hat(p - delta * v, A1, b1, phi, kind=kind, bulge=bulge,
                              R=R_list[0], pc=pc_list[0], eps=eps, r=r, h=h)

        # Scalar Hessian-vector product approximation
        hess_scalar_fd = (f_plus - 2 * f0 + f_minus) / (delta**2)

        # Compare to vᵀ @ H @ v from analytical Hessian
        _, _, hess = e_s_hat(p, A1, b1, phi, kind=kind, bulge=bulge,
                             R=R_list[0], pc=pc_list[0], eps=eps, r=r, h=h)
        hess_scalar_ana = float(v.T @ hess @ v)
        num_hessians.append(hess_scalar_fd)
        hessians.append(hess_scalar_ana)

err_hess = [np.linalg.norm(h - num_hessians[i]) for i, h in enumerate(hessians)]
print(test)
print(grads)
print(err_hess)
print(hessians)
print(num_hessians)
px.line(err_hess)
# %%
""" IPOPT SOLUTION """
from cyipopt import Problem
import time

kappa = 1
delta = 3 * np.linalg.norm(init_path[:, 0] - init_path[:, -1])

problem = OptimalPathProblem(
    obstacles,
    q0,
    qd,
    e_s_hat,
    phi,
    kappa,
    delta,
    n_points,
    bulge,
    R_list,
    pc_list,
    eps,
    min_path,
    r,
    h,
)

# Bounds: no bounds on x
x_L = np.full(problem.n, -np.inf)
x_U = np.full(problem.n, np.inf)

# Constraint bounds
c_L = np.zeros(problem.m)
c_U = np.zeros(problem.m)
# First 2*d are equalities
c_U[: 2 * 2] = 0.0
# Last N-1 are upper bounds only (||p_{i+1} - p_i|| ≤ δ)
c_U[2 * 2 :] = delta

nlp = Problem(
    n=problem.n, m=problem.m, problem_obj=problem, lb=x_L, ub=x_U, cl=c_L, cu=c_U
)
# nlp.add_option('derivative_test', 'first-order')  # Checks gradient (first derivatives)
# nlp.add_option('derivative_test_print_all', 'yes')  # Prints detailed results
# nlp.add_option('print_level', 12)
options = {
    # Force first-order method (no second derivatives)
    # "hessian_approximation": "exact",
    # "derivative_test": "second-order",
    # "derivative_test_print_all": "yes",
    "hessian_approximation": "limited-memory",
    # 'gradient_approximation': 'finite-difference-values',
    # 'jacobian_approximation': 'finite-difference-values',
    # Configure L-BFGS parameters
    # 'limited_memory_update_type': 'bfgs',  # Standard BFGS update
    # 'limited_memory_max_history': 10,      # History size (10-50 is typical)
    # Disable second-order features
    # 'mehrotra_algorithm': 'no',           # Disable second-order correction
    # 'fast_step_computation': 'no',         # Disable advanced step calc
    "mu_strategy": "adaptive",
    # 'linear_solver': 'mumps',
    # Adjust convergence criteria for first-order method
    "tol": 1e-3,  # Relax tolerance (default 1e-8)
    "max_iter": 1000,  # Increase iteration limit
    "acceptable_iter": 10,  # Stop after 10 "good enough" iters
    # Output control
    "print_level": 5,
    "print_frequency_iter": 10,
}
for key in options.keys():
    nlp.add_option(key, options[key])
# nlp.add_option('max_iter', 100)
print(init_path.flatten().shape)

irow, jcol = problem.hessianstructure()
print(f"Hessian size: {problem.n} x {problem.n}")
print(f"Number of structure entries: {len(irow)}")
# time.sleep(5)
x_opt, info = nlp.solve(init_path.T.flatten())

print(x_opt, x_opt.shape)
print(init_path.shape)
# problem.hessian(init_path.flatten(), [1]*20, 1)
# %%
""" PRINT IPOPT"""
print(x_opt.shape)
print(info["obj_val"])
fig = create_plot(
    obstacles,
    q0,
    qd,
    R_list,
    pc_list,
    p1,
    p2,
    distances,
    x_opt.reshape(n_points, 2).T,
    init_path,
)
fig.show()


# %%
"""Create plotly animation of the path deformation"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


def add_polygon_plt(ax, A, b, alpha=0.5, color=(0.64, 0.62, 0.61)):
    """Add polygon to matplotlib axes"""
    vertices = get_polytope_vertices_opt(A, b)
    hull = ConvexHull(vertices)
    poly_vertices = vertices[hull.vertices]
    ax.fill(
        poly_vertices[:, 0],
        poly_vertices[:, 1],
        color=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1,
    )


def animate_deformation_matplotlib(
    ax,  # Pass axes instead of figure
    init_path,
    obstacles,
    q0,
    qd,
    p1,
    p2,
    distances,
    R_list=[],
    pc_list=[],
    n_iters=10,
    h=0.1,
    r=0.8,
    eps=5e-2,
    bulge=True,
    min_path=True,
    frame_delay=200,  # ms between frames
):
    # Precompute all paths
    paths = [init_path.copy()]
    for iter_ in range(n_iters):
        path, _, *_ = deform_path(
            paths[-1],
            obstacles,
            R_list=R_list,
            pc_list=pc_list,
            h=h,
            r=r,
            eps=eps,
            bulge=bulge,
            min_path=min_path,
        )
        paths.append(path)

    # Setup static elements
    # Create RdBu-like colormap
    cmap = LinearSegmentedColormap.from_list("RdBu", ["#2166ac", "#f7f7f7", "#b2182b"])
    contour = ax.contourf(p1, p2, distances, levels=20, cmap=cmap, alpha=1.0)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Distance")

    # Add polygons
    for A, b in obstacles:
        add_polygon_plt(ax, A, b)

    ax.plot(q0[0, 0], q0[1, 0], "gx", markersize=10, label="q0")
    ax.plot(qd[0, 0], qd[1, 0], "b*", markersize=10, label="qd")

    # Initialize path line
    (path_line,) = ax.plot([], [], "ok-", linewidth=2)

    # Add legend and set limits
    ax.legend()
    ax.set_xlim(np.min(p1), np.max(p1))
    ax.set_ylim(np.min(p2), np.max(p2))
    ax.set_aspect("equal", "box")
    ax.set_title("Path Deformation Animation")
    ax.grid(True)

    # Animation update function
    def update(frame):
        path = paths[frame]
        path_line.set_data(path[0, :], path[1, :])
        ax.set_title(f"Deformation Step: {frame}/{n_iters}")
        return (path_line,)

    # Create animation
    ani = FuncAnimation(
        ax.figure, update, frames=len(paths), interval=frame_delay, blit=True
    )

    return ani


# Create static figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# Generate animation
animation = animate_deformation_matplotlib(
    ax=ax,
    init_path=init_path,
    obstacles=obstacles,
    q0=q0,  # (2,1) array
    qd=qd,  # (2,1) array
    p1=p1,  # X grid coordinates
    p2=p2,  # Y grid coordinates
    distances=distances,  # 2D distance values
    R_list=R_list,  # List of radii for auxiliary circles
    pc_list=pc_list,  # List of center points for auxiliary circles
    n_iters=max_iters,
    h=h,  # Smoothing parameter
    r=r,  # Exponent for distance function
    eps=eps,  # Bulging parameter
    bulge=bulge,  # Whether to apply bulging
    min_path=min_path,
    frame_delay=500,  # Delay between frames in milliseconds
)

# To display in notebook
from IPython.display import HTML, display
import webbrowser
from pathlib import Path


def show_animation(animation, filename="anim.html"):
    path = Path(filename).absolute()
    animation.save(path, writer="html")
    webbrowser.open(f"file://{path}")


show_animation(animation)
# display(HTML(animation.to_jshtml()))

# %%
import numpy as np
import plotly.graph_objects as go

q0 = np.array([-15, -15])
qd = np.array([5, 5])
polygons = generate_random_polygon_set(10, False, q0, qd, max_vertices=20, radius_lim=(0.5, 6),
                                       bbox=(-20, -20, 20, 20), seed=42)

fig = go.Figure()
for A, b, original_vertices in polygons:
    fig.add_trace(go.Scatter(
        x=[q0[0]],
        y=[q0[1]],
        mode='markers',
        marker=dict(symbol='cross')
    ))
    fig.add_trace(go.Scatter(
        x=[qd[0]],
        y=[qd[1]],
        mode='markers',
        marker=dict(symbol='star')
    ))
 
    # seed = 42 + i
    # A, b, original_vertices = generate_random_polygon(
    #     max_vertices=20,
    #     radius_lim=(1, 6),
    #     bbox=(-20, -20, 20, 20),
    #     seed=seed)
    # Slight shrink for numerical safety
    epsilon = 1e-6 * 0
    b_strict = b - epsilon

    try:
        interior_point = find_strictly_feasible_point(A, b)
        halfspaces = np.hstack((A, -b[:, None]))
        hs = HalfspaceIntersection(halfspaces, interior_point)
        reconstructed_vertices = hs.intersections

        # Use ConvexHull to order them
        hull = ConvexHull(reconstructed_vertices)
        ordered_vertices = reconstructed_vertices[hull.vertices]

        x = np.append(ordered_vertices[:, 0], ordered_vertices[0, 0])
        y = np.append(ordered_vertices[:, 1], ordered_vertices[0, 1])

        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers', fill='toself',
            name=f'Polygon {i+1}', line=dict(width=2)
        ))

    except Exception as e:
        print(f"Polygon {i+1} failed to reconstruct: {e}")
# for i in range(4):
#     seed = 42 + i
#     _, _, vertices = generate_random_polygon(seed=seed)
#
#     # Get convex hull to order the vertices
#     hull = ConvexHull(vertices)
#     hull_vertices = vertices[hull.vertices]
#
#     # Close the polygon
#     x = np.append(hull_vertices[:, 0], hull_vertices[0, 0])
#     y = np.append(hull_vertices[:, 1], hull_vertices[0, 1])
#
#     fig.add_trace(go.Scatter(
#         x=x, y=y, mode='lines+markers', fill='toself',
#         name=f'Polygon {i+1}',
#         line=dict(width=2)
#     ))
#
fig.update_layout(
    title='Random Convex Polygons in 2D',
    xaxis_title='x', yaxis_title='y',
    xaxis=dict(scaleanchor='y', scaleratio=1),
    showlegend=True,
    width=700, height=700
)

#%%
# hull = ConvexHull(vertices)

# Get polygon description in the form Ax ≤ b
A = []
b = []
for i in range(num_vertices):
    j = (i + 1) % num_vertices
    edge = vertices[j] - vertices[i]
    normal = np.array([-edge[1], edge[0]])
    # Normalize for numerical stability
    norm = np.linalg.norm(normal)
    if norm > 1e-8:  # Avoid division by zero
        normal = normal / norm

    A.append(normal)
    b.append(np.dot(normal, vertices[i]))
A = np.array(A)
b = np.array(b)

fig.add_trace(
    go.Scatter(
        x=vertices[:, 0],
        y=vertices[:, 1],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Vertices",
    )
)

# # plot normals
# for ai in A:
#     ai = ai.reshape(-1, 1)
#     x_start = -ai[0, 0] * 10
#     y_start = -ai[1, 0] * 10
#     x_end = ai[0, 0] * 10
#     y_end = ai[1, 0] * 10
#     fig.add_trace(go.Scatter(
#         x=[x_start, x_end],
#         y=[y_start, y_end],
#         mode='lines',
#         line=dict(color='blue', width=2),
#         name='Normal'
#     ))


fig.show()

# %%
