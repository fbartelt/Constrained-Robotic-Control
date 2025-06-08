# %%
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from itertools import combinations
from scipy.spatial import ConvexHull


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

    return np.array(vertices)


# def phi(s, h=0.1, *args, **kwargs):
#     if s < 0:
#         val = 0
#         grad = 0
#     else:
#         val = (s**3) / (2 * (s + h))
#         grad = (3 * s**2) / (2 * (s + h)) - (s**3) / (2 * (s + h)**2)

#     return val, grad


def phi(s, h=0.5, *args, **kwargs):
    """Smooth approximation of the distance function
    h*log(cosh(x)), derivative is h*tanh(x), second derivative is h*sech(x)**2
    """
    if s < 0:
        val = 0
        grad = 0
    else:
        val = np.log(np.cosh(s / h)) * h
        grad = np.tanh(s / h)

    return val, grad


def inner_distance(f, p, A, b, r=0.1, *args, **kwargs):
    """Uses short-circuit algorithm"""
    N, m = A.shape
    in_dists, in_grad = [], np.zeros((1, m))

    for i, ai in enumerate(A):
        ai = ai.reshape(-1, 1)
        s = (b[i] - ai.T @ p).item()
        f_val, f_grad = f(s, *args, **kwargs)
        if f_val != 0:
            in_dists.append(f_val ** (-1 / r))
            in_grad += (-1 / r) * (f_val ** ((-1 / r) - 1)) * f_grad * (-ai.T)
        else:
            in_dists.append(np.inf)

    S = np.sum(in_dists)
    in_dist = (1 / N * S) ** (-r)
    first_chain = -r * (in_dist / S)  # -r/N * (S ** (-r - 1))
    in_grad = first_chain * in_grad.reshape(1, -1)
    return in_dist, in_grad


def outter_distance(f, p, A, b, *args, **kwargs):
    N, m = A.shape
    out_dist, out_grad = [], np.zeros((1, m))

    for i, ai in enumerate(A):
        ai = ai.reshape(-1, 1)
        s = (ai.T @ p - b[i]).item()
        f_val, f_grad = f(s, *args, **kwargs)
        out_dist.append(f_val)
        out_grad += f_grad * ai.T

    out_dist = (1 / N) * np.sum(out_dist)
    out_grad = (1 / N) * out_grad.reshape(1, -1)
    return out_dist, out_grad


def bulging(dist, grad_dist, p, pc, R, eps=1e-3, out=True):
    p_ = np.array(p).reshape(-1, 1)
    pc_ = np.array(pc).reshape(-1, 1)
    grad = np.array(grad_dist).reshape(1, -1)
    rho = 0.5 * (((p_ - pc_).T @ (p_ - pc_)) - R**2).item()
    if not out:
        rho = -rho
    grad_rho = (p_ - pc_).T
    sqrt_term = np.sqrt((eps**2 * rho**2) + (1 - 2 * eps) * (dist**2))
    bulge_dist = eps * rho + sqrt_term
    bulge_grad = eps * grad_rho + 0.5 * (1 / sqrt_term) * (
        2 * eps * rho * grad_rho + 2 * (1 - 2 * eps) * dist * grad
    )
    return bulge_dist, bulge_grad


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
    *args,
    **kwargs,
):
    out_dist, out_grad = outter_distance(f, p, A, b, *args, **kwargs)
    in_dist, in_grad = inner_distance(f, p, A, b, *args, **kwargs)
    if bulge:
        in_dist, in_grad = bulging(in_dist, in_grad, p, pc, R, eps=eps, out=False)
        out_dist, out_grad = bulging(out_dist, out_grad, p, pc, R, eps=eps)
    in_dist = -1 * in_dist

    if kind == "out":
        return out_dist, out_grad
    elif kind == "in":
        return in_dist, in_grad
    else:
        return out_dist + in_dist, out_grad + in_grad


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


def minminmin(x_fixed, x_vary, y_fixed, y_vary, A_list, b_list, func, *args, **kwargs):
    f_verticals = []
    f_horizontals = []

    for A, b in zip(A_list, b_list):
        ith_vertical, ith_horizontal = [], []

        for x in x_vary:
            p_hori = np.array([x, y_fixed])
            ith_horizontal.append(func(p_hori, A, b, *args, **kwargs))
        f_horizontals.append(np.min(ith_horizontal).item())

        for y in y_vary:
            p_vert = np.array([x_fixed, y])
            ith_vertical.append(func(p_vert, A, b, *args, **kwargs))
        f_verticals.append(np.min(ith_vertical).item())

    return np.minimum(np.min(f_verticals), np.min(f_horizontals))


q0 = np.array([-1.15, 1.6]).reshape(-1, 1)
qd = np.array([2.3, -1.75]).reshape(-1, 1)
qd = np.array([1.7, -1.75]).reshape(-1, 1)

# q0 = np.array([-1.1, 0.45]).reshape(-1, 1)
# qd = np.array([1.1, 0.45]).reshape(-1, 1)

obstacles = [(A1, b1), (A2, b2)]
pc_list = [[0, 0.25], [1.5, -1]]
R_list = [0.91, 0.71]
# obstacles = [(A1, b1)]

lambda_ = np.linspace(0, 1, 100)
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
min_path = False
k = 5e-2

def deform_path(
    init_path,
    obstacles,
    R_list=[],
    pc_list=[],
    h=0.1,
    r=0.8,
    eps=5e-2,
    bulge=False,
    min_path=False,
    k=1.0,
):
    path = init_path.copy()
    dists, grads = [], []
    for j, p_ in enumerate(init_path.T):
        p = p_.copy().reshape(-1, 1)
        dist, grad = 0, np.zeros((1, path.shape[0]))
        for i, obstacle in enumerate(obstacles):
            pc = pc_list[i]
            R = R_list[i]
            A, b = obstacle
            dist_, grad_ = e_s_hat(
                p, A, b, phi, kind="in", bulge=bulge, R=R, pc=pc, eps=eps, r=r, h=h
            )
            dist += dist_
            grad += grad_.reshape(1, -1)
        dists.append(dist)
        grads.append(grad.T / (np.linalg.norm(grad) + 1e-8))
    min_idx = np.argmin(dists)
    min_dist, min_grad = dists[min_idx], grads[min_idx]
    # path = path - (min_grad * np.abs(dists))
    grad_length = np.zeros((1, path.shape[1]))
    if min_path:
        direction = (path[:, 1:] - path[:, :-1])
        grad_length = k *  direction / (np.linalg.norm(direction, axis=0) + 1e-8)
        grad_length = np.hstack([np.array([0, 0]).reshape(2, -1), grad_length])
        grad_length[:, -1] = np.array([0, 0])


    # path = path + (min_grad * dists)
    for j in range(path.shape[1]):
        const_obs = [b - A @ path[:, j].reshape(-1, 1) for A, b in obstacles]
        err = np.max(const_obs)
        coeff = dists[j]
        path[:, j] += (grads[j] * coeff).ravel() + grad_length[:, j].ravel()

    return path, min_dist, min_grad


for _ in range(10):
    path, min_dist, min_grad = deform_path(
        path,
        obstacles,
        R_list=R_list,
        pc_list=pc_list,
        h=h,
        r=r,
        eps=eps,
        bulge=bulge,
        min_path=min_path,
        k=k
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
for j, pi in enumerate(P):
    # print(j)
    pi = pi.reshape(-1, 1)
    kind = "both"
    bulge = True
    d1, grad_d1 = e_s_hat(
        pi,
        A1,
        b1,
        phi,
        kind=kind,
        bulge=bulge,
        pc=pc_list[0],
        R=R_list[0],
        eps=eps,
        r=r,
        h=h,
    )
    d2, grad_d2 = e_s_hat(
        pi,
        A2,
        b2,
        phi,
        kind=kind,
        bulge=bulge,
        pc=pc_list[1],
        R=R_list[1],
        eps=eps,
        r=r,
        h=h,
    )
    d1_list.append(d1)
    d2_list.append(d2)
    e_s = d1 + d2
    # k = 1e-6
    # e_s = -1/k * np.log(np.exp(-k * d1) + np.exp(-k * d2))
    e_s = np.minimum(d1, d2)  # OK outside, terrible inside
    x, y = pi.ravel()
    # e_s = minminmin(x.item(), p1, y.item(), p2, [A1, A2], [b1, b2], e_s_hat, f=phi, kind=kind, r=0.1, h=0.1)
    distances.append(e_s)

distances = np.array(distances).reshape(P1.shape)


# Plot polygon and level sets (2D)
def add_polygon(fig, A, b):
    vertices = get_polytope_vertices_opt(A, b)
    hull = ConvexHull(vertices)
    # Add polygon
    fig.add_trace(
        go.Scatter(
            x=np.append(vertices[hull.vertices, 0], vertices[hull.vertices[0], 0]),
            y=np.append(vertices[hull.vertices, 1], vertices[hull.vertices[0], 1]),
            fill="toself",
            fillcolor="rgba(163, 159, 158, 0.2)",
            line=dict(color="rgba(163, 159, 158, 1)"),
        )
    )


fig = go.Figure()
add_polygon(fig, A1, b1)
add_polygon(fig, A2, b2)

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


# fig.add_trace(go.Scatter(
#     x=A[:, 0],
#     y=A[:, 1],
#     mode='markers',
#     marker=dict(color='red', size=8),
#     name='Vertices'
# ))

# Add level sets
contour = go.Contour(
    x=p1,
    y=p2,
    z=distances,
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
    frame_delay=200,  # ms between frames
):
    # Precompute all paths
    paths = [init_path.copy()]
    for iter_ in range(n_iters):
        path, _, _ = deform_path(
            paths[-1],
            obstacles,
            R_list=R_list,
            pc_list=pc_list,
            h=h,
            r=r,
            eps=eps,
            bulge=bulge,
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
    n_iters=10,
    h=h,  # Smoothing parameter
    r=r,  # Exponent for distance function
    eps=eps,  # Bulging parameter
    bulge=bulge,  # Whether to apply bulging
    frame_delay=500,  # Delay between frames in milliseconds
)

# To display in notebook
from IPython.display import HTML

HTML(animation.to_jshtml())
# %%


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
        np.maximum(xmin + radius, ymin + radius),
        np.minimum(xmax - radius, ymax - radius),
    )
    vertices += np.array([offset, offset])
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
    return A, b, vertices


fig = go.Figure()

max_vertices = 20
radius_lim = (5, 10.0)
bbox = (-500, -500, 500, 500)
seed = None
fig = go.Figure()
rng = np.random.default_rng(seed)
num_vertices = rng.integers(3, max_vertices + 1).item()
radius = rng.uniform(radius_lim[0], radius_lim[1])
angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
vertices = np.array([radius * np.cos(angles), radius * np.sin(angles)]).T
# Calculate safe translation boundaries
xmin, ymin, xmax, ymax = bbox
offset = rng.uniform(
    np.maximum(xmin + radius, ymin + radius), np.minimum(xmax - radius, ymax - radius)
)
vertices += np.array([offset, offset])
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
