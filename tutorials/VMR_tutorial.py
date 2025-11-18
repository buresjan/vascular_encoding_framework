"""
Basic tutorial of the Vascular Encoding Framework using a model publicly available on the
Vascular Model Repository (https://www.vascularmodel.com/index.html).

To run this tutorial the user need to donwload the case 0008_H_AO_H It can be found
using the search functionality on the filters tab of the repository web.

After downloading and unziping it, the user can either move the directory inside the tutorials
directory or modify this file to set the path to the unziped directory.

"""

import os

import numpy as np
import pyvista as pv
import vascular_encoding_framework as vef

# Prefer bundled dataset; fall back to user path if not found.
_here = os.path.dirname(__file__)
case_path = os.path.join(_here, "0008_H_AO_H")
if not os.path.isdir(case_path):
    case_path = f"{os.path.expanduser('~')}/tmp/0008_H_AO_H"  # Modify if needed
mesh_path = os.path.join(case_path, "Meshes", "0091_0001.vtp")


# Since the mesh as it is provided is a closed mesh, let us load it with
# pyvista and open the caps.
mesh = pv.read(mesh_path)
mesh = mesh.threshold(value=0.1, scalars="CapID", method="lower").extract_surface()

# Initialize vef.VascularMesh with the opened mesh.
vmesh = vef.VascularMesh(mesh)

# Once initialized, since the mesh is open,the boundary tree is initialized without hierarchy
# Hence, let us inspect the ids attributed to each boundary to define the hierarchy.
SHOW_BOUNDARIES = True
if SHOW_BOUNDARIES:
    vmesh.plot_boundary_ids()
print("Available boundary ids:", vmesh.boundaries.enumerate())

# After visualizing it, we can define the desired hierarchy as follows:
# Note: Current VEF assigns ids like 'B0', 'B1', ...
# Adjust the mapping below to match the ids shown above.
hierarchy = {
    "B5": {
        "id": "B5",
        "parent": None,
        "children": {"B0"},
    },
    "B0": {"id": "B0", "parent": "B5", "children": {"B3", "B4", "B1"}},
    "B3": {"id": "B3", "parent": "B0", "children": {"B2"}},
    "B4": {"id": "B4", "parent": "B0", "children": {}},
    "B1": {"id": "B1", "parent": "B0", "children": {}},
    "B2": {"id": "B2", "parent": "B3", "children": {}},
}
vmesh.set_boundary_data(hierarchy)


# Let's compute the centerline for the encoding
# First we extract a domain approximation of the lumen, here we chose de seekers mode, it works
# well with more tubelike geometries and we can add some parameter tunning using the method_params
# argument. If debug argument is set to True, some plots are shown of the process.
cl_domain = vef.centerline.extract_centerline_domain(
    vmesh=vmesh, params={"method": "seekers", "reduction_rate": 0.85, "eps": 1e-3}, debug=False
)


# Once the centerline domain has been extracted, we can compute the path tree according to the hierarchy we defined.
cp_xtractor = vef.centerline.CenterlinePathExtractor()
cp_xtractor.adjacency_factor = 0.5
# Again setting this argument to true, results in some plots of the procedure..
cp_xtractor.debug = False
cp_xtractor.set_centerline_domain(cl_domain)
cp_xtractor.set_vascular_mesh(vmesh, update_boundaries=True)
cp_xtractor.compute_paths()

# Before jumping into analytic/spline encoding let us define the knots for each branch, i.e.
# degrees of freedom for the fitting procedure.
knot_params = {
    "5": {"cl_knots": None, "tau_knots": None, "theta_knots": None},
    "0": {"cl_knots": 15, "tau_knots": 19, "theta_knots": 19},
    "3": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "4": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "1": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "2": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
}

# We are in conditions of defining the centerline tree.
# In the current API, branch-specific parameters are passed under each branch id.
branch_specific = {
    k: {"n_knots": v["cl_knots"]}
    for k, v in knot_params.items()
    if v["cl_knots"] is not None
}
cl_tree = vef.CenterlineTree.from_multiblock_paths(cp_xtractor.paths, **branch_specific)

# The centerline on its own allows us to compute some useful fields like the Vessel Coordinates, but
# first, let's check that centerline has been well computed and let us inspect how the adapted frame
# looks like. The plotting helper is now a method of CenterlineTree.
cl_tree.plot_adapted_frame(vmesh=vmesh, scale=0.5)

# The computation of centerline association and the vessel coordinates usually takes a while.
bid = [
    cl_tree.get_centerline_association(
        p=vmesh.points[i],
        n=vmesh.get_array(name="Normals", preference="point")[i],
        method="scalar",
        thrs=60,
    )
    for i in range(vmesh.n_points)
]
vcs = np.array(
    [cl_tree.cartesian_to_vcs(p=vmesh.points[i], cl_id=bid[i]) for i in range(vmesh.n_points)]
)
vmesh["cl_association"] = bid
vmesh["tau"] = vcs[:, 0]
vmesh["theta"] = vcs[:, 1]
vmesh["rho"] = vcs[:, 2]

# Now, lets visualize the computations.
vmesh.plot(scalars="cl_association")
vmesh.plot(scalars="tau")
vmesh.plot(scalars="theta")
vmesh.plot(scalars="rho")
