"""
mesh_gen.py -- Gmsh Python API mesh generator for the triangle+rectangle
"active nematic in an elastic solid" domain.

Geometry (right-angle vertex at the origin):
    A = (0, 0)      90 deg vertex, on the rectangle's near edge
    B = (L, 0)      45 deg vertex, on the rectangle's near edge
    C = (0, -L)     45 deg vertex, the free tip (sharp point)
    Rectangle: x in [0, L], y in [0, H],  H >> L

A lies on the straight segment between (0, H) and C (both on x = 0), so the
union of the triangle {A, B, C} and the rectangle [0, L] x [0, H] is the
simply-connected quadrilateral

    C -> B -> (L, H) -> (0, H) -> C

Boundary physical groups:
    actuated : (0, H) -> C     length H + L, the moving wall
    fixed    : B -> (L, H)     the fixed wall
    free     : C -> B (hypotenuse) and (L, H) -> (0, H) (far short edge)
"""
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile

# ---------------------------------------------------------------------------
# Geometry / mesh-size parameters -- tune here
# ---------------------------------------------------------------------------
L = 1.0                    # triangle leg length / rectangle width
H = 5.0 * L                 # rectangle height (H >> L)

LC_TIP = L / 60.0           # target element size near the tip
LC_BULK = L / 20.0          # target element size in the bulk
R_REFINE = 0.5 * L          # radius within which element size ~= LC_TIP
R_TRANSITION = 1.0 * L      # radius beyond which element size ~= LC_BULK

# Physical group tags
TAG_ACTUATED = 1
TAG_FIXED = 2
TAG_FREE = 3
TAG_DOMAIN = 100

MESH_FILENAME = "output/mesh.xdmf"


def build_mesh(comm: MPI.Comm = MPI.COMM_WORLD, verbose: bool = False):
    """Build the tagged mesh with the Gmsh Python API and hand it to dolfinx.

    Returns (mesh, cell_tags, facet_tags). facet_tags carries TAG_ACTUATED,
    TAG_FIXED, TAG_FREE on the corresponding boundary facets.
    """
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    if comm.rank == 0:
        gmsh.model.add("muscle_domain")

        C = gmsh.model.geo.addPoint(0.0, -L, 0.0, LC_BULK)
        B = gmsh.model.geo.addPoint(L, 0.0, 0.0, LC_BULK)
        P3 = gmsh.model.geo.addPoint(L, H, 0.0, LC_BULK)
        P4 = gmsh.model.geo.addPoint(0.0, H, 0.0, LC_BULK)

        l_hyp = gmsh.model.geo.addLine(C, B)      # free: hypotenuse
        l_fixed = gmsh.model.geo.addLine(B, P3)   # fixed wall
        l_top = gmsh.model.geo.addLine(P3, P4)    # free: far short edge
        l_act = gmsh.model.geo.addLine(P4, C)     # actuated wall

        loop = gmsh.model.geo.addCurveLoop([l_hyp, l_fixed, l_top, l_act])
        surf = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [l_act], TAG_ACTUATED, name="actuated")
        gmsh.model.addPhysicalGroup(1, [l_fixed], TAG_FIXED, name="fixed")
        gmsh.model.addPhysicalGroup(1, [l_hyp, l_top], TAG_FREE, name="free")
        gmsh.model.addPhysicalGroup(2, [surf], TAG_DOMAIN, name="domain")

        # --- local refinement at the tip C: Distance + Threshold fields ---
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "PointsList", [C])

        thresh_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", LC_TIP)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", LC_BULK)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", R_REFINE)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", R_TRANSITION)

        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh, cell_tags, facet_tags


def save_mesh(mesh, cell_tags, facet_tags, filename: str = MESH_FILENAME):
    """Save the tagged mesh to XDMF for inspection in ParaView."""
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with XDMFFile(mesh.comm, filename, "w") as xdmf:
        xdmf.write_mesh(mesh)
        cell_tags.name = "cell_tags"
        facet_tags.name = "facet_tags"
        xdmf.write_meshtags(cell_tags, mesh.geometry)
        xdmf.write_meshtags(facet_tags, mesh.geometry)


if __name__ == "__main__":
    mesh, cell_tags, facet_tags = build_mesh(verbose=True)
    save_mesh(mesh, cell_tags, facet_tags)
    if mesh.comm.rank == 0:
        print(f"Mesh built: {mesh.topology.index_map(2).size_global} cells, "
              f"{mesh.topology.index_map(0).size_global} vertices")
        print(f"Saved to {MESH_FILENAME}")
