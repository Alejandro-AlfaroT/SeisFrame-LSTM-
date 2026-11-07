"""Independent hand traction patches; AREF-MEMBRANE-UNIT-20261103-01."""
import copy
import unittest

import numpy as np

from Design.SMRF_Diaphragm_Actions import (
    compare_mesh_actions, conventional_design_cases, footprint_traction,
    inertial_nodal_forces, rectangular_mesh, resultant, solve_diaphragm_actions,
)


def mesh(n=1, beams=()):
    return rectangular_mesh(np.linspace(0, 12, n+1), np.linspace(0, 8, n+1),
                            thickness_in=2, elastic_modulus_ksi=3000,
                            poisson=0.2, beams=beams)


def patch_force(m, sx=0.0, sy=0.0, tau=0.0):
    """Integrate prescribed constant stress tractions on four physical edges."""
    xy = np.asarray(m["nodes"])
    f = np.zeros_like(xy)
    nx, ny, t = len(m["x_in"]), len(m["y_in"]), m["thickness_in"]
    stress = np.array([[sx, tau], [tau, sy]])
    edges = []
    for j in range(ny-1):
        edges.extend([(j*nx, (j+1)*nx, [-1, 0]),
                      (j*nx+nx-1, (j+1)*nx+nx-1, [1, 0])])
    for i in range(nx-1):
        edges.extend([(i, i+1, [0, -1]),
                      ((ny-1)*nx+i, (ny-1)*nx+i+1, [0, 1])])
    for i, j, normal in edges:
        edge_force = stress @ normal * t * np.linalg.norm(xy[j]-xy[i])
        f[i] += edge_force/2
        f[j] += edge_force/2
    return f


def component(m, force, ident="p", role="transfer"):
    z = np.zeros((len(m["nodes"]), 2))
    return {"id": ident, "role": role, "provenance": "independent constant-stress boundary traction",
            "inertia": z.tolist(), "columns_above_on_floor": force.tolist(),
            "columns_below_on_floor": z.tolist()}


def solve_patch(m, stress=(0.1, 0.05, 0.025), extra=None):
    f = patch_force(m, *stress)
    if extra is not None:
        f += extra
    return solve_diaphragm_actions(m, [component(m, f)], [{"id": "patch", "factors": {"p": 1}}],
                                  cuts=[{"axis": "x", "coordinate_in": 5.0},
                                        {"axis": "y", "coordinate_in": 3.0}])


class MembraneActions(unittest.TestCase):
    def test_constant_stress_patch_irregular_mesh_and_signed_shear(self):
        m = rectangular_mesh([0, 2, 7, 12], [0, 3, 8], thickness_in=2,
                             elastic_modulus_ksi=3000, poisson=0.2)
        r = solve_patch(m, (0.1, -0.05, -0.025))
        case = r["cases"][0]
        for s in case["slab_samples"]:
            np.testing.assert_allclose([s["Nxx_kip_per_in"], s["Nyy_kip_per_in"], s["Nxy_kip_per_in"]],
                                       [0.2, -0.1, -0.05], atol=2e-12)
        for cut in case["cuts"]:
            np.testing.assert_allclose(cut["balance_error"], [0, 0, 0], atol=1e-10)
        self.assertLess(case["gauge_reaction_norm_kip"], 1e-10)
        self.assertFalse(r["strength_verified"])
        self.assertFalse(r["mesh_convergence_verified"])

    def test_axial_beam_web_compatible_strain_no_slab_overlap(self):
        # sx=.3 ksi, sy=0 gives eps_x=.0001. Web EA=3000*4:
        # Nweb=1.2 kip, while the full slab retains its own .6 kip/in.
        beam = {"id": "b0", "member_id": "frame_beam", "i": 0, "j": 1,
                "area_web_in2": 4.0}
        m = mesh(beams=[beam])
        extra = np.zeros((4, 2))
        extra[0, 0], extra[1, 0] = -1.2, 1.2
        r = solve_patch(m, (0.3, 0, 0), extra)
        self.assertAlmostEqual(r["cases"][0]["beam_axial"][0]["N_kip"], 1.2, places=11)
        for s in r["cases"][0]["slab_samples"]:
            self.assertAlmostEqual(s["Nxx_kip_per_in"], 0.6, places=11)

    def test_mesh_witness_common_physical_locations(self):
        coarse, fine = solve_patch(mesh(1)), solve_patch(mesh(3))
        witness = compare_mesh_actions(coarse, fine,
                                      [{"id": "physical", "case_id": "patch", "x_in": 5.0, "y_in": 3.0}])
        self.assertTrue(witness["passed"])
        with self.assertRaisesRegex(ValueError, "interface"):
            compare_mesh_actions(coarse, fine, [{"case_id": "patch", "x_in": 4.0, "y_in": 3.0}])

    def test_unbalanced_component_cannot_be_hidden_by_other_component(self):
        m = mesh()
        f = patch_force(m, .1)
        a, b = component(m, f, "a"), component(m, -f, "b")
        a["columns_above_on_floor"][0][0] += 1
        b["columns_above_on_floor"][0][0] -= 1
        with self.assertRaisesRegex(ValueError, "not an equilibrated"):
            solve_diaphragm_actions(m, [a, b], [{"id": "cancel", "factors": {"a": 1, "b": 1}}])

    def test_inertia_and_traction_helpers_preserve_force_and_torsion(self):
        m = mesh(2)
        w = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        f = inertial_nodal_forces(m, w, [10, -4], 21)
        np.testing.assert_allclose(resultant(m["nodes"], f, [6, 4]), [10, -4, 21], atol=1e-12)
        traction = footprint_traction(m, {0: 1, 1: 1, 3: 1, 4: 1}, [-3, 2],
                                      application_xy_in=[3, 2], moment_kip_in=7)
        np.testing.assert_allclose(resultant(m["nodes"], traction, [3, 2]), [-3, 2, 7], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "centroid"):
            footprint_traction(m, {0: 1}, [1, 0], application_xy_in=[3, 2])

    def test_transfer_cannot_contain_local_inertia(self):
        m = mesh()
        c = component(m, patch_force(m, .1))
        c["inertia"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "no local floor inertia"):
            solve_diaphragm_actions(m, [c], [{"id": "bad", "factors": {"p": 1}}])

    def test_separate_signed_components_superpose_before_envelope(self):
        m = mesh()
        a = component(m, patch_force(m, .1), "local", "local_inertia")
        b = component(m, patch_force(m, -.08), "transfer", "transfer")
        cases = [{"id": "raw", "factors": {"local": 1, "transfer": 1}},
                 {"id": "fpx", "factors": {"local": 2, "transfer": 1}}]
        r = solve_diaphragm_actions(m, [a, b], cases)
        self.assertAlmostEqual(r["cases"][0]["slab_samples"][0]["Nxx_kip_per_in"], .04)
        self.assertAlmostEqual(r["cases"][1]["slab_samples"][0]["Nxx_kip_per_in"], .24)
        # Scaling the cancelling net would have given .08, which is wrong.

    def test_conventional_collector_cases_do_not_cap_or_merge_transfer(self):
        cases = conventional_design_cases(local_component_ids=["a"], transfer_component_ids=["t"],
                                          elf_floor_force_kip=100, fpx_eq_kip=250,
                                          fpx_min_kip=80, fpx_max_kip=160, omega0=3)
        byid = {c["id"]: c for c in cases}
        self.assertEqual(byid["diaphragm_fpx"]["factors"], {"a": 1.6, "t": 1})
        self.assertEqual(byid["collector_eq_omega"]["factors"], {"a": 7.5, "t": 3})
        self.assertEqual(byid["collector_minimum"]["factors"], {"a": .8, "t": 1})

    def test_bad_mesh_overlap_proxy_and_nonfinite_load_rejected(self):
        with self.assertRaises(ValueError):
            rectangular_mesh([0, 1, 1], [0, 1], thickness_in=1, elastic_modulus_ksi=3000)
        with self.assertRaises(ValueError):
            mesh(beams=[{"id": "diag", "i": 0, "j": 3, "area_web_in2": 2}])
        m = mesh()
        c = component(m, patch_force(m, .1))
        c["columns_below_on_floor"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            solve_diaphragm_actions(m, [c], [{"id": "nan", "factors": {"p": 1}}])


if __name__ == "__main__":
    unittest.main()
