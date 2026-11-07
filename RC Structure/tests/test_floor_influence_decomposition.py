"""Hand statics only; never imports or calls OpenSees."""
import copy
import unittest

import numpy as np

from Design.SMRF_Floor_Influences import (
    decompose_floor_influences, floor_free_bodies, membrane_components,
    reconstruct_from_basis, subtract_response, verify_superposition,
)
from Design.SMRF_Diaphragm_Actions import rectangular_mesh, resultant


def hand_response(loads):
    """Two stories/two resisting lines, changing share .8/.2 -> .5/.5.

    The hand equilibrium model deliberately has nonzero self-equilibrated floor1
    transfer under a floor2 force. It is a decomposition reference, not frame FE.
    """
    fx = np.asarray(loads)[:, 0]
    rows = []
    for story, shares in ((1, (.8, .2)), (2, (.5, .5))):
        shear = fx[story-1:].sum()
        for line, share in enumerate(shares):
            force = [0.]*12
            force[0], force[6] = -share*shear, share*shear
            rows.append({"tag": str((story-1)*2+line+1), "i_level": story-1, "j_level": story,
                         "i_node": (story-1)*2+line, "j_node": story*2+line,
                         "i_xyz": [line*12., 4., (story-1)*120.],
                         "j_xyz": [line*12., 4., story*120.], "global_force": force})
    delta = .01*np.array([[2.,1.],[1.,3.]]) @ fx
    return {"columns": rows, "floor_masters": [{"node": 100001, "xyz": [6.,4.,120.]},
                                                 {"node": 100002, "xyz": [6.,4.,240.]}],
            "master_displacements": [[float(d),0.,0.] for d in delta]}


def basis():
    out={"num_floor":2,"source_identity":"hand-statics-changing-frame-share", "basis":[],"witnesses":[]}
    for floor in (1,2):
        load=np.zeros((2,3));load[floor-1,0]=1
        out["basis"].append({"source_floor":floor,"dof":"x","amplitude":1.,"response":hand_response(load)})
    return out


class FloorInfluences(unittest.TestCase):
    def test_column_on_floor_sign_and_separate_above_below(self):
        loads=[[10,0,0],[20,0,0]]
        floors=floor_free_bodies(hand_response(loads),loads)
        self.assertTrue(all(f["equilibrium_passed"] for f in floors))
        self.assertEqual([r["Fx_kip"] for r in floors[0]["columns_above_on_floor"]],[10.,10.])
        self.assertEqual([r["Fx_kip"] for r in floors[0]["columns_below_on_floor"]],[-24.,-6.])
        self.assertEqual(floors[1]["columns_above_on_floor"],[])

    def test_verified_decomposition_retains_transfer_despite_zero_floor_resultant(self):
        inf=basis();loads=[[10,0,0],[20,0,0]]
        check=verify_superposition(inf,loads,hand_response(loads))
        self.assertTrue(check["passed"])
        inf["witnesses"].append({"loads":loads,"verification":check})
        local,transfer=decompose_floor_influences(inf,loads,1)
        self.assertEqual(local["applied_load"],[10.,0.,0.])
        self.assertEqual(transfer["applied_load"],[0.,0.,0.])
        above=[r["Fx_kip"] for r in transfer["columns_above_on_floor"]]
        below=[r["Fx_kip"] for r in transfer["columns_below_on_floor"]]
        np.testing.assert_allclose(np.array(above)+below,[-6,6])
        self.assertTrue(transfer["equilibrium_passed"])

    def test_missing_full_signed_witness_and_nonlinear_change_rejected(self):
        inf=basis();loads=[[10,0,0],[20,0,0]]
        with self.assertRaisesRegex(ValueError,"witness"):
            decompose_floor_influences(inf,loads,1)
        direct=hand_response(loads)
        direct["columns"][0]["global_force"][6]+=.01
        self.assertFalse(verify_superposition(inf,loads,direct)["passed"])
        inf["witnesses"]=[{"loads":loads,"verification":{"passed":True}}]
        with self.assertRaisesRegex(ValueError,"witness"):
            decompose_floor_influences(inf,[[-10,0,0],[20,0,0]],1)

    def test_gravity_subtraction_and_geometry_guard(self):
        base=hand_response([[3,0,0],[4,0,0]])
        total=hand_response([[13,0,0],[24,0,0]])
        inc=subtract_response(total,base)
        expected=hand_response([[10,0,0],[20,0,0]])
        for a,b in zip(inc["columns"],expected["columns"]):
            np.testing.assert_allclose(a["global_force"],b["global_force"])
        total["columns"][0]["i_xyz"][0]=1
        with self.assertRaisesRegex(ValueError,"geometry"):
            subtract_response(total,base)

    def test_nonzero_load_cannot_silently_miss_basis(self):
        with self.assertRaisesRegex(ValueError,"no corresponding"):
            reconstruct_from_basis(basis(),[[1,1,0],[2,0,0]])

    def test_mapping_to_membrane_preserves_footprints_and_local_transfer(self):
        inf=basis();loads=[[10,0,0],[20,0,0]]
        check=verify_superposition(inf,loads,hand_response(loads))
        inf["witnesses"]=[{"loads":loads,"verification":check}]
        comps=decompose_floor_influences(inf,loads,1)
        m=rectangular_mesh([0,6,12],[0,4,8],thickness_in=2,elastic_modulus_ksi=3000)
        # Physical vertical footprint on each frame line has centroid y4.
        footprints={"1":{0:1,3:2,6:1},"2":{2:1,5:2,8:1},
                    "3":{0:1,3:2,6:1},"4":{2:1,5:2,8:1}}
        mapped=membrane_components(m,comps,column_footprints=footprints,nodal_mass_weights=[1]*9)
        for comp in mapped:
            total=sum(np.asarray(comp[k]) for k in ("inertia","columns_above_on_floor","columns_below_on_floor"))
            np.testing.assert_allclose(resultant(m["nodes"],total),[0,0,0],atol=1e-12)
        self.assertGreater(np.max(np.abs(mapped[1]["columns_above_on_floor"])),0)
        self.assertEqual(np.max(np.abs(mapped[1]["inertia"])),0)


if __name__ == "__main__":
    unittest.main()
