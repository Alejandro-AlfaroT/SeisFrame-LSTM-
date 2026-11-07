import copy
import unittest
import numpy as np
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem
from Design.SMRF_Common_Compatibility import displacement_audit
from test_coupled_floor_geometry import fixture


class CompatibilityAdapterTests(unittest.TestCase):
    def test_actual_imposed_response_and_independent_rotated_node_audit(self):
        definition=fixture()
        model=CoupledFloorSystem(definition,[dict(id='G',dead=1.,live=.5)],source_identity='actual_fixture',
            stiffness_basis='independent physical fixture',subdivisions_per_bay=2,
            beam_bending_factor=.5,slab_bending_factor=.5,web_torsion_factor=1.,column_bending_factor=.5,
            max_shells_per_floor=1024,max_coupled_dofs=1500)
        levels=[dict(floor=k,design_in=dict(ux=.01*k,uy=-.004*k,rz_rad=.0002*k)) for k in (1,2)]
        result=model.solve_imposed_floor_displacements([dict(id='actual_compat',gravity_case_id='G',
            reference_xy_in=[3.,7.],sign=-1.,axis='x',levels=levels)],pdelta=False)
        rows=displacement_audit(definition,result,0)
        self.assertEqual(len(rows),2)
        self.assertLess(max(v for r in rows for v in r['residual_in'].values()),1e-12)
        ids=[list(v) for v in result['frame']['physical_dof_ids']];u=result['frame']['cases'][0]['physical_displacements']
        for node in definition['nodes']:
            x,y,z=node['xyz']
            if not z:continue
            k=z/definition['geometry']['story_h']
            self.assertAlmostEqual(u[ids.index([node['id'],1])],-.01*k+(y-7)*.0002*k,places=12)
        changed=copy.deepcopy(result);at=ids.index([definition['floors'][0]['supports'][0]['id'],1])
        changed['frame']['cases'][0]['physical_displacements'][at]+=.001
        with self.assertRaisesRegex(ValueError,'Cd/Ie targets'):displacement_audit(definition,changed,0)
        changed=copy.deepcopy(result);changed['frame']['physical_dof_ids'][0]=ids[1]
        with self.assertRaisesRegex(ValueError,'unique finite'):displacement_audit(definition,changed,0)
