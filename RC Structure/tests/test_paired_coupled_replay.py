import copy,json,unittest
import numpy as np
from test_coupled_floor_geometry import fixture,build
from test_coupled_floor_compatibility import imposed


class PairedCoupledReplayTests(unittest.TestCase):
    def test_actual_gravity_imposed_and_extra_load_json_replay(self):
        system=build(fixture(),[dict(id='DL',dead=1.,live=1.)])
        def saved(value):return json.loads(json.dumps(value))
        gravity=system.solve(pdelta=True);self.assertEqual(gravity['status'],'equilibrated')
        self.assertTrue(all(type(w['node_id']) is int for f in saved(gravity)['floors'] for w in f['responses'][0]['joint_wrenches']))
        self.assertEqual(system.solve(pdelta=True,saved_response=saved(gravity))['status'],'equilibrated')
        compatibility=system.solve_imposed_floor_displacements([imposed()],pdelta=True)
        self.assertEqual(system.solve_imposed_floor_displacements([imposed()],pdelta=True,saved_response=saved(compatibility))['status'],'equilibrated')
        extra={}
        for row in system.floors:
            f=np.zeros((len(row['assembly'].dof_ids),1));f[0]=.01;f[6]=-.01;extra[row['assembly'].id]=f
        kwargs=dict(baseline_case_id='DL',case_ids=['extra'],input_identity='paired_replay260',pdelta=True)
        additional=system.solve_additional_floor_loads(extra,**kwargs)
        replay=system.solve_additional_floor_loads(extra,**kwargs,saved_response=saved(additional['total_response']))
        self.assertEqual(additional['status'],'equilibrated');self.assertEqual(replay['status'],'equilibrated')
        for tag,c in additional['increments']['extra']['columns'].items():self.assertEqual(c,replay['increments']['extra']['columns'][tag])
        bad=copy.deepcopy(gravity);bad['frame']['cases'][0]['floor_states'][0]['displacements_lo'][0]+=1e-20
        with self.assertRaises(ValueError):system.solve(saved_response=bad)


if __name__=='__main__':unittest.main()
