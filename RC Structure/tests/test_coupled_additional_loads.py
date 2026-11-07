"""Additional floor free bodies retain every coupled force and actual P."""
import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from test_coupled_floor_geometry import fixture,build

class AdditionalFloorLoadTests(unittest.TestCase):
    def test_increment_matches_independent_full_matrix_and_other_floor_coupling(self):
        system=build(fixture(),[dict(id='DL',dead=1.,live=1.)]);loads={}
        for row in system.floors:
            floor=row['assembly'];f=np.zeros((len(floor.dof_ids),2));ids=floor.node_ids
            # Pure balanced planar horizontal force pair away from joint centers.
            candidates=[(i,floor.node_positions[n]) for i,n in enumerate(ids) if isinstance(n,str)]
            for i,p in candidates:
                if p[1]==8. and p[0] in (10.,30.):f[6*i,0]=1. if p[0]==10 else -1.
            if row['description']['level']!=2:f[:]=0.
            f[:,1]=-f[:,0];loads[row['description']['id']]=f
        result=system.solve_additional_floor_loads(loads,baseline_case_id='DL',case_ids=['plus','minus'],
            input_identity='independent_full_matrix_pair',pdelta=False)
        self.assertEqual(result['status'],'equilibrated')
        # Independent full unconden​sed system, actual floor matrices and columns.
        ids=[d for row in system.floors for d in row['assembly'].dof_ids]
        ids +=[(n['id'],j) for n in system.definition['nodes'] if n['xyz'][2]==0 for j in range(1,7)]
        self.assertEqual(len(ids),len(set(ids)));index={d:i for i,d in enumerate(ids)}
        matrix=sparse.lil_matrix((len(ids),len(ids)));rhs=np.zeros(len(ids))
        for row in system.floors:
            ix=np.array([index[d] for d in row['assembly'].dof_ids]);matrix[np.ix_(ix,ix)]+=row['model'].stiffness
            rhs[ix]+=loads[row['description']['id']][:,0]
        for element,_,_ in system.frame.columns:
            ix=np.array([index[(element.definition[end],j)] for end in ('i','j') for j in range(1,7)])
            matrix[np.ix_(ix,ix)]+=element.stiffness
        fixed={index[(n['id'],j)] for n in system.definition['nodes'] if n['xyz'][2]==0 for j in range(1,7)}
        free=np.array([i for i in range(len(ids)) if i not in fixed]);u=np.zeros(len(ids))
        u[free]=spsolve(matrix.tocsr()[free,:][:,free],rhs[free])
        expected=[u[index[tuple(d)]] for d in result['total_response']['frame']['physical_dof_ids']]
        plus=result['increments']['plus'];minus=result['increments']['minus']
        np.testing.assert_allclose(plus['displacement_increment'],expected,rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(plus['displacement_increment'],-np.array(minus['displacement_increment']),atol=1e-10)
        self.assertGreater(np.max(abs(np.array(expected))),1e-6)
        self.assertTrue(plus['all_six_column_force_components_retained'])
        self.assertGreater(max(abs(v) for row in plus['columns'].values() for v in row['global_force_increment']),1e-7)
        for floor in plus['floors']:self.assertLess(floor['field_subtraction_max_error_kip_kipin'],1e-8)

    def test_complete_source_inventory_and_actual_baseline_are_required(self):
        system=build(fixture(),[dict(id='DL',dead=1.,live=1.)])
        with self.assertRaisesRegex(ValueError,'Complete physical floor loads'):
            system.solve_additional_floor_loads({},baseline_case_id='DL',case_ids=['x'],input_identity='test')

if __name__=='__main__':unittest.main()
