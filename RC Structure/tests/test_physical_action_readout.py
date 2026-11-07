import copy
import unittest
import numpy as np
from test_primary_frame_transfer import PrimaryTransferTests
from Design.SMRF_Physical_Action_Readout import primary_beam_cut,primary_flexural_projection,physical_web_loads
from Design.SMRF_Full_T_Projection import FullTIncrementProjector

class PhysicalActionReadoutTests(unittest.TestCase):
    def test_actual_primary_cut_derivatives_and_complete_moments(self):
        system,physical,primary=PrimaryTransferTests().system();primary.transfer_gravity(physical,case_id='DL')
        increment=primary.solve_increment([[1.,-.3,2.],[2.,.5,-3.]])
        nodes={n['id']:n['xyz'] for n in primary.definition['nodes']}
        for b in primary.definition['beams']:
            pi,pj=np.array(nodes[b['i']]),np.array(nodes[b['j']]);j=0 if pj[0]>pi[0] else 1
            station=(pi[j]+pj[j])/2;h=.001
            mid=primary_beam_cut(primary.definition,increment,b['id'],station)
            left=primary_beam_cut(primary.definition,increment,b['id'],station-h)
            right=primary_beam_cut(primary.definition,increment,b['id'],station+h)
            fields=('N_tension_kip','M_centroid_kipin','Ms_transverse_first_moment_kipin')
            derivative=[(right[k]-left[k])/(2*h) for k in fields]
            np.testing.assert_allclose(derivative,mid['normal_derivative_per_in'],rtol=1e-8,atol=1e-8)
            center=pi[1-j]
            p=FullTIncrementProjector(slab_thickness_in=1.,beam_depth_in=5.,flange_interval_in=[center-1,center+5],
                web_interval_in=[center-1,center+1],slab_E_ksi=200.,web_E_ksi=200.,axis=mid['axis'])
            projected=primary_flexural_projection(p,mid,external_axial_owner='actual_P5_fixture')
            np.testing.assert_allclose(projected['reconstructed_resultants'],[0.,mid[fields[1]],mid[fields[2]]],atol=1e-8)
            self.assertTrue(projected['source_partition']['measured_rigid_beam_axial_zero'])
            self.assertFalse(projected['source_partition']['gravity_C0_added'])
        bad=copy.deepcopy(increment);bad['primary_definition_sha256']='stale'
        with self.assertRaisesRegex(ValueError,'different physical model'):
            primary_beam_cut(primary.definition,bad,primary.definition['beams'][0]['id'],5.)

    def test_actual_web_distributed_loads_match_element_end_force_sum(self):
        system,physical,_=PrimaryTransferTests().system()
        for row,floor in zip(system.floors,physical['floors']):
            loads=physical_web_loads(row,0)
            for element in floor['responses'][0]['elements']:
                if element['kind']!='web':continue
                f=element['global_resisting_force'];p=np.array(element['positions_in']);length=np.linalg.norm(p[1]-p[0])
                self.assertAlmostEqual(f[2]+f[8]+loads[element['id']]*length,0.,places=8)

if __name__=='__main__':unittest.main()
