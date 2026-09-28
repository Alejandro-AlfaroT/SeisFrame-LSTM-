import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Design.SMRF_Cut_Regions import effective_flange_regions,integrate_shell_interval,recover_floor_regions
from Design.SMRF_Coupled_Analysis import analyze_coupled_gravity
from Design.SMRF_Composite_Sections import transport_wrench
from tests.test_smrf_coupled_analysis import SLAB,GEOMETRY,SECTIONS,DEAD,ZERO


def shell(x0=0.,x1=100.,y0=0.,y1=40.,field=None):
    field=field or (lambda x,y:[2.,3.,.5,10.,20.,4.,1.,2.])
    values=[]
    for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1)):
        x=(x0+x1)/2+sx*(x1-x0)/(2*math.sqrt(3));y=(y0+y1)/2+sy*(y1-y0)/(2*math.sqrt(3))
        values.extend(field(x,y))
    return dict(element=1,node_positions_in=[[x0,y0,0.],[x1,y0,0.],[x1,y1,0.],[x0,y1,0.]],gauss_resultants_raw=values)


class ShellIntervalTests(unittest.TestCase):
    def test_constant_traction_and_opposite_faces_have_correct_global_moments(self):
        for axis,cut,bounds,ref,expected in (
            ('x',50.,[0.,40.],[50.,20.,0.],[80.,20.,40.,-160.,400.,0.]),
            ('y',20.,[0.,100.],[50.,20.,0.],[50.,300.,200.,-2000.,400.,0.])):
            for side,sign in (('left',1),('right',-1)):
                r=integrate_shell_interval([shell()],axis=axis,cut_in=cut,interval_in=bounds,reference_in=ref,side=side)
                for a,b in zip(r['global_wrench'],expected):self.assertAlmostEqual(a,sign*b,places=8)

    def test_partial_cell_partition_and_reference_transport_are_exact(self):
        # N(y) varies; transporting it produces a quadratic integrand.
        s=shell(field=lambda x,y:[1.+y,.2*x,.3,4.+y,3.,2.,.5*y,.4])
        total=integrate_shell_interval([s],axis='x',cut_in=37.,interval_in=[0.,40.],reference_in=[37.,20.,0.],side='left')
        parts=[]
        for lo,hi in ((0.,7.3),(7.3,22.6),(22.6,40.)):
            ref=[37.,(lo+hi)/2,-8.]
            r=integrate_shell_interval([s],axis='x',cut_in=37.,interval_in=[lo,hi],reference_in=ref,side='left')
            parts.append(transport_wrench(r['global_wrench'],ref,[37.,20.,0.]))
        for i,v in enumerate(total['global_wrench']):self.assertAlmostEqual(v,sum(p[i] for p in parts),places=8)
        self.assertAlmostEqual(total['global_wrench'][0],40.+40.**2/2)
        # Independent integral -int((y-20)*(1+y),y=0..40).
        self.assertAlmostEqual(total['global_wrench'][5],-(40.**3/3-19*40.**2/2-20*40.))

    def test_one_sided_discontinuous_recovery_is_not_averaged(self):
        cells=[shell(x0=0.,x1=50.),shell(x0=50.,x1=100.,field=lambda x,y:[8.,3.,.5,10.,20.,4.,1.,2.])]
        left=integrate_shell_interval(cells,axis='x',cut_in=50.,interval_in=[0.,40.],reference_in=[50.,20.,0.],side='left')
        right=integrate_shell_interval(cells,axis='x',cut_in=50.,interval_in=[0.,40.],reference_in=[50.,20.,0.],side='right')
        self.assertAlmostEqual(left['global_wrench'][0],80.)
        self.assertAlmostEqual(right['global_wrench'][0],-320.)

    def test_missing_duplicate_and_rotated_cells_are_refused(self):
        for cells in ([],[shell(),shell()],[shell(y1=39.)]):
            with self.assertRaises(ValueError):
                integrate_shell_interval(cells,axis='x',cut_in=50.,interval_in=[0.,40.],reference_in=[50.,20.,0.],side='left')
        s=shell();s['node_positions_in'][1][1]=1.
        with self.assertRaises(ValueError):
            integrate_shell_interval([s],axis='x',cut_in=50.,interval_in=[0.,40.],reference_in=[50.,20.,0.],side='left')

    def test_nonfinite_resultants_are_refused(self):
        s=shell();s['gauss_resultants_raw'][2]=float('nan')
        with self.assertRaises(ValueError):
            integrate_shell_interval([s],axis='x',cut_in=50.,interval_in=[0.,40.],reference_in=[50.,20.,0.],side='left')

    def test_native_shell_patch_resultant_signs_and_units(self):
        import openseespy.opensees as ops
        if ops.getNodeTags():raise RuntimeError('Test requires an empty model')
        E,nu,t=4000.,.2,6.;ex,ey,gxy=1e-5,-.5e-5,.7e-5
        kx,ky,kxy=2e-6,-1e-6,.6e-6;gx,gy=3e-6,-2e-6
        positions=[[0.,0.,0.],[100.,0.,0.],[100.,40.,0.],[0.,40.,0.]]
        try:
            ops.model('basic','-ndm',3,'-ndf',6)
            ops.section('ElasticMembranePlateSection',1,E,nu,t,0.)
            ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
            for tag,(x,y,z) in enumerate(positions,1):
                ops.node(tag,x,y,z)
                u=[ex*x+gxy*y/2,ey*y+gxy*x/2,-.5*kx*x*x-.5*ky*y*y-kxy*x*y+gx*x+gy*y,-ky*y-kxy*x,kx*x+kxy*y,0.]
                for i,v in enumerate(u,1):ops.sp(tag,i,v)
            ops.element('ShellMITC4',1,1,2,3,4,1)
            # One uncoupled elastic equation avoids a zero-size global system.
            ops.node(5,200.,0.,0.);ops.node(6,200.,0.,0.);ops.fix(5,1,1,1,1,1,1);ops.fix(6,0,1,1,1,1,1)
            ops.uniaxialMaterial('Elastic',1,1.);ops.element('zeroLength',2,5,6,'-mat',1,'-dir',1)
            ops.constraints('Transformation');ops.numberer('Plain');ops.system('UmfPack');ops.algorithm('Linear')
            ops.integrator('LoadControl',1.);ops.analysis('Static');self.assertEqual(ops.analyze(1),0)
            raw=list(ops.eleResponse(1,'stresses'))
            D=E*t**3/(12*(1-nu**2));A=E*t/(1-nu**2);G=E/(2*(1+nu))
            expected=[A*(ex+nu*ey),A*(ey+nu*ex),G*t*gxy,D*(kx+nu*ky),D*(ky+nu*kx),D*(1-nu)*kxy,(5/6)*G*t*gx,(5/6)*G*t*gy]
            for k in range(4):
                for a,b in zip(raw[8*k:8*k+8],expected):self.assertAlmostEqual(a,b,places=9)
        finally:ops.wipe()


class FloorRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g=dict(GEOMETRY,num_bay_x=2,num_bay_y=2)
        case=dict(DEAD,dead_factor=1.2,live_factor=1.6,live_pattern=[[0,0]])
        cls.result=analyze_coupled_gravity(SLAB,cls.g,SECTIONS,[ZERO,case],mesh_per_bay=4)

    def test_edge_geometry_mismatch_is_explicit_and_slab_is_not_dropped(self):
        for axis in ('x','y'):
            p=effective_flange_regions(self.g,SECTIONS,SLAB,axis=axis)
            beams=[r for r in p['regions'] if r['kind']=='beam_flange']
            self.assertEqual(len(beams),3)
            for r in (beams[0],beams[-1]):
                self.assertAlmostEqual(r['missing_flange_width_in'],SECTIONS['b_beam_in']/2)
                self.assertFalse(r['strength_geometry_matches'])
            self.assertTrue(beams[1]['strength_geometry_matches'])
            self.assertFalse(p['geometry_matches_all_strength_sections'])
            self.assertFalse(p['reinforcement_ownership_assigned'])
            self.assertAlmostEqual(sum(r['interval_in'][1]-r['interval_in'][0] for r in p['regions']),p['domain_interval_in'][1])

    def test_regions_reassemble_but_do_not_claim_equilibrium_from_that(self):
        before=copy.deepcopy(self.result)
        for axis,cut in (('x',120.),('y',100.)):
            r=recover_floor_regions(self.result,2,axis,cut)
            self.assertTrue(r['numerical_partition_passed'])
            self.assertTrue(r['native_whole_cut_balanced'])
            self.assertFalse(r['capacity_pairing_authorized'])
            self.assertFalse(r['production_enabled'])
            self.assertGreater(max(abs(v) for v in r['whole_cut_face_sum']),1e-3)
            self.assertTrue(any(v['kind']=='complementary_slab' for v in r['rows']))
        self.assertEqual(before,self.result)

    def test_failed_missing_native_and_rigid_restraint_records_are_refused(self):
        for key,value in (('status','failed'),('section_action_schema',None)):
            broken=dict(self.result,**{key:value})
            with self.assertRaises(ValueError):recover_floor_regions(broken,2,'x',120.)
        broken=copy.deepcopy(self.result);broken['inputs']['inplane_restraint']='rigid_floor'
        with self.assertRaises(ValueError):recover_floor_regions(broken,2,'x',120.)


if __name__=='__main__':unittest.main()
