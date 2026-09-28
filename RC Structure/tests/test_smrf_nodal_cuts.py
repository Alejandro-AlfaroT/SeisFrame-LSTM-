"""Independent analytical plate/line fixtures for a diagnostic recovery operator."""
import math
import copy
import unittest
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Nodal_Cuts import line_projection,integrate_line,recover_shell_trace,recover_nodal_regions


def plate(mode,axis='x',nx=8,ny=4):
    """nu=0 cantilever strip: exact one-way static resultants without a beam."""
    length,width=120.,48.;pressure=.001;end_moment=2.;end_tension=1.5
    positions={};shells=[];loads={}
    def xyz(x,y):return [x,y,0.] if axis=='x' else [y,x,0.]
    def node(i,j):return 1+j*(nx+1)+i
    try:
        ops.model('basic','-ndm',3,'-ndf',6)
        ops.section('ElasticMembranePlateSection',1,3600.,0.,4.,0.)
        for j in range(ny+1):
            for i in range(nx+1):
                n=node(i,j);positions[n]=xyz(i*length/nx,j*width/ny)
                ops.node(n,*positions[n])
                if i==0:ops.fix(n,1,1,1,1,1,1)
        for j in range(ny):
            for i in range(nx):
                # Reorder for global CCW when the longitudinal axis is Y.
                ns=[node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)]
                if axis=='y':ns=[ns[k] for k in (0,3,2,1)]
                tag=len(shells)+1;ops.element('ShellMITC4',tag,*ns,1)
                applied=np.zeros((4,6))
                if mode=='pressure':applied[:,2]=-pressure*(length/nx)*(width/ny)/4
                for n,value in zip(ns,applied):loads.setdefault(n,np.zeros(6));loads[n]+=value
                shells.append(dict(tag=tag,nodes=ns,applied_nodal_force_kip_kip_in=applied.ravel().tolist()))
        ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
        for j in range(ny+1):
            n=node(nx,j);tributary=width/ny*(.5 if j in (0,ny) else 1.)
            loads.setdefault(n,np.zeros(6))
            if mode=='moment':loads[n][4 if axis=='x' else 3]=(1 if axis=='x' else -1)*end_moment*tributary
            if mode=='membrane':loads[n][0 if axis=='x' else 1]=end_tension*tributary
        for n,f in loads.items():ops.load(n,*f)
        ops.constraints('Transformation');ops.numberer('RCM');ops.system('UmfPack')
        ops.algorithm('Linear');ops.integrator('LoadControl',1.);ops.analysis('Static')
        if ops.analyze(1):raise RuntimeError('Analytical plate fixture failed to solve')
        for s in shells:
            s['node_positions_in']=[positions[n] for n in s.pop('nodes')]
            s['global_nodal_force_kip_kip_in']=list(ops.eleForce(s['tag']))
        return shells
    finally:ops.wipe()


class NodalCutTests(unittest.TestCase):
    def setUp(self):ops.wipe();self.addCleanup(ops.wipe)

    def test_constant_and_linear_trace_on_nonuniform_grid(self):
        coords=np.array([-3.,0.,2.,8.,11.]);a=np.array([2.,-1.,3.,4.,-5.,6.]);b=np.array([.2,.3,-.4,.1,.7,-.2])
        f=np.zeros((len(coords),6))
        for i,(lo,hi) in enumerate(zip(coords,coords[1:])):
            h=hi-lo;t0=a+b*lo;t1=a+b*hi
            f[i]+=h*(2*t0+t1)/6;f[i+1]+=h*(t0+2*t1)/6
        trace=line_projection(coords,f)
        np.testing.assert_allclose(trace['wrench_density'],a+coords[:,None]*b,rtol=1e-12,atol=1e-12)
        lo,hi=-1.2,9.3;cut,z,refy=17.,-2.,4.
        # Exact polynomial integrals, independent of the production quadrature.
        I0=hi-lo;I1=(hi**2-lo**2)/2;I2=(hi**3-lo**3)/3
        force=a[:3]*I0+b[:3]*I1;moment=a[3:]*I0+b[3:]*I1
        moment+=np.array([a[2]*(I1-refy*I0)+b[2]*(I2-refy*I1),0.,-a[0]*(I1-refy*I0)-b[0]*(I2-refy*I1)])
        actual=integrate_line(trace,axis='x',cut_in=cut,z_in=z,interval_in=[lo,hi],reference_in=[cut,refy,z])
        np.testing.assert_allclose(actual,np.r_[force,moment],rtol=1e-12,atol=1e-12)

    def test_plate_static_resultants_in_both_axes_and_arbitrary_partial_width(self):
        for axis in ('x','y'):
            for mode in ('pressure','moment','membrane'):
                with self.subTest(axis=axis,mode=mode):
                    shells=plate(mode,axis)
                    for side,sign in (('left',1.),('right',-1.)):
                        trace=recover_shell_trace(shells,axis=axis,cut_in=60.,side=side)
                        for lo,hi in ((0.,48.),(7.3,39.1)):
                            width=hi-lo;ref=[60.,(lo+hi)/2,0.] if axis=='x' else [(lo+hi)/2,60.,0.]
                            f=integrate_line(trace,axis=axis,cut_in=60.,z_in=0.,interval_in=[lo,hi],reference_in=ref)
                            expected=np.zeros(6)
                            if mode=='pressure':
                                expected[2]=-.001*width*60
                                expected[4 if axis=='x' else 3]=(1 if axis=='x' else -1)*.001*width*60**2/2
                            elif mode=='moment':expected[4 if axis=='x' else 3]=(1 if axis=='x' else -1)*2.*width
                            else:expected[0 if axis=='x' else 1]=1.5*width
                            np.testing.assert_allclose(f,sign*expected,rtol=1e-8,atol=1e-7)

    def test_invalid_trace_inputs_and_unmeshed_cut(self):
        for coords,forces in (([0,0],np.zeros((2,6))),([0,1],np.zeros((2,5))),([0,float('nan')],np.zeros((2,6)))):
            with self.assertRaises(ValueError):line_projection(coords,forces)
        with self.assertRaisesRegex(ValueError,'mesh boundary'):
            recover_shell_trace(plate('pressure'),axis='x',cut_in=61.,side='left')

    def test_composite_flange_axial_and_bending_components_match_beam_theory(self):
        from tools.review_composite_sections import analytical_cantilever
        result=analytical_cantilever(8)
        trace=recover_shell_trace(result['shells'],axis='x',cut_in=120.,side='left')
        analytical=result['analytical']
        for lo,hi in ((-30.,30.),(-17.,22.)):
            f=integrate_line(trace,axis='x',cut_in=120.,z_in=0.,interval_in=[lo,hi],reference_in=[120.,(lo+hi)/2,0.])
            fraction=(hi-lo)/60.
            self.assertAlmostEqual(f[0],fraction*analytical['flange_axial_kip'],places=7)
            self.assertAlmostEqual(f[4],fraction*analytical['left_cut_my_components_kip_in']['shell_native'],places=7)

    def test_complete_generated_inventory_and_physical_offset_are_required(self):
        from Design.SMRF_Floor_Compatibility import CompatibleFloor
        from tests.test_smrf_floor_compatibility import SLAB,GEOMETRY,SECTIONS,CASE
        model=CompatibleFloor(SLAB,GEOMETRY,SECTIONS,2,slab_perimeter='beam_outer_faces')
        solved=model.solve([CASE,CASE]);response=solved['floors'][0]
        args=dict(axis='x',cut_in=120.,slab_perimeter='beam_outer_faces')
        original=recover_nodal_regions(response,GEOMETRY,SECTIONS,SLAB,**args)
        self.assertTrue(original['native_reassembly_passed']);self.assertFalse(original['capacity_pairing_authorized'])
        for key in ('shells','webs'):
            for duplicate in (False,True):
                bad=copy.deepcopy(response)
                if duplicate:bad[key].append(copy.deepcopy(bad[key][0]))
                else:bad[key].pop(0)
                with self.assertRaisesRegex(ValueError,'inventory'):recover_nodal_regions(bad,GEOMETRY,SECTIONS,SLAB,**args)
        with self.assertRaisesRegex(ValueError,'physical offset'):
            recover_nodal_regions(response,GEOMETRY,dict(SECTIONS,h_beam_in=20.),SLAB,**args)

    def test_uniform_membrane_shear_and_plate_twisting_patch(self):
        for mode in ('membrane_shear','twist'):
            with self.subTest(mode=mode):
                ops.wipe();ops.model('basic','-ndm',3,'-ndf',6)
                e,nu,h,gradient=3600.,.2,4.,1e-5
                ops.section('ElasticMembranePlateSection',1,e,nu,h,0.)
                pos={};shells=[];nx,ny=4,4
                ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
                for j in range(ny+1):
                    for i in range(nx+1):
                        n=1+j*(nx+1)+i;x,y=i*30.,j*12.;pos[n]=[x,y,0.];ops.node(n,x,y,0.)
                        u=[gradient*y/2,gradient*x/2,0.,0.,0.,0.] if mode=='membrane_shear' else [0.,0.,gradient*x*y,gradient*x,-gradient*y,0.]
                        # One free interior node makes this a solved patch test;
                        # all other nodes carry its exact compatible kinematics.
                        if (i,j)!=(2,2):
                            for d,v in enumerate(u):ops.sp(n,d+1,v)
                for j in range(ny):
                    for i in range(nx):
                        n=1+j*(nx+1)+i;ns=[n,n+1,n+nx+2,n+nx+1];tag=len(shells)+1
                        ops.element('ShellMITC4',tag,*ns,1)
                        # Known continuum tractions/couples on the OUTER edges
                        # belong to the cut-side load ledger. At a cut endpoint,
                        # native element forces also contain these edge actions.
                        # Derive them from the imposed analytical field, never
                        # from the computed nodal reactions.
                        applied=np.zeros((4,6))
                        for edge,normal,edge_length,active in (((0,1),(0,-1),30.,j==0),((1,2),(1,0),12.,i==nx-1),
                                                                ((2,3),(0,1),30.,j==ny-1),((3,0),(-1,0),12.,i==0)):
                            if not active:continue
                            density=np.zeros(6)
                            if mode=='membrane_shear':
                                shear=e/(2*(1+nu))*h*gradient
                                density[:2]=[shear*normal[1],shear*normal[0]]
                            else:
                                twist=e*h**3/(12*(1+nu))*gradient
                                density[3:5]=[twist*normal[0],-twist*normal[1]]
                            for k in edge:applied[k]+=density*edge_length/2
                        shells.append(dict(tag=tag,node_positions_in=[pos[n] for n in ns],applied_nodal_force_kip_kip_in=applied.ravel().tolist()))
                ops.constraints('Transformation');ops.numberer('RCM');ops.system('UmfPack')
                ops.algorithm('Linear');ops.integrator('LoadControl',1.);ops.analysis('Static');self.assertEqual(ops.analyze(1),0)
                for s in shells:s['global_nodal_force_kip_kip_in']=list(ops.eleForce(s['tag']))
                for axis,cut,interval in (('x',60.,[7.3,39.1]),('y',24.,[19.3,101.1])):
                    n=0 if axis=='x' else 1;ref=[0.,0.,0.];ref[n]=cut;ref[1-n]=sum(interval)/2;width=interval[1]-interval[0]
                    for side,sign in (('left',1.),('right',-1.)):
                        trace=recover_shell_trace(shells,axis=axis,cut_in=cut,side=side)
                        f=integrate_line(trace,axis=axis,cut_in=cut,z_in=0.,interval_in=interval,reference_in=ref)
                        expected=np.zeros(6)
                        if mode=='membrane_shear':expected[1-n]=e/(2*(1+nu))*h*gradient*width
                        else:expected[3+n]=(1 if axis=='x' else -1)*e*h**3/(12*(1+nu))*gradient*width
                        np.testing.assert_allclose(f,sign*expected,rtol=1e-8,atol=1e-7)

if __name__=='__main__':unittest.main()
