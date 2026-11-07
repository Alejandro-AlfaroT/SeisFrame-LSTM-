"""Independent uncondensed physical systems, not preconditioner self-checks."""
import copy
import json
from types import SimpleNamespace
import unittest
import numpy as np
from scipy import sparse
from scipy.optimize import root
from Design.SMRF_Static_Substructure import condense
from Design.SMRF_Substructure_Frame import SubstructureFrame
from Design.SMRF_Paired_Arithmetic import paired_dense_action
from Design.SMRF_Paired_Coupled_Equilibrium import PairedCoupledEquilibrium,paired_column_response


class ExplicitFloor:
    def __init__(self,floor,matrix):
        self.floor=floor;self.matrix=matrix;self.metadata=dict(method='independent_dense_test_fixture',physical_matrix=matrix.tolist())
    def apply_pair(self,h,l):return paired_dense_action(self.matrix,h,l)


def fixture(*,imposed=False,tolerance_scale=.01):
    rng=np.random.default_rng(252);nodes=[dict(id=i,xyz=[2.*i,3.*i,100.*i]) for i in range(3)]
    rows=[];interfaces=[]
    for level in (1,2):
        dofs=[(level,d) for d in range(1,7)]+[(f'internal{level}',d) for d in (1,2)]
        a=rng.normal(size=(8,8));k=a.T@a+np.eye(8)*3
        # A deliberately imperfect Schur preconditioner must NOT determine
        # accepted physical equilibrium or freeze the boundary displacement.
        p=k+np.diag(np.arange(1,9)*.03)
        floor=SimpleNamespace(id=f'f{level}',dof_ids=dofs,boundary_dofs=dofs[:6])
        f=rng.normal(size=(8,2))*5;f[2,:]-=50
        sub=condense(p,dofs,dofs[:6],f,input_identity=f'preconditioner{level}')
        rows.append(dict(assembly=floor,model=ExplicitFloor(floor,k),condensed=sub,
            loads=[dict(loads=f[:,i]) for i in range(2)]))
        interfaces.append(dict(id=floor.id,dof_ids=dofs[:6],stiffness=sub.stiffness,loads=sub.loads,input_identity=f'interface{level}'))
    columns=[dict(id=i,i=i-1,j=i,A=20.,E=4000.,G=1600.,J=100.,Iy=200.,Iz=250.,vecxz=[1.,0.,0.],
        axial_load_kip_per_in=[-.2,-.3]) for i in (1,2)]
    nodal=np.zeros((18,2));nodal[6,:]=[5.,7.];nodal[13,:]=[-2.,3.]
    kwargs=dict(supports=[(0,d) for d in range(1,7)])
    if imposed:
        # Vertical displacement and two joint rotations remain free; actual
        # horizontal translations/torsion are prescribed at both floors.
        free=[6*i+d for i in (1,2) for d in (2,3,4)]
        transform=sparse.csr_matrix((np.ones(len(free)),(free,np.arange(len(free)))),shape=(18,len(free)))
        lift=np.zeros((18,2));lift[6,:]=[.1,.15];lift[12,:]=[.2,.3];lift[17,:]=[.001,-.002]
        kwargs=dict(kinematic_transform=transform,reduced_dof_ids=[(nodes[i//6]['id'],i%6+1) for i in free],support_lift=lift)
    frame=SubstructureFrame(nodes,interfaces,columns,loads=nodal,input_identity='paired_coupled_252',**kwargs)
    engine=PairedCoupledEquilibrium(frame,rows,input_identity='independent_full_physical',nodal_loads=nodal,tolerance_scale=tolerance_scale)
    return engine


def independent_full(engine,pdelta):
    frame=engine.frame;n=len(frame.dof_ids);nr=frame.transform.shape[1];size=n+4
    t=sparse.block_diag((frame.transform,sparse.eye(4))).toarray();full=np.zeros((size,size));f=np.zeros((size,2));f[:n]=engine.nodal
    maps=[]
    for row,(indices,loads),offset in zip(engine.floors,engine.maps,(n,n+2)):
        ix=np.r_[indices,[offset,offset+1]];maps.append(ix);full[np.ix_(ix,ix)]+=row['model'].matrix;f[ix]+=loads
    for element,indices,weights in frame.columns:
        full[np.ix_(indices,indices)]+=element.stiffness
        for case in range(2):f[indices,case]+=element.equivalent_load(weights[case])
    answers=[]
    for case in range(2):
        lift=np.r_[frame.lift[:,case],np.zeros(4)]
        q=np.linalg.solve(t.T@full@t,t.T@(f[:,case]-full@lift))
        if pdelta:
            def residual(q):
                u=t@q+lift;r=full@u-f[:,case]
                for element,indices,weights in frame.columns:
                    r[indices]+=element.response(u[indices],weights[case],pdelta=True)['geometric_global_force']
                return t.T@r
            solved=root(residual,q,method='hybr',options={'xtol':1e-10})
            if np.max(abs(residual(solved.x)))>1e-8:raise AssertionError('Independent uncondensed nonlinear reference did not equilibrate')
            q=solved.x
        answers.append(t@q+lift)
    return answers,maps


class PairedCoupledTests(unittest.TestCase):
    def compare(self,imposed,pdelta):
        engine=fixture(imposed=imposed);expected,maps=independent_full(engine,pdelta);record=engine.solve(pdelta=pdelta)
        self.assertEqual(record['status'],'equilibrated');self.assertTrue(record['floor_and_joint_physical_equilibrium'])
        for result,reference in zip(record['cases'],expected):
            np.testing.assert_allclose(np.array(result['physical_displacements'])+result['physical_displacements_lo'],reference[:18],rtol=2e-8,atol=2e-9)
            self.assertLessEqual(result['maximum_free_equilibrium_residual'],result['equilibrium_tolerance'])
            for floor,ix in zip(result['floor_states'],maps):
                np.testing.assert_allclose(np.array(floor['displacements_hi'])+floor['displacements_lo'],reference[ix],rtol=2e-8,atol=2e-9)
                self.assertTrue(floor['equilibrium']['passed'])
        # JSON round trip and force recomputation, without any new global solve.
        saved=json.loads(json.dumps(record));self.assertTrue(engine.verify_saved(saved,pdelta=pdelta)['passed'])
        with self.assertRaises(ValueError):engine.verify_saved(saved,pdelta=not pdelta)
        return engine,record

    def test_full_linear_joint_and_interior_equilibrium(self):
        engine,record=self.compare(False,False)
        approx=engine.frame.solve(pdelta=False)
        self.assertGreater(np.max(abs(np.array(record['cases'][0]['physical_displacements'])-approx['cases'][0]['physical_displacements'])),1e-5)

    def test_pdelta_uncondensed_nonlinear_reference(self):self.compare(False,True)
    def test_imposed_kinematics_with_linear_and_pdelta(self):
        for pdelta in (False,True):
            with self.subTest(pdelta=pdelta):self.compare(True,pdelta)

    def test_saved_pair_boundary_force_and_low_part_tampering_refused(self):
        engine,record=self.compare(False,True)
        for field,index in [('displacements_hi',0),('displacements_lo',7)]:
            bad=copy.deepcopy(record);bad['cases'][0]['floor_states'][0][field][index]+=1e-12
            with self.assertRaises(ValueError):engine.verify_saved(bad)
        bad=copy.deepcopy(record);bad['cases'][0]['columns']['1']['local_force'][0]+=.01
        with self.assertRaises(ValueError):engine.verify_saved(bad)
        bad=copy.deepcopy(record);bad['identity_sha256']='stale'
        with self.assertRaises(ValueError):engine.verify_saved(bad)

    def test_wrong_load_inventory_and_invalid_iteration_budget_refused(self):
        engine=fixture()
        with self.assertRaisesRegex(ValueError,'loads do not match'):
            PairedCoupledEquilibrium(engine.frame,engine.floors,input_identity='omitted nodal load')
        for limit in (0,41,1.5):
            with self.assertRaises(ValueError):engine.solve(max_iterations=limit)
        short=engine.solve(max_iterations=1,pdelta=True)
        self.assertEqual(short['status'],'unresolved')
        with self.assertRaises(ValueError):engine.verify_saved(short)
        for scale in (0.,1.01,float('nan')):
            with self.assertRaisesRegex(ValueError,'only be tightened'):fixture(tolerance_scale=scale)

    def test_default_criteria_verified_on_independent_full_equations(self):
        engine=fixture(tolerance_scale=1.);record=engine.solve(pdelta=False)
        self.assertEqual(record['status'],'equilibrated')
        for case,result in enumerate(record['cases']):
            joint=np.array(result['physical_displacements'])+result['physical_displacements_lo'];rj=-engine.nodal[:,case].copy()
            for row,(indices,loads),fs in zip(engine.floors,engine.maps,result['floor_states']):
                u=np.array(fs['displacements_hi'])+fs['displacements_lo'];res=row['model'].matrix@u-loads[:,case]
                rj[indices]+=res[:6]
                self.assertLessEqual(np.max(abs(res[6:])),1e-9*fs['equilibrium']['force_scale'])
            for element,indices,weights in engine.frame.columns:
                rj[indices]+=element.response(joint[indices],weights[case],pdelta=False)['global_resisting_force']
            self.assertLessEqual(np.max(abs(engine.frame.transform.T@rj)),result['equilibrium_tolerance'])

    def test_native_column_law_with_weight_and_signed_axial_force(self):
        engine=fixture();element=engine.frame.columns[0][0]
        for sign in (-1,1):
            h=np.array([.1,.2,-.03,.001,-.002,.003,.3,-.1,sign*.5,-.001,.004,.002]);l=np.zeros(12)
            for pdelta in (False,True):
                _,_,actual=paired_column_response(element,h,l,-.3,pdelta=pdelta)
                expected=element.response(h,-.3,pdelta=pdelta)
                for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-10)


if __name__=='__main__':unittest.main()
