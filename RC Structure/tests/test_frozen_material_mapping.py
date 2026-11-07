"""Service-family material/identity/replay contract, no global analysis.

Actual cell material integration is exercised. The enclosing system is a
deterministic orchestration double; these tests assert no structural accuracy.
"""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from Design.SMRF_Frozen_Gravity_Design import frozen_gravity_family
from Design.SMRF_Physical_Floor_Materials import FrozenGravityReferenceBuilder
from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH
from Design.SMRF_Coupled_Floor_Design import _hash
from test_bar_pitch_layers import fixture


class TestSystem:
    __test__=False
    def __init__(self,*,cases=None,case_identity='predictor',u=None,log=None):
        self.cases=cases or [dict(id='service_DL',dead=1.,live=1.,live_pattern='all')]
        self.case_identity=case_identity;self.u=np.zeros(24) if u is None else u;self.log=[] if log is None else log
        self.source_identity='source';self.definition=dict(state_identity='installed',identity_sha256='geometry')
        raw=np.zeros((4,8,24))
        # Objective constant-strain patch for the orchestration double. A
        # rigid motion must not become a material strain when the production
        # builder retains/removes paired rigid-motion components.
        for i in range(4):
            raw[i,0,[0,6]]=[-1/30,1/30];raw[i,1,[1,19]]=[-1/30,1/30]
            raw[i,2,[1,7,0,18]]=[-1/30,1/30,-1/30,1/30]
            raw[i,3,[4,10]]=[1/30,-1/30];raw[i,4,[3,21]]=[-1/30,1/30]
            raw[i,5,[3,9,4,22]]=[-1/30,1/30,1/30,-1/30]
        elements=[dict(kind='slab',id=str(i),points=np.array([[0,lo,0],[30,lo,0],[30,hi,0],[0,hi,0]]),native=dict(strain=raw),
            transform=np.eye(24),indices=np.arange(24)) for i,(lo,hi) in enumerate(((0.,8.),(8.,19.),(19.,30.)))]
        floor=SimpleNamespace(id='f1',x=[0.,30.],y=[0.,8.,19.,30.],thickness=8.,ec=4000.,nu=.2,elements=elements,
            dof_ids=[(n,d) for n in range(4) for d in range(1,7)],node_ids=list(range(4)),
            node_positions=dict(enumerate([[0,0,0],[30,0,0],[30,30,0],[0,30,0]])))
        self.floors=[dict(assembly=floor,description=dict(id='f1',level=1,fc_ksi=5.))]
    def solve(self,*,pdelta,saved_response=None):
        self.log.append(('solve',self.case_identity))
        dofs=self.floors[0]['assembly'].dof_ids
        result=dict(status='equilibrated',frame=dict(identity_sha256=_hash([self.case_identity,self.cases]),physical_dof_ids=dofs,
            cases=[dict(physical_displacements=self.u.tolist(),physical_displacements_lo=[0.]*24) for c in self.cases]),
            floors=[dict(id='f1',responses=[dict(case_id=c['id'],dof_ids=dofs,retained_displacements=self.u.tolist(),
                retained_displacements_lo=[0.]*24) for c in self.cases])])
        if saved_response is not None and saved_response!=result:raise ValueError('Test response reconstruction differs')
        return result
    def with_cases(self,cases,*,patterns,case_identity):
        return TestSystem(cases=cases,case_identity=case_identity,u=self.u,log=self.log)
    def set_floor_operators(self,operators,*,iteration_identity):
        self.log.append(('operators',iteration_identity,operators))


class FrozenMappingTests(unittest.TestCase):
    def run_family(self,mapping='clipped_cylinders_v2',*,saved=None,updates=1,u=None):
        inv=fixture();inv['hook_stiffness_basis']='actual straight portions only';system=TestSystem(u=u)
        def operator(builder,row,reference,**kwargs):
            self.assertEqual(builder.material_mapping,mapping)
            self.assertEqual(reference['audit']['material_mapping'],mapping)
            self.assertEqual(reference['status'],'reference_constructed')
            return dict(material_mapping=mapping,matrices={k:v.tolist() for k,v in reference['slab_bending_matrices'].items()})
        with patch.object(FrozenGravityReferenceBuilder,'frozen_operator',operator):
            out,design=frozen_gravity_family(system,inv,[dict(id='design_G',dead=1.2,live=1.)],protocol_id='same_protocol',
                reference_updates=updates,maximum_wall_seconds=60,maximum_coupled_dofs=100,saved_result=saved,material_mapping=mapping)
        return out,design,system

    def test_actual_material_choice_reaches_every_update_and_saved_replay(self):
        legacy,_,ls=self.run_family();fixed,_,fs=self.run_family(PITCH)
        for result,mapping in ((legacy,'clipped_cylinders_v2'),(fixed,PITCH)):
            self.assertEqual(result['status'],'responses_computed');self.assertEqual(result['material_mapping'],mapping)
            self.assertEqual(len(result['references']),2);self.assertFalse(result['qualification_asserted'])
            for ref in result['references']:
                self.assertEqual(ref['material_mapping'],mapping)
                self.assertEqual(ref['floor_references']['f1']['material_mapping'],mapping)
                self.assertTrue(ref['floor_references']['f1']['ownership']['verified'])
        self.assertNotEqual(legacy['references'][0]['reference_identity'],fixed['references'][0]['reference_identity'])
        lm=[r for r in ls.log if r[0]=='operators'][0][2];fm=[r for r in fs.log if r[0]=='operators'][0][2]
        self.assertNotEqual(lm['f1']['matrices'],fm['f1']['matrices'])
        replay,_,_=self.run_family(PITCH,saved=fixed)
        self.assertEqual(replay['references'],fixed['references'])

    def test_cross_mapping_or_old_method_refused_before_saved_predictor_use(self):
        fixed,_,_=self.run_family(PITCH)
        with self.assertRaisesRegex(ValueError,'Saved actual gravity'):self.run_family(saved=fixed)
        old=copy.deepcopy(fixed);old.pop('material_mapping')
        with self.assertRaisesRegex(ValueError,'Saved actual gravity'):self.run_family(PITCH,saved=old)
        old=copy.deepcopy(fixed);old['method']='actual_mat_frozen_static_gravity_reference_family_v1'
        with self.assertRaisesRegex(ValueError,'Saved actual gravity'):self.run_family(PITCH,saved=old)
        with self.assertRaisesRegex(ValueError,'mapping'):self.run_family('auto_qualified')

    def test_mapping_does_not_bypass_material_domain_failure(self):
        u=np.zeros(24);u[6]=-.004*30 # same -0.004 physical compressive strain
        out,design,system=self.run_family(PITCH,u=u)
        self.assertIsNone(design);self.assertEqual(out['status'],'unknown')
        self.assertFalse(out['references'][0]['floor_references']['f1']['material_domain_valid'])
        self.assertFalse(any(r[0]=='operators' for r in system.log))

    def test_saved_reference_audits_cannot_be_changed_or_omitted(self):
        fixed,_,_=self.run_family(PITCH)
        changes=[('material_domain_valid',False),('quadrature_converged',False),('cracked_point_ids',['invented_history']),
                 ('physical_material_ownership_verified',False)]
        for key,value in changes:
            bad=copy.deepcopy(fixed);bad['references'][0]['floor_references']['f1'][key]=value
            with self.assertRaisesRegex(ValueError,'material audit'):self.run_family(PITCH,saved=bad)
        bad=copy.deepcopy(fixed);bad['references'][0]['floor_references'].clear()
        with self.assertRaisesRegex(ValueError,'material audit'):self.run_family(PITCH,saved=bad)
        for key in ('source_identity','definition_sha256'):
            bad=copy.deepcopy(fixed);bad[key]='changed'
            with self.assertRaisesRegex(ValueError,'Saved actual gravity'):self.run_family(PITCH,saved=bad)

    def test_common_input_signature_binds_mapping_independently(self):
        from Design.SMRF_Common_Action_Analysis import common_input_identity
        args=('state',{'identity_sha256':'geometry'},{'identity_sha256':'inventory'},'source',{},[],1.,{}, {},{})
        default=common_input_identity(*args)
        self.assertEqual(default,common_input_identity(*args,material_mapping='clipped_cylinders_v2'))
        self.assertNotEqual(default,common_input_identity(*args,material_mapping=PITCH))

if __name__=='__main__':unittest.main()
