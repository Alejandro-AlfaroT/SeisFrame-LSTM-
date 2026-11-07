"""Independent native retained-frame baseline/ELF comparison after P transfer."""
import json
import unittest
import numpy as np
import openseespy.opensees as ops
from test_primary_frame_transfer import PrimaryTransferTests


class NativePrimaryTransferTests(unittest.TestCase):
    def tearDown(self):ops.wipe()

    def native(self,primary,floor_loads):
        ops.wipe();ops.model('basic','-ndm',3,'-ndf',6)
        definition=primary.definition;g=definition['geometry']
        for n in primary.nodes:
            ops.node(n['id'],*n['xyz'])
            if n['xyz'][2]==0:ops.fix(n['id'],1,1,1,1,1,1)
        for floor in range(1,g['nf']+1):
            master=100000+floor;ops.node(master,*primary.master_xy,floor*g['story_h']);ops.fix(master,0,0,1,1,1,0)
            ops.rigidDiaphragm(3,master,*[n['id'] for n in primary.nodes if n['xyz'][2]==floor*g['story_h']])
        ops.geomTransf('PDelta',1,1,0,0);ops.geomTransf('Linear',2,0,0,1);tags={};tag=1
        for kind,rows in (('column',definition['columns']),('beam',definition['beams'])):
            for row in rows:
                release=[]
                for key in ('releasey','releasez'):
                    if row.get(key):release += ['-'+key,row[key]]
                ops.element('elasticBeamColumn',tag,row['i'],row['j'],*[row[k] for k in ('A','E','G','J','Iy','Iz')],
                            1 if kind=='column' else 2,*release)
                tags[str(row['id'])]=tag;tag+=1
        ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
        for n in primary.nodes:
            i=primary.index[(n['id'],1)];ops.load(n['id'],*primary.baseline['auxiliary_nodal_loads'][i:i+6])
        for row in definition['columns']:
            w=float(np.atleast_1d(row['axial_load_kip_per_in'])[0])
            ops.eleLoad('-ele',tags[str(row['id'])],'-type','-beamUniform',0.,0.,w)
        for floor,load in enumerate(floor_loads,1):ops.load(100000+floor,load[0],load[1],0.,0.,0.,load[2])
        ops.constraints('Transformation');ops.numberer('RCM');ops.system('UmfPack');ops.test('NormDispIncr',1e-12,100)
        ops.algorithm('Newton');ops.integrator('LoadControl',.05);ops.analysis('Static')
        self.assertEqual(ops.analyze(20),0)
        return dict(u=[ops.nodeDisp(node,dof) for node,dof in primary.dof_ids],
                    global_force={name:ops.eleResponse(tag,'globalForce') for name,tag in tags.items()},
                    local_force={name:ops.eleResponse(tag,'localForce') for name,tag in tags.items()})

    def test_native_baseline_and_signed_ELF_match_all_physical_member_forces(self):
        _,physical,primary=PrimaryTransferTests().system();primary.transfer_gravity(physical,case_id='DL')
        loads=[[1.,-.3,2.],[2.,.5,-3.]]
        base=self.native(primary,[[0,0,0],[0,0,0]]);loaded=self.native(primary,loads);actual=primary.solve_increment(loads)
        expected_u=np.array(loaded['u'])-base['u']
        np.testing.assert_allclose(actual['displacement_increment'],expected_u,rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(primary.baseline['result']['cases'][0]['physical_displacements'],base['u'],rtol=1e-7,atol=1e-10)
        errors=[]
        for family in ('columns','beams'):
            for name,row in actual[family].items():
                for force in ('global','local'):
                    expected=np.array(loaded[force+'_force'][name])-base[force+'_force'][name]
                    got=np.array(row[force+'_force_increment']);errors.extend(abs(got-expected))
                    np.testing.assert_allclose(got,expected,rtol=1e-7,atol=1e-8)
        print(json.dumps(dict(witness='primary_affine_baseline_native',models=2,static_increments=40,
            maximum_displacement_error_in_rad=float(np.max(abs(np.array(actual['displacement_increment'])-expected_u))),
            maximum_member_force_error_kip_kipin=float(max(errors)),
            maximum_initial_axial_transfer_error_kip=max(abs(v) for r in primary.baseline['column_axial_ledger'] for v in r['error_kip']))))


if __name__=='__main__':unittest.main()
