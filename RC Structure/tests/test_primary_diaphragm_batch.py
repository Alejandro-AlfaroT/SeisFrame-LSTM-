"""Complete multi-floor, multi-G batching and immutable source readout."""
import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
from Design.SMRF_Physical_Diaphragm import identity
from Design import SMRF_Diaphragm_Actions as actions
from Design.SMRF_Primary_Diaphragm_Batch import (estimate_primary_diaphragm_batch,analyze_primary_diaphragms,
    primary_diaphragm_cut,PrimaryDiaphragmReadout,replay_primary_diaphragms)
from tests.test_physical_diaphragm import inputs,hand_response


def common_fixture():
    d,mass,source,inf=inputs();sources={'synthetic_mechanics_fixture':'0'*64};source=identity(sources)
    combos=[dict(id='Ehigh',dead=1.2,live=1.6,ex=1.,ey=-.3),dict(id='Elow',dead=.9,live=0.,ex=-1.,ey=.3),
            dict(id='Ghigh',dead=1.2,live=1.6,ex=0.,ey=0.)]
    groups={}
    for gid,case in zip(('high','low'),combos):
        row=copy.deepcopy(inf);row.update(protocol_id='actual_fixture_inputs',source_identity=source,
            physical_gravity_case_id=gid,grid_mass_by_floor=mass,gravity_combination=dict(id=gid,dead=case['dead'],live=case['live']))
        row['witnesses'][0]['id']=case['id']
        loads=np.array(row['witnesses'][0]['loads']);loads[:,:2]=np.array([10.,20.])[:,None]*[case['ex'],case['ey']]
        row['witnesses'][0]['loads']=loads.tolist()
        row['witnesses'][0]['direct_increment']=hand_response(d,loads,residual=.8)
        groups[gid]=row
    return dict(definition=d,state_identity=d['state_identity'],definition_sha256=d['identity_sha256'],source_identity=sources,
        analysis_source_identity=source,input_signature='actual_fixture_inputs',combinations=combos,grid_masses=mass,
        gravity_case_mapping={'Ehigh':'high','Elow':'low','Ghigh':'high'},p5_influences=groups,
        elf=dict(story_forces_kip=[10.,20.],story_weights_kip=[sum(r['weight_kip'] for r in mass[str(k)]) for k in (1,2)]))


def settings():return dict(sds=1.,importance_factor=1.,omega0=3.,seismic_live_load_fraction=.25,subdivisions_per_bay=2,
    maximum_nodes_per_floor=1000,demand_factors_by_floor={k:dict(transfer_factor=1.,irregularity_factor=1.) for k in (1,2)},
    max_witness_floor_pairs=4,max_wall_seconds=60.)


class PrimaryBatchTests(unittest.TestCase):
    def test_actual_signed_elf_force_and_base_force_saved_replay(self):
        common=common_fixture()
        baseline=analyze_primary_diaphragms(common,**settings())
        for case in common['combinations']:
            if not (case['ex'] or case['ey']):continue
            case['ex']*=1.3;case['ey']*=1.3
            witness=common['p5_influences'][common['gravity_case_mapping'][case['id']]]['witnesses'][0]
            base=copy.deepcopy(witness);witness['loads']=(1.3*np.array(base['loads'])).tolist()
            witness['direct_increment']=hand_response(common['definition'],witness['loads'],residual=1.1)
            witness['conventional_base_witness']=dict(base,strength_factor=1.3)
        result=analyze_primary_diaphragms(common,**settings())
        policy={k:v for k,v in settings().items() if k in ('sds','importance_factor','omega0','seismic_live_load_fraction','demand_factors_by_floor')}
        self.assertTrue(replay_primary_diaphragms(result,common,**policy)['numerical_equilibrium_verified'])
        for floor in result['floors']:
            for cid in result['floors'][floor]['witnesses']:
                for a,b in zip(baseline['floors'][floor]['witnesses'][cid]['cases'][2:4],result['floors'][floor]['witnesses'][cid]['cases'][2:4]):
                    np.testing.assert_allclose(a['displacements_in'],b['displacements_in'],rtol=1e-11,atol=1e-12)
        common['p5_influences']['low']['witnesses'][0]['loads'][0][0]*=-1.
        with self.assertRaisesRegex(ValueError,'signed original ELF'):estimate_primary_diaphragm_batch(common,subdivisions_per_bay=2,maximum_nodes_per_floor=1000)

    def test_replay_rebuilds_every_signed_force_and_operator_without_a_solve(self):
        common=common_fixture();result=analyze_primary_diaphragms(common,**settings())
        policy={k:v for k,v in settings().items() if k in ('sds','importance_factor','omega0','seismic_live_load_fraction','demand_factors_by_floor')}
        with mock.patch.object(actions,'splu',side_effect=AssertionError('Readback must not factor or solve')):
            audit=replay_primary_diaphragms(result,common,**policy)
        self.assertEqual(audit['force_cases_checked'],28);self.assertEqual(audit['factorizations'],0)
        self.assertTrue(audit['numerical_equilibrium_verified']);self.assertFalse(audit['mesh_convergence_verified'])
        self.assertLess(audit['maximum_normalized_nodal_residual'],1e-8)
        for kind in ('displacement','web','missing_branch','factors'):
            changed=copy.deepcopy(result);floor=changed['floors']['1'];case=floor['witnesses']['Ehigh']['cases'][0]
            if kind=='displacement':case['displacements_in'][0][0]+=.01
            elif kind=='web':case['beam_axial'][0]['N_kip']+=1.
            elif kind=='missing_branch':floor['witnesses']['Ehigh']['cases'].pop()
            else:case['factors'][next(iter(case['factors']))]*=1.5
            # Recompute all container hashes: numeric replay must reject even
            # when an incorrect payload is self-consistently serialized.
            floor['payload_sha256']=identity({k:v for k,v in floor.items() if k!='payload_sha256'})
            changed['payload_sha256']=identity({k:v for k,v in changed.items() if k!='payload_sha256'})
            with self.subTest(kind=kind),self.assertRaises(ValueError):replay_primary_diaphragms(changed,common,**policy)

    def test_estimate_has_exact_inventory_and_never_assembles_a_matrix(self):
        common=common_fixture()
        with mock.patch.object(actions,'_assemble',side_effect=AssertionError('Estimate cannot assemble')):
            estimate=estimate_primary_diaphragm_batch(common,subdivisions_per_bay=2,maximum_nodes_per_floor=1000)
        self.assertEqual(estimate['factorizations'],2);self.assertEqual(estimate['total_force_cases'],28)
        self.assertFalse(estimate['sparse_factor_memory_included']);self.assertGreater(estimate['estimated_retained_python_bytes'],0)

    def test_every_gravity_and_signed_witness_retained_one_factor_per_floor(self):
        common=common_fixture();real=actions.splu
        with mock.patch.object(actions,'splu',wraps=real) as factor:result=analyze_primary_diaphragms(common,**settings())
        self.assertEqual(factor.call_count,2);self.assertEqual(result['completed_witness_floor_pairs'],4)
        for floor,row in result['floors'].items():
            self.assertEqual(set(row['witnesses']),{'Ehigh','Elow'})
            self.assertEqual(row['prepared_operator']['factorizations'],1);self.assertEqual(row['prepared_operator']['case_rhs_solved'],14)
            for case in row['witnesses'].values():
                self.assertNotIn('components',case);self.assertNotIn('mesh',case);self.assertEqual(len(case['cases']),7)
                self.assertTrue(all(c['slab_samples_omitted'] for c in case['cases']))
        reader=PrimaryDiaphragmReadout(result)
        cut=reader.cut(1,'Ehigh','collector_original_direct_omega',axis='x',station_in=11.7,
            interval_in=[-1.,1.],trace='plus',transverse_reference_in=0.)
        self.assertEqual(cut['combination_id'],'Ehigh');self.assertEqual(cut['gravity_case_id'],'high')
        self.assertFalse(cut['physical_out_of_plane_compatibility_claimed'])
        self.assertLess(np.max(np.abs(cut['floor_freebody_residual'])),1e-8)
        self.assertTrue(np.all(np.abs(cut['floor_freebody_residual'])<=np.array(cut['floor_freebody_tolerance'])))
        witness=result['floors']['1']['witnesses']['Ehigh'];case=next(c for c in witness['cases'] if c['id']==cut['branch_id'])
        f=sum(abs(case['factors'].get(a['id'],0.))*a['equilibrium_tolerance']*a['force_normalization_kip'] for a in witness['component_audit'])
        self.assertAlmostEqual(cut['floor_freebody_tolerance'][0],f)

    def test_per_floor_artifacts_replay_exact_fields_and_detect_changed_bytes(self):
        common=common_fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd().resolve()) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(Path.cwd().resolve()))
            stored=analyze_primary_diaphragms(common,**settings(),artifact_directory=tmp)
            self.assertTrue(all('artifact_file' in r for r in stored['floors'].values()))
            reader=PrimaryDiaphragmReadout(stored);first=reader.floor(1);self.assertIs(reader.floor(1),first)
            self.assertEqual(reader.floor(2)['adapter']['floor'],2)
            path=Path(stored['floors']['1']['artifact_file']);path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'artifact content'):reader.floor(1)

    def test_missing_signed_case_wrong_gravity_and_wrong_mass_remain_rejected(self):
        for mutate in ('missing','gravity','mass'):
            common=common_fixture()
            if mutate=='missing':common['p5_influences']['high']['witnesses']=[]
            elif mutate=='gravity':common['p5_influences']['high']['gravity_combination']['dead']=1.
            else:common['elf']['story_weights_kip'][0]+=.1
            with self.assertRaises(ValueError):estimate_primary_diaphragm_batch(common,subdivisions_per_bay=2,maximum_nodes_per_floor=1000)
        common=common_fixture()
        with self.assertRaisesRegex(ValueError,'pair count'):analyze_primary_diaphragms(common,**{**settings(),'max_witness_floor_pairs':3})


if __name__=='__main__':unittest.main()
