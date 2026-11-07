"""Actual physical-column force transfer and retained primary ELF increments."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from Design.SMRF_Primary_Frame_Transfer import PrimaryFrame,_hash,transferred_floor_influences,verify_transferred_floor_influences
from Design.SMRF_Floor_Influences import floor_free_bodies
from test_coupled_floor_geometry import fixture,build


def primary_fixture(system):
    d=system.definition;g=d['geometry'];positions={tuple(n['xyz']):n['id'] for n in d['nodes']};beams=[]
    for floor in d['floors']:
        for beam in floor['beams']:
            b,h,e=beam['b_in'],beam['h_in'],beam['E_ksi'];t=d['slab_thickness_in'];bf=b+12
            af,aw=bf*t,b*(h-t);zc=-h/2
            inertia=bf*t**3/12+b*(h-t)**3/12+af*aw/(af+aw)*zc**2
            xyz0=([beam['start'],beam['line'],floor['z_in']] if beam['axis']=='x' else [beam['line'],beam['start'],floor['z_in']])
            xyz1=([beam['end'],beam['line'],floor['z_in']] if beam['axis']=='x' else [beam['line'],beam['end'],floor['z_in']])
            release=3 if beam['axis']=='y' and beam['line']==g['lx'] else 0
            beams.append(dict(id=beam['id'],i=positions[tuple(xyz0)],j=positions[tuple(xyz1)],level=floor['level'],
                A=b*h,E=e,G=e/2.4,J=.35*h*b**3/4,Iy=.35*inertia,Iz=.35*h*b**3/12,
                vecxz=[0,0,1],releasey=release,releasez=release,kind=f'beam_{beam["axis"]}'))
    result=dict(method='independent_primary_fixture',physical_definition_sha256=d['identity_sha256'],
        state_identity=d['state_identity'],lateral_design_model='mixed_released_primary_fixture',nodes=d['nodes'],
        geometry=g,beams=beams,columns=system._columns,gravity_case=system.cases[0])
    result['identity_sha256']=_hash(result)
    return result


class PrimaryTransferTests(unittest.TestCase):
    def test_saved_primary_gravity_and_increment_replay_without_frame_solve(self):
        system,physical,primary=self.system();transfer=primary.transfer_gravity(physical,case_id='DL')
        loads=[[1.,-.3,2.],[2.,.5,-3.]];increment=primary.solve_increment(loads)
        other=PrimaryFrame(primary_fixture(system),source_identity=system.source_identity)
        with patch('Design.SMRF_Substructure_Frame.SubstructureFrame.solve',side_effect=AssertionError('No frame solve on saved readback')):
            other.transfer_gravity(physical,case_id='DL',saved_transfer=transfer)
            rebuilt=other.solve_increment(loads,saved_increment=increment)
        self.assertEqual(rebuilt['beams'],increment['beams'])
        wrong=copy.deepcopy(increment);wrong['beams'][next(iter(wrong['beams']))]['local_force_increment'][0]+=.1
        with self.assertRaisesRegex(ValueError,'increment differs'):other.solve_increment(loads,saved_increment=wrong)

    def system(self):
        system=build(fixture(),[dict(id='DL',dead=1.,live=1.)]);physical=system.solve(pdelta=True)
        primary=PrimaryFrame(primary_fixture(system),source_identity=system.source_identity)
        return system,physical,primary

    def test_every_actual_column_axial_end_is_reproduced_and_auxiliary_cancels(self):
        system,physical,primary=self.system()
        result=primary.transfer_gravity(physical,case_id='DL')
        self.assertTrue(all(r['passed'] for r in result['column_axial_ledger']))
        self.assertEqual(len(result['column_axial_ledger']),len(system.definition['columns']))
        self.assertLess(np.max(np.abs(result['target_displacement_error_in_rad'])),1e-10)
        zero=primary.solve_increment([[0,0,0],[0,0,0]])
        self.assertLess(np.max(np.abs(zero['displacement_increment'])),1e-10)
        self.assertTrue(all(np.max(np.abs(r['global_force_increment']))<1e-9 for r in zero['beams'].values()))
        loaded=primary.solve_increment([[1.,-.3,2.],[2.,.5,-3.]])
        self.assertEqual(loaded['status'],'equilibrated')
        self.assertEqual(max(map(abs,loaded['auxiliary_load_increment'])),0.)
        bodies=floor_free_bodies(primary.influence_response(loaded),loaded['floor_loads'])
        self.assertTrue(all(r['equilibrium_passed'] for r in bodies))
        self.assertLess(max(r['normalized_balance_error'] for r in bodies),1e-8)

    def test_actual_source_case_and_column_force_mismatches_are_refused(self):
        _,physical,primary=self.system()
        bad=copy.deepcopy(physical);bad['source_identity']='old_source'
        with self.assertRaisesRegex(ValueError,'source identities'):primary.transfer_gravity(bad,case_id='DL')
        bad=copy.deepcopy(physical);bad['cases'][0]['dead']=1.2
        with self.assertRaisesRegex(ValueError,'declaration differs'):primary.transfer_gravity(bad,case_id='DL')
        bad=copy.deepcopy(physical);next(iter(bad['frame']['cases'][0]['columns'].values()))['local_force'][0]+=1.
        with self.assertRaisesRegex(ValueError,'every actual physical column axial'):primary.transfer_gravity(bad,case_id='DL')

    def test_saved_input_requires_actual_payload_and_keeps_prior_source(self):
        _,physical,primary=self.system();physical['source_identity']='prior_source_snapshot'
        binding=dict(source_identity=physical['source_identity'],payload_sha256=_hash(physical),
                     input_manifest_sha256='frozen_input_manifest')
        result=primary.transfer_gravity(physical,case_id='DL',saved_input_binding=binding)
        self.assertEqual(result['physical_result_source_identity'],'prior_source_snapshot')
        self.assertEqual(result['saved_input_binding'],binding)
        altered=copy.deepcopy(physical);altered['frame']['cases'][0]['physical_displacements'][0]+=.01
        with self.assertRaisesRegex(ValueError,'actual content/source'):
            primary.transfer_gravity(altered,case_id='DL',saved_input_binding=binding)

    def test_rigid_projection_changes_no_vertical_displacement(self):
        _,physical,primary=self.system();dofs=list(map(tuple,physical['frame']['physical_dof_ids']))
        u=physical['frame']['cases'][0]['physical_displacements']
        result=primary.transfer_gravity(physical,case_id='DL');new=result['result']['cases'][0]['physical_displacements']
        for i,(_,dof) in enumerate(dofs):
            if dof in (3,4,5):self.assertAlmostEqual(new[i],u[i],places=10)
        # Beam releases are retained verbatim in the actual force adapter.
        self.assertTrue(any(e.definition['releasey']==3 for e,_ in primary.beams))

    def test_exact_tangent_matches_independent_signed_increment_derivative(self):
        _,physical,primary=self.system();primary.transfer_gravity(physical,case_id='DL')
        loads=np.array([[1.,-.3,2.],[2.,.5,-3.]])
        reference=primary.incremental_reference(loads);h=.001
        plus=primary.solve_increment(h*loads);minus=primary.solve_increment(-h*loads)
        central=(np.array(plus['displacement_increment'])-minus['displacement_increment'])/(2*h)
        np.testing.assert_allclose(central,reference['exact_tangent_displacement'],rtol=1e-6,atol=1e-9)
        self.assertEqual(len(reference['initial_column_P']),len(primary.definition['columns']))
        self.assertTrue(all('delta_basic_axial_tension_kip' in r for r in plus['columns'].values()))

    def test_complete_central_basis_retains_actual_second_order_free_body(self):
        from Design.SMRF_Floor_Influences import decompose_with_second_order_residual
        system,physical,primary=self.system();primary.transfer_gravity(physical,case_id='DL')
        masses={'1':{'fixture':1.},'2':{'fixture':1.}}
        result=transferred_floor_influences(primary,system.definition,masses,
            witness_cases=[dict(id='mixed',loads=[[1.,-.3,2.],[2.,.5,-3.]])],
            protocol_id='two_floor_unit_fixture',max_direct_solves=25,max_wall_seconds=30.)
        self.assertTrue(result['all_linearizations_converged']);self.assertEqual(result['model_solves'],25)
        self.assertFalse(result['auxiliary_loads_included_as_design_demand'])
        with patch('Design.SMRF_Substructure_Frame.SubstructureFrame.solve',side_effect=AssertionError('No frame solve on influence replay')):
            proof=verify_transferred_floor_influences(primary,system.definition,masses,result)
        self.assertEqual(proof['raw_primary_increments_verified'],25)
        corrupt=copy.deepcopy(result);corrupt['raw_increment_witnesses'][0]['increment']['frame']['cases'][0]['physical_displacements'][0]+=.01
        with self.assertRaises(ValueError):verify_transferred_floor_influences(primary,system.definition,masses,corrupt)
        for floor in (1,2):
            audit=decompose_with_second_order_residual(result,'mixed',floor)
            self.assertLess(audit['direct_reconstruction_max_abs_kip_kipin'],1e-8)
        with self.assertRaisesRegex(ValueError,'bound is insufficient'):
            transferred_floor_influences(primary,system.definition,masses,
                witness_cases=[dict(id='mixed',loads=[[1.,0,0],[2.,0,0]])],
                protocol_id='unit',max_direct_solves=24,max_wall_seconds=30.)


if __name__=='__main__':unittest.main()
