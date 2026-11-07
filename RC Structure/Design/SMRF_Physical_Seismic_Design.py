"""Complete physical floor action branches with the retained ELF load basis.

ACI318-19 6.3.1.1 and8.4.1.9 permit consistent separate analysis models;
6.6.3.1.2 supplies0.5Ig for every member under factored lateral loading.
This branch includes ALL columns and the physical slab/web eccentricities.
It supplies simultaneous N/M/Q and column actions, not a replacement for
the required perimeter-SFRS primary forces or full Mn/Mpr capacity design.
The final design must separately retain those primary requirements and the
signed common-section primary-moment guards. No primary force is added here.

Mass distribution is actual and source-bound; external inertia is applied
once. Internal column tractions are recovered, never applied as extra loads.
ELF coefficients, accidental torsion and load-combination factors are supplied
explicitly by the retained demand generator; no R/ELF substitution occurs.
"""
from __future__ import annotations

import time
import numpy as np
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem,_hash
from Design.SMRF_Physical_Diaphragm import PhysicalDiaphragm,installed_grid_masses
from Design.SMRF_Diaphragm_Embedding import prepare_external_inertia_cases

METHOD='complete_physical_floor_ELF_and_design_displacement_branches_v1'


class PhysicalSeismicSystem:
    def __init__(self,definition,gravity_cases,*,source_identity,
                 seismic_live_load_fraction,subdivisions_per_bay,
                 max_shells_per_floor,max_coupled_dofs,max_cases,
                 patterns=(),punching_offset_in=None):
        if type(max_cases) is not int or max_cases<1:
            raise ValueError('Explicit positive finite case-count bound required')
        self.max_cases=max_cases;self.definition=definition;self.source_identity=source_identity
        self.system=CoupledFloorSystem(definition,gravity_cases,source_identity=source_identity,
            stiffness_basis='ACI318-19 6.6.3.1.2: all-member0.5Ig factored lateral branch; actual eccentric geometry',
            subdivisions_per_bay=subdivisions_per_bay,max_shells_per_floor=max_shells_per_floor,
            max_coupled_dofs=max_coupled_dofs,beam_bending_factor=.5,slab_bending_factor=.5,
            web_torsion_factor=1.,column_bending_factor=.5,patterns=patterns,
            punching_offset_in=punching_offset_in)
        self.mass=installed_grid_masses(definition);self.adapters={};self.floors={}
        for row in self.system.floors:
            level=row['description']['level'];floor=row['assembly'];self.floors[level]=floor
            self.adapters[level]=PhysicalDiaphragm(definition,level,grid_mass_by_floor=self.mass,
                seismic_live_load_fraction=seismic_live_load_fraction,source_identity=source_identity,
                mesh_axes=(floor.x,floor.y),maximum_nodes=max_shells_per_floor*4)

    def solve(self,cases,*,input_identity,pdelta=True,saved_result=None):
        """Return complete signed gravity+ELF branches, grouped by actual G.

        Every case has id, gravity_case_id, loads[nf,3], reference_xy_in
        and demand_basis_identity. loads are signed Fx,Fy,Mz about that
        reference, with the final combination factors already applied.
        Include actual ELF/Fpx/collector cases explicitly; no implicit scaling
        or selective component mixing is performed by this mechanical API.
        """
        started=time.perf_counter();cases=list(cases)
        if not input_identity or not cases or len(cases)>self.max_cases or len({c['id'] for c in cases})!=len(cases):
            raise ValueError('Unique nonempty bounded physical cases and source identity required')
        gravity={c['id'] for c in self.system.cases};groups={}
        for case in cases:
            if case.get('gravity_case_id') not in gravity or not case.get('demand_basis_identity'):
                raise ValueError('Every physical case needs its actual simultaneous gravity and ELF demand identity')
            if case['id'] in gravity:raise ValueError('Physical seismic and original gravity case identities must differ')
            groups.setdefault(case['gravity_case_id'],[]).append(case)
        output={};mapping={};complete=True
        if saved_result is not None:
            if saved_result['case_descriptions']!=cases or set(saved_result['groups'])!=set(groups):
                raise ValueError('Saved complete physical ELF case inventory differs')
        for baseline,selected in groups.items():
            prepared=prepare_external_inertia_cases(self.adapters,self.floors,selected)
            result=self.system.solve_additional_floor_loads(prepared['loads'],baseline_case_id=baseline,
                case_ids=prepared['case_ids'],input_identity=_hash(dict(input=input_identity,
                    external_inertia=prepared['input_identity'],cases=selected)),pdelta=pdelta,
                saved_response=None if saved_result is None else saved_result['groups'][baseline]['total_response'])
            output[baseline]=result;mapping[baseline]=prepared['mapping']
            complete=complete and result['status']=='equilibrated'
        return dict(method=METHOD,status='responses_computed' if complete else 'unresolved',
            definition_sha256=self.definition['identity_sha256'],state_identity=self.definition['state_identity'],
            source_identity=self.source_identity,input_identity=str(input_identity),case_descriptions=cases,
            groups=output,external_load_mapping=mapping,elapsed_seconds=time.perf_counter()-started,
            primary_force_policy='Every retained perimeter-SFRS primary requirement remains separately required',
            stiffness_basis=dict(beam_I_factor=.5,slab_I_factor=.5,column_I_factor=.5,
                web_GJ_factor=1.,equilibrium_torsion_retained=True),
            primary_moment_guard_complete=False,local_mesh_qualified=False,
            qualification_asserted=False,official_pilot_started=False)

    def compatibility(self,cases,*,input_identity,pdelta=True,saved_result=None):
        """Complete G+design-displacement local response, kept as its own case.

        Input signed displacements must already contain Cd/Ie from the
        retained primary analysis. The coupled API keeps vertical movement
        and both joint bending rotations free. This branch is not summed
        with either a primary moment or a complete physical ELF response.
        """
        cases=list(cases)
        if not input_identity or not cases or len(cases)>self.max_cases:
            raise ValueError('Explicit bounded complete compatibility case inventory required')
        result=self.system.solve_imposed_floor_displacements(cases,pdelta=pdelta,
            saved_response=None if saved_result is None else saved_result['response'])
        return dict(method=METHOD+'/compatibility',input_identity=str(input_identity),
            response=result,qualification_asserted=False,official_pilot_started=False)
