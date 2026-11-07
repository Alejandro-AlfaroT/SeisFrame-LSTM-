"""Atomic, physical-inventory slab/web six-resultant strength adapter.

This module consumes an ALREADY COMPOSED common cut from action recovery.
It never adds C0, ELF, diaphragm or frame resultants itself. The physical-core
route uses recovered web actions plus only the slab directly above that web;
the remote slab is checked separately. It does not move a full effective-T
cut into a narrower web or reduce nominal/probable beam strengths.

Units: inch, kip, ksi. Nxx,Nyy,Nxy [kip/in], Mxx,Myy,Mxy [kip-in/in];
tension positive; z=0 at slab midpoint; M=int(z*sigma dz). Engineering
gamma/kxy are conjugate to Nxy/Mxy without an extra factor of two.
The common strain is constant across the declared physical cut width and
linear through depth. Upstream recovery owns spatial integration, mesh
convergence, actual support compatibility and complete core/remote coverage.

Pass means this simultaneous N/M/membrane-shear section check passed. It
does not replace transverse shear, torsion, development, detailing or global
stability checks. A failed inverse solve returns unknown, not a capacity
failure. capacity_exceeded requires a rigorously relaxed resultant bound.
"""
from __future__ import annotations

import math
import numpy as np

from Design.SMRF_Layered_Slab_Section import (
    LayeredSlabSection, design_concrete_peak_strain,
)

METHOD = 'atomic_physical_slab_web_common_section_v1'
_IDENTITY = ('state_identity', 'combination_id', 'location_id', 'recovery_id')
_MODES = ('actual_core', 'actual_common_width', 'slab_patch')
_OUTSIDE = ('transverse_shear_Vxz_Vyz', 'beam_torsion', 'development_and_anchorage',
            'member_detailing_and_deformation_capacity', 'global_stability')


def _positive(v, name):
    v=float(v)
    if not math.isfinite(v) or v<=0:raise ValueError(f'{name} must be positive and finite')
    return v


def _identity(value):
    if not isinstance(value,dict) or any(value.get(k) is None or not str(value.get(k,'')).strip() for k in _IDENTITY):
        raise ValueError('Atomic state, combination, location and recovery identities required')
    return {k:str(value[k]) for k in _IDENTITY}


def mat_layers_from_global_placement(placement, *, bounds_in, thickness_in,
                                   fy_ksi, es_ksi=29000.,bar_area_by_size=None,floor=None):
    """Homogenize actual globally phased bars over one explicit physical patch.

    bounds_in=(xmin,xmax,ymin,ymax). Each layer's actual integer bar count is
    divided by the width perpendicular to its bars. No fresh phase or assumed
    bar count is generated. A bar cut by a transverse patch boundary is refused
    rather than credited fully in two patches; move the accounting boundary
    between bars or integrate its physical area explicitly upstream. A bar
    runs through the complete axial patch interval, otherwise this adapter
    requires an upstream segmented/development treatment. Patch dimensions are
    material accounting dimensions, not an assertion about Gauss-point size.
    """
    if placement.get('status')!='pass':
        raise ValueError('A complete successful global physical mat placement is required')
    box=np.asarray(bounds_in,dtype=float)
    if box.shape!=(4,) or not np.isfinite(box).all() or box[1]<=box[0] or box[3]<=box[2]:
        raise ValueError('Positive finite x/y physical patch bounds required')
    h=_positive(thickness_in,'Thickness');fy=_positive(fy_ksi,'Steel yield')
    es=_positive(es_ksi,'Steel modulus')
    if not placement.get('layout_sha256') or not placement.get('state_identity'):
        raise ValueError('Global mat layout hash and state identity required')
    layer_source=placement
    prefix=''
    if 'floors' in placement:
        if floor is None or str(floor) not in placement['floors']:
            raise ValueError('An actual floor identity in the global physical mat placement is required')
        layer_source=placement['floors'][str(floor)]
        prefix=f'floor_{floor}/'
    intervals={'x':box[:2], 'y':box[2:]}
    rows=[]
    for name,mat in layer_source.get('layers',{}).items():
        if mat.get('status',placement['status'])!='pass':raise ValueError('Every installed mat layer must have a physical phase')
        axis=mat.get('runs_along')
        if axis not in ('x','y'):raise ValueError('Orthogonal global mat axis required')
        perpendicular='y' if axis=='x' else 'x'
        lo,hi=intervals[perpendicular];alo,ahi=intervals[axis]
        radius=_positive(mat['db_in'],'Bar diameter')/2
        # Actual ACI nominal area governs strength; db governs
        # geometric boundary containment. The kernel retains its documented
        # equal-area circular concrete subtraction convention.
        area=mat.get('bar_area_in2',mat.get('area_in2'))
        if area is None and bar_area_by_size is not None:
            area=bar_area_by_size.get(mat.get('bar_size'))
        if area is None:raise ValueError('Actual nominal bar area or explicit bar-size area mapping required')
        area=float(area)
        if not math.isfinite(area) or area<=0:raise ValueError('Positive actual bar area required')
        selected=[]
        for i,coordinate in enumerate(mat['global_transverse_coordinates_in']):
            coordinate=float(coordinate)
            if not math.isfinite(coordinate):raise ValueError('Finite actual bar coordinates required')
            if coordinate+radius>lo+1e-10 and coordinate-radius<hi-1e-10:
                if coordinate-radius<lo-1e-10 or coordinate+radius>hi+1e-10:
                    raise ValueError('Physical patch boundary cuts a mat bar; explicit split accounting required')
                selected.append((i,coordinate))
        if not selected:continue
        run=mat.get('straight_run_interval_in',mat.get('run_interval_in'))
        if run is None:raise ValueError('Actual straight mat run interval required')
        if len(run)!=2 or alo<run[0]-1e-10 or ahi>run[1]+1e-10:
            raise ValueError('Actual mat does not run through complete patch; segment/development input required')
        ids=[f'{prefix}slab/{name}/{i}' for i,_ in selected]
        rows.append(dict(id=f'slab/{name}',axis=axis,
                         z_in=h/2-float(mat['centroid_from_top_in']),
                         area_in2_per_in=len(ids)*area/(hi-lo),bar_area_in2=area,
                         db_in=2*radius,
                         fy_ksi=fy,es_ksi=es,concrete_owner='slab',
                         physical_bar_ids=ids,density_width_in=float(hi-lo),
                         physical_transverse_coordinates_in=[v for _,v in selected],
                         physical_run_segment_in=[float(alo),float(ahi)],
                         material_inventory_id=str(placement['layout_sha256']),
                         state_identity=str(placement['state_identity'])))
    if not rows:raise ValueError('No physical steel occupies the requested patch')
    return rows


def _positive_linear_integral(a,b,lo,hi):
    """Exact integral max(a+b*z,0) dz over a finite interval."""
    if b==0:return max(a,0.)*(hi-lo)
    crossing=-a/b
    if lo<crossing<hi:
        if b>0:lo=crossing
        else:hi=crossing
    elif a+b*(lo+hi)/2<=0:return 0.
    return max(0.,a*(hi-lo)+b*(hi*hi-lo*lo)/2)


class SharedFloorSection:
    """One physical slab patch, optionally with one disjoint downstand web.

    mat_layers use the LayeredSlabSection format plus physical_bar_ids,
    material_inventory_id, state_identity and density_width_in. The
    actual integer count*bar_area/density_width must equal area_in2_per_in.
    beam_rows use total area_in2 over cut_width, not area per inch, and require
    the same provenance/physical bar IDs. Beam bars above the web can declare
    concrete_owner='slab'; downstand bars name the actual web. A physical bar
    ID may occur only once in this inventory. This is local inventory ownership;
    checking the same physical bar at another cut is not duplicate capacity.
    """
    def __init__(self,thickness_in,fc_ksi,*,ec_ksi,cut_width_in,mat_layers,
                 beam_rows=(),drop_webs=(),mode,phi_requirements,
                 concrete_peak_strain=None,steel_strain_limit=.01):
        if mode not in _MODES:raise ValueError('Explicit actual_core, actual_common_width or slab_patch mode required')
        self.mode=mode;self.width=_positive(cut_width_in,'Physical cut width')
        if not phi_requirements or not isinstance(phi_requirements,dict):
            raise ValueError('All applicable named strength-reduction factors required')
        self.phi_requirements={str(k):_positive(v,'Applicable phi') for k,v in phi_requirements.items()}
        if any(v>1 for v in self.phi_requirements.values()):raise ValueError('Phi cannot exceed one')
        self.phi=min(.65,*self.phi_requirements.values())
        if mode=='slab_patch' and (drop_webs or beam_rows):
            raise ValueError('Pure slab patch cannot also own a beam web or beam bars')
        if mode!='slab_patch' and len(drop_webs)!=1:
            raise ValueError('A physical common beam cut requires exactly one nonintersecting drop web')
        if mode=='actual_core' and not math.isclose(float(drop_webs[0]['width_in']),self.width,rel_tol=0,abs_tol=1e-9):
            raise ValueError('Actual core width must equal the physical web width')
        self.ownership=[];layers=[];seen=set();state_ids=set()
        for source,rows in (('mat',mat_layers),('beam',beam_rows)):
            for raw in rows:
                row=dict(raw);ids=row.get('physical_bar_ids')
                if not isinstance(ids,(list,tuple)) or not ids or any(not isinstance(v,str) or not v.strip() for v in ids):
                    raise ValueError('Every layer must identify its actual physical bars')
                if len(ids)!=len(set(ids)) or seen.intersection(ids):
                    raise ValueError('One physical steel bar was credited twice in a common cut')
                seen.update(ids)
                if not row.get('material_inventory_id') or not row.get('state_identity'):
                    raise ValueError('Each physical steel layer needs inventory and state provenance')
                state_ids.add(str(row['state_identity']))
                if source=='beam':
                    area=_positive(row.pop('area_in2'),'Actual beam row total area')
                    if not math.isclose(area,len(ids)*float(row['bar_area_in2']),rel_tol=1e-9,abs_tol=1e-10):
                        raise ValueError('Actual beam row area differs from its physical bar inventory')
                    row['area_in2_per_in']=area/self.width
                if source=='mat':
                    if 'density_width_in' not in row:
                        raise ValueError('Actual mat count requires its physical accounting width')
                    density_width=_positive(row['density_width_in'],'Mat accounting width')
                    expected=len(ids)*float(row['bar_area_in2'])/density_width
                    if not math.isclose(float(row['area_in2_per_in']),expected,rel_tol=1e-9,abs_tol=1e-10):
                        raise ValueError('Mat density differs from actual counted bars and accounting width')
                    if mode!='slab_patch' and row['axis']==drop_webs[0]['axis'] and not math.isclose(
                            density_width,self.width,rel_tol=0,abs_tol=1e-9):
                        raise ValueError('Longitudinal mat bar count must cover this actual common-cut width')
                self.ownership.append(dict(layer_id=row['id'],source=source,
                    concrete_owner=row.get('concrete_owner','slab'),physical_bar_ids=list(ids),
                    material_inventory_id=str(row['material_inventory_id']),
                    state_identity=str(row['state_identity']),
                    area_in2_per_in=float(row['area_in2_per_in']),
                    physical_run_segment_in=row.get('physical_run_segment_in')))
                layers.append(row)
        if len(state_ids)!=1:raise ValueError('All physical steel must belong to one actual design state')
        self.state_identity=next(iter(state_ids))
        if concrete_peak_strain is None:
            self.peak_basis=design_concrete_peak_strain(fc_ksi,ec_ksi)
        else:
            peak=_positive(concrete_peak_strain,'Explicit concrete peak')
            self.peak_basis=dict(peak_strain=peak,provenance='explicit_source_backbone_peak',measured=None)
        self.kernel=LayeredSlabSection(thickness_in,fc_ksi,layers,cut_width_in=self.width,
            drop_webs=drop_webs,phi=self.phi,steel_strain_limit=steel_strain_limit,
            concrete_model='polak1993_thorenfeldt',
            concrete_peak_strain=self.peak_basis['peak_strain'],
            compression_strain_limit=self.peak_basis['peak_strain'])

    def support_upper_bound(self,direction):
        """Proved relaxed upper bound on direction dot design resultant.

        Concrete stresses lie in [-fc,0] for each normal component and
        |tau|<=fc/2 for the published no-enhancement compression-only law.
        Treating these component limits independently enlarges the admissible
        set. Integrating gross concrete (without bar subtraction), unlimited
        strain compatibility and independent +/-fy steel further enlarges it.
        Thus an exceedance proves this declared section model cannot carry the
        target. This is deliberately not an estimated capacity or safety fit.
        """
        d=np.asarray(direction,dtype=float)
        if d.shape!=(6,) or not np.isfinite(d).all():raise ValueError('Six finite dual coefficients required')
        k=self.kernel;lo,hi=-k.h/2,k.h/2
        concrete=k.fc*sum(_positive_linear_integral(-d[j],-d[j+3],lo,hi) for j in (0,1))
        concrete+=k.fc/2*sum(_positive_linear_integral(sign*d[2],sign*d[5],lo,hi) for sign in (-1,1))
        for web in k.webs:
            j=0 if web['axis']=='x' else 1
            concrete+=k.fc*web['width_in']/self.width*_positive_linear_integral(
                -d[j],-d[j+3],web['z_bottom_in'],web['z_top_in'])
        steel=0.
        for row in k.layers:
            j=0 if row['axis']=='x' else 1
            steel+=abs(d[j]+row['z_in']*d[j+3])*row['area_in2_per_in']*row['fy_ksi']
        return self.phi*(concrete+steel)

    def _exceedance(self,target):
        records=[]
        for j in range(6):
            for sign in (-1.,1.):
                direction=np.zeros(6);direction[j]=sign
                upper=self.support_upper_bound(direction);demand=float(direction@target)
                if demand>upper+1e-8+1e-9*max(abs(demand),abs(upper)):
                    records.append(dict(direction=direction.tolist(),demand=demand,
                        proved_design_upper_bound=upper,excess=demand-upper))
        return records

    def evaluate(self,common_cut,*,max_evaluations=250,path_steps=0,initial_strain=None):
        """Check one atomic state/combination/location and physical common cut.

        Required input keys: identity, resultants, units ('per_unit_width' or
        'integrated_strip'), reference_z_in=0, cut_width_in, source_identities,
        accounting. Each source identity must match the atomic common cut.
        accounting requires method='disjoint_actual_fields', evidence_id,
        equilibrium_residual and equilibrium_tolerance (nonnegative arrays of
        matching units), common_width_strain_projection_verified and
        core_remote_coverage_verified. These record upstream numerical/model
        evidence; missing evidence returns unknown even if a section root exists.
        A new core receiving all effective-flange actions is not this method.
        """
        identity=_identity(common_cut.get('identity'))
        if identity['state_identity']!=self.state_identity:
            raise ValueError('Action state differs from installed physical steel state')
        sources=common_cut.get('source_identities')
        if not isinstance(sources,list) or not sources:
            raise ValueError('Atomic common-cut source identities required')
        for source in sources:
            # Different source recovery IDs are legitimate after projection;
            # the physical state, concurrent combination and final location are not.
            normalized=_identity(source)
            if any(normalized[k]!=identity[k] for k in _IDENTITY[:-1]):
                raise ValueError('Independent combinations/locations cannot be mixed into one common cut')
        if float(common_cut.get('reference_z_in',math.nan))!=0:
            raise ValueError('All six actions must already be about the slab-midplane reference')
        if not math.isclose(float(common_cut.get('cut_width_in',math.nan)),self.width,rel_tol=0,abs_tol=1e-9):
            raise ValueError('Action and material cut widths differ')
        target=np.asarray(common_cut.get('resultants'),dtype=float)
        if target.shape!=(6,) or not np.isfinite(target).all():raise ValueError('Six finite concurrent actions required')
        units=common_cut.get('units')
        if units=='integrated_strip':target=target/self.width
        elif units!='per_unit_width':raise ValueError('Explicit action units required')
        a=common_cut.get('accounting',{});pending=[]
        if a.get('method')!='disjoint_actual_fields':pending.append('actual_disjoint_spatial_action_accounting_not_established')
        if not a.get('evidence_id'):pending.append('missing_upstream_action_evidence_identity')
        err=np.asarray(a.get('equilibrium_residual',[]),dtype=float)
        tol=np.asarray(a.get('equilibrium_tolerance',[]),dtype=float)
        if not (err.size and err.shape==tol.shape and np.isfinite(err).all() and np.isfinite(tol).all()
                and np.all(tol>=0) and np.all(abs(err)<=tol)):
            pending.append('upstream_force_moment_equilibrium_not_verified')
        if a.get('common_width_strain_projection_verified') is not True:
            pending.append('common_width_spatial_strain_projection_not_verified')
        if a.get('core_remote_coverage_verified') is not True:
            pending.append('physical_core_remote_coverage_not_verified')
        result=dict(method=METHOD,identity=identity,mode=self.mode,cut_width_in=self.width,
                    factored_resultants_per_in=target.tolist(),phi=self.phi,
                    phi_requirements=self.phi_requirements,material_ownership=self.ownership,
                    backbone=self.peak_basis,accounting=a,pending=pending,
                    scope='simultaneous_slab_web_N_M_membrane_shear',
                    material_method_basis='ACI31819_22.2.2.3_PolakVecchio1993_restricted_strength_and_primary_test_comparisons',
                    outside_scope=list(_OUTSIDE),
                    nominal_probable_effective_flange_inventory_modified=False)
        exceeded=self._exceedance(target)
        if exceeded:
            result.update(status='unknown' if pending else 'capacity_exceeded',
                section_status='capacity_exceeded',capacity_bound_violations=exceeded,
                reason='Demand exceeds a proved relaxed upper bound of the declared material model',
                numerical_state=None)
            return result
        path=None
        if path_steps:
            path=self.kernel.solve_load_path(target,required_shear_phi=self.phi,
                steps=path_steps,max_evaluations_per_step=max_evaluations)
            state=path['final_state'];verified=path['proportional_force_path_verified']
        else:
            state=self.kernel.solve(target,required_shear_phi=self.phi,
                initial_strain=initial_strain,max_evaluations=max_evaluations)
            verified=state['restricted_model_state_verified']
        if not verified:
            pending.append('common_section_equilibrium_or_material_domain_not_verified')
        result.update(status='pass' if verified and not pending else 'unknown',
            section_status='pass' if verified else 'unknown',numerical_state=state,
            path=path,capacity_bound_violations=[],
            reason='Compatible simultaneous section equilibrium verified' if verified else
                'Inverse solve/domain failure is not proof of capacity exceedance')
        return result


class SpatialSharedFloorSection:
    """Atomic seven-action adapter for actual transverse strain/bar positions.

    section is a SMRF_Spatial_Floor_Section.SpatialFloorSection. Physical
    state/inventory identity is declared once and must match every action
    source. Sixth-mode averaging flags do not qualify a seventh-mode field.
    """
    def __init__(self,section,*,state_identity,material_inventory_id):
        from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection
        if not isinstance(section,SpatialFloorSection):raise ValueError('Actual spatial seven-resultant section required')
        if not str(state_identity).strip() or not str(material_inventory_id).strip():
            raise ValueError('Actual design state and physical material inventory identities required')
        self.section=section;self.state_identity=str(state_identity);self.inventory_id=str(material_inventory_id)

    def evaluate(self,common_cut,*,initial_strain=None,max_evaluations=250):
        s=self.section;identity=_identity(common_cut.get('identity'))
        if identity['state_identity']!=self.state_identity:raise ValueError('Action and physical material states differ')
        sources=common_cut.get('source_identities')
        if not isinstance(sources,list) or not sources:raise ValueError('Atomic projected action source identities required')
        for source in sources:
            sid=_identity(source)
            if any(sid[k]!=identity[k] for k in _IDENTITY[:-1]):raise ValueError('Nonconcurrent state/combination/location in spatial cut')
        if common_cut.get('axis')!=s.axis:raise ValueError('Actual beam axis mismatch')
        if float(common_cut.get('reference_z_in',math.nan))!=0 or not math.isclose(
                float(common_cut.get('reference_s_in',math.nan)),s.sref,rel_tol=0,abs_tol=1e-10):
            raise ValueError('Spatial action reference differs from slab-midplane/beam reference')
        if not math.isclose(float(common_cut.get('cut_width_in',math.nan)),s.width,rel_tol=0,abs_tol=1e-9):
            raise ValueError('Spatial action/material widths differ')
        target=np.asarray(common_cut.get('resultants'),dtype=float)
        if target.shape!=(7,) or not np.isfinite(target).all():raise ValueError('All seven concurrent resultants including transverse first moment required')
        if common_cut.get('units')=='integrated_strip':target=target/s.width
        elif common_cut.get('units')!='per_unit_width':raise ValueError('Explicit spatial action units required')
        a=common_cut.get('accounting',{});pending=[]
        if a.get('method')!='disjoint_actual_fields' or not a.get('evidence_id'):
            pending.append('actual_disjoint_spatial_action_evidence_not_established')
        err=np.asarray(a.get('equilibrium_residual',[]),dtype=float);tol=np.asarray(a.get('equilibrium_tolerance',[]),dtype=float)
        if not (err.size and err.shape==tol.shape and np.isfinite(err).all() and np.isfinite(tol).all() and
                np.all(tol>=0) and np.all(abs(err)<=tol)):
            pending.append('upstream_force_moment_equilibrium_not_verified')
        weak=float(a.get('weak_first_moment_residual_kip_in',math.nan))
        weak_tol=float(a.get('weak_first_moment_tolerance_kip_in',math.nan))
        if not math.isfinite(weak) or not math.isfinite(weak_tol) or weak_tol<0 or abs(weak)>weak_tol:
            pending.append('weak_first_moment_action_reconstruction_not_verified')
        if a.get('spatial_seven_field_projection_verified') is not True:
            pending.append('seven_resultant_physical_field_transfer_not_verified')
        if a.get('physical_diaphragm_embedding_verified') is False:
            pending.append('planar_diaphragm_force_datum_does_not_establish_physical_slab_web_bending_response')
        if a.get('primary_force_composition_authorized') is False:
            pending.append('independent_physical_additional_load_branch_is_not_a_primary_response_correction')
        if a.get('core_remote_coverage_verified') is not True:pending.append('physical_core_remote_coverage_not_verified')
        result=dict(method='atomic_spatial_seven_shared_section_v1',identity=identity,axis=s.axis,
            cut_width_in=s.width,reference_s_in=s.sref,material_inventory_id=self.inventory_id,
            factored_resultants_per_in=target.tolist(),phi=s.phi,phi_requirements=s.phi_requirements,
            physical_longitudinal_bar_ids=[v['id'] for v in s.longitudinal],
            physical_transverse_bar_ids=[v for row in s.transverse for v in row['physical_bar_ids']],
            pending=pending,accounting=a,outside_scope=list(_OUTSIDE),
            nominal_probable_effective_flange_inventory_modified=False)
        bounds=s.component_capacity_upper_bounds();violations=[]
        for j in range(7):
            value=float(target[j]);bound=bounds['positive'][j] if value>=0 else bounds['negative_magnitude'][j]
            if abs(value)>bound+1e-8+1e-9*max(abs(value),bound):
                violations.append(dict(component=j,demand=value,proved_design_upper_bound_magnitude=bound))
        if not violations:
            certificate=s.capacity_separation_certificate(target)
            result['combined_capacity_upper_bound_certificate']=certificate
            if certificate['capacity_exceeded']:violations.append(certificate)
        if violations:
            result.update(status='unknown' if pending else 'capacity_exceeded',section_status='capacity_exceeded',
                capacity_bound_violations=violations,numerical_state=None)
            return result
        state=s.solve(target,initial_strain=initial_strain,max_evaluations=max_evaluations)
        if not state['restricted_model_state_verified']:pending.append('spatial_section_equilibrium_or_material_domain_not_verified')
        result.update(status='pass' if not pending else 'unknown',section_status=state['status'],
            numerical_state=state,capacity_bound_violations=[])
        complete=a.get('all_concurrent_demands_included') is True
        result['longitudinal_state']=dict(status='pass' if result['status']=='pass' and complete else 'unknown',
            case_id=identity['combination_id'],cut_id=identity['location_id'],state_identity=identity['state_identity'],
            recovery_id=identity['recovery_id'],strength_phi=s.phi,all_concurrent_demands_included=complete,
            bars=[{**{k:bar[k] for k in ('id','nominal_stress_ksi','area_in2','fy_ksi','s_in','z_in')},
                   'strength_credited':bar.get('strength_credited',True)}
                  for bar in state['steel'] if 's_in' in bar])
        return result
