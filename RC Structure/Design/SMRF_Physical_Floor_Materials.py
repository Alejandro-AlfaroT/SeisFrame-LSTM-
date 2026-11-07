"""Actual installed straight-bar inventory and frozen static floor references.

The fixed reference is a declared linear analysis under ACI318-19 6.3.1.1,
R6.3.1.1 and6.6.3.1.1, not a universal nonlinear-convergence requirement or
force bound. Actual service/reference gravity strains determine a local
nominal secant; its zero-incremental-N Schur3x3 bending matrix is frozen.
Root's exact generalized energy assembly retains gross membrane/geometric
coupling and the separately declared beam/column/torsion assumptions.

Cell material homogenization uses actual finite straight-bar cylinders,
clipped by BOTH plan boundaries, actual straight-run ends and slab faces.
No integer density is assumed in a partial cell. Volume and first/second
depth moments are integrated analytically on each circular-segment interval.
Hook/anchorage steel is separately owned by the authoritative anchorage
inventory; its stiffness is not credited by this straight-bar section model.
This scope is explicit in the reference audit and requires the existing
physical hook/development checks. It is not a complete3D anchorage constitutive
model. Final spatial7strength/anchorage/torsion checks remain separate.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np

from Design.SMRF_Slab_Secant import LayeredSlabSecant

METHOD='actual_clipped_inventory_frozen_gravity_reference_v2_exact_circle_limits'


class PhysicalInventoryUnresolved(ValueError):
    """The installed mat or cage inventory could not be frozen; ``evidence`` says what was unresolved and where."""
    def __init__(self,message,evidence=None):
        super().__init__(message);self.evidence=evidence or {}


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sqrt_primitives(r,z,center):
    # The integration limits are constructed as center +/- r. Subtracting
    # center again can move that exact limit just inside the circle, where
    # asin's endpoint sensitivity loses full-cylinder area/first moments.
    # Recognize the actual limits directly; do not snap interior points or
    # change a geometric/integration tolerance.
    u=(-r if z<=center-r else r if z>=center+r else z-center)
    root=math.sqrt(max(0.,r*r-u*u));angle=math.atan2(u,root)
    j0=(u*root+r*r*angle)/2
    j1=-(max(0.,r*r-u*u))**1.5/3
    j2=(r**4*angle+u*root*(2*u*u-r*r))/8
    return np.array([j0,center*j0+j1,center*center*j0+2*center*j1+j2])


class ClippedBar:
    """One nominal-area circular straight cylinder clipped to a real cell."""
    def __init__(self,bar,bounds,slab_thickness):
        self.bar=dict(bar);self.bounds=tuple(map(float,bounds));self.h=float(slab_thickness)
        if len(self.bounds)!=4 or not all(math.isfinite(v) for v in self.bounds):raise ValueError('Finite cell xmin,xmax,ymin,ymax required')
        x0,x1,y0,y1=self.bounds
        if x1<=x0 or y1<=y0 or self.h<=0:raise ValueError('Positive physical cell/slab dimensions required')
        if bar.get('axis') not in ('x','y') or not bar.get('id'):raise ValueError('Actual bar identity/axis required')
        self.axis=bar['axis'];self.sc=float(bar['s_in']);self.zc=float(bar['z_in'])
        area=float(bar['area_in2'])
        if not all(math.isfinite(v) for v in (self.sc,self.zc,area)) or area<=0:raise ValueError('Finite actual bar geometry required')
        self.r=math.sqrt(area/math.pi)
        along=(x0,x1) if self.axis=='x' else (y0,y1);cross=(y0,y1) if self.axis=='x' else (x0,x1)
        run=tuple(map(float,bar['straight_run_interval_in']))
        if len(run)!=2 or not all(math.isfinite(v) for v in run) or run[1]<=run[0]:raise ValueError('Positive finite actual straight run required')
        self.along=(max(along[0],run[0]),min(along[1],run[1]))
        self.length=max(0.,self.along[1]-self.along[0]);self.cross=cross;self.cell_area=(x1-x0)*(y1-y0)
        lo=max(-self.h/2,self.zc-self.r);hi=min(self.h/2,self.zc+self.r)
        breaks=[lo,hi] if hi>lo else []
        for bound in cross:
            dist=bound-self.sc
            if abs(dist)<self.r:
                delta=math.sqrt(self.r*self.r-dist*dist)
                breaks.extend(v for v in (self.zc-delta,self.zc+delta) if lo<v<hi)
        self.breaks=sorted(set(breaks));self.segments=[];mom=np.zeros(3);first_s=0.
        for za,zb in zip(self.breaks[:-1],self.breaks[1:]):
            middle=(za+zb)/2;rad=math.sqrt(max(0.,self.r*self.r-(middle-self.zc)**2))
            if min(cross[1],self.sc+rad)<=max(cross[0],self.sc-rad):continue
            ch,ah=(cross[1],0.) if cross[1]<=self.sc+rad else (self.sc,1.)
            cl,al=(cross[0],0.) if cross[0]>=self.sc-rad else (self.sc,-1.)
            constant,coefficient=ch-cl,ah-al
            poly=np.array([(zb**(i+1)-za**(i+1))/(i+1) for i in range(3)])
            sqrt=_sqrt_primitives(self.r,zb,self.zc)-_sqrt_primitives(self.r,za,self.zc)
            mom+=constant*poly+coefficient*sqrt
            radial_squared=self.r*self.r*(zb-za)-((zb-self.zc)**3-(za-self.zc)**3)/3
            first_s+=(ch*ch-cl*cl)/2*(zb-za)+(ch*ah-cl*al)*sqrt[0]+(ah*ah-al*al)/2*radial_squared
            self.segments.append((za,zb,constant,coefficient))
        self.moments=self.length*mom
        self.first_transverse=self.length*first_s
        self.first_longitudinal=self.moments[0]*sum(self.along)/2

    def occupied(self,z):
        z=np.asarray(z,dtype=float);radius=np.sqrt(np.maximum(0.,self.r*self.r-(z-self.zc)**2))
        width=np.maximum(0.,np.minimum(self.cross[1],self.sc+radius)-np.maximum(self.cross[0],self.sc-radius))
        active=(abs(z-self.zc)<self.r)&(z>=-self.h/2)&(z<=self.h/2)
        return np.where(active,self.length*width/self.cell_area,0.)

    def record(self):
        mx,my=(self.first_longitudinal,self.first_transverse) if self.axis=='x' else (self.first_transverse,self.first_longitudinal)
        return dict(id=self.bar['id'],axis=self.axis,operator_owner=self.bar['operator_owner'],
            straight_volume_in3=float(self.moments[0]),first_x_in4=float(mx),first_y_in4=float(my),
            first_z_in4=float(self.moments[1]),second_z_in5=float(self.moments[2]),
            clipped_run_interval_in=list(self.along),bounds_in=list(self.bounds))


class CellVoidProfile:
    def __init__(self,bounds_in,slab_thickness_in,bars,*,state_identity):
        if not state_identity:raise ValueError('Actual installed-state identity required')
        self.bounds=tuple(map(float,bounds_in));self.h=float(slab_thickness_in)
        if len(self.bounds)!=4 or not all(math.isfinite(v) for v in self.bounds) or self.bounds[1]<=self.bounds[0] or self.bounds[3]<=self.bounds[2] or not math.isfinite(self.h) or self.h<=0:
            raise ValueError('Finite positive physical cell/slab dimensions required')
        if len({b['id'] for b in bars})!=len(bars):raise ValueError('Physical steel identities cannot be duplicated')
        if any(b['operator_owner'] not in ('slab','web_line') for b in bars):raise ValueError('Explicit slab or web_line steel ownership required')
        self.parts=[ClippedBar(b,self.bounds,self.h) for b in bars]
        self.parts=[p for p in self.parts if p.moments[0]>1e-14]
        self.breaks=sorted(set(v for p in self.parts for v in p.breaks))
        self.records=[p.record() for p in self.parts]
        self.void_only_ids=sorted(p.bar['id'] for p in self.parts if p.bar['operator_owner']=='web_line')
        self.identity=_hash({'state_identity':str(state_identity),'bounds':self.bounds,'h':self.h,'parts':self.records})
        self.area=(self.bounds[1]-self.bounds[0])*(self.bounds[3]-self.bounds[2])
        self.exact_depth_moments=np.sum([p.moments for p in self.parts],axis=0) if self.parts else np.zeros(3)

    def occupied(self,z):
        return sum((p.occupied(z) for p in self.parts),np.zeros_like(np.asarray(z,dtype=float)))

    def slab_steel_layers(self):
        rows=[]
        for p in self.parts:
            b=p.bar
            if b['operator_owner']!='slab':continue
            # The nominal steel is homogenized by its actual clipped volume;
            # standard section steel strain remains at the bar centroid.
            rows.append(dict(id=b['id'],axis=b['axis'],z_in=b['z_in'],
                area_in2_per_in=float(p.moments[0]/self.area),bar_area_in2=b['area_in2'],
                db_in=b['db_in'],fy_ksi=b['fy_ksi'],es_ksi=b['es_ksi']))
        return rows


def _brief_failures(failures,limit=8):
    """At most ``limit`` failing rules of one member end, one of every rule before any repeat, collisions cut to 4."""
    seen=set();first=[];rest=[]
    for c in failures:
        (rest if c.get('rule') in seen else first).append(c);seen.add(c.get('rule'))
    return [{k:v for k,v in c.items() if k!='collisions'}|({'collisions':c['collisions'][:4]} if c.get('collisions') else {})
            for c in (first+rest)[:limit]]


def installed_inventory(state=None,*,beam_interval_provider=None,allow_unqualified_geometry=False):
    """Freeze authoritative installed mat phase and actual member cage lanes."""
    import Structure_Parameters as sp
    from Model import Member_Groups as mg
    from Design.SMRF_Joint_Mat_Physical import global_mat_placement
    from Design.SMRF_Splice_Physical import group_beam_lanes
    state=state or mg.active()
    if state is None or mg.active() is None or state.identity()!=mg.active().identity():
        raise ValueError('The exact grouped state must be installed before material freezing')
    placement=global_mat_placement(state);lanes=group_beam_lanes(state)
    if placement['status']!='pass':
        raise PhysicalInventoryUnresolved('Authoritative actual global mat placement is unresolved',
            dict(placement_status=placement['status'],layers={n:dict(status=v.get('status'),phase_trials=v.get('phase_trials'))
                 for n,v in (placement.get('layers') or {}).items()},reason=placement.get('reason')))
    if beam_interval_provider is None:
        from Design import SMRF_Transition_Context
        beam_interval_provider=getattr(SMRF_Transition_Context,'installed_beam_straight_intervals',None)
    if not callable(beam_interval_provider):
        raise ValueError('Authoritative installed beam straight intervals required; grid endpoints can overstate hooked bars')
    h=float(sp.SLAB_THICKNESS_IN);floors={};beamrows={};beam_geometry={};physical_failures=[];unresolved=[]
    rule_counts={};rule_examples={}
    for floor,record in placement['floors'].items():
        bars=[];hooks=[]
        for name,mat in record['layers'].items():
            for i,coordinate in enumerate(mat['global_transverse_coordinates_in']):
                ident=f'floor_{floor}/slab/{name}/{i}'
                bars.append(dict(id=ident,axis=mat['runs_along'],s_in=float(coordinate),
                    mat_name=name,
                    z_in=h/2-mat['centroid_from_top_in'],area_in2=sp.rebar_area(mat['bar_size']),
                    db_in=mat['db_in'],fy_ksi=sp.SLAB_FY_KSI,es_ksi=sp.ES_KSI,
                    straight_run_interval_in=list(mat['straight_run_interval_in']),operator_owner='slab'))
                hooks.append(dict(id=ident,hook_vertical_axes_in=mat['hook_vertical_axes_in'],
                    hook_radius_in=mat['hook_radius_in'],hook_extension_in=mat['hook_extension_in'],
                    stiffness_owner='anchorage_idealization_no_straight_section_credit'))
        floors[str(floor)]=dict(bars=bars,hook_inventory=hooks,footprint=record['footprint'])
    for member in state.members():
        if member.is_column:continue
        lane=lanes[member.group_id]
        if lane['status']!='pass':
            raise PhysicalInventoryUnresolved('Actual beam cage lanes are unresolved',
                dict(group_id=member.group_id,member_tag=member.member_tag,lane_status=lane['status'],lane_method=lane.get('method')))
        design=member.design;axis='x' if member.member_type=='beam_x' else 'y'
        center=member.grid_j*sp.BAY_Y if axis=='x' else member.grid_i*sp.BAY_X
        intervals=beam_interval_provider(state,member.member_tag)
        if not intervals.get('evidence_id') or (intervals.get('status')!='pass' and not
                (allow_unqualified_geometry and intervals.get('geometry_complete') is True)):
            # Every unresolved member is collected before the refusal, so its evidence names them all.
            unresolved.append(dict(member_tag=member.member_tag,group_id=member.group_id,member_type=member.member_type,
                floor=member.story_or_floor,grid=[member.grid_i,member.grid_j],status=intervals.get('status'),
                cage=dict(b_in=design.b_in,h_in=design.h_in,bar_size=design.bar_size,top_bars=design.top_bars,
                          bot_bars=design.bot_bars,stirrup_bar_size=design.stirrup_bar_size),
                ends=[dict(joint_id=e.get('joint_id'),beam_context_status=e.get('beam_context_status'),
                           failing_rules=_brief_failures(e.get('geometry_failures') or []))
                      for e in intervals.get('end_evidence') or []]))
            for e in intervals.get('end_evidence') or []:
                for c in e.get('geometry_failures') or []:
                    rule_counts[c.get('rule')]=rule_counts.get(c.get('rule'),0)+1
                    rule_examples.setdefault(c.get('rule'),len(unresolved)-1)
            continue
        if intervals.get('status')!='pass':
            physical_failures.append(dict(member_id=str(member.member_tag),physical_detail_status=intervals.get('physical_detail_status',intervals['status']),
                evidence_id=intervals['evidence_id'],end_evidence=intervals.get('end_evidence')))
        runs={v['source_lane_id']:v for v in intervals['bars']}
        if set(runs)!={p['id'] for p in lane['coordinates']}:
            raise ValueError('Authoritative beam intervals must cover the exact actual cage lane inventory')
        bars=[]
        for p in lane['coordinates']:
            bars.append(dict(id=f'member_{member.member_tag}/beam/{p["id"]}',source_lane_id=p['id'],
                member_id=str(member.member_tag),axis=axis,s_in=center+p['x_in'],
                z_in=h/2-(design.h_in/2-p['y_in']),area_in2=sp.rebar_area(design.bar_size),
                db_in=sp.rebar_diameter(design.bar_size),fy_ksi=sp.FY_KSI,es_ksi=sp.ES_KSI,
                straight_run_interval_in=list(runs[p['id']]['owned_straight_interval_in']),
                physical_straight_run_interval_in=list(runs[p['id']]['straight_run_interval_in']),
                straight_run_evidence_id=intervals['evidence_id'],operator_owner='web_line'))
        beamrows[str(member.member_tag)]=bars
        beam_geometry[str(member.member_tag)]=dict(axis=axis,floor=member.story_or_floor,
            line_in=center,b_in=design.b_in,h_in=design.h_in,fc_ksi=design.fc_ksi,
            ec_ksi=sp.concrete_ec_ksi(design.fc_ksi),straight_run_evidence=intervals)
        floors[str(member.story_or_floor)]['bars'].extend(bars)
    if unresolved:
        # Counts are of every failing rule (before any list is cut); the listed members are the first 24 and one
        # example of each rule not among them.
        listed=sorted(set(range(min(24,len(unresolved))))|set(rule_examples.values()))
        raise PhysicalInventoryUnresolved('Authoritative installed beam straight intervals are unresolved',
            dict(unresolved_member_count=len(unresolved),failing_rule_counts=rule_counts,
                 failing_rule_example_member={str(rule):unresolved[k]['member_tag'] for rule,k in rule_examples.items()},
                 groups=sorted({r['group_id'] for r in unresolved}),members=[unresolved[k] for k in listed]))
    record=dict(method=METHOD,state_identity=state.identity(),layout_sha256=placement['layout_sha256'],
        slab_thickness_in=h,floors=floors,beam_bars=beamrows,beam_geometry=beam_geometry,
        physical_detail_status='fail' if physical_failures else 'pass',physical_detail_failures=physical_failures,
        unqualified_geometry_diagnostic=bool(allow_unqualified_geometry),
        hook_stiffness_basis='No hooked-leg stiffness credit; physical geometry/development checked by authoritative anchorage modules')
    record['identity_sha256']=_hash(record)
    return record


def spatial_section_from_inventory(inventory,*,floor,axis,station_in,interval_in,
        axial_accounting_interval_in,fc_ksi,ec_ksi,phi_requirements,member_id=None,
        transverse_reference=None,development_evidence=(),quadrature_order=12,
        apply_development_domains=False,material_scope='common_slab_web'):
    """Reproducible actual physical cut inventory for the common seven modes.

    Both transverse strip boundaries and axial accounting boundaries must
    clear actual bar circles. The latter gives an integer transverse-bar row
    over one explicitly declared physical length. No phase/fractional steel
    is invented. Strength action averaging over that length remains upstream.
    Development evidence is reported separately and is never inferred merely
    from a straight bar crossing the cut.
    """
    from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection
    if material_scope not in ('common_slab_web','slab_only'):
        raise ValueError('Explicit common-section or slab-only material ownership required')
    slab_only=material_scope=='slab_only'
    if slab_only and member_id is not None:raise ValueError('Slab-only resistance cannot also own a beam web')
    if axis not in ('x','y'):raise ValueError('Actual x or y section axis required')
    lo,hi=map(float,interval_in);a,b=map(float,axial_accounting_interval_in);station=float(station_in)
    if not all(math.isfinite(v) for v in (lo,hi,a,b,station)) or hi<=lo or b<=a or not a<=station<=b:
        raise ValueError('Positive finite physical strip/accounting interval containing the cut required')
    reference=(lo+hi)/2 if transverse_reference is None else float(transverse_reference)
    if not math.isfinite(reference):raise ValueError('Finite actual transverse reference required')
    material=inventory['floors'][str(floor)];bars=material['bars'];h=float(inventory['slab_thickness_in'])
    longitudinal=[];groups={};issues=[];excluded=[];selected_ids=[];web=None;window_issues=[]
    strength_fc=float(fc_ksi);strength_ec=float(ec_ksi);actual_grades={'slab_fc_ksi':strength_fc}
    if member_id is not None:
        member_id=str(member_id);g=inventory['beam_geometry'][member_id]
        if g['axis']!=axis or int(g['floor'])!=int(floor):raise ValueError('Actual member axis/floor differs from the cut')
        # A uniform resistance model uses the lower specified grade in both
        # disjoint regions. Higher actual web grade receives no extra credit;
        # physical analysis stiffness still uses each actual material.
        actual_grades['web_fc_ksi']=float(g['fc_ksi'])
        strength_fc=min(strength_fc,float(g['fc_ksi']));strength_ec=min(strength_ec,float(g['ec_ksi']))
        web=dict(id='member_'+member_id,interval_in=[g['line_in']-g['b_in']/2,g['line_in']+g['b_in']/2],
                 z_interval_in=[h/2-g['h_in'],-h/2])
    for r in bars:
        radius=float(r['db_in'])/2;s=float(r['s_in']);run=tuple(map(float,r['straight_run_interval_in']))
        void_only=slab_only and r['operator_owner']=='web_line'
        if void_only and (r['z_in']+radius<=-h/2+1e-9 or r['z_in']-radius>=h/2-1e-9):continue
        if void_only and (r['z_in']-radius<-h/2-1e-9 or r['z_in']+radius>h/2+1e-9):
            issues.append(dict(reason='beam_bar_circle_crosses_slab_only_depth_boundary',bar_id=r['id']));continue
        if r['axis']==axis:
            if (min(b,run[1])>max(a,run[0])+1e-9 and s+radius>lo+1e-9 and s-radius<hi-1e-9 and
                    (run[0]>a+1e-9 or run[1]<b-1e-9)):
                window_issues.append(dict(reason='longitudinal_physical_bar_ends_inside_axial_window',bar_id=r['id'],
                    physical_run_interval_in=list(run)))
            if not run[0]-1e-9<=station<=run[1]+1e-9:continue
            if s+radius<=lo+1e-9 or s-radius>=hi-1e-9:continue
            if s-radius<lo-1e-9 or s+radius>hi+1e-9:
                issues.append(dict(reason='longitudinal_circle_crosses_strip_boundary',bar_id=r['id']));continue
            if r['operator_owner']=='web_line' and not slab_only and str(r.get('member_id'))!=member_id:
                issues.append(dict(reason='beam_steel_has_no_matching_disjoint_web',bar_id=r['id']));continue
            longitudinal.append(dict({key:r[key] for key in ('id','s_in','z_in','area_in2','db_in','fy_ksi','es_ksi')},strength_credited=not void_only))
            if void_only:excluded.append(dict(id=r['id'],reason='beam_owned_steel_void_retained_no_slab_resistance_credit'))
            else:selected_ids.append(r['id'])
        else:
            if s+radius<=a+1e-9 or s-radius>=b-1e-9:continue
            if s-radius<a-1e-9 or s+radius>b+1e-9:
                issues.append(dict(reason='transverse_circle_crosses_axial_accounting_boundary',bar_id=r['id']));continue
            start,end=max(lo,run[0]),min(hi,run[1])
            if end<=start:continue
            if r['operator_owner']!='slab' and not slab_only:
                issues.append(dict(reason='crossing_beam_joint_requires_separate_physical_section',bar_id=r['id']));continue
            key=(r['z_in'],r['area_in2'],r['db_in'],r['fy_ksi'],r['es_ksi'],start,end,void_only)
            groups.setdefault(key,[]).append(r['id'])
            if void_only:excluded.append(dict(id=r['id'],reason='beam_owned_steel_void_retained_no_slab_resistance_credit'))
            else:selected_ids.append(r['id'])
    transverse=[]
    for values,ids in groups.items():
        z,area,db,fy,es,start,end,void_only=values
        row=dict(id='transverse_row_'+_hash(sorted(ids))[:20],physical_bar_ids=sorted(ids),
            area_in2=area,db_in=db,fy_ksi=fy,es_ksi=es,z_in=z,accounting_length_in=b-a,
            run_interval_in=[start,end])
        if void_only:row['strength_run_interval_in']=None
        transverse.append(row)
    covered=set();failed=set();development_records=[];domains={}
    for e in development_evidence:
        erow=dict(e);span=erow.get('station_interval_in');ids=erow.get('physical_bar_ids',[])
        scoped=(bool(erow.get('evidence_id')) and erow.get('state_identity')==inventory['state_identity'] and
                  erow.get('material_inventory_id')==inventory['identity_sha256'] and
                  str(erow.get('floor'))==str(floor) and erow.get('axis')==axis and
                  np.asarray(erow.get('interval_in',[])).shape==(2,) and
                  np.allclose(erow['interval_in'],[lo,hi],rtol=0,atol=1e-9) and
                  np.asarray(erow.get('axial_accounting_interval_in',[])).shape==(2,) and
                  np.allclose(erow['axial_accounting_interval_in'],[a,b],rtol=0,atol=1e-9) and
                  isinstance(span,(list,tuple)) and len(span)==2 and float(span[0])<=station<=float(span[1]))
        accepted=scoped and erow.get('status')=='pass'
        if accepted:covered.update(ids)
        if scoped and erow.get('status')=='fail':failed.update(set(ids)&set(selected_ids))
        domain=erow.get('full_fy_development_domain',{})
        if scoped and apply_development_domains and domain.get('status')=='computed':
            dr=domain.get('interval_in');physical=domain.get('physical_straight_run_interval_in')
            if domain.get('basis')!='full_fy_both_signs_both_sides_no_25_4_10_reduction':
                raise ValueError('Actual full-fy development-domain basis required')
            if dr is not None and (np.asarray(dr).shape!=(2,) or not np.isfinite(dr).all() or dr[1]<=dr[0]):
                raise ValueError('Finite positive or explicitly empty developed domain required')
            if np.asarray(physical).shape!=(2,) or not np.isfinite(physical).all() or physical[1]<=physical[0]:
                raise ValueError('Development domain must retain actual installed straight run')
            if dr is not None and (dr[0]<physical[0]-1e-9 or dr[1]>physical[1]+1e-9):
                raise ValueError('Credited development domain exceeds actual physical straight run')
            for bid in set(ids)&set(selected_ids):
                if bid in domains and domains[bid]!=dr:raise ValueError('Conflicting actual bar development domains')
                domains[bid]=dr
            covered.update(set(ids)&set(selected_ids))
        development_records.append(dict(evidence=erow,accepted_for_station=bool(accepted),identity_and_station_scoped=bool(scoped)))
    zero_credit=[];partial_credit=[]
    if apply_development_domains:
        for bar in longitudinal:
            if bar['id'] in domains:
                dr=domains[bar['id']];bar['strength_credited']=dr is not None and dr[0]-1e-9<=station<=dr[1]+1e-9
                if not bar['strength_credited']:zero_credit.append(bar['id'])
        split=[]
        for row in transverse:
            grouped={}
            for bid in row['physical_bar_ids']:
                physical=row['run_interval_in'];dr=domains.get(bid,physical)
                if 'strength_run_interval_in' in row and row['strength_run_interval_in'] is None:dr=None
                credit=None if dr is None else (max(physical[0],dr[0]),min(physical[1],dr[1]))
                if credit is not None and credit[1]<=credit[0]:credit=None
                grouped.setdefault(credit,[]).append(bid)
                if credit is None:zero_credit.append(bid)
                elif list(credit)!=list(physical):partial_credit.append(bid)
            for credit,ids in grouped.items():
                split.append(dict(row,id='transverse_row_'+_hash(sorted(ids))[:20],physical_bar_ids=ids,
                                  strength_run_interval_in=None if credit is None else list(credit)))
        transverse=split
    missing=sorted(set(selected_ids)-covered)
    for bar in longitudinal:
        if bar.get('strength_credited') is False and bar['id'] not in domains:
            # Another owner has already removed this whole bar's strength;
            # its physical concrete void was checked above.
            continue
        if bar['id'] not in domains:
            window_issues.append(dict(reason='longitudinal_development_domain_missing_for_axial_window',bar_id=bar['id']))
            continue
        domain=domains[bar['id']]
        if domain is not None and min(b,domain[1])>max(a,domain[0])+1e-9 and (domain[0]>a+1e-9 or domain[1]<b-1e-9):
            window_issues.append(dict(reason='longitudinal_strength_credit_changes_inside_axial_window',bar_id=bar['id'],
                full_fy_domain_interval_in=list(domain)))
    arguments=dict(thickness_in=h,fc_ksi=strength_fc,ec_ksi=strength_ec,axis=axis,
        slab_interval_in=[lo,hi],web=web,reference_s_in=reference,longitudinal_bars=longitudinal,
        transverse_rows=transverse,phi_requirements=dict(phi_requirements),quadrature_order=quadrature_order)
    identity_record=dict(state_identity=inventory['state_identity'],material_inventory_id=inventory['identity_sha256'],
        floor=int(floor),axis=axis,station_in=station,member_id=member_id,interval_in=[lo,hi],
        axial_accounting_interval_in=[a,b],reference_s_in=reference,reference_z_in=0.,constructor=arguments,
        material_scope=material_scope)
    output=dict(status='unknown' if issues else 'inventory_constructed',section=None,constructor=arguments,
        identity=identity_record,identity_sha256=_hash(identity_record),issues=issues,
        physical_bar_ids=sorted(selected_ids),excluded=excluded,
        development=dict(status='fail' if failed-set(domains) else 'pass' if not missing else 'unknown',
                         failed_physical_bar_ids=sorted(failed),missing_physical_bar_ids=missing,evidence=development_records,
                         development_domains_applied=bool(apply_development_domains),
                         zero_strength_credit_ids=sorted(zero_credit),partial_strength_credit_ids=sorted(partial_credit),
                         installed_concrete_voids_preserved=True,
                         full_fy_only=bool(development_records) and all(r['evidence'].get('full_fy_only') is True for r in development_records),
                         actual_demand_failure_established=False),
        action_accounting=dict(integer_transverse_bar_count=sum(len(v['physical_bar_ids']) for v in transverse),
            axial_accounting_length_in=b-a,corresponding_action_average_verified=False),
        axial_material_uniformity=dict(status='unknown' if window_issues else 'pass',issues=window_issues,
            interval_in=[a,b],transverse_bar_homogenization_qualified=False),
        concrete_strength_basis=dict(actual_specified_grades=actual_grades,uniform_strength_fc_ksi=strength_fc,
            uniform_strength_ec_ksi=strength_ec,higher_grade_resistance_credit=False),
        complete_design_qualified=False)
    output['physical_detail_status']=inventory.get('physical_detail_status','unknown')
    output['unqualified_geometry_diagnostic']=inventory.get('unqualified_geometry_diagnostic',False)
    if not issues:
        try:output['section']=SpatialFloorSection(**arguments)
        except ValueError as exc:output.update(status='unknown',issues=[dict(reason=str(exc))])
    return output


class FrozenGravityReferenceBuilder:
    """Executable fixed-D builder from one coherent actual reference solution.

    No OpenSees/native query is made. row is an existing CoupledFloorSystem
    row; response carries its actual retained displacement vector. The result
    supplies exact cell inventory of four3x3 D matrices to root's generalized
    elastic assembly. This builder does not label a force sensitivity comparison
    or complete global gravity design qualified merely because it returned D.
    """
    def __init__(self,inventory,*,reference_identity,quadrature_order=12,
                 reference_branch_policy='virgin_cracking_history',prior_cracked_point_ids=(),
                 material_mapping='clipped_cylinders_v2'):
        if not inventory.get('state_identity') or not reference_identity:raise ValueError('Actual inventory/reference identities required')
        self.inventory=inventory;self.reference_identity=str(reference_identity);self.order=int(quadrature_order)
        if reference_branch_policy not in ('virgin_cracking_history','all_cracked_declared'):
            raise ValueError('Explicit virgin/history or precracked reference policy required')
        self.branch_policy=reference_branch_policy;self.cracked=set(prior_cracked_point_ids)
        from Design.SMRF_Bar_Pitch_Layers import METHOD as PITCH_METHOD
        if material_mapping not in ('clipped_cylinders_v2',PITCH_METHOD):raise ValueError('Explicit supported material mapping required')
        self.material_mapping=material_mapping;self.cache={};self._pitch_fields={}

    def cell_inventory(self,floor_row):
        floor=floor_row['assembly'];level=str(floor_row['description']['level'])
        key=(floor.id,self.inventory['identity_sha256'],self.material_mapping,tuple(floor.x),tuple(floor.y))
        if key in self.cache:return self.cache[key]
        bars=self.inventory['floors'][level]['bars'];cells={};total={}
        if self.material_mapping=='clipped_cylinders_v2':
            profile_for=lambda box:CellVoidProfile(box,floor.thickness,bars,state_identity=self.inventory['state_identity'])
            mapping_identity=None
        else:
            from Design.SMRF_Bar_Pitch_Layers import FixedPitchFloorMaterial
            if level not in self._pitch_fields:self._pitch_fields[level]=FixedPitchFloorMaterial(self.inventory,level)
            field=self._pitch_fields[level]
            if field.h!=floor.thickness:raise ValueError('Fixed material layer thickness differs from physical floor')
            profile_for=field.profile;mapping_identity=field.identity
        for row in floor.elements:
            if row['kind']!='slab':continue
            points=np.asarray(row['points']);box=(float(points[:,0].min()),float(points[:,0].max()),
                                                float(points[:,1].min()),float(points[:,1].max()))
            profile=profile_for(box)
            cells[row['id']]=profile
            for record in profile.records:
                entry=total.setdefault(record['id'],np.zeros(5))
                entry+=np.array([record[k] for k in ('straight_volume_in3','first_x_in4','first_y_in4','first_z_in4','second_z_in5')])
        whole=profile_for((floor.x[0],floor.x[-1],floor.y[0],floor.y[-1]))
        errors=[]
        for record in whole.records:
            actual=total.get(record['id'],np.zeros(5));target=np.array([record[k] for k in
                ('straight_volume_in3','first_x_in4','first_y_in4','first_z_in4','second_z_in5')])
            error=actual-target;passed=bool(np.all(abs(error)<=1e-7+1e-9*np.maximum(abs(actual),abs(target))))
            errors.append(dict(id=record['id'],residual=error.tolist(),passed=passed))
        audit=dict(method='actual_straight_material_cell_volume_first_moment_partition_v2',
            material_mapping=self.material_mapping,mapping_identity=mapping_identity,
            verified=all(v['passed'] for v in errors),bar_checks=errors,
            inventory_identity=self.inventory['identity_sha256'],whole_floor_profile=whole.identity,
            hook_stiffness_basis=self.inventory['hook_stiffness_basis'])
        self.cache[key]=(cells,audit);return cells,audit

    def build(self,floor_row,response,*,reference_loading):
        from Design.SMRF_Material_Floor import common_shell_strain_matrix
        if not reference_loading:raise ValueError('Coherent signed service/reference gravity loading identity required')
        floor=floor_row['assembly'];desc=floor_row['description'];u=np.asarray(response['retained_displacements'],dtype=float)
        if u.shape!=(len(floor.dof_ids),) or not np.isfinite(u).all():raise ValueError('Actual coherent reference floor displacement field required')
        paired='retained_displacements_lo' in response
        if paired:
            from Design.SMRF_Paired_Physical_Equilibrium import remove_rigid_pair
            from Design.SMRF_Paired_Arithmetic import paired_dense_action
            uh,ul,_=remove_rigid_pair(floor,u,np.asarray(response['retained_displacements_lo'],float))
        cells,ownership=self.cell_inventory(floor_row);matrices={};records=[];domain=True;quadrature=True
        for row in floor.elements:
            if row['kind']!='slab':continue
            profile=cells[row['id']];model=LayeredSlabSecant(floor.thickness,desc['fc_ksi'],profile.slab_steel_layers(),
                ec_ksi=floor.ec,poisson=floor.nu,quadrature_order=self.order,concrete_void_profile=profile)
            if paired:
                lh,ll=paired_dense_action(row['transform'],uh[row['indices']],ul[row['indices']])
                qs=np.array([sum(paired_dense_action(bg,lh,ll)) for bg in common_shell_strain_matrix(row)])
            else:
                local=row['transform']@u[row['indices']];qs=common_shell_strain_matrix(row)@local
            values=[];points=[]
            for gp,q in enumerate(qs):
                point_id=f'{floor.id}|{row["id"]}|gp{gp}';predictor_reasons=[]
                if self.branch_policy=='virgin_cracking_history' and point_id not in self.cracked:
                    elastic=model.evaluate(q,branch='uncracked_elastic')
                    predictor_reasons=list(elastic['domain_reasons'])
                    material_reasons=[v for v in predictor_reasons if v!='secant_quadrature_not_converged']
                    if not material_reasons:out=elastic
                    else:self.cracked.add(point_id);out=model.evaluate(q,branch='restricted_cracked')
                else:
                    self.cracked.add(point_id);out=model.evaluate(q,branch='restricted_cracked')
                d=np.array(out['zero_incremental_N_relaxed_D']);values.append(d)
                domain &=out['constitutive_state_verified'];quadrature &=out['quadrature']['passed']
                deig=np.linalg.eigvalsh(d);dscale=max(1.,float(max(abs(deig))))
                points.append(dict(reference_generalized_strain=q.tolist(),D=d.tolist(),
                    secant_method=out['method'],relaxed_bending_energy_recovery=out['relaxed_bending_energy_recovery'],
                    membrane_relaxation_recovery=out['membrane_relaxation_recovery'],
                    point_id=point_id,constitutive_branch=out['branch'],elastic_predictor_domain_reasons=predictor_reasons,
                    constitutive_state_verified=out['constitutive_state_verified'],domain_reasons=out['domain_reasons'],
                    rank=out['rank'],quadrature=out['quadrature'],nominal_resultants=out['nominal_resultants'],
                    bending_eigenvalues=deig.tolist(),bending_rank=int(np.sum(deig>1e-11*dscale)),
                    neutral_axis_relaxation=out['zero_incremental_N_membrane_response']))
            matrices[row['id']]=np.array(values)
            records.append(dict(cell_id=row['id'],void_profile_identity=profile.identity,
                exact_void_depth_moments=profile.exact_depth_moments.tolist(),
                slab_steel_ids=[v['id'] for v in profile.slab_steel_layers()],
                external_beam_void_ids=profile.void_only_ids,reference_points=points))
        audit=dict(method=METHOD,state_identity=self.inventory['state_identity'],reference_identity=self.reference_identity,
            material_mapping=self.material_mapping,material_mapping_engineering_qualification_inferred=False,
            reference_loading=reference_loading,reference_case_id=response.get('case_id'),
            paired_reference_displacements_used=paired,
            material_domain_valid=bool(domain),quadrature_converged=bool(quadrature),
            reference_branch_policy=self.branch_policy,cracked_point_ids=sorted(self.cracked),
            physical_material_ownership_verified=ownership['verified'],ownership=ownership,cells=records,
            stiffness_basis='ACI31819_6.3.1.1_R6.3.1.1_6.6.3.1.1_frozen_actual_mat_zero_incremental_N_full_D/'+self.material_mapping,
            gross_membrane_geometric_coupling_preserved_by_caller=True,
            final_combination_nonlinear_equilibrium_required=False,force_bound_claimed=False,
            targeted_coupled_sensitivity_verified=False,complete_design_qualified=False)
        return dict(slab_bending_matrices=matrices,audit=audit,
                    status='reference_constructed' if domain and quadrature and ownership['verified'] else 'unknown')

    def frozen_operator(self,floor_row,reference,*,beam_bending_factor,web_torsion_factor,
                        source_identity,maximum_coupled_dofs=2400):
        """Construct root's exact elastic assembly from a verified reference.

        Calling this performs a sparse membrane condensation; the caller owns
        its bounded structural protocol. This helper makes no OpenSees query.
        Member/column/torsion factors remain explicit source-based choices.
        """
        from Design.SMRF_Floor_Substructure import GeneralizedFloorStiffness
        audit=reference['audit']
        if (reference['status']!='reference_constructed' or audit['state_identity']!=self.inventory['state_identity']
                or audit['reference_identity']!=self.reference_identity or audit.get('material_mapping')!=self.material_mapping):
            raise ValueError('Verified matching actual frozen reference required')
        return GeneralizedFloorStiffness(floor_row['assembly'],beam_bending_factor=beam_bending_factor,
            slab_bending_factors=None,slab_bending_matrices=reference['slab_bending_matrices'],
            web_torsion_factor=web_torsion_factor,
            stiffness_basis=audit['stiffness_basis'],input_identity=source_identity,
            maximum_coupled_dofs=maximum_coupled_dofs)


def compare_frozen_references(first,second):
    """Record matrix sensitivity without inventing a universal force bound.

    A numerical stiffness change is not a design-action envelope. The caller
    still compares complete equilibrated floor responses under both models.
    """
    if first['audit']['state_identity']!=second['audit']['state_identity']:
        raise ValueError('Reference sensitivity must retain the same physical state')
    aa,bb=first['slab_bending_matrices'],second['slab_bending_matrices']
    if set(aa)!=set(bb):raise ValueError('Reference sensitivity requires identical physical cell inventory')
    records=[]
    for ident in sorted(aa):
        a,b=np.asarray(aa[ident],float),np.asarray(bb[ident],float)
        if a.shape!=(4,3,3) or b.shape!=a.shape:raise ValueError('Every physical cell requires four full D matrices')
        for gp,(da,db) in enumerate(zip(a,b)):
            scale=max(1.,float(np.linalg.norm(da,2)),float(np.linalg.norm(db,2)))
            records.append(dict(cell_id=ident,gp=gp,relative_spectral_change=float(np.linalg.norm(db-da,2)/scale),
                                first_eigenvalues=np.linalg.eigvalsh(da).tolist(),second_eigenvalues=np.linalg.eigvalsh(db).tolist()))
    return dict(method='same_inventory_frozen_full_D_reference_comparison',
        first_reference=first['audit']['reference_identity'],second_reference=second['audit']['reference_identity'],
        state_identity=first['audit']['state_identity'],points=records,
        maximum_relative_spectral_change=max((v['relative_spectral_change'] for v in records),default=0.),
        response_envelope_verified=False,force_bound_claimed=False)
