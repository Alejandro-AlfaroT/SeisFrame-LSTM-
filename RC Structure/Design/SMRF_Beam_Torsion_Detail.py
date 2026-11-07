"""Actual support development and splice-zone transverse torsion inventory."""
from __future__ import annotations
from dataclasses import replace
import Structure_Parameters as sp
from Model import Member_Properties as mp
from Design.SMRF_Splice_Physical import group_beam_lanes,verify_beam_mechanical_splice,beam_cage_geometry
from Design.SMRF_Actual_Bar_Development import actual_bar_development
from Design.SMRF_Physical_Floor_Cuts import bar_clear_interval
from Design.SMRF_Coupled_Floor_Design import _hash

METHOD='actual_beam_torsion_full_span_and_splice_detail_v1'


def transverse_variants(design,splice_detail,*,station_from_left_face_in,clear_span_in):
    """Installed local detail, including both cages at a splice transition.

The modified spacing is NOT replaced by the baseline spacing merely because
the splice preserves total Av/s: outer torsion-leg Av/s may be smaller.
The verified splice construction includes one baseline cage on each side.
"""
    base=dict(kind='baseline',bar_size=design.stirrup_bar_size,legs=design.stirrup_legs,
        spacing_in=design.stirrup_spacing_in)
    if not 0<=station_from_left_face_in<=clear_span_in:raise ValueError('Torsion station outside actual clear span')
    if splice_detail.get('status')!='pass':return dict(status=splice_detail.get('status','not_evaluated'),variants=[],reason='Actual splice geometry is not passing')
    local=splice_detail.get('transverse_detail');positions=splice_detail.get('hoop_stations_in')
    if not local or not positions:return dict(status='not_evaluated',variants=[],reason='Complete actual modified hoop zone required')
    lo,hi=positions[0],positions[-1];s=design.stirrup_spacing_in;x=station_from_left_face_in
    changed={k:local[k] for k in ('bar_size','legs','spacing_in')};changed['kind']='mechanical_splice'
    if x<lo-s or x>hi+s:variants=[base]
    elif lo<x<hi:variants=[changed]
    else:variants=[base,changed]
    return dict(status='pass',variants=variants,splice_zone_in=local['zone_in'],splice_hoop_stations_in=positions,
        basis='actual closed perimeter hoops; baseline/local transition checked with both cages')


def member_detail(state,inventory,member,*,slab_fc_ksi):
    """Reconstruct full-span bars, both actual support hooks, and local hoops.

Torsion needs tension anchorage at the support faces. The full-fy compression
domain used by the seven-action material kernel remains a separate check.
"""
    tag=str(member.member_tag);design=member.design;axis=member.member_type[-1];floor=member.story_or_floor
    geometry=inventory['beam_geometry'][tag];bars=inventory['beam_bars'][tag];ids=[b['id'] for b in bars]
    lanes=group_beam_lanes(state)[member.group_id]
    clear=mp.beam_clear_span_in(member);fi,fj=mp.beam_face_distances_in(member)
    start=(member.grid_i if axis=='x' else member.grid_j)*(sp.BAY_X if axis=='x' else sp.BAY_Y)
    ends=[start+fi,start+mp.beam_span_in(member)-fj];fp=inventory['floors'][str(floor)]['footprint']
    families=[dict(db_in=b['db_in'],global_transverse_coordinates_in=[b['s_in']]) for b in inventory['floors'][str(floor)]['bars'] if b['axis']==axis]
    cross='y' if axis=='x' else 'x';line=geometry['line_in']
    interval=bar_clear_interval([line-design.b_in/2,line+design.b_in/2],[fp[cross+'_min_in'],fp[cross+'_max_in']],families)
    specs=[dict(cut_id=f'beam_{tag}/torsion_support_{i}',floor=floor,axis=axis,station_in=x,interval_in=interval,
                axial_accounting_interval_in=[max(fp[axis+'_min_in'],x-1.),min(fp[axis+'_max_in'],x+1.)],
                physical_bar_ids=ids,member_id=tag) for i,x in enumerate(ends)]
    development=actual_bar_development(state,inventory,specs,slab_fc_ksi=slab_fc_ksi)
    anchors=[]
    for row in development['cuts']:
        supplied={e['physical_bar_ids'][0]:e for e in row['evidence']}
        for ident in ids:
            e=supplied.get(ident);checks=[] if e is None else e['checks']
            passed=bool(checks) and all(c.get('physical_detail_status')=='pass' and
                len(c.get('sides',[]))==2 and all(s.get('tension_status')=='pass' for s in c['sides']) for c in checks)
            anchors.append(dict(cut_id=row['specification']['cut_id'],bar_id=ident,tension_developed=passed,
                                evidence_id=None if e is None else e['evidence_id']))
    fullspan=bool(bars) and all(b['straight_run_interval_in'][0]<=ends[0]+1e-8 and b['straight_run_interval_in'][1]>=ends[1]-1e-8 for b in bars)
    splice=verify_beam_mechanical_splice(design,lanes['coordinates'],clear_span_in=clear,spacing_in=design.stirrup_spacing_in,
        fy_ksi=sp.FY_KSI,specification=sp.REINFORCEMENT_SPECIFICATION,clear_cover_in=sp.BEAM_CLEAR_COVER_IN,aggregate_in=sp.AGGREGATE_MAX_SIZE_IN)
    cage=beam_cage_geometry(design,lanes['coordinates'],sp.BEAM_CLEAR_COVER_IN)
    detail=dict(status='pass' if fullspan and anchors and all(a['tension_developed'] for a in anchors) and splice['status']=='pass' and cage['passes'] else 'not_evaluated',
        source_check_ids=[a['evidence_id'] for a in anchors if a['evidence_id']],longitudinal_bar_ids=ids,
        longitudinal_extends_required_zone=fullspan,transverse_extends_required_zone=bool(cage['passes'] and splice['status']=='pass'),
        both_supports_developed=bool(anchors and all(a['tension_developed'] for a in anchors)),
        required_zone_basis='uncut full-clear-span reinforcement; tension development at both support faces; no interior cutoff',
        support_faces_in=ends,clear_span_in=clear,first_hoop_from_support_in=2.,support_anchorage=anchors)
    result=dict(method=METHOD,state_identity=state.identity(),material_inventory_id=inventory['identity_sha256'],member_tag=member.member_tag,
        coordinates=[{**p,'id':f'member_{member.member_tag}/beam/{p["id"]}'} for p in lanes['coordinates']],
        development=detail,raw_development=development,splice_detail=splice)
    result['evidence_id']=_hash(result);return result


def variant_design(design,variant):
    return replace(design,stirrup_bar_size=variant['bar_size'],stirrup_legs=variant['legs'],stirrup_spacing_in=variant['spacing_in'])
