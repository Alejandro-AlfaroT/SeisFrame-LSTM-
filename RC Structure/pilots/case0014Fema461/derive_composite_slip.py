"""Composite cracked-section and per-layer bond-slip targets. Diagnostic only."""
from pathlib import Path
import json,math,argparse
from scipy.optimize import brentq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INPUT=Path(__file__).with_name('inputs')

def derive():
 d=json.loads((INPUT/'bond_slip_design.json').read_text());f=d['resolved_family'];r=d['reinforcement'];h=d['sections']['h_beam_in'];bw=d['sections']['b_beam_in'];bf=f['effective_flange_width_in'];t=d['slab']['thickness_in'];fc=d['sections']['fc_beam_ksi'];ec=57*math.sqrt(fc*1000);es=d['materials']['es_ksi'];fy=d['materials']['fy_ksi'];n=es/ec
 layers=[{'name':'beam_top','y':r['beam_bar_stacking']['offsets_in']['y']['top'],'area':r['beam_top_bars']*r['beam_bar_area_in2'],'db':1.,'anchorage_fc':d['sections']['fc_col_ksi'],'embedment':d['anchorage']['directions']['y']['embedment_available_in']},{'name':'beam_bottom','y':h-r['beam_bar_stacking']['offsets_in']['y']['bottom'],'area':r['beam_bot_bars']*r['beam_bar_area_in2'],'db':1.,'anchorage_fc':d['sections']['fc_col_ksi'],'embedment':d['anchorage']['directions']['y']['embedment_available_in']}]
 for layer in f['slab_layers_from_top']:layers.append({'name':'slab_'+layer['face'],'y':layer['depth_in'],'area':layer['area_in2'],'db':.5,'anchorage_fc':fc,'embedment':f['exterior_anchorage']['embedment_available_in']})
 results={}
 for sign in ('sagging','hogging'):
  ll=[dict(layer,y=layer['y'] if sign=='sagging' else h-layer['y']) for layer in layers]
  def integrals(c):
   edges=sorted(set([0.,c]+[x for x in [t if sign=='sagging' else h-t] if 0<x<c]));F=0.;I=0.
   for a,b in zip(edges[:-1],edges[1:]):
    middle=(a+b)/2;width=bf if (middle<t if sign=='sagging' else middle>h-t) else bw
    F+=width*(c*(b-a)-(b*b-a*a)/2)
    I+=width*((c-a)**3-(c-b)**3)/3
   return F,I
  c=brentq(lambda c:integrals(c)[0]+n*sum(l['area']*(c-l['y']) for l in ll),1e-6,h-1e-6)
  inertia=integrals(c)[1]+n*sum(l['area']*(l['y']-c)**2 for l in ll);EI=ec*inertia
  maxdist=max(l['y']-c for l in ll);curv=fy/es/maxdist;moment=EI*curv
  per=[]
  for l in ll:
   stress=es*curv*(l['y']-c);item={**l,'stress_at_first_section_yield_ksi':stress,'tension_at_first_yield':stress>0}
   if stress>0:
    ub=12*math.sqrt(l['anchorage_fc']*1000)/1000;up=ub/2;ld=stress*l['db']/(4*ub);slip=(stress/es)*ld/2;lever=l['y']-c
    sy=(fy/es)*fy*l['db']/(8*ub);ld_y=fy*l['db']/(4*ub);ldpost=.25*fy*l['db']/(4*up);su=sy+(.02+fy/es)*ldpost/2
    item.update(slip_at_first_section_yield_in=slip,rotation_equivalent_at_first_section_yield_rad=slip/lever,slip_at_own_bar_yield_in=sy,rotation_at_own_bar_yield_rad=sy/lever,rotation_at_1_25fy_assuming_2pct_strain_rad=su/lever,developed_length_first_section_yield_in=ld,developed_length_1_25fy_in=ld_y+ldpost,within_embedment_1_25fy=ld_y+ldpost<=l['embedment'])
   per.append(item)
  rotations=[x['rotation_equivalent_at_first_section_yield_rad'] for x in per if x['tension_at_first_yield']]
  results[sign]={'neutral_axis_from_compression_face_in':c,'cracked_inertia_in4':inertia,'EI_kip_in2':EI,'curvature_first_tension_bar_yield_per_in':curv,'first_tension_bar_yield_moment_kip_in':moment,'compression_face_stress_at_first_yield_ksi':ec*curv*c,'compression_linear_assumption_below_fc':ec*curv*c<fc,'controlling_tension_bar':max(ll,key=lambda l:l['y'])['name'],'per_layer':per,'layer_rotation_spread_rad':[min(rotations),max(rotations)],'common_slip_rotation_compatibility':'Independent bar predictions differ; a single series rotation is not fitted yet.'}
 fixture=json.loads((INPUT/'fixture.json').read_text());original_ke=fixture['members']['beam']['ke'];nfac=json.loads((INPUT/'global_parameters.json').read_text())['imk_hinge_stiffness_factor'];L=fixture['members']['beam']['source_element']['length_in'];L=float(L)
 return {'status':'composite_literature_targets_not_connection_calibration','reproduced_saved_strengths':{'sagging':f['mn_positive_kip_in'],'hogging':f['mn_negative_kip_in']},'slab_anchorage':f['exterior_anchorage'],'branches':results,'flexure_stiffness_candidates':{s:6*nfac*v['EI_kip_in2']/L for s,v in results.items()},'original_aggregate_ke':original_ke,'limitations':['Cracked elastic section calculation is a first-yield diagnostic, not a postyield section model.','Slab and beam bars have separate anchorages and independent predicted slips; enforce section axial/moment equilibrium and displacement compatibility before condensation.','Developed straight-segment strain penetration excludes cyclic hook bearing, pullout, splitting and concrete deterioration.','Postyield bar strain 0.02 remains declared, not fitted.','Two directional cracked flexural stiffnesses do not uniquely specify the one-Ke IMK flexure spring.','Original cyclic energy fit includes aggregate behavior; cannot identify flexure-only deterioration from geometry alone.','No pipeline installation or experimental calibration.']}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);result=derive();(root/'composite_targets.json').write_text(json.dumps(result,indent=2))
 fig,axs=plt.subplots(1,2,figsize=(12,5))
 for ax,(sign,v) in zip(axs,result['branches'].items()):
  layers=[l for l in v['per_layer'] if l['tension_at_first_yield']];ax.bar([l['name'] for l in layers],[l['rotation_equivalent_at_first_section_yield_rad']*1000 for l in layers]);ax.set(title=sign,xlabel='Tension layer at first section yield',ylabel='Independent slip rotation estimate (mrad)');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Case 0014 — beam and slab anchorages need compatibility');fig.tight_layout();fig.savefig(root/'layer_slip_targets.png',dpi=160);plt.close(fig)
 print(json.dumps({s:{k:v[k] for k in ['controlling_tension_bar','first_tension_bar_yield_moment_kip_in','layer_rotation_spread_rad']} for s,v in result['branches'].items()},indent=2))
