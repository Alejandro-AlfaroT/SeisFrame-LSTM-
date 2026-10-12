"""Case-specific developed-bar strain penetration targets; diagnostic only."""
from pathlib import Path
import math,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parent

def derive():
 d=json.loads((ROOT/'inputs/bond_slip_design.json').read_text());sec=d['sections'];r=d['reinforcement'];mat=d['materials']
 fy=mat['fy_ksi'];es=mat.get('es_ksi',29000.);fc=sec['fc_col_ksi'];ec=57*math.sqrt(sec['fc_beam_ksi']*1000)
 db=r['beam_bar_diameter_in'];ub=12*math.sqrt(fc*1000)/1000;ubpost=6*math.sqrt(fc*1000)/1000;ey=fy/es
 available=d['anchorage']['directions']['y']['embedment_available_in'];out={}
 for face,opposite,count,compcount in [('top','bottom',r['beam_top_bars'],r['beam_bot_bars']),('bottom','top',r['beam_bot_bars'],r['beam_top_bars'])]:
  offsets=r['beam_bar_stacking']['offsets_in']['y'];depth=sec['h_beam_in']-offsets[face];dp=offsets[opposite];As=count*r['beam_bar_area_in2'];Asc=compcount*r['beam_bar_area_in2'];b=sec['b_beam_in'];n=es/ec
  rho=As/(b*depth);rhop=Asc/(b*depth);a=n*(rho+rhop);c=(math.sqrt(a*a+2*n*(rho+rhop*dp/depth))-a)*depth
  lever=depth-c;curv=ey/lever;cs=es*curv*(c-dp);C=.5*b*ec*curv*c*c
  # Moment about tension-steel level, with explicit compression-steel force.
  moment_y=C*(depth-c/3)+Asc*cs*(depth-dp)
  ld=fy*db/(4*ub);sy=ey*ld/2;probable=1.25*fy;eps_probable=.02;ldpost=(probable-fy)*db/(4*ubpost);su=sy+(eps_probable+ey)*ldpost/2
  assert c>0 and lever>0 and moment_y>0
  assert ld<available,'Yield development exceeds saved straight embedment; hook law needed'
  out['hogging' if face=='top' else 'sagging']={'tension_face':face,'d_in':depth,'compression_bar_depth_in':dp,'neutral_axis_in':c,'lever_in':lever,'first_bar_yield_moment_kip_in':moment_y,'compression_steel_stress_at_tension_yield_ksi':cs,'slip_at_yield_in':sy,'rotation_at_yield_rad':sy/lever,'slip_at_probable_in':su,'rotation_at_probable_rad':su/lever,'developed_length_yield_in':ld,'developed_length_probable_in':ld+ldpost,'developed_length_within_saved_embedment':ld+ldpost<=available,'elastic_slip_secant_kip_in_per_rad':moment_y/(sy/lever)}
 return {'status':'literature_based_prediction_not_experimental_calibration','source':'Sezen and Setzler 2008, DOI 10.14359/19787; implemented stepped bond formulation already in Analysis/Joint_Slip_Subassembly.py','anchorage':d['anchorage']['directions']['y'],'anchorage_fc_ksi':fc,'bar_diameter_in':db,'fy_ksi':fy,'es_ksi':es,'elastic_bond_stress_ksi':ub,'postyield_bond_stress_ksi':ubpost,'probable_steel_stress_ksi':probable,'assumed_steel_strain_at_probable':eps_probable,'branches':out,'limitations':['Developed-bar strain penetration approximation; 90-degree hook bearing/slip is not independently calibrated.','Neutral axis and first-bar-yield moment use a cracked rectangular beam section without slab bars or flange; saved aggregate slab-inclusive My is not substituted for first bar yield.','Postyield conversion retains yield neutral axis and assumes steel strain 0.02 at 1.25fy; sensitivity parameter, not specimen data.','No experimental cyclic pinching, bond-deterioration or anchorage-failure calibration supplied.','Elastic flexure/slip separation of original aggregate stiffness is not independently identified.']}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise FileExistsError(a.output)
 result=derive();a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
