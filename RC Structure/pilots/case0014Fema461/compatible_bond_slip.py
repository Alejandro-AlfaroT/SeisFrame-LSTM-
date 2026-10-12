"""Compatible tension-layer strain penetration about a declared fixed compression pivot."""
import math
import numpy as np
from derive_composite_slip import derive

def calibrate(strain_at_probable=.02):
 composite=derive();branches={};es=29000.;fy=60.
 if not math.isfinite(strain_at_probable) or strain_at_probable<=fy/es:raise ValueError('strain_at_probable must exceed elastic yield strain')
 hard=15/(strain_at_probable-fy/es)
 for sign,v in composite['branches'].items():
  c=v['neutral_axis_from_compression_face_in'];layers=[dict(l,lever=l['y']-c) for l in v['per_layer'] if l['tension_at_first_yield']]
  def stress(s,l):
   ub=12*math.sqrt(l['anchorage_fc']*1000)/1000;up=ub/2;sy=(fy/es)*fy*l['db']/(8*ub)
   if s<=sy:return math.sqrt(max(0,s)*8*ub*es/l['db'])
   a=l['db']/(8*up*hard);b=l['db']*(2*fy/es)/(8*up)
   return fy+(-b+math.sqrt(b*b+4*a*(s-sy)))/(2*a)
  def moment(theta):return sum(l['area']*stress(theta*l['lever'],l)*l['lever'] for l in layers)
  ty=min(l['slip_at_own_bar_yield_in']/l['lever'] for l in layers)
  def target_rotation(l):
   ub=12*math.sqrt(l['anchorage_fc']*1000)/1000;up=ub/2
   return (l['slip_at_own_bar_yield_in']+(strain_at_probable+fy/es)*(.25*fy*l['db']/(4*up))/2)/l['lever']
  tc=min(target_rotation(l) for l in layers)
  angles=np.linspace(0,tc,1001);moments=np.array([moment(x) for x in angles]);my=moment(ty);mc=moment(tc)
  # A compression contact force at the fixed pivot balances total bar tension.
  checks=[]
  for theta in angles:
   tension=sum(l['area']*stress(theta*l['lever'],l) for l in layers);contact=-tension
   checks.append(abs(contact+tension))
  branches[sign]={'pivot_from_compression_face_in':c,'yield_rotation':ty,'yield_moment':my,'cap_rotation':tc,'cap_moment':mc,'secant_ke':my/ty,'angles':angles.tolist(),'moments':moments.tolist(),'layers':layers,'maximum_axial_equilibrium_residual':max(checks),'contact':'rigid compression contact at fixed cracked-section neutral-axis pivot; compression steel omitted from interface'}
 # One Ke required by IMK: mean flexibility of branch secants, no orientation dependence.
 ke=2/sum(1/v['secant_ke'] for v in branches.values())
 for v in branches.values():
  v['installed_yield_rotation']=v['yield_moment']/ke;v['dp']=v['cap_rotation']-v['installed_yield_rotation'];assert v['dp']>0
  x=np.array(v['angles']);my=v['yield_moment'];mc=v['cap_moment'];y=np.where(x<=my/ke,ke*x,my+(mc-my)*(x-my/ke)/v['dp'])
  v['imk_envelope_normalized_rmse']=float(np.sqrt(np.mean((y-np.array(v['moments']))**2))/mc)
 return {'status':'literature_based_diagnostic_with_declared_cyclic_assumptions','strain_at_probable':strain_at_probable,'probable_bar_stress_ksi':75.,'ke':ke,'branches':branches,'composite':composite,'kappaF':.25,'kappaD':.50,'slip_deterioration':'suppressed; no cyclic bond deterioration calibration','assumptions':['Fixed compression-contact pivot, not evolving contact/neutral axis.','Common slip rotation imposed on all tension groups; bar stresses solved from their individual bond laws.','Developed straight-segment approximation for hooked bars; hook-bearing and pullout excluded.','Compression contact is rigid and compression-bar slip omitted.',f'Postyield steel reaches 75 ksi at {100*strain_at_probable:g} percent strain; declared diagnostic sensitivity.','Harmonic branch secant stiffness approximates nonlinear subyield slip envelope.','Pinching .25/.50 declared; not fit to measured cycles.','Slip post-cap extension is a numerical sensitivity extrapolation, not anchorage failure prediction.']}
