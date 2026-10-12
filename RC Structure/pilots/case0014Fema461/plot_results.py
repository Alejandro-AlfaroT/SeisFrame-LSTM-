from pathlib import Path
import json,shutil,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
ROOT=parser.parse_args().output.resolve();OUT=ROOT/'results'
read=lambda p:json.loads(p.read_text())
meta=read(OUT/'test_definition.json');checks=meta['checks'];members=meta['members'];H=meta['height_in'];L=2*meta['beam_half_span_in']
def load(name):
    a=np.load(OUT/(name+'.npz'));return {str(k):a['data'][:,i] for i,k in enumerate(a['columns'])}
A=load('fema461_installed');C=load('fema461_no_cyclic_deterioration');MP=load('monotonic_positive');MN=load('monotonic_negative')
ar=read(OUT/'fema461_installed.json');cr=read(OUT/'fema461_no_cyclic_deterioration.json')
BLUE='#126d93';RED='#c23d54';INK='#17334d';GOLD='#ba6b08';GREEN='#337d60'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold'})
def style(ax):ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
def save(fig,name):fig.savefig(ROOT/name,dpi=170,facecolor='white');plt.close(fig)
def ratio_of_yield(name,theta):
    v=members[name];s=v['source_spring'];fy=np.where(theta>=0,float(s['yield_moment_positive_kip_in']),float(s['yield_moment_negative_kip_in']))
    return abs(theta)/(fy/v['ke'])

fig,(ax,bx)=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1,1.2]})
ax.plot([0,0],[-H/2,H/2],color=INK,lw=7);ax.plot([-L/2,0],[0,0],color=BLUE,lw=6)
ax.add_patch(Rectangle((-13,-11),26,22,fc='#e4edf2',ec=INK,lw=1,zorder=3))
ax.scatter([0],[0],s=45,color=INK,zorder=5)
ax.add_patch(Polygon([(-7,-H/2-10),(7,-H/2-10),(0,-H/2)],fc='white',ec=INK))
ax.add_patch(Polygon([(-L/2-7,-10),(-L/2+7,-10),(-L/2,0)],fc='white',ec=INK));ax.plot([-L/2-9,-L/2+9],[-13,-13],color=INK,lw=1)
ax.annotate('',xy=(-30,H/2),xytext=(30,H/2),arrowprops={'arrowstyle':'<->','color':RED,'lw':2})
ax.text(0,H/2+8,'Imposed horizontal displacement',ha='center',color=RED,fontsize=10)
ax.annotate('',xy=(0,H/2+1),xytext=(0,H/2+30),arrowprops={'arrowstyle':'->','color':GOLD,'lw':2})
ax.text(8,H/2+22,f'P = {meta["column_top_axial_load_kip"]:.1f} kip',color=GOLD,fontsize=10)
ax.text(-L/2-10,16,'Beam 165, end j\n20 × 22 in; 5 ksi\n5 #8 top + 5 #8 bottom',color=BLUE,fontsize=10)
ax.text(18,H/4,'Column 37, end i\n26 × 26 in; 8 ksi\n12 #8; #5 @ 4 in',fontsize=10)
ax.text(18,-H/4,'Column 16, end j\n26 × 26 in; 8 ksi',fontsize=10)
ax.text(-L/2,-34,f'Half beam: {L/2:g} in\nFar end: vertical roller',fontsize=10)
ax.text(16,-H/2-7,'Bottom pin',fontsize=10)
ax.text(-L/2,-H/2-28,f'Column halves: {H/2:g} in each\nJoint vertical preload: {meta["joint_vertical_load_kip"]:.1f} kip',fontsize=10)
ax.set(xlim=(-L/2-20,122),ylim=(-H/2-42,H/2+40));ax.set_aspect('equal');ax.axis('off');ax.set_title('Exterior connection at floor 1 · Y–Z plane',pad=12)
amps=meta['protocol']['amplitude_drift_ratios'];targets=[0.]+[s*a*100 for a in amps for cycle in range(2) for s in [1,-1,0]];xx=np.arange(len(targets))/3
bx.plot(xx,targets,color=BLUE,lw=1);bx.axhline(3,color=GOLD,ls='--',lw=1);bx.axhline(-3,color=GOLD,ls='--',lw=1);bx.axvline(meta['protocol']['base_steps']*2,color=GOLD,ls=':',label='End of 3% reference sequence')
bx.set(xlabel='Loading cycle count (not time)',ylabel='Imposed fixture drift (%)',title='FEMA 461 displacement history');bx.legend(fontsize=8,loc='upper left');style(bx)
fig.suptitle('Case 0014 · isolated beam–column connection diagnostic',fontsize=16,weight='bold',color=INK)
fig.text(.5,.02,'Rigid centerline core; member flexure hinges. Shaded joint is illustrative, not a deformable panel. Columns include P–Δ.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.07,1,.92));save(fig,'setup_protocol.png')

fig,ax=plt.subplots(figsize=(12,6))
ax.plot(A['beam_rotation_rad'],A['beam_moment_kip_in']/12,color=BLUE,lw=.75,label='FEMA cycles · installed parameters')
for i,mono in enumerate([MP,MN]):ax.plot(mono['beam_rotation_rad'],mono['beam_moment_kip_in']/12,color=INK,lw=1.4,ls='--',label='Fresh monotonic loading' if i==0 else None)
ax.set(xlabel='Beam spring rotation (rad)',ylabel='Conjugate beam moment (kip-ft)',title='Beam 165 · exterior end j');ax.legend(fontsize=9,loc='lower right');style(ax)
fig.tight_layout();save(fig,'moment_rotation.png')

fig,bx=plt.subplots(figsize=(10,5))
for k,color,label in [('column_below',GOLD,'Column 16 · upper end'),('column_above',GREEN,'Column 37 · lower end')]:
    bx.plot(A[k+'_rotation_rad'],A[k+'_moment_kip_in']/12,color=color,lw=1.2,label=label)
bx.set(xlabel='Column spring rotation (rad)',ylabel='Conjugate column moment (kip-ft)',title='Column hinges stay below first yield');bx.ticklabel_format(axis='x',style='sci',scilimits=(0,0));bx.legend(fontsize=9);style(bx)
fig.tight_layout();save(fig,'column_rotation_checks.png')

fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.8))
ax.plot(C['beam_rotation_rad'],C['beam_moment_kip_in']/12,color='#aebbc5',lw=.6,label='Cyclic deterioration suppressed')
ax.plot(A['beam_rotation_rad'],A['beam_moment_kip_in']/12,color=BLUE,lw=.7,label='Installed deterioration')
ax.set(xlabel='Beam spring rotation (rad)',ylabel='Beam moment (kip-ft)',title='Same backbone, different cyclic deterioration');style(ax);ax.legend(fontsize=9)
ret=[]
for a,c in zip(ar['endpoints'],cr['endpoints']):
    if a['cycle']!=2 or abs(a['target_drift_ratio'])<1e-12:continue
    sign='Hogging' if a['target_drift_ratio']>0 else 'Sagging'
    ratio=abs(a['beam_moment_kip_in']/c['beam_moment_kip_in']) if c['beam_moment_kip_in'] else None
    ret.append({'level':a['level'],'cycle':2,'amplitude_drift_percent':abs(a['target_drift_ratio'])*100,'direction':sign,'installed_moment_kip_ft':abs(a['beam_moment_kip_in'])/12,'control_moment_kip_ft':abs(c['beam_moment_kip_in'])/12,'retained_fraction':ratio})
for direction,color in [('Hogging',BLUE),('Sagging',RED)]:
    r=[q for q in ret if q['direction']==direction];bx.plot([q['amplitude_drift_percent'] for q in r],[q['retained_fraction']*100 for q in r],'o-',ms=3,color=color,label=direction)
bx.set(xlabel='Imposed fixture drift amplitude (%)',ylabel='Moment / no-deterioration control (%)',title='Strength retained at second-cycle endpoints',ylim=(-3,110));style(bx);bx.legend(fontsize=9)
fig.suptitle('Cyclic strength loss is active in the current IMK model',fontsize=16,weight='bold',color=INK)
fig.text(.5,.02,'The control retains backbone softening and P–Δ; only energy-driven cyclic deterioration is suppressed. No parameters were fitted.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.07,1,.92));save(fig,'deterioration_comparison.png')

fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.8))
ax.plot(C['drift_ratio']*100,C['actuator_force_kip'],color='#b4c0c8',lw=.7,label='No cyclic deterioration')
ax.plot(A['drift_ratio']*100,A['actuator_force_kip'],color=BLUE,lw=.7,label='Installed deterioration');ax.axhline(0,color=INK,lw=.5)
ax.set(xlabel='Fixture drift (%)',ylabel='Actuator force (kip)',title='Connection force–displacement response');style(ax);ax.legend(fontsize=9)
for k,color,label in [('beam',BLUE,'Beam'),('column_below',GOLD,'Lower column'),('column_above',GREEN,'Upper column')]:
    bx.plot(A['step'],A[k+'_rotation_rad'],color=color,lw=.8,label=label)
bx.set(xlabel='Committed displacement substep (not seconds)',ylabel='Spring rotation (rad)',title='Where the nonlinear rotation accumulates');style(bx);bx.legend(fontsize=9)
fig.suptitle('Global fixture response and local rotation histories',fontsize=16,weight='bold',color=INK)
fig.text(.5,.02,'Negative resistance at positive drift reflects P–Δ after severe spring degradation; displacement control can continue tracing this unstable branch.',ha='center',fontsize=9.5)
fig.tight_layout(rect=(0,.07,1,.92));save(fig,'fixture_response.png')

ratio={k:float(max(ratio_of_yield(k,A[k+'_rotation_rad']))) for k in members}
severe=next((q for q in ret if q['retained_fraction']<.01),None)
summary={'case':'case_0014','connection':'first-floor exterior Y-Z connection, source joint 37','completed':True,'loading_levels':len(amps),'cycles':2*len(amps),'reference_drift_percent':3,'maximum_drift_percent':8.4,'column_hinge_maximum_yield_rotation_ratios':{k:v for k,v in ratio.items() if k!='beam'},'peak_beam_rotation_rad':float(max(abs(A['beam_rotation_rad']))),'first_second_cycle_endpoint_below_one_percent_control':severe,'strength_retention':ret,'checks':checks,'interpretation':'Numerical behavior of the existing V2 member hinges in an idealized fixture. Not experimental calibration, joint qualification, or a prediction of physical connection failure.'}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'cycle_strength.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ret[0]));w.writeheader();w.writerows(ret)
table=''.join(f'<tr><td>{q["amplitude_drift_percent"]:.3f}</td><td>{q["direction"]}</td><td>{q["installed_moment_kip_ft"]:.1f}</td><td>{q["control_moment_kip_ft"]:.1f}</td><td>{q["retained_fraction"]*100:.1f}%</td></tr>' for q in ret if q['amplitude_drift_percent']>=2.99)
css='body{background:#f1f5f8;color:#17334d;font:16px/1.55 system-ui;margin:0}main{max-width:1320px;margin:auto;padding:28px}section,header,footer{background:white;padding:24px;border-radius:12px;margin:20px 0}h1{font-size:35px;line-height:1.2}h2{font-size:24px}img{width:100%;height:auto}a{color:#08739a}.note{padding:15px;border-left:4px solid #ba6b08;background:#fff6e6}.stats{display:flex;gap:16px;flex-wrap:wrap}.stats div{padding:16px;background:#eaf2f7;border-radius:8px}.stats b{display:block;font-size:25px}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #dbe4eb;text-align:left}small{color:#587080}.scroll{overflow:auto}code{overflow-wrap:anywhere}@media(max-width:700px){main{padding:12px}section,header,footer{padding:16px}h1{font-size:28px}}'
picture=lambda file,alt:f'<a href="{file}"><img src="{file}" alt="{alt}"></a>'
html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case 0014 | FEMA 461 connection test</title><style>{css}</style><main>
<header>Frozen case 0014 diagnostic<h1>Case 0014 — reversed-cyclic connection test</h1><p>An isolated first-floor exterior beam–column fixture, using the <b>saved V2 IMKPeakOriented parameters</b> from the returned run. The beam hinge takes the nonlinear deformation and shows clear cyclic deterioration; both column hinges remain below first yield.</p><div class="stats"><div><b>{2*len(amps)} cycles</b>two at each amplitude</div><div><b>3% drift</b>FEMA reference target</div><div><b>8.4% drift</b>declared diagnostic extension</div><div><b>5 runs</b>cycles, controls, step check</div></div><p class="note">This tests the behavior of the existing numerical model. The rigid joint core cannot show joint shear damage, and the member hinges do not separately resolve bond-slip failure, bar buckling, fracture or concrete damage. This is not an experimental calibration or a connection acceptance test.</p></header>
<section><h2>Fixture and loading</h2>{picture('setup_protocol.png','Connection test fixture and imposed FEMA displacement cycles')}<p>Source joint <b>37</b> at (19.5, 50, 13.5) ft: beam <b>165, end j</b>, column <b>16, end j</b> below, and column <b>37, end i</b> above. The source building has four 13.5-ft stories. The test cuts each member at an assumed midpoint inflection point: 150-in beam arm, 81-in column arms.</p><p>The history follows <a href="https://www.atcouncil.org/pdfs/FEMA461.pdf">FEMA 461 §2.9.1, pp. 21–24</a>: two full reversals per amplitude, 1.4 growth to the reference amplitude, and extensions of 0.3 times that reference. Here 13 levels run from {amps[0]*100:.4f}% to 3%, followed by 3.9%, 4.8%, 5.7%, 6.6%, 7.5% and 8.4%. The 8.4% limit is a declared numerical exploration limit, not a FEMA acceptance threshold. More than six cycles occur before first modeled yield.</p><p>Column preloads: {meta['column_top_axial_load_kip']:.2f} kip at the upper cut and {meta['joint_vertical_load_kip']:.2f} kip at the joint. Columns use P–Δ transformations. Original slab contributions remain embedded in the saved beam strengths and stiffness; no explicit slab or transverse X-beam members are modeled.</p></section>
<section><h2>Beam hinge moment–rotation behavior</h2>{picture('moment_rotation.png','Beam 165 moment rotation history and fresh monotonic backbone')}<p>Fresh monotonic traces show the underlying backbone. Repeated cycling reduces the beam’s moment resistance well before its original monotonic rotation limit. Positive beam rotation/moment is sagging; negative is hogging.</p></section>
<section><h2>Isolate cyclic deterioration</h2>{picture('deterioration_comparison.png','Installed cyclic deterioration compared with suppressed deterioration')}<p>The comparison changes only the four cyclic deterioration capacities to a large finite value in the control. Initial stiffness, positive/negative strength, backbone softening, axial loads and P–Δ remain the same. Under the installed settings, unloading-stiffness and accelerated-reloading deterioration are already effectively suppressed; strength and post-capping deterioration remain active.</p><div class="scroll"><table><thead><tr><th>Drift amplitude</th><th>Direction</th><th>Installed moment (kip-ft)</th><th>Control moment (kip-ft)</th><th>Retained</th></tr></thead><tbody>{table}</tbody></table></div><p>Values are second-cycle endpoints at matching imposed drift. At the 7.5% level the beam spring loses its remaining modeled moment resistance. This is a constitutive-model result under this long cyclic history; it is not evidence that a physical case-0014 joint would fail at that drift.</p></section>
<section><h2>Fixture response</h2>{picture('fixture_response.png','Force displacement and local hinge rotation histories')}<p>After severe spring degradation, the P–Δ contribution can make the required actuator force negative at positive displacement. The test remains displacement-controlled and can trace this branch. Completion of these imposed steps does not demonstrate stable load-controlled behavior or full-building safety.</p></section>
<section><h2>Verification and assumptions</h2><details><summary>Column hinge checks — columns remain below yield</summary>{picture("column_rotation_checks.png","Column hinge moment rotation verification histories")}<p>Peak lower/upper column spring rotations reach {ratio["column_below"]*100:.1f}% / {ratio["column_above"]*100:.1f}% of yield rotation. These local spring rotations check the fixture response.</p></details><p>All imposed endpoints completed. The installed and half-step runs each needed one successful step subdivision. Halving the displacement increment changed endpoint force by at most <b>{checks['half_step_relative_endpoint_force_difference']*100:.5f}%</b> of the installed run’s peak force. The independent elastic stiffness calculation agrees to {checks['relative_elastic_error']:.2e} relative error. Each hinge’s positive and negative monotonic cap also reproduced its saved parameters.</p><p>Maximum horizontal equilibrium residual: {meta['runs']['fema461_installed']['max_horizontal_residual_kip']:.2e} kip. Joint moment residual: {meta['runs']['fema461_installed']['max_joint_moment_residual_kip_in']:.2e} kip-in. External/internal incremental work mismatch: {meta['runs']['fema461_installed']['max_work_balance_error_kip_in']:.2e} kip-in.</p><ul><li>Joint core is rigid and centered; there are no finite joint offsets or panel-shear springs.</li><li>Member backbones remain fixed at the original gravity-axial reference values. Changing axial force does not update the P–M surface.</li><li>Preloads and pin/roller boundaries idealize the subassembly. Original full-frame distributed loads, beam gravity moments and 3D interaction are not replayed.</li><li>No source design or production calibration parameter was changed.</li></ul></section>
<footer><h2>Data and reproducibility</h2><p><a href="results/fema461_installed.csv">Full installed response CSV</a> · <a href="cycle_strength.csv">Cycle strength table</a> · <a href="summary.json">Summary</a> · <a href="results/test_definition.json">Fixture, parameters, source hashes and checks</a> · Repository runner: pilots/case0014Fema461/run_connection.py</p><p>Geometry and member parameters: results/test_definition.json</p><small>The earthquake’s peak-damage hinge was column 16’s base. This connection test uses its upper end at the first floor and the connected beam; they are different hinges.</small></footer></main></html>'''
(ROOT/'index.html').write_text(html,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['strength_retention','checks']},indent=2))
