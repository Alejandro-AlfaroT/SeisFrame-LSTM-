"""Declared pinching sensitivity in the complete case 0014 connection fixture."""
from pathlib import Path
import argparse,importlib.util,sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
ref=args.reference.resolve();meta=json.loads((ref/'results/test_definition.json').read_text())
saved=sys.argv;sys.argv=['fixture','--output',str(root/'connection')]
spec=importlib.util.spec_from_file_location('fixture',Path(__file__).with_name('run_connection.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);sys.argv=saved
original=f.matargs
amps=meta['protocol']['amplitude_drift_ratios'];targets=[(sign*a,level,cycle) for level,a in enumerate(amps,1) for cycle in (1,2) for sign in (1,-1,0)]
reference=np.load(ref/'results/fema461_installed.npz');cols=list(reference['columns']);data=reference['data'];tx=data[:,cols.index('beam_rotation_rad')];my=data[:,cols.index('beam_moment_kip_in')]
cases=[('pinching_025_025',.25,.25),('pinching_010_050',.10,.50),('pinching_005_075',.05,.75)]
reports=[];histories=[]
for name,kf,kd in cases:
 def mapped(v,control=False):
  values=original(v,control)
  return ['IMKPinching',*values[1:],kf,kd] if v is f.MEMBERS['beam'] else values
 f.matargs=mapped
 checks=f.material_checks()
 report,h=f.run(name,targets)
 assert report['completed']
 assert report['max_horizontal_residual_kip']<1e-5 and report['max_vertical_residual_kip']<1e-5
 reports.append({'name':name,'kappaF':kf,'kappaD':kd,'installed_beam_arguments':mapped(f.MEMBERS['beam']),'material_cap_checks':checks,'report':report});histories.append(h)
 # Each curve is a separate full-fixture solution under the same imposed displacement.
fig,axs=plt.subplots(1,3,figsize=(18,6),sharex=True,sharey=True)
for ax,(name,kf,kd),h in zip(axs,cases,histories):
 ax.plot(tx,my/12,color='#426b95',lw=.7,alpha=.65,label='Original PeakOriented')
 ax.plot([r['beam_rotation_rad'] for r in h],[r['beam_moment_kip_in']/12 for r in h],color='#cf5d24',lw=.8,label='IMKPinching beam')
 ax.set(title=f'κF={kf:.2f}, κD={kd:.2f}',xlabel='Beam spring rotation (rad)');ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('Beam moment (kip-ft)');fig.suptitle('Case 0014 — declared pinching sensitivity, same FEMA connection loading');fig.tight_layout();fig.savefig(root/'pinching_comparison.png',dpi=160);plt.close(fig)
# Zoom early inelastic cycles to make reloading breakpoints visible without late failure paths.
fig,axs=plt.subplots(1,3,figsize=(18,6),sharex=True,sharey=True)
mask=(data[:,cols.index('level')]>=10)&(data[:,cols.index('level')]<=13)
for ax,(name,kf,kd),h in zip(axs,cases,histories):
 chosen=[r for r in h if 10<=r['level']<=13]
 ax.plot(tx[mask],my[mask]/12,color='#426b95',lw=.9,label='Original PeakOriented')
 ax.plot([r['beam_rotation_rad'] for r in chosen],[r['beam_moment_kip_in']/12 for r in chosen],color='#cf5d24',lw=.9,label='IMKPinching beam')
 ax.set(title=f'κF={kf:.2f}, κD={kd:.2f}',xlabel='Beam spring rotation (rad)');ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('Beam moment (kip-ft)');fig.suptitle('Reloading paths — FEMA amplitude levels 10–13');fig.tight_layout();fig.savefig(root/'pinching_reloading.png',dpi=160);plt.close(fig)
(root/'sensitivity.json').write_text(json.dumps({'status':'declared_sensitivity_not_calibrated','scope':'Beam material changed inside complete fixture; columns PeakOriented and rigid joint retained. Backbone, slab contribution and all deterioration settings fixed. No fit to PeakOriented.','cases':reports},indent=2))
print('All three declared sensitivity cases completed and material/equilibrium checks passed.')
