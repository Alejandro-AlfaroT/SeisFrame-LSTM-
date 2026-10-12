"""Fit two IMKPinching shape parameters to simulated beam data, then replay the fixture."""
from pathlib import Path
import argparse, importlib.util, sys, json, hashlib, contextlib, io
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',type=Path,required=True,help='Completed FEMA runner output directory')
parser.add_argument('--output',type=Path,required=True,help='New diagnostic output directory')
args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
reference=args.reference.resolve();f=reference/'results/fema461_installed.npz'
a=np.load(f);cols=list(a['columns']);theta=a['data'][:,cols.index('beam_rotation_rad')];target=a['data'][:,cols.index('beam_moment_kip_in')]
# Import the fixture builder into a fresh protected child output directory.
saved=sys.argv;sys.argv=['run_connection.py','--output',str(root/'connection')]
spec=importlib.util.spec_from_file_location('fixture',Path(__file__).with_name('run_connection.py'));fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);sys.argv=saved
ops=fixture.ops;original=fixture.matargs;base=original(fixture.MEMBERS['beam']);scale=max(abs(target));evaluations=[]
def history(kf,kd,peak=False):
    ops.wipe();values=base if peak else ['IMKPinching',*base[1:],float(kf),float(kd)]
    ops.uniaxialMaterial(*values);ops.testUniaxialMaterial(values[1]);result=np.empty(len(theta))
    for i,t in enumerate(theta):ops.setStrain(float(t));result[i]=ops.getStress()
    if not np.isfinite(result).all():raise ValueError('Non-finite material history')
    return result
baseline=history(0,0,True);baseline_error=float(np.sqrt(np.mean((baseline-target)**2))/scale)
assert baseline_error<1e-6,('Reference material replay mismatch',baseline_error)
def objective(x):
    pred=history(*x);score=float(np.mean(((pred-target)/scale)**2));evaluations.append({'kappaF':float(x[0]),'kappaD':float(x[1]),'normalized_mse':score});return score
starts=[(.1,.1),(.5,.5),(.9,.1),(.1,.9),(.9,.9)]
best=min(starts,key=objective)
fit=minimize(objective,best,method='Powell',bounds=[(.001,.999),(.001,.999)],options={'maxiter':12,'xtol':.002,'ftol':1e-5})
kf,kd=map(float,fit.x);pred=history(kf,kd)
# Preserve columns as PeakOriented; substitute only the beam law.
def mapped(v,control=False):
    values=original(v,control)
    return ['IMKPinching',*values[1:],kf,kd] if v is fixture.MEMBERS['beam'] else values
fixture.matargs=mapped
meta=json.loads((reference/'results/test_definition.json').read_text());amps=meta['protocol']['amplitude_drift_ratios']
targets=[(sign*amp,level,cycle) for level,amp in enumerate(amps,1) for cycle in (1,2) for sign in (1,-1,0)]
report,hist=fixture.run('fema461_pinching_fitted',targets)
assert report['completed'],'Connection replay did not complete'
assert report['max_horizontal_residual_kip']<1e-5
assert report['max_vertical_residual_kip']<1e-5
assert report['max_joint_moment_residual_kip_in']<1e-4
cap_checks=fixture.material_checks()
refined,_=fixture.run('fema461_pinching_half_step',targets,step_drift=.000025)
assert refined['completed']
step_error=max(abs(x['actuator_force_kip']-y['actuator_force_kip']) for x,y in zip(report['endpoints'],refined['endpoints']))/report['max_abs_actuator_force_kip']
assert step_error<.02

# Compare complete fixture solutions at matching requested endpoints.
ref=json.loads((reference/'results/fema461_installed.json').read_text());assert len(report['endpoints'])==len(ref['endpoints'])
endpoint_rmse=float(np.sqrt(np.mean([(x['beam_moment_kip_in']-y['beam_moment_kip_in'])**2 for x,y in zip(report['endpoints'],ref['endpoints'])]))/scale)
work=lambda moment,rotation:float(np.sum((moment[1:]+moment[:-1])*.5*np.diff(rotation)))
result={'status':'simulation_to_simulation_fit_not_experimental_calibration','reference_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'fit_parameters':{'kappaF':kf,'kappaD':kd},'bounds':[.001,.999],'optimized_parameters_only':['kappaF','kappaD'],'installed_beam_arguments':mapped(fixture.MEMBERS['beam']),'argument_order':['material','tag','Ke','dp_pos','dpc_pos','du_pos','Fy_pos','FmaxFy_pos','FresFy_pos','dp_neg','dpc_neg','du_neg','Fy_neg','FmaxFy_neg','FresFy_neg','Lamda_S','Lamda_C','Lamda_A','Lamda_K','c_S','c_C','c_A','c_K','D_pos','D_neg','kappaF','kappaD'],'normalized_material_rmse':float(np.sqrt(fit.fun)),'reference_peakoriented_replay_rmse':baseline_error,'connection_endpoint_normalized_rmse':endpoint_rmse,'target_signed_work_kip_in':work(target,theta),'fitted_signed_work_kip_in':work(pred,theta),'optimizer_success':bool(fit.success),'optimizer_message':str(fit.message),'evaluations':evaluations,'connection_report':report,'material_cap_checks':cap_checks,'half_step_relative_endpoint_force_difference':step_error,'limitations':['Target is existing simulated beam response, not measured connection data.','Backbone, slab contribution and all deterioration parameters are fixed.','Columns remain IMKPeakOriented; rigid joint has no calibrated panel-shear or slip spring.','Parameters do not generalize to other connections without validation.']}
(root/'fit_parameters.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(root/'material_comparison.npz',rotation_rad=theta,target_moment_kip_in=target,pinching_moment_kip_in=pred)
fig,ax=plt.subplots(figsize=(11,6));ax.plot(theta,target/12,label='Existing PeakOriented target',lw=1);ax.plot(theta,pred/12,label='Fitted Pinching, prescribed beam rotation',lw=.8,alpha=.8);ax.set(xlabel='Beam spring rotation (rad)',ylabel='Moment (kip-ft)',title='Case 0014 beam 165 — simulated history fit');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(root/'material_fit.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,6));ax.plot(theta,target/12,label='Original connection',lw=1);ax.plot([r['beam_rotation_rad'] for r in hist],[r['beam_moment_kip_in']/12 for r in hist],label='Connection with fitted Pinching beam',lw=.8,alpha=.8);ax.set(xlabel='Beam spring rotation (rad)',ylabel='Moment (kip-ft)',title='Same FEMA fixture displacement history');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(root/'connection_comparison.png',dpi=160);plt.close(fig)
print(json.dumps({k:result[k] for k in ['fit_parameters','normalized_material_rmse','connection_endpoint_normalized_rmse','optimizer_success']},indent=2))
