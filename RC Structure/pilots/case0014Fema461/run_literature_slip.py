"""Read-only extraction and isolated FEMA-461-history diagnostic. No repository imports/edits."""
from pathlib import Path
import csv,json,hashlib,math,time
import numpy as np
import openseespy.opensees as ops

import argparse
parser=argparse.ArgumentParser(description='Split case 0014 FEMA 461 connection diagnostic')
parser.add_argument('--output',type=Path,required=True,help='New output directory')
parser.add_argument('--strain-at-probable',type=float,default=.02,help='Declared bar strain at 75 ksi, e.g. 0.08; diagnostic only')
args=parser.parse_args()
ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=False)
OUT=ROOT/'results';OUT.mkdir()
INPUT=Path(__file__).resolve().parent/'inputs'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
fixture=read(INPUT/'fixture.json');gp=read(INPUT/'global_parameters.json')
H=fixture['height_in'];L=fixture['beam_half_span_in']*2
n=gp['imk_hinge_stiffness_factor'];Ifactor=(n+1)/n
MEMBERS=fixture['members']
Ptop=fixture['column_top_axial_load_kip'];Plower=fixture['column_below_reference_axial_kip'];Pjoint=fixture['joint_vertical_load_kip']
COORDS={1:(0,0),2:(0,H),3:(-L/2,H/2),10:(0,H/2),11:(0,H/2),12:(0,H/2),13:(0,H/2)}
ELE_NODES={'column_below':(1,11),'column_above':(12,2),'beam':(3,13)}
CY=gp['imk_cyclic_parameters'];DM=.03;MAX_DRIFT=.084
from compatible_bond_slip import calibrate
calibration=calibrate(args.strain_at_probable)
beam=MEMBERS['beam'];original_ke=beam['ke'];original_dp=float(beam['source_spring']['theta_p'])
flex_dp=original_dp/1.55
# Cracked section flexural stiffness replaces aggregate elastic properties consistently.
flex_candidates=calibration['composite']['flexure_stiffness_candidates']
beam['ke']=2/sum(1/x for x in flex_candidates.values())
beam['EI_effective']=beam['ke']*float(beam['source_element']['length_in'])/(6*n)
beam['EI_segment']=beam['EI_effective']*Ifactor
beam['source_spring']=dict(beam['source_spring'],theta_p=str(flex_dp),theta_u=str(flex_dp+float(beam['source_spring']['theta_pc'])+.005))
SLIP_KE=calibration['ke']
def slip_branch(sign):
    v=calibration['branches'][sign];dp=v['dp']
    return [dp,10*dp,v['installed_yield_rotation']+11*dp,v['yield_moment'],v['cap_moment']/v['yield_moment'],.2]
SLIP_ARGS=['IMKPinching',104,SLIP_KE,*slip_branch('sagging'),*slip_branch('hogging'),1e12,1e12,1e12,1e12,1,1,1,1,1,1,.25,.50]
COORDS[14]=COORDS[10]


def matargs(v,control=False):
    s=v['source_spring'];lam=json.loads(s['installed_lamda_by_mode_rad']);dp=float(s['theta_p']);pc=float(s['theta_pc']);du=float(s['theta_u'])
    pos=[dp,pc,du,float(s['yield_moment_positive_kip_in']),gp['imk_fmaxfy_pos'],gp['imk_fresfy_pos']]
    neg=[dp,pc,du,float(s['yield_moment_negative_kip_in']),gp['imk_fmaxfy_neg'],gp['imk_fresfy_neg']]
    return ['IMKPeakOriented',v['material_tag'],v['ke'],*pos,*neg,*([1e12]*4 if control else [lam[k] for k in ['S','C','A','K']]),*[CY[k] for k in ['c_s','c_c','c_a','c_k','d_pos','d_neg']]]

def build(control=False,elastic=False,axial=True,pdelta=True):
    ops.wipe();ops.model('basic','-ndm',2,'-ndf',3)
    for tag,xy in COORDS.items():ops.node(tag,*xy)
    ops.fix(1,1,1,0);ops.fix(3,0,1,0)
    ops.geomTransf('PDelta' if pdelta else 'Linear',1);ops.geomTransf('Linear',2)
    for name,v in MEMBERS.items():
        hn=v['hinge_node'];ops.equalDOF(10,hn,1,2)
        if elastic:ops.uniaxialMaterial('Elastic',v['material_tag'],v['ke'])
        else:ops.uniaxialMaterial(*matargs(v,control))
        ends=(hn,10) if v['reverse_nodes'] else (10,hn)
        if name=='beam':
            ops.equalDOF(10,14,1,2)
            ops.element('zeroLength',103,14,10,'-mat',103,'-dir',3)
            if elastic:ops.uniaxialMaterial('Elastic',104,SLIP_KE)
            else:ops.uniaxialMaterial(*SLIP_ARGS)
            ops.element('zeroLength',104,13,14,'-mat',104,'-dir',3)
        else:ops.element('zeroLength',v['spring_element'],*ends,'-mat',v['material_tag'],'-dir',3)
        ops.element('elasticBeamColumn',v['elastic_element'],*ELE_NODES[name],v['A'],v['E'],v['EI_segment']/v['E'],2 if name=='beam' else 1)
    ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
    ops.load(2,0,-Ptop if axial else 0,0);ops.load(10,0,-Pjoint if axial else 0,0)
    ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral');ops.test('NormUnbalance',1e-4,100,0);ops.algorithm('Newton');ops.integrator('LoadControl',.1);ops.analysis('Static')
    assert ops.analyze(10)==0,'Gravity failure'
    ops.loadConst('-time',0);ops.wipeAnalysis();ops.timeSeries('Linear',2);ops.pattern('Plain',2,2);ops.load(2,1,0,0)
    ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral');ops.test('NormUnbalance',1e-4,100,0);ops.algorithm('Newton');ops.analysis('Static')
    return float(ops.nodeDisp(2,1))

def condensed_elastic_stiffness():
    tags=list(COORDS);ix={tag:i for i,tag in enumerate(tags)};K=np.zeros((3*len(tags),3*len(tags)))
    def assemble(k,ends):
        ids=[3*ix[t]+j for t in ends for j in range(3)];K[np.ix_(ids,ids)]+=k
    for name,v in MEMBERS.items():
        a,b=ELE_NODES[name];delta=np.array(COORDS[b])-COORDS[a];ell=float(np.linalg.norm(delta));c,si=delta/ell;EA=v['E']*v['A'];EI=v['EI_segment'];a0=EA/ell;b0=12*EI/ell**3;c0=6*EI/ell**2;d0=4*EI/ell;e0=2*EI/ell
        k=np.array([[a0,0,0,-a0,0,0],[0,b0,c0,0,-b0,c0],[0,c0,d0,0,-c0,e0],[-a0,0,0,a0,0,0],[0,-b0,-c0,0,b0,-c0],[0,c0,e0,0,-c0,d0]])
        rot=np.array([[c,si,0],[-si,c,0],[0,0,1]]);T=np.zeros((6,6));T[:3,:3]=rot;T[3:,3:]=rot;assemble(T.T@k@T,[a,b])
        k=np.zeros((6,6));k[2,2]=k[5,5]=v['ke'];k[2,5]=k[5,2]=-v['ke'];assemble(k,[10,v['hinge_node']])
    keys=[];mapping=[]
    for tag in tags:
        for j in range(3):
            key=(10 if tag in [11,12,13] and j<2 else tag,j)
            if (tag==1 and j<2) or (tag==3 and j==1):mapping.append(None);continue
            if key not in keys:keys.append(key)
            mapping.append(keys.index(key))
    T=np.zeros((len(mapping),len(keys)))
    for i,j in enumerate(mapping):
        if j is not None:T[i,j]=1
    Kr=T.T@K@T;F=np.zeros(len(keys));F[keys.index((2,0))]=1;U=np.linalg.solve(Kr,F)
    return 1/U[keys.index((2,0))]

def material_checks():
    checks=[]
    for name,v in MEMBERS.items():
        for sign in [1,-1]:
            ops.wipe();ops.uniaxialMaterial(*matargs(v));ops.testUniaxialMaterial(v['material_tag']);s=v['source_spring'];fy=float(s['yield_moment_positive_kip_in' if sign==1 else 'yield_moment_negative_kip_in']);cap=fy/v['ke']+float(s['theta_p']);target=fy*gp['imk_fmaxfy_pos' if sign==1 else 'imk_fmaxfy_neg']
            for x in np.linspace(0,sign*cap,2001):ops.setStrain(float(x))
            actual=abs(ops.getStress());error=abs(actual-target)/target;assert error<1e-5,(name,sign,error)
            checks.append({'member':name,'sign':sign,'initial_capping_rotation_rad':cap,'capping_moment_kip_in':actual,'relative_error':error})
    return checks

def snapshot(u0,level=0,cycle=0,target=0):
    ops.reactions();F=float(ops.getLoadFactor(2));ut=float(ops.nodeDisp(2,1));vs=np.array([ops.nodeDisp(2,2),ops.nodeDisp(10,2)])
    row={'step':0,'level':level,'cycle':cycle,'target_drift_ratio':target,'u_top_in':ut-u0,'drift_ratio':(ut-u0)/H,'actuator_force_kip':F,'top_vertical_in':vs[0],'joint_vertical_in':vs[1]}
    row['horizontal_residual_kip']=F+ops.nodeReaction(1,1)+ops.nodeReaction(3,1)
    row['vertical_residual_kip']=ops.nodeReaction(1,2)+ops.nodeReaction(3,2)-Ptop-Pjoint
    ef=[];ed=[]
    for name,v in MEMBERS.items():
        se=v['spring_element'];mom=float(ops.eleResponse(se,'material',1,'stress')[0]);theta=float(ops.eleResponse(se,'material',1,'strain')[0]);row[name+'_moment_kip_in']=mom;row[name+'_rotation_rad']=theta
        ee=v['elastic_element'];ef.extend(ops.eleForce(ee));ed.extend([z for node in ELE_NODES[name] for z in ops.nodeDisp(node)])
    row['beam_slip_rotation_rad']=float(ops.eleResponse(104,'material',1,'strain')[0])
    row['beam_slip_moment_kip_in']=float(ops.eleResponse(104,'material',1,'stress')[0])
    row['beam_total_rotation_rad']=row['beam_rotation_rad']+row['beam_slip_rotation_rad']
    row['series_moment_residual_kip_in']=row['beam_moment_kip_in']-row['beam_slip_moment_kip_in']
    row['series_rotation_residual_rad']=row['beam_total_rotation_rad']-(ops.nodeDisp(10,3)-ops.nodeDisp(13,3))
    row['joint_moment_residual_kip_in']=row['beam_moment_kip_in']-row['column_below_moment_kip_in']-row['column_above_moment_kip_in']
    return row,np.array(ef),np.array(ed)

def run(name,targets,step_drift=.00005,control=False,save=True):
    u0=build(control=control);rec,prevf,prevu=snapshot(u0);hist=[rec];ends=[];recoveries=0;external=0.;internal=0.;hingework={k:0. for k in MEMBERS};complete=True;reason='completed';t0=time.perf_counter()
    rec.update(external_work_kip_in=0,internal_work_kip_in=0,work_balance_error_kip_in=0)
    for target,level,cycle in targets:
        targetu=u0+target*H
        while abs(ops.nodeDisp(2,1)-targetu)>1e-10:
            if time.perf_counter()-t0>90 or len(hist)>400000:complete=False;reason='declared maximum committed steps exceeded';break
            du=np.clip(targetu-ops.nodeDisp(2,1),-step_drift*H,step_drift*H);success=False
            for divisor in [1,2,5,10,50,100]:
                ops.integrator('DisplacementControl',2,1,float(du/divisor));ops.algorithm('Newton');ok=ops.analyze(1)
                if ok==0:success=True;recoveries+=int(divisor>1);break
            if not success:complete=False;reason='solver failed after step subdivision';break
            row,ef,eu=snapshot(u0,level,cycle,target);row['step']=len(hist);prev=hist[-1]
            external+=(row['actuator_force_kip']+prev['actuator_force_kip'])/2*(row['u_top_in']-prev['u_top_in'])-Ptop*(row['top_vertical_in']-prev['top_vertical_in'])-Pjoint*(row['joint_vertical_in']-prev['joint_vertical_in'])
            internal+=float(np.dot((ef+prevf)/2,eu-prevu))
            for k in MEMBERS:
                dw=(row[k+'_moment_kip_in']+prev[k+'_moment_kip_in'])/2*(row[k+'_rotation_rad']-prev[k+'_rotation_rad']);internal+=dw;hingework[k]+=dw
                row[k+'_cumulative_work_kip_in']=hingework[k]
            internal+=(row['beam_slip_moment_kip_in']+prev['beam_slip_moment_kip_in'])/2*(row['beam_slip_rotation_rad']-prev['beam_slip_rotation_rad'])
            row.update(external_work_kip_in=external,internal_work_kip_in=internal,work_balance_error_kip_in=external-internal)
            hist.append(row);prevf=ef;prevu=eu
            branch='sagging' if row['beam_slip_rotation_rad']>=0 else 'hogging'
            if abs(row['beam_slip_rotation_rad'])>calibration['branches'][branch]['cap_rotation']:
                complete=False;reason='derived slip envelope exceeded; stop before unsupported anchorage extrapolation';break
            if not np.isfinite(list(row.values())).all():raise RuntimeError('Non-finite response')
        if not complete:break
        ends.append(dict(hist[-1]))
    cols=list(hist[-1]);array=np.array([[r.get(k,0) for k in cols] for r in hist]);report={'name':name,'completed':complete,'stop_reason':reason,'committed_steps':len(hist)-1,'endpoint_count':len(ends),'requested_endpoints':len(targets),'step_drift_ratio':step_drift,'step_reductions':recoveries,'seconds':time.perf_counter()-t0,'max_abs_actuator_force_kip':max(abs(r['actuator_force_kip']) for r in hist),'max_abs_drift_ratio':max(abs(r['drift_ratio']) for r in hist),'max_horizontal_residual_kip':max(abs(r['horizontal_residual_kip']) for r in hist),'max_vertical_residual_kip':max(abs(r['vertical_residual_kip']) for r in hist),'max_joint_moment_residual_kip_in':max(abs(r['joint_moment_residual_kip_in']) for r in hist),'max_work_balance_error_kip_in':max(abs(r['work_balance_error_kip_in']) for r in hist),'final_external_work_kip_in':external,'final_member_hinge_work_kip_in':hingework,'peak_rotations_rad':{k:max(abs(r[k+'_rotation_rad']) for r in hist) for k in MEMBERS},'endpoints':ends}
    if save:
        with (OUT/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r.get(k,0) for k in cols} for r in hist)
        np.savez_compressed(OUT/(name+'.npz'),data=array,columns=cols)
        (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['endpoints']}),flush=True);return report,hist

if __name__=='__main__':
    checks=material_checks()
    amps=fixture['protocol']['amplitude_drift_ratios']
    targets=[(sign*a,level,cycle) for level,a in enumerate(amps,1) for cycle in (1,2) for sign in (1,-1,0)]
    report,hist=run('literature_split',targets)
    # Preserve failed/partial histories; completion is reported, never inferred.
    series=max(abs(r['series_moment_residual_kip_in']) for r in hist)
    rotation=max(abs(r['series_rotation_residual_rad']) for r in hist)
    assert series/max(v['cap_moment'] for v in calibration['branches'].values())<1e-6 and rotation<1e-10
    fine,fh=run('literature_split_half_step',targets,step_drift=.000025)
    step_error=None
    if fine['completed'] and report['completed']:
        step_error=max(abs(a['actuator_force_kip']-b['actuator_force_kip']) for a,b in zip(report['endpoints'],fine['endpoints']))/report['max_abs_actuator_force_kip']
    maxslip=max(abs(r['beam_slip_rotation_rad']) for r in hist)
    beyond=sum(abs(r['beam_slip_rotation_rad'])>calibration['branches']['sagging' if r['beam_slip_rotation_rad']>=0 else 'hogging']['cap_rotation'] for r in hist)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(18,6))
    for ax,key,title in zip(axs,['beam_rotation_rad','beam_slip_rotation_rad','beam_total_rotation_rad'],['PeakOriented flexure','Pinching strain penetration','Combined series rotation']):
        ax.plot([r[key] for r in hist],[r['beam_moment_kip_in']/12 for r in hist],lw=.8)
        ax.set(title=title,xlabel='Rotation (rad)',ylabel='Moment (kip-ft)');ax.grid(alpha=.2)
    fig.suptitle('Case 0014 — literature-based diagnostic with declared assumptions');fig.tight_layout();fig.savefig(ROOT/'response.png',dpi=160);plt.close(fig)
    calibration.update(flexure_arguments=matargs(beam),slip_arguments=SLIP_ARGS,original_ke=original_ke,material_cap_checks=checks,coarse_run=report,half_step_run=fine,half_step_relative_endpoint_force_difference=step_error,half_step_qualified=bool(report['completed'] and fine['completed'] and step_error is not None and step_error<.02),max_slip_rotation_rad=maxslip,steps_beyond_derived_slip_envelope=beyond,series_moment_residual_kip_in=series,series_rotation_residual_rad=rotation,pipeline_installed=False,flexure_deterioration='Original aggregate energy settings retained provisionally; flexure-only experimental identification unavailable')
    (ROOT/'parameters.json').write_text(json.dumps(calibration,indent=2))
    print(json.dumps({k:calibration[k] for k in ['ke','half_step_qualified','half_step_relative_endpoint_force_difference','max_slip_rotation_rad','steps_beyond_derived_slip_envelope']},indent=2))
