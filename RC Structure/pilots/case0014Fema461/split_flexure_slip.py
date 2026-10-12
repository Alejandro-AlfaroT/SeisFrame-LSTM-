"""Read-only extraction and isolated FEMA-461-history diagnostic. No repository imports/edits."""
from pathlib import Path
import csv,json,hashlib,math,time
import numpy as np
import openseespy.opensees as ops

import argparse
parser=argparse.ArgumentParser(description='Split case 0014 FEMA 461 connection diagnostic')
parser.add_argument('--output',type=Path,required=True,help='New output directory')
ROOT=parser.parse_args().output.resolve();ROOT.mkdir(parents=True,exist_ok=False)
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
# Declared sensitivity partition, not fitted bond-slip calibration.
SLIP_ELASTIC_SHARE=.20
beam=MEMBERS['beam'];original_ke=beam['ke'];original_dp=float(beam['source_spring']['theta_p'])
flex_dp=original_dp/1.55;slip_dp=original_dp-flex_dp
beam['ke']=original_ke/(1-SLIP_ELASTIC_SHARE)
beam['source_spring']=dict(beam['source_spring'],theta_p=str(flex_dp),theta_u=str(flex_dp+float(beam['source_spring']['theta_pc'])+.005))
SLIP_KE=original_ke/SLIP_ELASTIC_SHARE
SLIP_ARGS=['IMKPinching',104,SLIP_KE,slip_dp,10*slip_dp,11*slip_dp+.005,float(beam['source_spring']['yield_moment_positive_kip_in']),1.25,.2,slip_dp,10*slip_dp,11*slip_dp+.005,float(beam['source_spring']['yield_moment_negative_kip_in']),1.25,.2,1e12,1e12,1e12,1e12,1,1,1,1,1,1,.25,.25]
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
            if len(hist)>400000:complete=False;reason='declared maximum committed steps exceeded';break
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
    report,hist=run('fema461_split_flexure_slip',targets)
    assert report['completed']
    series=max(abs(r['series_moment_residual_kip_in']) for r in hist)
    rotation=max(abs(r['series_rotation_residual_rad']) for r in hist)
    assert series/max(float(beam['source_spring']['yield_moment_positive_kip_in']),float(beam['source_spring']['yield_moment_negative_kip_in']))<1e-6 and rotation<1e-10
    assert report['max_horizontal_residual_kip']<1e-4 and report['max_vertical_residual_kip']<1e-4
    step_error=None  # Half-step attempt stalled at reversals; qualification remains open.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(18,6))
    for ax,key,title in zip(axs,['beam_rotation_rad','beam_slip_rotation_rad','beam_total_rotation_rad'],['PeakOriented flexure','Pinching slip interface','Combined series rotation']):
        ax.plot([r[key] for r in hist],[r['beam_moment_kip_in']/12 for r in hist],lw=.8)
        ax.set(title=title,xlabel='Rotation (rad)',ylabel='Moment (kip-ft)');ax.grid(alpha=.2)
    fig.suptitle('Case 0014 — separate flexure and slip, declared diagnostic partition');fig.tight_layout();fig.savefig(ROOT/'split_response.png',dpi=160);plt.close(fig)
    summary={'status':'provisional_completed_coarse_run_half_step_unresolved','flexure_material':'IMKPeakOriented','slip_material':'IMKPinching','solver_test':{'type':'NormUnbalance','tolerance':1e-4},'slip_elastic_flexibility_share':SLIP_ELASTIC_SHARE,'combined_initial_ke':1/(1/beam['ke']+1/SLIP_KE),'original_initial_ke':original_ke,'flexure_theta_p':flex_dp,'allocated_slip_theta_p':slip_dp,'slip_arguments':SLIP_ARGS,'flexure_arguments':matargs(beam),'series_moment_residual_kip_in':series,'series_rotation_residual_rad':rotation,'half_step_relative_endpoint_force_difference':step_error,'material_cap_checks':checks,'run':report,'limitations':['20 percent elastic slip share is declared, not section-derived or fitted.','1.55 plastic-capacity partition follows the repository empirical slip-factor concept, not independent bond-slip calibration.','Slip deterioration suppressed; no cyclic bond-deterioration evidence.','Flexure deterioration settings retained as a sensitivity assumption, not recalibrated after decomposition.','Columns retain aggregate hinges; joint panel stays rigid; no panel shear or column-slip interface added.','Slab contributions remain embedded in beam strengths.']}
    (ROOT/'split_parameters.json').write_text(json.dumps(summary,indent=2))
    print('Coarse split fixture completed; half-step qualification unresolved.',series,rotation)
