"""Read-only extraction and isolated FEMA-461-history diagnostic. No repository imports/edits."""
from pathlib import Path
import csv,json,hashlib,math,time
import numpy as np
import openseespy.opensees as ops

import argparse
parser=argparse.ArgumentParser(description='Frozen case 0014 FEMA 461 connection diagnostic')
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
        ops.element('zeroLength',v['spring_element'],*ends,'-mat',v['material_tag'],'-dir',3)
        ops.element('elasticBeamColumn',v['elastic_element'],*ELE_NODES[name],v['A'],v['E'],v['EI_segment']/v['E'],2 if name=='beam' else 1)
    ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
    ops.load(2,0,-Ptop if axial else 0,0);ops.load(10,0,-Pjoint if axial else 0,0)
    ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral');ops.test('NormDispIncr',1e-11,100,0);ops.algorithm('Newton');ops.integrator('LoadControl',.1);ops.analysis('Static')
    assert ops.analyze(10)==0,'Gravity failure'
    ops.loadConst('-time',0);ops.wipeAnalysis();ops.timeSeries('Linear',2);ops.pattern('Plain',2,2);ops.load(2,1,0,0)
    ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral');ops.test('NormDispIncr',1e-11,100,0);ops.algorithm('Newton');ops.analysis('Static')
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
    row['joint_moment_residual_kip_in']=row['beam_moment_kip_in']-row['column_below_moment_kip_in']-row['column_above_moment_kip_in']
    return row,np.array(ef),np.array(ed)

def run(name,targets,step_drift=.00005,control=False,save=True):
    u0=build(control=control);rec,prevf,prevu=snapshot(u0);hist=[rec];ends=[];recoveries=0;external=0.;internal=0.;hingework={k:0. for k in MEMBERS};complete=True;reason='completed';t0=time.perf_counter()
    rec.update(external_work_kip_in=0,internal_work_kip_in=0,work_balance_error_kip_in=0)
    for target,level,cycle in targets:
        targetu=u0+target*H
        while abs(ops.nodeDisp(2,1)-targetu)>1e-10:
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
    materials=material_checks();u0=build(elastic=True,axial=False,pdelta=False);ops.integrator('DisplacementControl',2,1,1e-4);assert ops.analyze(1)==0
    observed=ops.getLoadFactor(2)/1e-4;expected=condensed_elastic_stiffness();rel=abs(observed/expected-1);assert rel<1e-7,rel
    checks={'material_backbone_checks':materials,'elastic_fixture_stiffness_kip_per_in':observed,'independent_stiffness_kip_per_in':expected,'relative_elastic_error':rel}
    mono_plus,mp=run('monotonic_positive',[(MAX_DRIFT,0,0)]);mono_minus,mn=run('monotonic_negative',[(-MAX_DRIFT,0,0)])
    first_yield=[]
    for hist in [mp,mn]:
        for row in hist:
            if any(abs(row[k+'_rotation_rad'])>=float(v['source_spring']['yield_moment_positive_kip_in' if row[k+'_rotation_rad']>=0 else 'yield_moment_negative_kip_in'])/v['ke'] for k,v in MEMBERS.items()):first_yield.append(abs(row['drift_ratio']));break
    levels=13
    while min(first_yield)<=DM/(1.4**(levels-3)):levels+=1
    amps=[DM/(1.4**i) for i in reversed(range(levels))]+[DM*(1+.3*i) for i in range(1,7)]
    targets=[(sign*a,level,cycle) for level,a in enumerate(amps,1) for cycle in [1,2] for sign in [1,-1,0]]
    active,ha=run('fema461_installed',targets);control,hc=run('fema461_no_cyclic_deterioration',targets,control=True)
    refined,hr=run('fema461_half_step',targets,step_drift=.000025)
    compare=[]
    for a,b in zip(active['endpoints'],refined['endpoints']):
        compare.append(abs(a['actuator_force_kip']-b['actuator_force_kip']))
    checks['half_step_peak_endpoint_force_difference_kip']=max(compare,default=0)
    checks['half_step_relative_endpoint_force_difference']=max(compare,default=0)/active['max_abs_actuator_force_kip']
    checks['at_least_six_cycles_before_monotonic_first_yield']=amps[2]<min(first_yield)
    checks['monotonic_first_yield_drift_ratios']=first_yield
    for result in [mono_plus,mono_minus,active,control,refined]:
        assert result['completed'],result['name']+' did not complete'
        assert result['max_horizontal_residual_kip']<1e-5
        assert result['max_vertical_residual_kip']<1e-5
        assert result['max_joint_moment_residual_kip_in']<1e-4
    assert checks['half_step_relative_endpoint_force_difference']<.02
    meta={'case':'case_0014','source_joint_node':37,'source_joint_xyz_in':[234,600,162],'plane':'global Y-Z','members':MEMBERS,'joint_model':'rigid centerline node; no explicit panel spring','column_geometry':'PDelta','beam_geometry':'Linear','height_in':H,'beam_half_span_in':L/2,'column_half_height_in':H/2,'column_top_axial_load_kip':Ptop,'joint_vertical_load_kip':Pjoint,'column_below_reference_axial_kip':Plower,'elastic_segment_inertia_factor':Ifactor,'ke_policy':'Exact saved full-member spring Ke; elastic EI recovered from Ke L / (6n) and same (n+1)/n correction as V2; segment cut at midpoint.','beam_sign_policy':'Saved beam local-y axis is negative global X: reverse spring node order, retaining both branches and installed lambda unchanged. Column local y is positive global X.','protocol':{'name':'FEMA 461 §2.9.1 displacement-history diagnostic','source':'https://www.atcouncil.org/pdfs/FEMA461.pdf','printed_pages':[21,22,23,24],'base_steps':levels,'cycles_per_amplitude':2,'ratio':1.4,'reference_drift_ratio':DM,'extension_increment_ratio':.3*DM,'amplitude_drift_ratios':amps,'max_imposed_drift_ratio':MAX_DRIFT,'stop':'declared 8.4% fixture-drift limit or unresolved numerical failure; not an acceptance criterion'},'scope':'Idealized single-plane exterior connection fixture with source member IMK parameters; not an experimental fit or full-frame replay.','limitations':['Rigid centerline joint has no joint shear damage model.','Member hinge retains the existing aggregate flexure/bar-slip scope; no separate anchorage or bond-slip failure model.','Column strengths/backbones remain frozen at saved gravity-axial reference values, without evolving P-M interaction.','Axial preloads reproduce nominal upper/lower reference split approximately; original full-building beam gravity moments and distributed loads are not reproduced.','Transverse X beams, torsion and three-dimensional/slab kinematics are omitted; source beam slab strength and stiffness contributions remain embedded in saved parameters.','Midpoint pin/roller fixture boundaries idealize inflection points, not a prediction of the full building.','FEMA history is applied numerically; concrete cracking, spalling, bar buckling and rupture are not resolved.'], 'opensees_version':ops.version(),'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [INPUT/'fixture.json',INPUT/'global_parameters.json']},'design_sha256':fixture['design_sha256'],'checks':checks,'runs':{q['name']:{k:v for k,v in q.items() if k!='endpoints'} for q in [mono_plus,mono_minus,active,control,refined]}}
    (OUT/'test_definition.json').write_text(json.dumps(meta,indent=2));(OUT/'checks.json').write_text(json.dumps(checks,indent=2));print('All verification checks passed.',flush=True)
