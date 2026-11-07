"""Independent full-matrix and native imposed floor displacement witnesses."""
import json
import math
import os
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import openseespy.opensees as ops
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem
from tests.test_coupled_floor_geometry import fixture,build


def imposed(ident='plus',gravity='DL',sign=1.,zero=False):
    return dict(id=ident,gravity_case_id=gravity,sign=sign,reference_xy_in=[17.,7.],
                levels=[dict(floor=k,design_in=dict(ux=0. if zero else .003*k,
                    uy=0. if zero else -.002*k,rz_rad=0. if zero else .00007*k)) for k in (1,2)])


def full_oracle(system,row):
    """Assemble and solve every uncondensed floor and column DOF directly."""
    definition=system.definition
    ids=[dof for floor in system.floors for dof in floor['assembly'].dof_ids]
    ids += [(n['id'],k) for n in definition['nodes'] if n['xyz'][2]==0 for k in range(1,7)]
    index={dof:i for i,dof in enumerate(ids)}
    k=sparse.lil_matrix((len(ids),len(ids)));f=np.zeros(len(ids))
    case=next(i for i,c in enumerate(system.cases) if c['id']==row['gravity_case_id'])
    for floor in system.floors:
        ix=np.array([index[i] for i in floor['assembly'].dof_ids])
        k[np.ix_(ix,ix)]+=floor['model'].stiffness;f[ix]+=floor['loads'][case]['loads']
    for element,_,weights in system.frame.columns:
        c=element.definition;ix=np.array([index[(c[end],j)] for end in ('i','j') for j in range(1,7)])
        k[np.ix_(ix,ix)]+=element.stiffness;f[ix]+=element.equivalent_load(weights[case])
    lift=np.zeros(len(ids));fixed=set();origin=row['reference_xy_in'];sign=row.get('sign',1.)
    design={r['floor']:r['design_in'] for r in row['levels']}
    for n in definition['nodes']:
        x,y,z=n['xyz']
        if z==0:
            fixed.update(index[n['id'],j] for j in range(1,7));continue
        v=design[int(z/definition['geometry']['story_h'])]
        values={1:sign*(v['ux']-(y-origin[1])*v['rz_rad']),
                2:sign*(v['uy']+(x-origin[0])*v['rz_rad']),6:sign*v['rz_rad']}
        for dof,value in values.items():
            ix=index[n['id'],dof];fixed.add(ix);lift[ix]=value
    free=np.array([i for i in range(len(ids)) if i not in fixed]);k=k.tocsr()
    rhs=f-k@lift;u=lift.copy();u[free]+=spsolve(k[free,:][:,free],rhs[free])
    return ids,u,np.asarray(k@u-f),float(np.max(np.abs((k@u-f)[free]),initial=0.))


class CoupledCompatibilityTests(unittest.TestCase):
    def tearDown(self):ops.wipe()

    def test_all_free_physical_dofs_and_reordered_gravity_rhs_match_full_oracle(self):
        system=build(fixture(),[dict(id='D',dead=1.,live=0.),dict(id='DL',dead=1.,live=1.)])
        rows=[imposed('plus','DL'),imposed('zero','D',zero=True),imposed('minus','DL',-1.)]
        result=system.solve_imposed_floor_displacements(rows,pdelta=False)
        self.assertEqual(result['status'],'equilibrated')
        self.assertFalse(result['qualification_asserted'])
        for j,row in enumerate(rows):
            ids,u,r,residual=full_oracle(system,row);index={d:i for i,d in enumerate(ids)}
            self.assertLess(residual,1e-9)
            response=result['frame']['cases'][j]
            for dof,value in zip(map(tuple,result['frame']['physical_dof_ids']),response['physical_displacements']):
                self.assertAlmostEqual(value,u[index[dof]],places=10)
            for dof,value in zip(map(tuple,result['frame']['physical_dof_ids']),response['physical_residual_or_reaction']):
                self.assertAlmostEqual(value,r[index[dof]],places=8)
            actual=dict(zip(map(tuple,result['frame']['physical_dof_ids']),response['physical_displacements']))
            case=next(i for i,c in enumerate(system.cases) if c['id']==row['gravity_case_id'])
            for floor in system.floors:
                boundary=np.array([actual[d] for d in floor['assembly'].boundary_dofs])
                recovered=floor['condensed'].recover(boundary,loads=floor['loads'][case]['loads'])
                np.testing.assert_allclose(recovered,[u[index[d]] for d in floor['assembly'].dof_ids],rtol=1e-9,atol=1e-10)

    def test_zero_lift_is_exact_constrained_gravity_and_signs_are_affine(self):
        system=build(fixture(),[dict(id='DL',dead=1.,live=1.)])
        rows=[imposed(),imposed('zero',zero=True),imposed('minus',sign=-1.)]
        result=system.solve_imposed_floor_displacements(rows,pdelta=False)
        u=np.array([r['physical_displacements'] for r in result['frame']['cases']])
        np.testing.assert_allclose((u[0]+u[2])/2,u[1],rtol=1e-10,atol=1e-11)
        ids,zero,_,_=full_oracle(system,rows[1]);index={d:i for i,d in enumerate(ids)}
        np.testing.assert_allclose(u[1],[zero[index[tuple(d)]] for d in result['frame']['physical_dof_ids']],atol=1e-10)
        # Zero imposed horizontal motion still restrains the joints. It is
        # not silently replaced by the horizontally free gravity solution.
        for dof,value in zip(result['frame']['physical_dof_ids'],u[1]):
            if dof[1] in (1,2,6):self.assertEqual(value,0.)

    def test_missing_or_duplicate_levels_and_wrong_gravity_are_refused(self):
        system=build(fixture(),[dict(id='DL',dead=1.,live=1.)])
        rows=[imposed(gravity='missing'),dict(imposed(),levels=imposed()['levels'][:1]),
              dict(imposed(),levels=[imposed()['levels'][0]]*2),imposed(sign=0.)]
        for row in rows:
            with self.assertRaises(ValueError):system.solve_imposed_floor_displacements([row],pdelta=False)


@unittest.skipUnless(os.environ.get('SEISFRAME_NATIVE_SUBSTRUCTURE_WITNESS')=='1','Exclusive native structural witness protocol required')
class NativeCompatibilityTests(unittest.TestCase):
    reports=[]
    solves=0

    @classmethod
    def tearDownClass(cls):
        path=os.environ.get('SEISFRAME_NATIVE_SUBSTRUCTURE_RESULT')
        if path:Path(path).write_text(json.dumps({'native_version':ops.version(),'cases':cls.reports,
            'structural_solves':cls.solves},indent=2)+'\n',encoding='utf-8')
        ops.wipe()

    def test_gross_two_floor_native_linear_and_pdelta_physical_actions(self):
        for pdelta in (False,True):
            with self.subTest(pdelta=pdelta):self.run_native(pdelta)

    def run_native(self,pdelta):
        d=fixture();row=imposed();gamma=d['gamma_kip_in3']
        system=CoupledFloorSystem(d,[dict(id='DL',dead=1.,live=1.)],source_identity='native prescribed floor witness',
            stiffness_basis='gross exact native witness only',subdivisions_per_bay=2,max_shells_per_floor=1024,
            max_coupled_dofs=1500,beam_bending_factor=1.,slab_bending_factor=1.,web_torsion_factor=1.,column_bending_factor=1.)
        result=system.solve_imposed_floor_displacements([row],pdelta=pdelta)
        self.assertEqual(result['status'],'equilibrated')
        frame=result['frame']['cases'][0]
        actual=dict(zip(map(tuple,result['frame']['physical_dof_ids']),frame['physical_displacements']))
        # Native geometry, loads, properties, joint patch ownership and
        # rigid offsets are rebuilt from physical inputs, without reading
        # assembly matrices, transforms or element load vectors.
        ops.wipe();ops.model('basic','-ndm',3,'-ndf',6)
        native={n['id']:1000+i for i,n in enumerate(d['nodes'])}
        for n in d['nodes']:
            ops.node(native[n['id']],*n['xyz'])
            if n['xyz'][2]==0:ops.fix(native[n['id']],1,1,1,1,1,1)
        ops.geomTransf('PDelta' if pdelta else 'Linear',1,1.,0.,0.)
        def torsion(b,h):
            a,b=max(b,h),min(b,h)
            return a*b**3/3*(1-192*b/(math.pi**5*a)*sum(math.tanh(n*math.pi*a/(2*b))/n**5 for n in range(1,160,2)))
        coltags={};column_weights={};eccentric={}
        for ci,c in enumerate(d['columns']):
            b,h,e=c['b_in'],c['h_in'],c['E_ksi'];tag=2000+ci;coltags[c['id']]=tag
            ops.element('elasticBeamColumn',tag,native[c['i']],native[c['j']],b*h,e,e/2.4,torsion(b,h),b*h**3/12,h*b**3/12,1)
            bw=2. if c['story']==1 else 2.5;x=c['grid_i']*20.;y=c['grid_j']*16.
            x0,x1=max(x-h/2,-bw/2),min(x+h/2,40+bw/2)
            y0,y1=max(y-b/2,-bw/2),min(y+b/2,16+bw/2)
            removed=(x1-x0)*(y1-y0)*gamma
            column_weights[tag]=-(b*h*10*gamma-removed)/10
            eccentric[native[c['j']]]=[0.,0.,0.,removed*((y0+y1)/2-y),-removed*((x0+x1)/2-x),0.]
        slabnodes={};slabtags={};webtags={};webends={};webnodes={};weboffsets={};shellloads=[];webweights={};count=3000
        independent_rotations=0
        for desc in d['floors']:
            level=desc['level'];z=level*10.;cb=4. if level==1 else 3.;bw=2. if level==1 else 2.5;bh=5. if level==1 else 6.
            xgrid=sorted({-bw/2,40+bw/2,0.,10.,20.,30.,40.,2.,18.,22.,38.,bw/2,20-bw/2,20+bw/2,40-bw/2})
            ygrid=sorted({-bw/2,16+bw/2,0.,8.,16.,cb/2,16-cb/2,bw/2,16-bw/2})
            nx=len(xgrid);owners={};tags={};positions={}
            for j,y in enumerate(ygrid):
                for i,x in enumerate(xgrid):
                    number=j*nx+i;positions[number]=(x,y,z)
                    inside=[s for s in desc['supports'] if abs(x-s['x'])<=2.+1e-9 and abs(y-s['y'])<=cb/2+1e-9]
                    exact=next((s for s in inside if abs(x-s['x'])<1e-9 and abs(y-s['y'])<1e-9),None)
                    if exact:tag=native[exact['id']]
                    else:
                        count+=1;tag=count;ops.node(tag,x,y,z)
                        if inside:ops.rigidLink('beam',native[inside[0]['id']],tag)
                    tags[number]=tag;owners[number]=native[inside[0]['id']] if inside else tag
                    slabnodes[level,number]=tag
            ops.section('ElasticMembranePlateSection',level,200.,.2,1.,0.)
            for j in range(len(ygrid)-1):
                for i in range(nx-1):
                    nodes=[j*nx+i,j*nx+i+1,(j+1)*nx+i+1,(j+1)*nx+i]
                    count+=1;tag=count;ops.element('ShellMITC4',tag,*[tags[n] for n in nodes],level)
                    area=(xgrid[i+1]-xgrid[i])*(ygrid[j+1]-ygrid[j]);force=-.19/144*area/4
                    slabtags[level,i,j]=(tag,nodes,force)
                    shellloads.extend((tags[n],force) for n in nodes)
            localweb={}
            for b in desc['beams']:
                grid=xgrid if b['axis']=='x' else ygrid;perp=ygrid if b['axis']=='x' else xgrid
                line=perp.index(b['line']);first=grid.index(b['start']);last=grid.index(b['end'])
                for station in range(first,last):
                    nodes=[line*nx+station,line*nx+station+1] if b['axis']=='x' else [station*nx+line,(station+1)*nx+line]
                    if owners[nodes[0]]==owners[nodes[1]]:continue
                    endtags=[];offsets=[]
                    for n in nodes:
                        x,y,_=positions[n]
                        if owners[n] in native.values():
                            tag=owners[n];origin=np.array(ops.nodeCoord(tag));offset=np.array([x,y,z-bh/2])-origin
                        else:
                            # The physical beam Rz is independent of numerical
                            # shell drilling. Use one beam node per clear-span
                            # station; share translations/Rx/Ry only. End offsets
                            # encode the centroid eccentricity without chained
                            # Transformation-handler multi-point constraints.
                            key=(b['id'],n)
                            if key not in localweb:
                                count+=1;tag=count;ops.node(tag,x,y,z)
                                ops.equalDOF(tags[n],tag,1,2,3,4,5);localweb[key]=tag;independent_rotations+=1
                            tag=localweb[key];offset=np.array([0.,0.,-bh/2])
                        endtags.append(tag);offsets.append(offset)
                    count+=1;tag=count;depth=bh-1
                    transform=100000+tag;ops.geomTransf('Linear',transform,0.,0.,1.,'-jntOffset',*offsets[0],*offsets[1])
                    ops.element('elasticBeamColumn',tag,*endtags,bw*depth,200.,200/2.4,
                                torsion(bw,depth),bw*depth**3/12,depth*bw**3/12,transform)
                    webtags[level,f'{b["id"]}:{station-first}']=tag;webweights[tag]=-gamma*bw*depth
                    webends[tag]=endtags;weboffsets[tag]=offsets
        ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
        for tag,force in shellloads:ops.load(tag,0.,0.,force,0.,0.,0.)
        for tag,weight in webweights.items():ops.eleLoad('-ele',tag,'-type','-beamUniform',0.,weight)
        for tag,weight in column_weights.items():ops.eleLoad('-ele',tag,'-type','-beamUniform',0.,0.,weight)
        for tag,wrench in eccentric.items():ops.load(tag,*wrench)
        for n in d['nodes']:
            x,y,z=n['xyz']
            if z==0:continue
            level=int(z/10);rz=.00007*level
            ops.sp(native[n['id']],1,.003*level-(y-7)*rz)
            ops.sp(native[n['id']],2,-.002*level+(x-17)*rz)
            ops.sp(native[n['id']],6,rz)
        ops.constraints('Transformation');ops.numberer('RCM');ops.system('UmfPack')
        ops.test('NormUnbalance',1e-9,40,0);ops.algorithm('Newton');ops.integrator('LoadControl',1.);ops.analysis('Static')
        self.assertEqual(ops.analyze(1),0)
        type(self).solves+=1
        max_u=max(abs(value-ops.nodeDisp(native[tag],dof)) for (tag,dof),value in actual.items())
        max_force=max(float(np.max(np.abs(np.asarray(ops.eleResponse(coltags[tag],'localForce'))-c['local_force'])))
                      for tag,c in frame['columns'].items())
        max_force=max(max_force,max(float(np.max(np.abs(np.asarray(ops.eleForce(coltags[tag]))-c['global_resisting_force'])))
                                   for tag,c in frame['columns'].items()))
        for floor in result['floors']:
            level=floor['level']
            for element in floor['responses'][0]['elements']:
                if element['kind']=='slab':
                    i,j=element['cell'];tag,nodes,force=slabtags[level,i,j]
                    native_force=np.asarray(ops.eleForce(tag));native_force[2::6]-=force
                    tags=[slabnodes[level,n] for n in nodes]
                    max_force=max(max_force,float(np.max(np.abs(np.asarray(ops.eleResponse(tag,'stresses')).reshape(4,8)-element['raw_resultants']))))
                else:
                    tag=webtags[level,element['id']];native_force=np.asarray(ops.eleForce(tag))
                    tags=webends[tag]
                    # Native end forces are work-conjugate at the offset's
                    # retained node. Transport them to the physical centroid.
                    for end,offset in enumerate(weboffsets[tag]):
                        native_force[6*end+3:6*end+6]-=np.cross(offset,native_force[6*end:6*end+3])
                max_force=max(max_force,float(np.max(np.abs(native_force-element['global_resisting_force']))))
                displacements=np.array([ops.nodeDisp(t) for t in tags])
                if element['kind']=='web':
                    for end,offset in enumerate(weboffsets[tag]):displacements[end,:3]+=np.cross(displacements[end,3:],offset)
                max_u=max(max_u,float(np.max(np.abs(displacements.ravel()-element['displacements']))))
        report=dict(pdelta=pdelta,maximum_physical_displacement_error=max_u,maximum_physical_force_error=max_force,
                    shells=len(slabtags),webs=len(webtags),columns=len(coltags),independent_beam_normal_rotations=independent_rotations,
                    connection='translations/Rx/Ry shared; interior beam Rz independent; actual centroid offsets')
        self.reports.append(report)
        self.assertGreater(independent_rotations,0)
        self.assertLess(max_u,1e-8,report);self.assertLess(max_force,1e-7,report)
        ops.wipe()


if __name__=='__main__':unittest.main()
