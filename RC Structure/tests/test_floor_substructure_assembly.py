"""Explicit opt-in native witness; requires the exclusive structural slot."""
import json,math,os,unittest
from pathlib import Path
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Floor_Substructure import FloorAssembly
from Design.SMRF_Static_Substructure import condense
from Design.SMRF_Substructure_Frame import SubstructureFrame


@unittest.skipUnless(os.environ.get('SEISFRAME_NATIVE_SUBSTRUCTURE_WITNESS')=='1','Exclusive native structural witness protocol required')
class FloorAssemblyNativeWitness(unittest.TestCase):
    reports=[]
    independent_normal_rotation=False
    retained_normal_rotation=False
    @classmethod
    def tearDownClass(cls):
        path=os.environ.get('SEISFRAME_NATIVE_SUBSTRUCTURE_RESULT')
        if path:Path(path).write_text(json.dumps({'native_version':ops.version(),'cases':cls.reports,'structural_solves':len(cls.reports)},indent=2)+'\n',encoding='utf-8')
        ops.wipe()

    def run_case(self,columns,pattern):
        ops.wipe();grid=np.array([0.,2.,10.,18.,20.,22.,30.,38.,40.]);nx=len(grid)
        supports=[dict(id=f'joint:{i}:{j}',x=20.*i,y=20.*j,b_in=4.,h_in=4.) for j in range(3) for i in range(3)]
        beams=[dict(id=f'{axis}:{line}:{span}',axis=axis,line=20.*line,start=20.*span,end=20.*(span+1),b_in=2.,h_in=5.,E_ksi=200.)
               for axis in ('x','y') for line in range(3) for span in range(2)]
        assembly=FloorAssembly(grid,grid,floor_id='test',thickness_in=1.,elastic_modulus_ksi=200.,supports=supports,beams=beams)
        if self.retained_normal_rotation:
            from Design.SMRF_Beam_Shell_Rotation_Dofs import retain_independent_beam_rotations
            retain_independent_beam_rotations(assembly)
        elif self.independent_normal_rotation:
            from Design.SMRF_Beam_Shell_Connection import connect_independent_beam_rotations
            connect_independent_beam_rotations(assembly)
        pressure=np.array([-.01*(1 if pattern==0 else (1. if x<20. else .35)) for y in grid[:-1] for x in grid[:-1]])
        wrenches={'joint:1:1':[.3,-.2,-.7,.02,-.03,.05]}
        loading=assembly.loads(pressure,joint_wrenches=wrenches)
        reduced=condense(assembly.stiffness,assembly.dof_ids,assembly.boundary_dofs,loading['loads'],input_identity={'scope':'native assembly witness','columns':columns,'pattern':pattern})
        column_rows=[];frame_nodes=[]
        if columns:
            for k,s in enumerate(supports):
                base=f'base:{k}';frame_nodes.extend([{'id':s['id'],'xyz':[s['x'],s['y'],0.]},{'id':base,'xyz':[s['x'],s['y'],-10.]}])
                column_rows.append(dict(id=f'column:{k}',i=base,j=s['id'],A=16.,E=200.,G=200/2.4,J=36.,Iy=64/3,Iz=64/3,vecxz=[1.,0.,0.],axial_load_kip_per_in=-.001))
            frame=SubstructureFrame(frame_nodes,[{'id':'floor','dof_ids':assembly.boundary_dofs,'stiffness':reduced.stiffness,'loads':reduced.loads,'input_identity':reduced.record()['identity_sha256']}],
                    column_rows,supports=[(f'base:{k}',d) for k in range(9) for d in range(1,7)],input_identity='floor plus independent columns')
            solved=frame.solve(pdelta=False);self.assertEqual(solved['status'],'equilibrated')
            by_dof={tuple(d):v for d,v in zip(solved['physical_dof_ids'],solved['cases'][0]['physical_displacements'])}
            boundary=np.array([by_dof[d] for d in assembly.boundary_dofs])
        else:boundary=np.zeros(len(assembly.boundary_dofs))
        displacement=reduced.recover(boundary);recovered=assembly.recover(displacement,loading)
        # Build the native model independently from geometric inputs. Do not
        # reuse assembly indices, transforms, matrices or element loads.
        ops.wipe();ops.model('basic','-ndm',3,'-ndf',6)
        masters={s['id']:1000+k for k,s in enumerate(supports)}
        for s in supports:ops.node(masters[s['id']],s['x'],s['y'],0.)
        native_nodes={};physical_owner={}
        for j,y in enumerate(grid):
            for i,x in enumerate(grid):
                node=j*nx+i;inside=[s for s in supports if abs(x-s['x'])<=2.+1e-9 and abs(y-s['y'])<=2.+1e-9]
                self.assertLessEqual(len(inside),1)
                exact=next((s for s in inside if abs(x-s['x'])<1e-9 and abs(y-s['y'])<1e-9),None)
                tag=masters[exact['id']] if exact else 2000+node
                native_nodes[node]=tag
                if exact:physical_owner[node]=tag
                else:
                    ops.node(tag,float(x),float(y),0.)
                    if inside:
                        physical_owner[node]=masters[inside[0]['id']];ops.rigidLink('beam',physical_owner[node],tag)
                    else:physical_owner[node]=tag
        if columns:
            ops.geomTransf('Linear',1,1.,0.,0.)
            for k,s in enumerate(supports):
                base=4000+k;ops.node(base,s['x'],s['y'],-10.);ops.fix(base,1,1,1,1,1,1)
                ops.element('elasticBeamColumn',5000+k,base,masters[s['id']],16.,200.,200/2.4,36.,64/3,64/3,1)
        else:
            for tag in masters.values():ops.fix(tag,1,1,1,1,1,1)
        ops.section('ElasticMembranePlateSection',1,200.,.2,1.,0.)
        shell_tags={};shell_loads={};ci=0
        for j in range(nx-1):
            for i in range(nx-1):
                nodes=[j*nx+i,j*nx+i+1,(j+1)*nx+i+1,(j+1)*nx+i]
                tag=6000+ci;ops.element('ShellMITC4',tag,*[native_nodes[n] for n in nodes],1)
                area=(grid[i+1]-grid[i])*(grid[j+1]-grid[j]);f=np.zeros(24);f[2::6]=pressure[ci]*area/4
                shell_tags[(i,j)]=tag;shell_loads[(i,j)]=(nodes,f);ci+=1
        ops.geomTransf('Linear',2,0.,0.,1.)
        web_nodes={};web_tags={};count=0
        # Independent Saint-Venant rectangle series, same explicitly declared
        # 80 odd terms; no assembly capacity/section result is reused.
        a,b=4.,2.;torsion=a*b**3/3*(1-192*b/(math.pi**5*a)*sum(math.tanh(n*math.pi*a/(2*b))/n**5 for n in range(1,160,2)))
        for beam in beams:
            line=int(np.flatnonzero(grid==beam['line'])[0]);first=int(np.flatnonzero(grid==beam['start'])[0]);last=int(np.flatnonzero(grid==beam['end'])[0])
            for station in range(first,last):
                nodes=[line*nx+station,line*nx+station+1] if beam['axis']=='x' else [station*nx+line,(station+1)*nx+line]
                if physical_owner[nodes[0]]==physical_owner[nodes[1]]:continue
                for node in nodes:
                    if node not in web_nodes:
                        tag=8000+node;web_nodes[node]=tag;x=grid[node%nx];y=grid[node//nx]
                        ops.node(tag,float(x),float(y),-2.5)
                        if self.independent_normal_rotation and physical_owner[node] not in masters.values():
                            carrier=10000+node;ops.node(carrier,float(x),float(y),-2.5)
                            ops.rigidLink('beam',physical_owner[node],carrier)
                            ops.equalDOF(carrier,tag,1,2,3,4,5)
                        else:ops.rigidLink('beam',physical_owner[node],tag)
                tag=9000+count;count+=1
                ops.element('elasticBeamColumn',tag,web_nodes[nodes[0]],web_nodes[nodes[1]],8.,200.,200/2.4,torsion,2*4**3/12,4*2**3/12,2)
                web_tags[f'{beam["id"]}:{station-first}']=tag
        ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
        for nodes,f in shell_loads.values():
            for j,node in enumerate(nodes):ops.load(native_nodes[node],*f[6*j:6*j+6])
        for tag in web_tags.values():ops.eleLoad('-ele',tag,'-type','-beamUniform',0.,-.150/1728*8.)
        for ident,wrench in wrenches.items():ops.load(masters[ident],*wrench)
        if columns:
            for k in range(9):ops.eleLoad('-ele',5000+k,'-type','-beamUniform',0.,0.,-.001)
        # Exact Lagrange equations handle the independent offset-carrier and
        # five-DOF couplings simultaneously; no Transformation MPC chaining.
        ops.constraints('Lagrange' if self.independent_normal_rotation else 'Transformation')
        ops.numberer('RCM');ops.system('UmfPack');ops.test('NormUnbalance',1e-9,40,0)
        ops.algorithm('Newton');ops.integrator('LoadControl',1.);ops.analysis('Static')
        self.assertEqual(ops.analyze(1),0)
        retained={str(s['id']):masters[s['id']] for s in supports}
        retained.update({f'test:slab:{node}':native_nodes[node] for node in range(nx*nx)})
        if self.retained_normal_rotation:
            retained.update({r['node_id']:web_nodes[r['physical_node']] for r in assembly.web_rotation_connection['internal_rotations']})
        native_u=np.array([ops.nodeDisp(retained[str(ident)],d) for ident,d in assembly.dof_ids])
        displacement_error=float(np.max(np.abs(native_u-displacement)))
        force_error=resultant_error=physical_u_error=0.
        for element in recovered['elements']:
            if element['kind']=='slab':
                key=tuple(element['cell']);tag=shell_tags[key];nodes,load=shell_loads[key]
                native_force=np.asarray(ops.eleForce(tag))-load
                resultant_error=max(resultant_error,float(np.max(np.abs(np.asarray(ops.eleResponse(tag,'stresses')).reshape(4,8)-element['raw_resultants']))))
                tags=[native_nodes[n] for n in nodes]
            else:
                tag=web_tags[element['id']];native_force=np.asarray(ops.eleForce(tag));tags=[web_nodes[n] for n in element['nodes']]
            force_error=max(force_error,float(np.max(np.abs(native_force-element['global_resisting_force']))))
            physical_u_error=max(physical_u_error,float(np.max(np.abs(np.concatenate([ops.nodeDisp(t) for t in tags])-element['displacements']))))
        column_error=0.
        if columns:
            for k in range(9):column_error=max(column_error,float(np.max(np.abs(np.asarray(ops.eleResponse(5000+k,'localForce'))-solved['cases'][0]['columns'][f'column:{k}']['local_force']))))
        report=dict(columns=columns,pattern=pattern,shells=64,web_segments=len(web_tags),
                    independent_beam_normal_rotation=self.independent_normal_rotation,
                    retained_internal_normal_rotations=self.retained_normal_rotation,
                    retained_dofs=len(assembly.dof_ids),boundary_dofs=len(assembly.boundary_dofs),
                    max_retained_displacement_error=displacement_error,max_physical_displacement_error=physical_u_error,
                    max_physical_force_error=force_error,max_shell_resultant_error=resultant_error,max_column_local_force_error=column_error)
        self.reports.append(report)
        self.assertLess(max(displacement_error,physical_u_error,force_error,resultant_error,column_error),1e-7,report)
        ops.wipe()

    def test_fixed_joint_floor_patch_and_patterned_loads(self):
        for pattern in (0,1):self.run_case(False,pattern)

    def test_floor_condensed_onto_actual_elastic_columns(self):
        for pattern in (0,1):self.run_case(True,pattern)


class IndependentRotationNativeWitness(FloorAssemblyNativeWitness):
    reports=[]
    independent_normal_rotation=True


class RetainedRotationNativeWitness(FloorAssemblyNativeWitness):
    reports=[]
    independent_normal_rotation=True
    retained_normal_rotation=True


if __name__=='__main__':unittest.main()
