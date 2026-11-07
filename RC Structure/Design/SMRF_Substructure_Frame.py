"""Sparse assembly of exact supplied floor interfaces and elastic columns.

The optional column PDelta force law follows OpenSees v3.8.0
PDeltaCrdTransf3d::getGlobalResistingForce and ElasticBeam3d::addLoad.
It uses the basic J-end axial force, including its uniform-load fixed-end
term, not an average axial force. LocalForce output excludes the geometric
shear term just as native ElasticBeam3d::getResponse(3) does; the separate
global resisting force includes it. Initial displacement/joint offsets,
nonuniform column loads and shear-flexible members are outside this API.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from Design.SMRF_Static_Substructure import _rhs,_json_hash,_sparse_hash,_symmetric

METHOD='supplied_floor_schur_interfaces_elastic_columns_native_pdelta_v1'
PRIMARY_SOURCES=[
 'https://raw.githubusercontent.com/OpenSees/OpenSees/v3.8.0/SRC/coordTransformation/PDeltaCrdTransf3d.cpp',
 'https://raw.githubusercontent.com/OpenSees/OpenSees/v3.8.0/SRC/element/elasticBeamColumn/ElasticBeam3d.cpp']


def _token(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class ElasticColumn:
    """Physical 12-DOF element; node order and vecxz are explicit inputs."""
    def __init__(self,definition,xyz_i,xyz_j):
        self.definition=dict(definition);self.id=definition['id']
        delta=np.asarray(xyz_j,dtype=float)-np.asarray(xyz_i,dtype=float)
        self.length=float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or self.length<=0:raise ValueError('Positive finite column length required')
        if any(definition.get(k) for k in ('joint_offset_i','joint_offset_j','initial_displacement','Avy','Avz')):
            raise ValueError('Offsets, initial displacement and shear-flexible columns require an explicit supported formulation')
        self.properties={k:float(definition[k]) for k in ('A','E','G','J','Iy','Iz')}
        if any(not np.isfinite(v) or v<=0 for v in self.properties.values()):raise ValueError('Positive finite elastic column properties required')
        x=delta/self.length;vec=np.asarray(definition['vecxz'],dtype=float)
        if vec.shape!=(3,) or not np.isfinite(vec).all():raise ValueError('Explicit finite three-component vecxz required')
        y=np.cross(vec,x);norm=np.linalg.norm(y)
        if norm<=1e-12*np.linalg.norm(vec):raise ValueError('Column vecxz is parallel to its local X axis')
        y/=norm;z=np.cross(x,y);self.rotation=np.array([x,y,z]);self.transform=sparse.block_diag([self.rotation]*4).toarray()
        length=self.length;p=self.properties
        basic=np.zeros((6,12));basic[0,[0,6]]=[-1.,1.];basic[5,[3,9]]=[-1.,1.]
        for row,rotation in ((1,5),(2,11)):
            basic[row,rotation]=1.;basic[row,[1,7]]=[1/length,-1/length]
        for row,rotation in ((3,4),(4,10)):
            basic[row,rotation]=1.;basic[row,[2,8]]=[-1/length,1/length]
        kb=np.zeros((6,6));kb[0,0]=p['E']*p['A']/length;kb[5,5]=p['G']*p['J']/length
        for name,rows,inertia in [('releasez',(1,2),'Iz'),('releasey',(3,4),'Iy')]:
            release=definition.get(name,0)
            if type(release) is not int or release not in (0,1,2,3):raise ValueError('End releases use native integer codes 0,1,2,3')
            block=({0:np.array([[4.,2.],[2.,4.]]),1:np.diag([0.,3.]),2:np.diag([3.,0.]),3:np.zeros((2,2))}[release])*p['E']*p[inertia]/length
            kb[np.ix_(rows,rows)]=block
        self.basic=basic;self.kb=kb;self.local_stiffness=basic.T@kb@basic
        self.stiffness=self.transform.T@self.local_stiffness@self.transform
        geo=np.zeros((12,12))
        for i,j in ((1,7),(2,8)):geo[np.ix_([i,j],[i,j])]=np.array([[1.,-1.],[-1.,1.]])/length
        self.geometric_unit=self.transform.T@geo@self.transform
        self.axial_derivative=self.transform.T@basic[0]*kb[0,0]

    def equivalent_load(self,axial_load_per_length):
        q=float(axial_load_per_length)
        if not np.isfinite(q):raise ValueError('Finite signed local-X column distributed load required')
        f=np.zeros(12);f[[0,6]]=q*self.length/2
        return self.transform.T@f

    def response(self,displacement,axial_load_per_length=0.,*,pdelta=True):
        u=np.asarray(displacement,dtype=float)
        if u.shape!=(12,) or not np.isfinite(u).all():raise ValueError('Finite twelve-component column displacement required')
        w=float(axial_load_per_length);equivalent=self.equivalent_load(w)
        ul=self.transform@u;basic=self.kb@(self.basic@ul);basic[0]-=w*self.length/2
        local=self.local_stiffness@ul-self.transform@equivalent
        n=float(basic[0]);geo=n*(self.geometric_unit@u) if pdelta else np.zeros(12)
        return {'local_force':local,'global_resisting_force':self.transform.T@local+geo,
                'basic_force':basic,'pdelta_basic_axial_tension_positive':n,
                'geometric_global_force':geo,'local_displacement':ul}


class SubstructureFrame:
    """One global K0, explicit constraints, multiple independently solved loads.

    nodes: [{id,xyz:[x,y,z]}]; each node has physical DOFs (id,1..6).
    floors: [{id,dof_ids,stiffness,loads,input_identity}]; all provided couplings
    are assembled unchanged. columns: [{id,i,j,A,E,G,J,Iy,Iz,vecxz,
    axial_load_kip_per_in:<scalar or per-case list>,releasey:0,releasez:0}].
    Signed uniform axial load is positive from I toward J. Supports list
    physical DOFs fixed to zero, or supply explicit transform/lift instead.
    A single RHS load or scalar column weight applies to every load case.
    """
    def __init__(self,nodes,floors,columns,*,input_identity,supports=(),loads=None,
                 kinematic_transform=None,reduced_dof_ids=None,support_lift=None):
        if not input_identity:raise ValueError('Global frame source/input identity is mandatory')
        self.nodes=list(nodes);self.dof_ids=[(n['id'],d) for n in nodes for d in range(1,7)]
        node_map={_token(n['id']):n for n in nodes}
        if len(node_map)!=len(nodes) or not nodes:raise ValueError('Unique nonempty physical node inventory required')
        index={_token(d):i for i,d in enumerate(self.dof_ids)};self._index=index;size=len(index)
        sources=[];local_loads=[];columns=list(columns);self.columns=[]
        row=[];col=[];data=[]
        def assemble(indices,matrix):
            mat=sparse.coo_matrix(matrix)
            if mat.shape!=(len(indices),len(indices)):raise ValueError('Interface stiffness/DOF dimensions differ')
            row.extend(indices[mat.row]);col.extend(indices[mat.col]);data.extend(mat.data)
        def indices_of(dofs):
            tokens=[_token(d) for d in dofs]
            if len(set(tokens))!=len(tokens):raise ValueError('Duplicate physical DOF in an interface')
            try:return np.asarray([index[d] for d in tokens],dtype=int)
            except KeyError as exc:raise ValueError('Floor/support physical DOF absent from frame') from exc
        for floor in floors:
            if not floor.get('input_identity'):raise ValueError('Each floor interface requires source/input identity')
            indices=indices_of(floor['dof_ids']);matrix,_=_symmetric(floor['stiffness'],1e-10)
            assemble(indices,matrix);f,_=_rhs(floor['loads'],len(indices),'Floor interface loads');local_loads.append((indices,f))
            sources.append({'id':floor['id'],'input_identity':floor['input_identity'],'dof_ids':floor['dof_ids'],
                            'stiffness_sha256':_sparse_hash(matrix),'loads_sha256':_json_hash(f.tolist())})
        if len({_token(c['id']) for c in columns})!=len(columns):raise ValueError('Duplicate physical column identity')
        for definition in columns:
            ni=node_map[_token(definition['i'])];nj=node_map[_token(definition['j'])]
            element=ElasticColumn(definition,ni['xyz'],nj['xyz'])
            indices=indices_of([(definition[end],d) for end in ('i','j') for d in range(1,7)])
            assemble(indices,element.stiffness)
            w=np.atleast_1d(np.asarray(definition.get('axial_load_kip_per_in',0.),dtype=float))
            if w.ndim!=1 or not np.isfinite(w).all():raise ValueError('Column distributed loads must be scalar or per-case vector')
            self.columns.append((element,indices,w))
        nodal=np.zeros((size,1)) if loads is None else _rhs(loads,size,'Global nodal loads')[0]
        self.case_count=max([nodal.shape[1],*[f.shape[1] for _,f in local_loads],*[len(w) for _,_,w in self.columns]])
        def repeated(array):
            if array.shape[1]==self.case_count:return array
            if array.shape[1]==1:return np.repeat(array,self.case_count,axis=1)
            raise ValueError('Load case counts differ between floor, nodal or column inputs')
        self.loads=repeated(nodal.copy())
        for indices,f in local_loads:self.loads[indices]+=repeated(f)
        for element,indices,w in self.columns:
            if len(w) not in (1,self.case_count):raise ValueError('Column load case count differs')
            for case in range(self.case_count):self.loads[indices,case]+=element.equivalent_load(w[0 if len(w)==1 else case])
        self.stiffness=sparse.coo_matrix((data,(row,col)),shape=(size,size)).tocsr();self.stiffness.sum_duplicates()
        if kinematic_transform is None:
            fixed=set(indices_of(supports));free=[i for i in range(size) if i not in fixed]
            if not free:raise ValueError('No free global frame DOFs remain')
            self.transform=sparse.csr_matrix((np.ones(len(free)),(free,np.arange(len(free)))),shape=(size,len(free)))
            self.reduced_dof_ids=[self.dof_ids[i] for i in free]
        else:
            if supports:raise ValueError('Explicit kinematic transform must already include support elimination')
            self.transform=sparse.csr_matrix(kinematic_transform,dtype=float)
            if self.transform.shape[0]!=size or not np.isfinite(self.transform.data).all():raise ValueError('Global kinematic transform shape/values differ')
            self.reduced_dof_ids=list(reduced_dof_ids or [])
            if len(self.reduced_dof_ids)!=self.transform.shape[1] or len({_token(d) for d in self.reduced_dof_ids})!=len(self.reduced_dof_ids):raise ValueError('Explicit reduced DOF inventory differs')
        self.lift=np.zeros_like(self.loads) if support_lift is None else repeated(_rhs(support_lift,size,'Global prescribed displacement lift')[0])
        self.reduced_stiffness=(self.transform.T@self.stiffness@self.transform).tocsc()
        try:self._linear_factor=splu(self.reduced_stiffness)
        except RuntimeError as exc:raise ValueError('Global linear frame is singular under supplied constraints') from exc
        self.identity={'method':METHOD,'input_identity':input_identity,'nodes':nodes,'columns':columns,'floor_interfaces':sources,
                       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'primary_sources':PRIMARY_SOURCES,
                       'stiffness_sha256':_sparse_hash(self.stiffness),'loads_sha256':_json_hash(self.loads.tolist()),
                       'kinematic_transform_sha256':_sparse_hash(self.transform),'support_lift_sha256':_json_hash(self.lift.tolist()),
                       'reduced_dof_ids':self.reduced_dof_ids}
        self.identity_sha256=_json_hash(self.identity)

    def _geometric(self,u,case,*,tangent):
        force=np.zeros(len(self.dof_ids));rows=[];cols=[];values=[]
        for element,indices,w in self.columns:
            ue=u[indices];weight=w[0 if len(w)==1 else case]
            n=float(element.axial_derivative@ue-weight*element.length/2)
            direction=element.geometric_unit@ue;force[indices]+=n*direction
            if tangent:
                matrix=n*element.geometric_unit+np.outer(direction,element.axial_derivative)
                rr,cc=np.nonzero(matrix);rows.extend(indices[rr]);cols.extend(indices[cc]);values.extend(matrix[rr,cc])
        matrix=sparse.coo_matrix((values,(rows,cols)),shape=self.stiffness.shape).tocsr() if tangent else None
        return force,matrix

    def solve(self,*,pdelta=True,max_iterations=40,absolute_tolerance=1e-8,relative_tolerance=1e-10):
        if type(max_iterations) is not int or max_iterations<1:raise ValueError('Positive bounded iteration count required')
        if min(absolute_tolerance,relative_tolerance)<=0 or not np.isfinite([absolute_tolerance,relative_tolerance]).all():raise ValueError('Positive finite equilibrium tolerances required')
        start=time.perf_counter();results=[]
        for case in range(self.case_count):
            f=self.loads[:,case];lift=self.lift[:,case]
            fr=np.asarray(self.transform.T@(f-self.stiffness@lift)).ravel()
            q=self._linear_factor.solve(fr);history=[];converged=False
            limit=absolute_tolerance+relative_tolerance*max(1.,float(np.max(np.abs(fr),initial=0.)))
            for iteration in range(max_iterations):
                u=np.asarray(self.transform@q).ravel()+lift
                geo,tangent=self._geometric(u,case,tangent=True) if pdelta else (np.zeros_like(u),None)
                residual=self.stiffness@u+geo-f;reduced=np.asarray(self.transform.T@residual).ravel()
                norm=float(np.max(np.abs(reduced),initial=0.));history.append(norm)
                if norm<=limit:converged=True;break
                jac=self.reduced_stiffness if not pdelta else (self.reduced_stiffness+self.transform.T@tangent@self.transform).tocsc()
                try:step=splu(jac).solve(-reduced)
                except RuntimeError:break
                if not np.isfinite(step).all():break
                # Backtracking on the actual native resisting-force residual;
                # never declare convergence from displacement size alone.
                accepted=False
                for power in range(12):
                    trial=q+step/(2**power);ut=np.asarray(self.transform@trial).ravel()+lift
                    gt=self._geometric(ut,case,tangent=False)[0] if pdelta else np.zeros_like(ut)
                    rt=np.asarray(self.transform.T@(self.stiffness@ut+gt-f)).ravel()
                    if float(np.max(np.abs(rt),initial=0.))<norm:q=trial;accepted=True;break
                if not accepted:break
            # Keep the saved forces/displacements at the same final iterate,
            # including an update performed on the last allowed iteration.
            u=np.asarray(self.transform@q).ravel()+lift
            geo=self._geometric(u,case,tangent=False)[0] if pdelta else np.zeros_like(u)
            residual=self.stiffness@u+geo-f;reduced=np.asarray(self.transform.T@residual).ravel()
            norm=float(np.max(np.abs(reduced),initial=0.));converged=norm<=limit
            columns={}
            for element,indices,w in self.columns:
                response=element.response(u[indices],w[0 if len(w)==1 else case],pdelta=pdelta)
                columns[str(element.id)]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in response.items()}
            results.append({'status':'equilibrated' if converged else 'unresolved','case_index':case,
                            'physical_displacements':u.tolist(),'reduced_displacements':q.tolist(),
                            'physical_residual_or_reaction':np.asarray(residual).tolist(),'maximum_free_equilibrium_residual':norm,
                            'equilibrium_tolerance':limit,'residual_history':history,'columns':columns,
                            'iterations':iteration+1})
        return {'method':METHOD,'identity_sha256':self.identity_sha256,'physical_dof_ids':self.dof_ids,
                'status':'equilibrated' if all(r['status']=='equilibrated' for r in results) else 'unresolved',
                'pdelta':bool(pdelta),'cases':results,'elapsed_seconds':time.perf_counter()-start,
                'scope':'Exact static solution of supplied linear floor interfaces with native elastic-column PDelta force law; stability/strength qualification remains separate.'}

    def verify_saved(self,record,*,pdelta=True,absolute_tolerance=1e-8,relative_tolerance=1e-10):
        """Recompute K@u, native column forces and constraints; no frame solve."""
        if (record.get('identity_sha256')!=self.identity_sha256 or record.get('pdelta') is not bool(pdelta)
            or list(map(tuple,record['physical_dof_ids']))!=list(map(tuple,self.dof_ids))
            or len(record['cases'])!=self.case_count):raise ValueError('Saved frame operator/load/DOF identity differs')
        audits=[]
        for case,saved in enumerate(record['cases']):
            q=np.asarray(saved['reduced_displacements'],float);u=np.asarray(saved['physical_displacements'],float)
            if q.shape!=(self.transform.shape[1],) or u.shape!=(len(self.dof_ids),) or not np.isfinite(q).all() or not np.isfinite(u).all():
                raise ValueError('Saved frame displacement dimensions/values differ')
            expected=np.asarray(self.transform@q).ravel()+self.lift[:,case]
            if not np.allclose(u,expected,rtol=1e-11,atol=1e-12):raise ValueError('Saved frame constraints do not reconstruct physical displacement')
            f=self.loads[:,case];geo=self._geometric(u,case,tangent=False)[0] if pdelta else np.zeros_like(u)
            residual=self.stiffness@u+geo-f;reduced=np.asarray(self.transform.T@residual).ravel()
            fr=np.asarray(self.transform.T@(f-self.stiffness@self.lift[:,case])).ravel()
            limit=absolute_tolerance+relative_tolerance*max(1.,float(np.max(abs(fr),initial=0.)))
            norm=float(np.max(abs(reduced),initial=0.))
            if norm>limit:raise ValueError('Saved frame fails reconstructed native force equilibrium')
            if not np.allclose(saved['physical_residual_or_reaction'],residual,rtol=1e-10,atol=1e-8):raise ValueError('Saved frame reaction differs from reconstructed resisting force')
            if set(saved['columns'])!={str(e.id) for e,_,_ in self.columns}:raise ValueError('Saved frame column inventory differs')
            for element,indices,w in self.columns:
                current=element.response(u[indices],w[0 if len(w)==1 else case],pdelta=pdelta);prior=saved['columns'][str(element.id)]
                if set(current)!=set(prior):raise ValueError('Saved column response fields differ')
                for key,value in current.items():
                    if not np.allclose(np.asarray(value,float),np.asarray(prior[key],float),rtol=1e-10,atol=1e-8):
                        raise ValueError('Saved column native force/deformation differs: '+key)
            audits.append(dict(case_index=case,maximum_free_equilibrium_residual=norm,equilibrium_tolerance=limit))
        return dict(method=METHOD+'/saved_replay',identity_sha256=self.identity_sha256,rows=audits,
            frame_solves=0,native_analyze_calls=0,equilibrium_verified=True)

