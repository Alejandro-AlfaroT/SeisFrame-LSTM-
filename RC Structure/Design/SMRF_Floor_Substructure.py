"""Physical slab/downstand-web floor operators for static substructuring.

Linear native element response queries extract element operators; they are
not structural solves. No analysis is run on import. Native queries require
an empty OpenSees domain and clear only the scratch domain they create.
Units: kip, inch, ksi, radians; global axes X,Y,Z, slab normal +Z.
Raw ShellMITC4 section order is Nxx,Nyy,Nxy,mxx,myy,mxy,Qx,Qy.
Its physical sagging moments are the negatives of the raw m entries.

This module builds and audits mechanical operators. It does not assert code
qualification, mesh convergence or applicability to a candidate by itself.
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse
from scipy.sparse.linalg import splu
from Design.SMRF_Linear_Refinement import refine_gauged_rhs
from Design.SMRF_Stable_Sparse_Assembly import accurate_triplet_matrix

METHOD_VERSION = 'physical_slab_eccentric_web_static_substructure_v2_accurate_assembly'


def _positive(value, name):
    x=float(value)
    if not math.isfinite(x) or x<=0:
        raise ValueError(f'{name} must be positive and finite')
    return x


def _scratch():
    import openseespy.opensees as ops
    if ops.getNodeTags() or ops.getEleTags():
        raise RuntimeError('Native floor kernel requires an empty isolated OpenSees domain')
    ops.model('basic','-ndm',3,'-ndf',6)
    return ops


def _queries(ops, nodes, tag, count, capture_sections=False):
    k=np.zeros((count,count)); strains=np.zeros((32,count)); stresses=np.zeros((32,count))
    amplitude=1e-4
    previous=None
    for c in range(count):
        if previous is not None:
            ops.setNodeDisp(nodes[previous//6],previous%6+1,0.,'-commit')
        ops.setNodeDisp(nodes[c//6],c%6+1,amplitude,'-commit')
        # ShellMITC4 forms its resisting force from trial nodal displacements;
        # calling eleForce also updates its four section stress resultants.
        k[:,c]=np.asarray(ops.eleForce(tag),float)/amplitude
        if capture_sections:
            strains[:,c]=np.asarray(ops.eleResponse(tag,'strains'),float)/amplitude
            stresses[:,c]=np.asarray(ops.eleResponse(tag,'stresses'),float)/amplitude
        previous=c
    asym=np.max(np.abs(k-k.T))/max(1.,np.max(np.abs(k)))
    if not np.isfinite(k).all() or asym>1e-10:
        raise RuntimeError(f'Native linear element stiffness is nonfinite/asymmetric: {asym}')
    return k,strains,stresses


@lru_cache(maxsize=256)
def native_shell_kernel(dx, dy, thickness, poisson=.2):
    """Unit-E rectangular ShellMITC4 K and exact section recovery operators.

    One 24-column native linear query, with no analyze/eigen calls. Retains
    separate membrane, bending, transverse-shear and drilling energies so a
    cracked bending assumption cannot accidentally scale axial/shear stiffness.
    """
    dx,dy,t=(_positive(v,n) for v,n in ((dx,'dx'),(dy,'dy'),(thickness,'thickness')))
    nu=float(poisson)
    if not math.isfinite(nu) or not -1<nu<.5:
        raise ValueError('Invalid Poisson ratio')
    ops=_scratch()
    try:
        points=[(0.,0.,0.),(dx,0.,0.),(dx,dy,0.),(0.,dy,0.)]
        for n,p in enumerate(points,1):ops.node(n,*p)
        ops.section('ElasticMembranePlateSection',1,1.,nu,t,0.)
        ops.element('ShellMITC4',1,1,2,3,4,1)
        k,b,s=_queries(ops,[1,2,3,4],1,24,True)
        d=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])/(1-nu*nu)
        dm,db,dq=t*d,t**3/12*d,5/6*t/(2*(1+nu))*np.eye(2)
        parts={name:np.zeros((24,24)) for name in ('membrane','bending','shear')}
        expected=np.zeros_like(s)
        for gp in range(4):
            bm,bb,bq=b[8*gp:8*gp+3],b[8*gp+3:8*gp+6],b[8*gp+6:8*gp+8]
            for name,bi,di in (('membrane',bm,dm),('bending',bb,db),('shear',bq,dq)):
                parts[name]+=dx*dy/4*(bi.T@di@bi)
            expected[8*gp:8*gp+3]=dm@bm
            expected[8*gp+3:8*gp+6]=-db@bb
            expected[8*gp+6:8*gp+8]=dq@bq
        error=np.max(np.abs(s-expected))/max(1.,np.max(np.abs(s)))
        if error>1e-10:
            raise RuntimeError(f'Native shell constitutive convention mismatch: {error}')
        parts['drilling']=k-sum(parts.values())
        oop=np.array([6*n+d for n in range(4) for d in (2,3,4)])
        drill_oop=np.max(np.abs(parts['drilling'][oop,:]))/max(1.,np.max(np.abs(k)))
        if drill_oop>1e-10:
            raise RuntimeError('Unexplained native out-of-plane stiffness component')
        return {'K':k,'strain':b,'raw_resultant':s,'parts':parts,
                'points':np.array(points),'opensees_version':ops.version(),
                'constitutive_relative_error':float(error),
                'out_of_plane_drilling_relative_error':float(drill_oop)}
    finally:
        ops.wipe()


@lru_cache(maxsize=512)
def native_web_kernel(length, width, depth, poisson=.2, axis='x', torsion_constant=None):
    """Unit-E elastic downstand rectangle. Returns actual 12 global DOF K.

    Uses the Saint-Venant rectangular series, not Iy+Iz, for torsion unless
    an explicit positive torsion constant is supplied. No flange is included.
    """
    length,b,h=(_positive(v,n) for v,n in ((length,'length'),(width,'width'),(depth,'depth')))
    if axis not in ('x','y'):raise ValueError('Web axis must be x or y')
    nu=float(poisson)
    if not math.isfinite(nu) or not -1<nu<.5:raise ValueError('Invalid Poisson ratio')
    j=rectangle_torsion_constant(b,h) if torsion_constant is None else _positive(torsion_constant,'torsion_constant')
    ops=_scratch()
    try:
        ops.node(1,0.,0.,0.);ops.node(2,length if axis=='x' else 0.,length if axis=='y' else 0.,0.)
        ops.geomTransf('Linear',1,0.,0.,1.)
        ops.element('elasticBeamColumn',1,1,2,b*h,1.,1/(2*(1+nu)),j,b*h**3/12,h*b**3/12,1)
        k,_,_=_queries(ops,[1,2],1,12)
        along=0 if axis=='x' else 1
        transverse=1-along
        parts={}
        for name,dofs in (('axial',[along,6+along]),
                          ('inplane_bending',[transverse,5,6+transverse,11]),
                          ('vertical_bending',[2,4-along,8,10-along]),
                          ('torsion',[3+along,9+along])):
            part=np.zeros_like(k);part[np.ix_(dofs,dofs)]=k[np.ix_(dofs,dofs)]
            parts[name]=part
        if not np.allclose(sum(parts.values()),k,rtol=1e-11,atol=1e-12):
            raise RuntimeError('Native web stiffness component inventory is incomplete')
        return {'K':k,'area_in2':b*h,'Iy_in4':b*h**3/12,'Iz_in4':h*b**3/12,
                'J_in4':j,'opensees_version':ops.version(),'parts':parts}
    finally:
        ops.wipe()


def rectangle_torsion_constant(width, depth, terms=80):
    """Saint-Venant odd Fourier series for a solid rectangle (positive J)."""
    a,b=sorted((_positive(width,'width'),_positive(depth,'depth')),reverse=True)
    if not isinstance(terms,int) or terms<8:raise ValueError('At least eight series terms required')
    correction=sum(math.tanh(n*math.pi*a/(2*b))/n**5 for n in range(1,2*terms,2))
    return a*b**3/3*(1-192*b/(math.pi**5*a)*correction)


def rigid_arm_matrix(offset_xyz):
    """u_slave=T u_retained, preserving all six DOFs and virtual work."""
    r=np.asarray(offset_xyz,float)
    if r.shape!=(3,) or not np.isfinite(r).all():raise ValueError('Finite 3D rigid-arm offset required')
    x,y,z=r
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    t=np.eye(6);t[:3,3:]=-skew
    return t


class FloorAssembly:
    """Disjoint slab and eccentric webs, with actual rigid column footprints.

    All grid coordinates, column center/face coordinates and beam endpoints
    are explicit inputs. support rows: id,x,y,b_in,h_in (h along X). Beam
    rows: id,axis,line,start,end,b_in,h_in,E_ksi. A beam is one physical span.
    Rectangular patches are clipped by the declared slab footprint. Nodes
    in a patch and web nodes below it link directly to its center: no MPC
    chain or penalty enters this assembly. Geometry does not assert that a
    rigid joint region is applicable; the joint verification remains required.
    """
    def __init__(self,x_in,y_in,*,floor_id,thickness_in,elastic_modulus_ksi,
                 supports,beams,poisson=.2,max_shells=8192,
                 concrete_weight_kip_in3=.150/1728):
        self.x=np.asarray(x_in,float);self.y=np.asarray(y_in,float)
        for row in (self.x,self.y):
            if row.ndim!=1 or len(row)<2 or not np.isfinite(row).all() or np.any(np.diff(row)<=1e-8):
                raise ValueError('Explicit increasing finite rectangular floor grid required')
        self.id=floor_id;self.thickness=_positive(thickness_in,'thickness')
        self.ec=_positive(elastic_modulus_ksi,'elastic_modulus');self.nu=float(poisson)
        self.gamma=_positive(concrete_weight_kip_in3,'concrete_weight')
        nx,ny=len(self.x),len(self.y)
        if type(max_shells) is not int or (nx-1)*(ny-1)>max_shells:
            raise ValueError('Declared floor shell budget exceeded')
        self.points=np.array([(x,y,0.) for y in self.y for x in self.x])
        self.supports=[dict(s) for s in supports];self.beams=[dict(b) for b in beams]
        if not self.supports or len({str(s['id']) for s in self.supports})!=len(self.supports):
            raise ValueError('Unique physical support IDs required')
        if len({str(b['id']) for b in self.beams})!=len(self.beams):
            raise ValueError('Unique physical beam IDs required')
        def coordinate(values,value):
            matches=np.flatnonzero(np.abs(values-float(value))<1e-8)
            if len(matches)!=1:raise ValueError('Every physical support/beam coordinate must be present in the mesh')
            return int(matches[0])
        owner=np.arange(len(self.points));joint_nodes={}
        patch_owner={}
        for s in self.supports:
            xc,yc=float(s['x']),float(s['y']);bx=_positive(s['h_in'],'support h');by=_positive(s['b_in'],'support b')
            center=coordinate(self.y,yc)*nx+coordinate(self.x,xc)
            if center in joint_nodes:raise ValueError('Two supports occupy the same physical center')
            joint_nodes[center]=s['id']
            for values,value in ((self.x,xc-bx/2),(self.x,xc+bx/2),(self.y,yc-by/2),(self.y,yc+by/2)):
                if values[0]<value<values[-1]:coordinate(values,value)
            inside=np.flatnonzero((abs(self.points[:,0]-xc)<=bx/2+1e-8)&(abs(self.points[:,1]-yc)<=by/2+1e-8))
            for node in inside:
                if int(node) in patch_owner:raise ValueError('Physical column footprints overlap or touch')
                patch_owner[int(node)]=center;owner[node]=center
        retained=sorted(set(owner));self.retained_nodes=retained;self.owner=owner
        inverse={node:i for i,node in enumerate(retained)}
        self.node_ids=[joint_nodes.get(node,f'{floor_id}:slab:{node}') for node in retained]
        self.dof_ids=[(ident,d) for ident in self.node_ids for d in range(1,7)]
        self.boundary_dofs=[(s['id'],d) for s in self.supports for d in range(1,7)]
        self.node_positions={ident:self.points[node].tolist() for ident,node in zip(self.node_ids,retained)}
        self.joint_centers=joint_nodes
        maps=[]
        for node in range(len(self.points)):
            own=owner[node]
            maps.append((np.arange(6*inverse[own],6*inverse[own]+6),rigid_arm_matrix(self.points[node]-self.points[own])))
        self.node_maps=maps
        self.elements=[]
        accum={name:[[],[],[]] for name in ('membrane','slab_bending','shear','web_bending','web_torsion')}
        def append_element(row,physical_nodes,local_points,k,parts):
            indices=np.concatenate([maps[n][0] for n in physical_nodes])
            # A web node is rigidly offset from the retained joint/slab node,
            # including the horizontal arm of a finite support patch.
            transform=sparse.block_diag([rigid_arm_matrix(p-self.points[owner[n]])
                                         for n,p in zip(physical_nodes,local_points)]).toarray()
            row.update(nodes=list(physical_nodes),points=np.asarray(local_points),indices=indices,
                       transform=transform,K=k,parts=parts)
            self.elements.append(row)
            for name,part in parts.items():
                mat=transform.T@part@transform
                rr,cc=np.nonzero(mat)
                accum[name][0].append(indices[rr]);accum[name][1].append(indices[cc]);accum[name][2].append(mat[rr,cc])
        for jj in range(ny-1):
            for ii in range(nx-1):
                nodes=[jj*nx+ii,jj*nx+ii+1,(jj+1)*nx+ii+1,(jj+1)*nx+ii]
                dx,dy=self.x[ii+1]-self.x[ii],self.y[jj+1]-self.y[jj]
                native=native_shell_kernel(float(dx),float(dy),self.thickness,self.nu)
                parts={'membrane':self.ec*(native['parts']['membrane']+native['parts']['drilling']),
                       'slab_bending':self.ec*native['parts']['bending'],'shear':self.ec*native['parts']['shear']}
                append_element({'kind':'slab','id':f'{floor_id}:cell:{ii}:{jj}','cell':[ii,jj],
                                'area_in2':float(dx*dy),'native':native},nodes,self.points[nodes],self.ec*native['K'],parts)
        seen_segments=set()
        for beam in self.beams:
            axis=beam['axis']
            if axis not in ('x','y'):raise ValueError('Beam axis must be x/y')
            along,across=(self.x,self.y) if axis=='x' else (self.y,self.x)
            line=coordinate(across,beam['line']);start=coordinate(along,beam['start']);end=coordinate(along,beam['end'])
            if end<=start:raise ValueError('Beam span must follow its positive physical axis')
            b=_positive(beam['b_in'],'beam width');h=_positive(beam['h_in'],'beam depth');e=_positive(beam['E_ksi'],'beam E')
            if h<=self.thickness:raise ValueError('Positive disjoint downstand web depth required')
            for at in range(start,end):
                nodes=([line*nx+at,line*nx+at+1] if axis=='x' else [at*nx+line,(at+1)*nx+line])
                if tuple(nodes) in seen_segments:raise ValueError('Overlapping physical beam segments')
                seen_segments.add(tuple(nodes))
                if owner[nodes[0]]==owner[nodes[1]]:
                    continue # this volume belongs to the rigid column joint
                length=float(along[at+1]-along[at]);offset=-h/2
                points=self.points[nodes].copy();points[:,2]=offset
                native=native_web_kernel(length,b,h-self.thickness,self.nu,axis)
                parts={'membrane':e*(native['parts']['axial']+native['parts']['inplane_bending']),
                       'web_bending':e*native['parts']['vertical_bending'],
                       'web_torsion':e*native['parts']['torsion']}
                append_element({'kind':'web','id':f'{beam["id"]}:{at-start}','member_id':beam['id'],
                                'axis':axis,'length_in':length,'section':dict(beam),'native':native,
                                'station_in':[float(along[at]-beam['start']),float(along[at+1]-beam['start'])]},
                               nodes,points,e*native['K'],parts)
        shape=(len(self.dof_ids),len(self.dof_ids));self.parts={}
        for name,(rr,cc,vv) in accum.items():
            self.parts[name]=(accurate_triplet_matrix(np.concatenate(vv),np.concatenate(rr),np.concatenate(cc),shape=shape)
                              if vv else csr_matrix(shape))
        self.stiffness=sum(self.parts.values(),csr_matrix(shape))
        self.membrane_dofs=np.array([6*n+d for n in range(len(retained)) for d in (0,1,5)])
        self.bending_dofs=np.array([6*n+d for n in range(len(retained)) for d in (2,3,4)])

    def loads(self,pressure_kip_in2,*,beam_weight_factor=1.,joint_wrenches=None):
        """Consistent signed gravity loads. Pressure positive upward.

        pressure is one scalar or exactly one value per cell in row-major Y/X
        order. Beam self-weight exists only outside actual support faces.
        All optional joint wrenches use actual support center/slab-plane axes.
        Returns a reusable load record, including each physical element load.
        """
        cells=[e for e in self.elements if e['kind']=='slab']
        pressure=np.asarray(pressure_kip_in2,float)
        if pressure.ndim==0:pressure=np.full(len(cells),float(pressure))
        if pressure.shape!=(len(cells),) or not np.isfinite(pressure).all():raise ValueError('One finite signed pressure per cell required')
        weight=float(beam_weight_factor)
        if not math.isfinite(weight):raise ValueError('Finite beam dead-load factor required')
        f=np.zeros(len(self.dof_ids));element_loads={};ci=0
        for row in self.elements:
            local=np.zeros(6*len(row['nodes']))
            if row['kind']=='slab':
                local[2::6]=pressure[ci]*row['area_in2']/4;ci+=1
            else:
                q=-weight*self.gamma*row['native']['area_in2'];length=row['length_in']
                local[[2,8]]=q*length/2
                rot=4 if row['axis']=='x' else 3
                sign=-1 if row['axis']=='x' else 1
                local[rot]=sign*q*length**2/12;local[rot+6]=-local[rot]
            element_loads[row['id']]=local
            np.add.at(f,row['indices'],row['transform'].T@local)
        locations={str(n):i for i,n in enumerate(self.node_ids)}
        supports={str(s['id']) for s in self.supports}
        for ident,wrench in (joint_wrenches or {}).items():
            if str(ident) not in supports:raise ValueError('Joint wrench has no actual support center')
            w=np.asarray(wrench,float)
            if w.shape!=(6,) or not np.isfinite(w).all():raise ValueError('Finite six-component joint wrench required')
            start=6*locations[str(ident)];f[start:start+6]+=w
        return {'loads':f,'element_loads':element_loads,'pressure_kip_in2':pressure,'beam_weight_factor':weight,
                'joint_wrenches':joint_wrenches or {}}

    def recover(self,displacements,load_record):
        u=np.asarray(displacements,float)
        if u.shape!=(len(self.dof_ids),) or not np.isfinite(u).all():raise ValueError('Finite physical floor displacement vector required')
        output=[];assembled=np.zeros_like(u)
        for row in self.elements:
            local=row['transform']@u[row['indices']]
            force=row['K']@local-load_record['element_loads'][row['id']]
            np.add.at(assembled,row['indices'],row['transform'].T@force)
            result={k:row[k] for k in ('id','kind','nodes')}
            result.update(positions_in=row['points'].tolist(),displacements=local.tolist(),
                          global_resisting_force=force.tolist())
            if row['kind']=='slab':
                result.update(cell=row['cell'],area_in2=row['area_in2'],
                              raw_resultants=(self.ec*row['native']['raw_resultant']@local).reshape(4,8).tolist())
            else:
                result.update(member_id=row['member_id'],axis=row['axis'],station_in=row['station_in'],section=row['section'])
            output.append(result)
        # The raw assembled element resistance is before explicitly supplied
        # joint wrenches. They remain external loads, never extra capacity.
        return {'method':METHOD_VERSION,'elements':output,'assembled_element_resistance':assembled.tolist(),
                'dof_ids':self.dof_ids,'boundary_dofs':self.boundary_dofs,
                'qualification_asserted':False}


class GeneralizedFloorStiffness:
    """Explicit cracked bending energies with gross axial/shear stiffness.

    Partition physical floor DOFs into in-plane m and bending b. The membrane
    energy is 1/2 [m,b]'[[A,B],[B',Dm]][m,b]. Its exact residual bending
    energy C= Dm-B'A+ B contains the self-equilibrated eccentric-web/slab
    couple. Preserve A,B; scale C and web own bending by the stated beam
    factor, slab own bending by its independently supplied factors, shear
    by 1, and web torsion by its separately supplied factor.

    A+ here is a three-rigid-mode gauge solve, not a repair of an unexplained
    singular matrix. Rigid compatibility and backward errors are checked.
    Physical element-force recovery uses the same energy decomposition.
    The caller must establish the engineering basis of each supplied factor.
    """
    def __init__(self,floor,*,beam_bending_factor,slab_bending_factors,
                 web_torsion_factor,stiffness_basis,input_identity,
                 solve_tolerance=1e-9,maximum_coupled_dofs=2400,slab_bending_matrices=None):
        if not stiffness_basis or not input_identity:raise ValueError('Explicit stiffness basis and input identity required')
        self.floor=floor;self.beta=_positive(beam_bending_factor,'beam_bending_factor')
        webs=[e for e in floor.elements if e['kind']=='web']
        if isinstance(web_torsion_factor,dict):
            if set(web_torsion_factor)!={e['member_id'] for e in webs}:
                raise ValueError('Every actual clear-span beam needs exactly one declared torsional stiffness factor')
            self.torsion={k:float(v) for k,v in web_torsion_factor.items()}
        else:
            value=float(web_torsion_factor)
            self.torsion={e['member_id']:value for e in webs}
        if any(not np.isfinite(v) or not 0<=v<=1 for v in self.torsion.values()):
            raise ValueError('Explicit beam torsional stiffness factors must be within[0,1]')
        cells=[e for e in floor.elements if e['kind']=='slab']
        self._slab_K={};self._slab_D={}
        if slab_bending_matrices is None:
            alpha=np.asarray(slab_bending_factors,float)
            if alpha.ndim==0:alpha=np.full(len(cells),float(alpha))
            if alpha.shape!=(len(cells),) or not np.isfinite(alpha).all() or np.any(alpha<=0):
                raise ValueError('One explicit positive bending factor per physical slab cell required')
            if float(alpha.max())>1:raise ValueError('Independent scalar factors cannot enhance gross stiffness')
            self.alpha={e['id']:float(a) for e,a in zip(cells,alpha)}
            for row in cells:self._slab_K[row['id']]=self.alpha[row['id']]*row['parts']['slab_bending']
        else:
            if slab_bending_factors is not None or set(slab_bending_matrices)!={r['id'] for r in cells}:
                raise ValueError('Complete explicit slab bending matrices replace, not supplement, scalar factors')
            self.alpha=None;alpha=None
            for row in cells:
                d=np.asarray(slab_bending_matrices[row['id']],float)
                if d.shape==(3,3):d=np.repeat(d[None,:,:],4,axis=0)
                if d.shape!=(4,3,3) or not np.isfinite(d).all():raise ValueError('Four actual nominal3x3 slab bending matrices required')
                for di in d:
                    scale=max(1.,float(np.max(abs(di))))
                    if np.max(abs(di-di.T))>1e-9*scale or np.linalg.eigvalsh((di+di.T)/2).min() < -1e-9*scale:
                        raise ValueError('Actual slab bending matrices require symmetric nonnegative energy')
                d=(d+d.transpose(0,2,1))/2;self._slab_D[row['id']]=d
                b=-row['native']['strain'].reshape(4,8,24)[:,3:6,:]
                self._slab_K[row['id']]=sum(bi.T@di@bi for bi,di in zip(b,d))*row['area_in2']/4
        if self.beta>1:raise ValueError('Independent beam bending factor cannot enhance gross stiffness')
        self.m,self.b=floor.membrane_dofs,floor.bending_dofs
        membrane=floor.parts['membrane'];a=membrane[self.m,:][:,self.m].tocsc()
        bmat=membrane[self.m,:][:,self.b].tocsr();dm=membrane[self.b,:][:,self.b].tocsr()
        coords=np.asarray([floor.node_positions[n] for n in floor.node_ids])
        first=int(np.lexsort((coords[:,1],coords[:,0]))[0])
        far=int(np.argmax(coords[:,0]))
        if abs(coords[far,0]-coords[first,0])<=1e-8:raise ValueError('A finite rectangular plan is required for in-plane gauges')
        fixed={3*first,3*first+1,3*far+1}
        free=np.array([i for i in range(len(self.m)) if i not in fixed])
        rigid=np.zeros((len(self.m),3))
        original_membrane_count=3*len(coords)
        rigid[:original_membrane_count:3,0]=1;rigid[1:original_membrane_count:3,1]=1
        rigid[:original_membrane_count:3,2]=-coords[:,1];rigid[1:original_membrane_count:3,2]=coords[:,0];rigid[2:original_membrane_count:3,2]=1
        if len(self.m)>original_membrane_count:
            extra=self.m[original_membrane_count:]
            declared=getattr(floor,'web_rotation_connection',{}).get('internal_rotation_dof_indices')
            if extra.tolist()!=declared or any(floor.dof_ids[i][1]!=6 for i in extra):raise ValueError('Declared internal beam normal rotations required')
            rigid[original_membrane_count:,2]=1.
        scale=max(1.,float(np.max(np.abs(membrane.data),initial=0.)))
        rigid_error=float(np.max(np.abs(a@rigid),initial=0.))/scale
        compat_error=float(np.max(np.abs(rigid.T@bmat),initial=0.))/scale
        if max(rigid_error,compat_error)>1e-8:raise ValueError('Membrane rigid modes or eccentric coupling do not balance')
        selected=np.flatnonzero(np.asarray(abs(bmat).sum(axis=0)).ravel()>0)
        if len(selected)>maximum_coupled_dofs:raise ValueError('Declared eccentric-coupling solve budget exceeded')
        try:factor=splu(a[free,:][:,free])
        except RuntimeError as exc:raise ValueError('Membrane has mechanisms beyond its three planar rigid modes') from exc
        # Form B' A+ B in bounded RHS blocks, never store the full dense A+ B.
        # The same factor applies A+ to each actual physical motion in fields().
        coupling=bmat[:,selected].tocsr();coupling_free=coupling[free,:]
        correction=np.empty((len(selected),len(selected)));backward=0.;refinements=[]
        full_free=a[:,free].tocsr()
        for start in range(0,len(selected),64):
            inds=selected[start:start+64];complete_rhs=bmat[:,inds].toarray();rhs=complete_rhs[free,:]
            answer,audit=refine_gauged_rhs(full_free,complete_rhs,free,factor.solve(rhs),factor.solve)
            refinements.append(dict(first_column=start,last_column=start+len(inds),**audit))
            correction[:,start:start+len(inds)]=np.asarray(coupling_free.T@answer)
            error=full_free@answer-complete_rhs
            norm=max(1.,float(np.max(np.abs(rhs),initial=0.)))
            backward=max(backward,float(np.max(np.abs(error),initial=0.))/norm)
        if backward>solve_tolerance:raise ValueError('Membrane relaxation equilibrium tolerance exceeded')
        self._selected=selected;self._relaxation_factor=factor
        self._relaxation_operator=full_free;self._relaxation_rhs=coupling
        self._relaxation_free=free;self._relaxation_tolerance=solve_tolerance
        rr,cc=np.meshgrid(selected,selected,indexing='ij')
        schur=sparse.coo_matrix((correction.ravel(),(rr.ravel(),cc.ravel())),shape=dm.shape).tocsr()
        # C is the Schur complement of the membrane/axial energy alone;
        # no plate shear or own bending is subtracted to manufacture it.
        self.couple_stiffness=dm-schur
        k=membrane.copy().tolil()
        change=((self.beta-1)*self.couple_stiffness).tocoo()
        k=k.tocsr()+coo_matrix((change.data,(self.b[change.row],self.b[change.col])),shape=k.shape).tocsr()
        k+=self.beta*floor.parts['web_bending']+floor.parts['shear']
        rows=[];cols=[];values=[]
        for row in cells:
            block=row['transform'].T@self._slab_K[row['id']]@row['transform']
            r,c=np.nonzero(block);rows.append(row['indices'][r]);cols.append(row['indices'][c]);values.append(block[r,c])
        for row in webs:
            block=self.torsion[row['member_id']]*row['transform'].T@row['parts']['web_torsion']@row['transform']
            r,c=np.nonzero(block);rows.append(row['indices'][r]);cols.append(row['indices'][c]);values.append(block[r,c])
        k+=accurate_triplet_matrix(np.concatenate(values),np.concatenate(rows),np.concatenate(cols),shape=k.shape)
        self.stiffness=k.tocsr()
        self.metadata={'method':METHOD_VERSION+'/generalized_cracked_energy_v6_bounded_influence_storage','input_identity':input_identity,
                       'stiffness_basis':stiffness_basis,'beam_bending_factor':self.beta,
                       'slab_bending_factors':None if alpha is None else alpha.tolist(),
                       'slab_bending_matrices':{k:v.tolist() for k,v in self._slab_D.items()},
                       'web_torsion_factor':self.torsion,
                       'axial_factor':1.,'transverse_shear_factor':1.,
                       'membrane_rigid_mode_error':rigid_error,'eccentric_coupling_balance_error':compat_error,
                       'relaxation_backward_error':backward,'coupled_bending_dofs':len(selected),
                       'relaxation_refinement':refinements,
                       'relaxation_storage_method':'blocked_schur_and_direct_physical_rhs_v1',
                       'dense_influence_storage_avoided_bytes':len(self.m)*len(selected)*8,
                       'membrane_factor_nnz':int(factor.nnz),
                       'gauge_indices':sorted(fixed),'qualification_asserted':False}
        if hasattr(floor,'web_rotation_connection'):
            self.metadata['beam_shell_connection']=floor.web_rotation_connection
            if floor.web_rotation_connection.get('internal_rotation_dof_count'):
                self.metadata['method']=METHOD_VERSION+'/generalized_cracked_energy_v7_retained_internal_normal_rotations'

    def fields(self,displacements):
        u=np.asarray(displacements,float)
        if u.shape!=(len(self.floor.dof_ids),) or not np.isfinite(u).all():raise ValueError('Finite floor displacement vector required')
        rhs=np.asarray(self._relaxation_rhs@u[self.b[self._selected]]).reshape(-1,1)
        free=self._relaxation_free
        answer,audit=refine_gauged_rhs(self._relaxation_operator,rhs,free,
            self._relaxation_factor.solve(rhs[free,:]),self._relaxation_factor.solve)
        norm=max(1.,float(np.max(abs(rhs[free,:]),initial=0.)))
        residual=float(audit['final_absolute_residuals'][0])/norm
        if residual>self._relaxation_tolerance:
            raise ValueError('Physical membrane relaxation equilibrium tolerance exceeded')
        shift=np.zeros(len(self.m));shift[free]=answer[:,0]
        self._last_relaxation_audit=dict(refinement=audit,relative_full_equation_residual=residual,
            tolerance=self._relaxation_tolerance,dense_influence_matrix_stored=False)
        membrane=np.zeros_like(u);membrane[self.m]=u[self.m]+shift
        bending=u-membrane
        return membrane,bending

    def recover(self,displacements,load_record):
        floor=self.floor;supplied=np.asarray(displacements,float)
        # All six infinitesimal rigid motions are exact null modes of this
        # free floor. Fit them to nodal translations and remove them before
        # large local/assembled products. This preserves deformation and the
        # force-balance tolerance, including under accidental floor rotation.
        # Supplied physical displacements remain unchanged in the record.
        if supplied.shape!=(len(floor.dof_ids),) or not np.isfinite(supplied).all():
            raise ValueError('Finite floor displacement vector required')
        original_dof_count=6*len(floor.node_ids)
        motion=supplied[:original_dof_count].reshape(-1,6)
        coords=np.asarray([floor.node_positions[n] for n in floor.node_ids],float)
        reference=coords.mean(axis=0);coords=coords-reference
        translation=motion[:,:3].mean(axis=0);deformed=motion[:,:3]-translation
        x,y=coords[:,0],coords[:,1]
        rotation_z=float(np.sum(x*deformed[:,1]-y*deformed[:,0])/np.sum(x*x+y*y))
        plane=np.column_stack((y,-x))
        rotation_x,rotation_y=np.linalg.solve(plane.T@plane,plane.T@deformed[:,2])
        rotation=np.array([rotation_x,rotation_y,rotation_z])
        u=motion.copy();u[:,:3]-=translation+np.cross(np.broadcast_to(rotation,coords.shape),coords)
        u[:,3:]-=rotation;u=u.ravel()
        if len(supplied)>original_dof_count:
            extra=list(range(original_dof_count,len(supplied)))
            if extra!=getattr(floor,'web_rotation_connection',{}).get('internal_rotation_dof_indices') or any(floor.dof_ids[i][1]!=6 for i in extra):
                raise ValueError('Only declared internal normal rotations may follow physical floor DOFs')
            u=np.r_[u,supplied[original_dof_count:]-rotation_z]
        um,ub=self.fields(u)
        assembled=np.zeros_like(u);result=[]
        for row in floor.elements:
            t=row['transform'];ind=row['indices'];m=t@um[ind];b=t@ub[ind];actual=t@u[ind]
            force=row['parts']['membrane']@(m+self.beta*b)
            if row['kind']=='slab':
                force+=self._slab_K[row['id']]@actual+row['parts']['shear']@actual
                # Raw section resultants inherit exactly the same signed
                # physical action partition as the nodal equilibrium proof.
                sm=(floor.ec*row['native']['raw_resultant']@(m+self.beta*b)).reshape(4,8)
                base=(floor.ec*row['native']['raw_resultant']@actual).reshape(4,8)
                if self.alpha is not None:sm[:,3:6]=self.alpha[row['id']]*base[:,3:6]
                else:
                    curvature=-row['native']['strain'].reshape(4,8,24)[:,3:6,:]@actual
                    sm[:,3:6]=np.einsum('nij,nj->ni',self._slab_D[row['id']],curvature)
                sm[:,6:]=base[:,6:]
            else:
                force+=self.beta*row['parts']['web_bending']@actual+self.torsion[row['member_id']]*row['parts']['web_torsion']@actual
            elastic=force.copy();force-=load_record['element_loads'][row['id']]
            np.add.at(assembled,ind,t.T@force)
            out={'id':row['id'],'kind':row['kind'],'positions_in':row['points'].tolist(),
                 'displacements':(t@supplied[ind]).tolist(),'global_resisting_force':force.tolist(),
                 'global_elastic_force':elastic.tolist()}
            if row['kind']=='slab':out.update(cell=row['cell'],area_in2=row['area_in2'],raw_resultants=sm.tolist())
            else:out.update(member_id=row['member_id'],axis=row['axis'],station_in=row['station_in'],section=row['section'])
            result.append(out)
        element_applied=np.zeros_like(u)
        for row in floor.elements:
            np.add.at(element_applied,row['indices'],row['transform'].T@load_record['element_loads'][row['id']])
        error=assembled-(self.stiffness@u-element_applied)
        scale=max(1.,float(np.max(np.abs(self.stiffness@u),initial=0.)))
        residual=float(np.max(np.abs(error),initial=0.))/scale
        if residual>1e-9:
            at=int(np.argmax(np.abs(error)))
            raise ValueError('Physical cracked force partition does not reproduce the assembled stiffness: '
                f'relative={residual:.17g}, absolute={abs(error[at]):.17g}, scale={scale:.17g}, dof={floor.dof_ids[at]}')
        return {'method':self.metadata['method'],'elements':result,'dof_ids':floor.dof_ids,
                'boundary_dofs':floor.boundary_dofs,'assembled_element_resistance':assembled.tolist(),
                'force_partition_relative_error':residual,'stiffness':self.metadata,
                'physical_relaxation':self._last_relaxation_audit,
                'removed_rigid_translation_in':translation.tolist(),
                'removed_rigid_rotation_rad':rotation.tolist(),'rigid_motion_reference_xyz_in':reference.tolist(),
                'qualification_asserted':False}
