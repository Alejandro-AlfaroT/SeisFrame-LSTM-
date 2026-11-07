"""Physical floor operator with explicit full slab A/B/D constitutive matrices.

Unlike independent x/y inertia multipliers, a full matrix retains membrane,
bending/twisting and membrane-bending couplings supplied by one actual layer
state. Matrices here are nominal constitutive stiffness, never phi-reduced
strength. This linearized operator is usable in a declared secant iteration;
it does not assert that a material iteration converged or its law is applicable.

The native MITC4 strain and transverse-shear operators are reused. Drilling
stabilization is retained separately as a numerical term, not extra capacity.
Every actual web has its own explicitly supplied twelve-DOF matrix; this API
does not silently give a cracked slab an uncracked or double-counted T beam.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.sparse import coo_matrix

METHOD='native_MITC4_actual_full_ABD_floor_v1'


def _matrix(value,n,name):
    value=np.asarray(value,float)
    if value.shape!=(n,n) or not np.isfinite(value).all():raise ValueError(f'{name} requires one finite{n}x{n} matrix')
    scale=max(1.,float(np.max(abs(value))));error=float(np.max(abs(value-value.T)))
    if error>1e-9*scale:raise ValueError(f'{name} has nonsymmetric constitutive energy')
    value=(value+value.T)/2
    if np.linalg.eigvalsh(value).min() < -1e-9*scale:raise ValueError(f'{name} has negative constitutive energy')
    return value


def common_shell_strain_matrix(element):
    """Four native GP matrices in the common physical curvature convention."""
    raw=np.asarray(element['native']['strain']).reshape(4,8,24)
    result=raw[:,:6,:].copy();result[:,3:6,:]*=-1
    return result


def web_normal_strain_matrices(element,stations,*,reference_z_in=0.,transverse_reference_in=None):
    """Euler-Bernoulli [eps0,kstrong,kweak] at explicit member fractions.

The actual web nodes lie at their centroid. The axial reference is transported
to the same physical slab-midplane/transverse datum used by the section law.
Both horizontal beam orientations retain the global native rotation signs.
"""
    axis=element['axis'];points=np.asarray(element['points'],float)
    if axis not in ('x','y') or points.shape!=(2,3):raise ValueError('Two actual aligned web nodes required')
    along=0 if axis=='x' else 1;across=1-along;length=points[1,along]-points[0,along]
    if length<=0:raise ValueError('Positive physical web length required')
    if transverse_reference_in is None:transverse_reference_in=points[0,across]
    zc=points[0,2]-float(reference_z_in);sc=points[0,across]-float(transverse_reference_in)
    matrices=[]
    for xi in stations:
        xi=float(xi)
        if not np.isfinite(xi) or not 0<=xi<=1:raise ValueError('Web stations must be finite physical fractions in[0,1]')
        h1,h2,h3,h4=(-6+12*xi)/length**2,(-4+6*xi)/length,(6-12*xi)/length**2,(-2+6*xi)/length
        b=np.zeros((3,12));b[0,along]=-1/length;b[0,6+along]=1/length
        rotation=4 if axis=='x' else 3;slope_sign=-1 if axis=='x' else 1
        b[1,[2,rotation,8,6+rotation]]=[-h1,-slope_sign*h2,-h3,-slope_sign*h4]
        weak_slope=1 if axis=='x' else -1
        b[2,[across,5,6+across,11]]=[-h1,-weak_slope*h2,-h3,-weak_slope*h4]
        b[0]-=zc*b[1]+sc*b[2]
        matrices.append(b)
    return np.asarray(matrices)


def web_operator_from_sections(element,section_matrices,*,stations,weights,torsion_matrix,
                               reference_z_in=0.,transverse_reference_in=None):
    """Integrate explicit nominal disjoint-web3 secants; torsion stays separate.

Weights integrate over fractions0..1 and must sum to1. No torsion law/factor
is invented here. The caller supplies a supported actual torsion operator and
keeps the separate concurrent shear/torsion checks in its required inventory.
"""
    stations=np.asarray(stations,float);weights=np.asarray(weights,float)
    if stations.ndim!=1 or stations.shape!=weights.shape or len(stations)<2 or not np.isfinite(weights).all() or np.any(weights<=0) or abs(weights.sum()-1)>1e-12:
        raise ValueError('Explicit positive complete web longitudinal quadrature required')
    matrices=list(section_matrices)
    if len(matrices)!=len(stations):raise ValueError('Every physical longitudinal station requires its nominal section matrix')
    b=web_normal_strain_matrices(element,stations,reference_z_in=reference_z_in,transverse_reference_in=transverse_reference_in)
    k=_matrix(torsion_matrix,12,'Actual torsion operator').copy()
    # A torsion operator can act only on rotations about the web's own axis.
    axis=3 if element['axis']=='x' else 4
    mask=np.ones((12,12),bool);mask[np.ix_([axis,axis+6],[axis,axis+6])]=False
    if np.max(abs(k[mask]),initial=0.)>1e-9*max(1.,float(np.max(abs(k)))):
        raise ValueError('Separate web torsion matrix includes already owned axial/bending stiffness')
    length=float(element['length_in'])
    for bi,ci,w in zip(b,matrices,weights):k+=length*w*bi.T@_matrix(ci,3,'Nominal physical web section')@bi
    return k


class MaterialFloorOperator:
    def __init__(self,floor,*,slab_ABD,web_matrices,transverse_shear_factors,
                 source_identity,material_state_identity):
        if not source_identity or not material_state_identity:raise ValueError('Actual source and constitutive state identities required')
        slab_ids={r['id'] for r in floor.elements if r['kind']=='slab'}
        web_ids={r['id'] for r in floor.elements if r['kind']=='web'}
        if set(slab_ABD)!=slab_ids or set(transverse_shear_factors)!=slab_ids or set(web_matrices)!=web_ids:
            raise ValueError('Every disjoint slab cell/web needs exactly one explicit physical constitutive operator')
        self.floor=floor;self.slab={};self.element_matrices={};self.shear={};rr=[];cc=[];vv=[]
        hashes=[]
        for row in floor.elements:
            if row['kind']=='slab':
                values=np.asarray(slab_ABD[row['id']],float)
                if values.shape==(6,6):values=np.repeat(values[None,:,:],4,axis=0)
                if values.shape!=(4,6,6):raise ValueError('One actual6x6 ABD matrix at each of four shell Gauss points required')
                values=np.array([_matrix(v,6,'Slab ABD') for v in values]);self.slab[row['id']]=values
                shear=float(transverse_shear_factors[row['id']])
                if not np.isfinite(shear) or not 0<shear<=1:raise ValueError('Explicit positive transverse shear factor <=1 required')
                self.shear[row['id']]=shear;b=common_shell_strain_matrix(row)
                k=sum(bg.T@c@bg for bg,c in zip(b,values))*row['area_in2']/4
                k+=floor.ec*(shear*row['native']['parts']['shear']+row['native']['parts']['drilling'])
            else:k=_matrix(web_matrices[row['id']],12,'Actual disjoint web')
            self.element_matrices[row['id']]=k
            hashes.append((row['id'],hashlib.sha256(np.ascontiguousarray(k,dtype='<f8').tobytes()).hexdigest()))
            transformed=row['transform'].T@k@row['transform'];r,c=np.nonzero(transformed)
            rr.append(row['indices'][r]);cc.append(row['indices'][c]);vv.append(transformed[r,c])
        size=len(floor.dof_ids)
        self.stiffness=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(size,size)).tocsr()
        self.metadata=dict(method=METHOD,source_identity=source_identity,material_state_identity=material_state_identity,
            element_operator_sha256=hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest(),
            shear_factors=self.shear,slab_matrices_are_nominal=True,
            material_iteration_converged=False,qualification_asserted=False)

    def section_strains(self,displacements):
        u=np.asarray(displacements,float)
        if u.shape!=(len(self.floor.dof_ids),) or not np.isfinite(u).all():raise ValueError('Actual finite physical floor displacement vector required')
        result={}
        for row in self.floor.elements:
            local=row['transform']@u[row['indices']]
            if row['kind']=='slab':result[row['id']]=(common_shell_strain_matrix(row)@local).tolist()
        return result

    def recover(self,displacements,load_record):
        floor=self.floor;u=np.asarray(displacements,float)
        if u.shape!=(len(floor.dof_ids),) or not np.isfinite(u).all():raise ValueError('Actual finite physical floor displacement vector required')
        assembled=np.zeros_like(u);applied=np.zeros_like(u);elements=[]
        for row in floor.elements:
            t=row['transform'];indices=row['indices'];local=t@u[indices]
            elastic=self.element_matrices[row['id']]@local
            load=np.asarray(load_record['element_loads'][row['id']]);force=elastic-load
            np.add.at(assembled,indices,t.T@force);np.add.at(applied,indices,t.T@load)
            result=dict(id=row['id'],kind=row['kind'],positions_in=row['points'].tolist(),
                        displacements=local.tolist(),global_resisting_force=force.tolist(),global_elastic_force=elastic.tolist())
            if row['kind']=='slab':
                q=common_shell_strain_matrix(row)@local
                r=np.einsum('nij,nj->ni',self.slab[row['id']],q)
                transverse=(floor.ec*row['native']['raw_resultant']@local).reshape(4,8)[:,6:]*self.shear[row['id']]
                result.update(cell=row['cell'],area_in2=row['area_in2'],raw_resultants=np.c_[r,transverse].tolist(),
                              common_generalized_strains=q.tolist())
            else:result.update(member_id=row['member_id'],axis=row['axis'],station_in=row['station_in'],section=row['section'])
            elements.append(result)
        residual=assembled-(self.stiffness@u-applied);scale=max(1.,float(np.max(abs(self.stiffness@u),initial=0.)))
        error=float(np.max(abs(residual),initial=0.))/scale
        if error>1e-9:raise ValueError('Physical material field does not reproduce the assembled nodal operator')
        return dict(method=METHOD,elements=elements,dof_ids=floor.dof_ids,boundary_dofs=floor.boundary_dofs,
                    assembled_element_resistance=assembled.tolist(),force_partition_relative_error=error,
                    stiffness=self.metadata,qualification_asserted=False)
