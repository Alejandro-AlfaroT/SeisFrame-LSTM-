"""Independent beam normal rotation for a flat shell stiffener connection.

The shell drilling rotation is a stabilization variable, not the beam's
physical weak-axis rotation. Keep actual support-end rotations and statically
eliminate the unloaded interior beam rotations along each clear span. All
three translations, both physical shell rotations, eccentricity and the full
beam stiffness remain. No hinge or stiffness reduction is introduced.

Reference: COMSOL6.3 Structural Mechanics, General Theory for the Shell and
Plate Interfaces, 'Connection Between Shells and Beams / Beam Edge to Shell
Edge'. This isolated implementation asserts no candidate/model qualification.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from Design.SMRF_Stable_Sparse_Assembly import accurate_triplet_matrix

METHOD='flat_shell_stiffener_independent_normal_rotation_v2_accurate_assembly_unqualified'


def rotation_map(blocks):
    """Stationary rotations for a chain of [v_i,rz_i,v_j,rz_j] energies.

    Retained inputs are every actual transverse translation, then the two
    physical end rotations. Interior rotations have no applied nodal couple.
    Blocks are supplied in global-axis signs; no sign convention is inferred.
    """
    blocks=np.asarray(blocks,float)
    if blocks.ndim!=3 or blocks.shape[1:]!=(4,4) or not len(blocks) or not np.isfinite(blocks).all():
        raise ValueError('Finite connected weak-bending chain blocks required')
    n=len(blocks)+1;k=np.zeros((2*n,2*n))
    for i,b in enumerate(blocks):
        if not np.allclose(b,b.T,rtol=1e-12,atol=1e-12):raise ValueError('Symmetric physical weak-bending blocks required')
        k[2*i:2*i+4,2*i:2*i+4]+=b
    retained=[*range(0,2*n,2),1,2*n-1];internal=list(range(3,2*n-1,2))
    result=np.zeros((n,n+2));result[0,-2]=1.;result[-1,-1]=1.
    if internal:
        a=k[np.ix_(internal,internal)];rhs=-k[np.ix_(internal,retained)]
        try:answer=np.linalg.solve(a,rhs)
        except np.linalg.LinAlgError as exc:raise ValueError('Independent beam rotation chain is singular') from exc
        residual=a@answer-rhs
        if np.max(abs(residual))>1e-10*max(1.,float(np.max(abs(rhs)))):
            raise ValueError('Independent beam rotations failed their equilibrium check')
        result[1:-1]=answer
    return result


def connect_independent_beam_rotations(floor):
    """Apply the explicit connection to an unqualified physical floor assembly.

    Each web's physical end forces are recovered with the same eliminated
    rotation map used in stiffness assembly. Supports retain their six DOFs.
    Only this model's clear-span beam normal rotations are eliminated; shell
    drilling variables remain internal to the native shell formulation.
    """
    if hasattr(floor,'web_rotation_connection'):raise ValueError('Beam connection already specified')
    grouped={}
    for row in floor.elements:
        if row['kind']=='web':grouped.setdefault(row['member_id'],[]).append(row)
    audits=[]
    for member,rows in grouped.items():
        rows.sort(key=lambda r:r['station_in'][0]);axis=rows[0]['axis'];across=1 if axis=='x' else 0
        nodes=[rows[0]['nodes'][0],*[r['nodes'][1] for r in rows]]
        if any(a['nodes'][1]!=b['nodes'][0] for a,b in zip(rows,rows[1:])):
            raise ValueError('One continuous actual clear-span chain required')
        if any(floor.owner[node] not in floor.joint_centers for node in (nodes[0],nodes[-1])):
            raise ValueError('Both physical beam-end rotations must belong to actual column joints')
        weak=[across,5,6+across,11]
        parts=[r['section']['E_ksi']*r['native']['parts']['inplane_bending'] for r in rows]
        mapping=rotation_map([p[np.ix_(weak,weak)] for p in parts])
        union=np.unique(np.concatenate([r['indices'] for r in rows]));loc={v:i for i,v in enumerate(union)}
        retained=np.zeros((len(nodes)+2,len(union)))
        def insert(target,row,component):
            np.add.at(target,[loc[j] for j in row['indices']],row['transform'][component])
        insert(retained[0],rows[0],across)
        for i,row in enumerate(rows):insert(retained[i+1],row,6+across)
        insert(retained[-2],rows[0],5);insert(retained[-1],rows[-1],11)
        rotations=mapping@retained
        for i,(row,part) in enumerate(zip(rows,parts)):
            new=np.zeros((12,len(union)))
            for component in range(12):insert(new[component],row,component)
            new[5]=rotations[i];new[11]=rotations[i+1]
            used=np.flatnonzero(np.any(new!=0.,axis=0));new=new[:,used];newind=union[used]
            row.update(transform=new,indices=newind,normal_rotation_connection=METHOD)
        audits.append(dict(member_id=member,segments=len(rows),independent_internal_rotations=max(0,len(nodes)-2)))
    # Assemble the final physical membrane directly. Subtracting an already
    # rounded old matrix and adding a connection delta would retain its lost
    # low-order terms, despite accurately summing the delta itself.
    rr=[];cc=[];vv=[]
    for row in floor.elements:
        block=row['transform'].T@row['parts']['membrane']@row['transform'];r,c=np.nonzero(block)
        if len(r):rr.append(row['indices'][r]);cc.append(row['indices'][c]);vv.append(block[r,c])
    floor.parts['membrane']=(accurate_triplet_matrix(np.concatenate(vv),np.concatenate(rr),np.concatenate(cc),shape=floor.stiffness.shape)
                            if vv else sparse.csr_matrix(floor.stiffness.shape))
    floor.stiffness=sum(floor.parts.values(),sparse.csr_matrix(floor.stiffness.shape))
    floor.web_rotation_connection=dict(method=METHOD,members=audits,qualification_asserted=False)
    return floor.web_rotation_connection
