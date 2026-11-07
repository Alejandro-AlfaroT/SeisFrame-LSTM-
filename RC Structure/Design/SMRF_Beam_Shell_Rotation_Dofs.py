"""Explicit internal beam rotations for the existing independent connection.

Keep each unloaded clear-span beam normal rotation in the floor equations,
instead of eliminating it into a dense interpolation map before assembly.
The physical Euler beam, shell, offsets and joint rotations are unchanged.
This optional algebraic representation requires equivalence/field verification;
it does not select a production floor model or qualify any engineering method.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from Design.SMRF_Stable_Sparse_Assembly import accurate_triplet_matrix

METHOD='flat_shell_stiffener_retained_normal_rotation_v2_accurate_assembly_unqualified'


def retain_independent_beam_rotations(floor):
    if hasattr(floor,'web_rotation_connection'):raise ValueError('Beam connection already specified')
    original_count=len(floor.dof_ids)
    if original_count!=6*len(floor.node_ids):raise ValueError('Original six-DOF physical floor nodes required')
    grouped={}
    for row in floor.elements:
        if row['kind']=='web':grouped.setdefault(row['member_id'],[]).append(row)
    extra=[];audits=[];physical_rotations=[]
    for member,rows in grouped.items():
        rows.sort(key=lambda row:row['station_in'][0]);nodes=[rows[0]['nodes'][0],*[r['nodes'][1] for r in rows]]
        if any(a['nodes'][1]!=b['nodes'][0] for a,b in zip(rows,rows[1:])):raise ValueError('One continuous clear-span beam chain required')
        if len(set(nodes))!=len(nodes):raise ValueError('Repeated physical beam chain node')
        if any(floor.owner[n] not in floor.joint_centers for n in (nodes[0],nodes[-1])):raise ValueError('Both beam ends must retain actual joint rotations')
        auxiliaries={}
        for node in nodes[1:-1]:
            if floor.owner[node] in floor.joint_centers:raise ValueError('Interior joint cannot be converted to an unloaded beam rotation')
            ident=f'{floor.id}:beam:{member}:normal_rotation:{node}'
            if ident in floor.node_positions:raise ValueError('Unique internal beam rotation identity required')
            index=len(floor.dof_ids);floor.dof_ids.append((ident,6));floor.node_positions[ident]=floor.points[node].tolist()
            auxiliaries[node]=index;extra.append(index)
            physical_rotations.append(dict(dof_index=index,node_id=ident,physical_node=int(node),member_id=member))
        for row in rows:
            old=row['transform'];old_indices=row['indices'];new_indices=np.unique(np.r_[old_indices,[auxiliaries[n] for n in row['nodes'] if n in auxiliaries]].astype(int))
            locations={int(v):i for i,v in enumerate(new_indices)};new=np.zeros((12,len(new_indices)))
            for column,index in enumerate(old_indices):new[:,locations[int(index)]]+=old[:,column]
            for end,node in enumerate(row['nodes']):
                if node in auxiliaries:
                    new[6*end+5]=0.;new[6*end+5,locations[auxiliaries[node]]]=1.
            row.update(transform=new,indices=new_indices,normal_rotation_connection=METHOD)
        audits.append(dict(member_id=member,segments=len(rows),retained_internal_rotations=len(auxiliaries)))
    # Assemble the final physical blocks directly; no subtract-old/add-new
    # updates and no elimination of these new equilibrium equations.
    shape=(len(floor.dof_ids),)*2;parts={}
    for name in floor.parts:
        rr=[];cc=[];vv=[]
        for row in floor.elements:
            if name not in row['parts']:continue
            block=row['transform'].T@row['parts'][name]@row['transform'];r,c=np.nonzero(block)
            if len(r):rr.append(row['indices'][r]);cc.append(row['indices'][c]);vv.append(block[r,c])
        parts[name]=(accurate_triplet_matrix(np.concatenate(vv),np.concatenate(rr),np.concatenate(cc),shape=shape)
                     if vv else sparse.csr_matrix(shape))
    floor.parts=parts;floor.stiffness=sum(parts.values(),sparse.csr_matrix(shape))
    floor.membrane_dofs=np.r_[floor.membrane_dofs,np.asarray(extra,dtype=int)]
    floor.web_rotation_connection=dict(method=METHOD,members=audits,original_physical_dof_count=original_count,
        internal_rotation_dof_indices=extra,internal_rotations=physical_rotations,internal_rotation_dof_count=len(extra),qualification_asserted=False)
    return floor.web_rotation_connection
