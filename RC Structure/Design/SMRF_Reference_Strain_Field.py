"""Freeze one rectangular shell reference strain field before mesh comparison.

Only the six kinematic strains are reconstructed. Constitutive matrices are
always recomputed with the destination cell's physical material; they are not
interpolated or projected. The field does not certify its reference loading,
material applicability or mesh convergence.
"""
from __future__ import annotations
import hashlib,json
import numpy as np

METHOD='frozen_rectangular_four_GP_reference_strain_field_v1'
GP=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])/np.sqrt(3.)


class RectangularReferenceStrainField:
    def __init__(self,*,x_in,y_in,cell_strains,reference_identity):
        if not reference_identity:raise ValueError('Bound physical reference identity required')
        self.x=np.array(x_in,float);self.y=np.array(y_in,float)
        for grid in (self.x,self.y):
            if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0):
                raise ValueError('Finite strictly increasing rectangular reference grid required')
        expected={(i,j) for i in range(len(self.x)-1) for j in range(len(self.y)-1)}
        if set(cell_strains)!=expected:raise ValueError('Exactly every reference cell must supply its four strains')
        self.strains={}
        for key,value in cell_strains.items():
            q=np.array(value,float)
            if q.shape!=(4,6) or not np.isfinite(q).all():raise ValueError('Four finite six-component reference strains required')
            q.setflags(write=False);self.strains[key]=q
        self.x.setflags(write=False);self.y.setflags(write=False)
        self.reference_identity=str(reference_identity)
        binding=dict(method=METHOD,reference_identity=self.reference_identity,x_in=self.x.tolist(),y_in=self.y.tolist(),
            cells=[dict(index=list(k),strains=self.strains[k].tolist()) for k in sorted(self.strains)])
        self.identity=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def at(self,x_in,y_in,*,trace=('plus','plus')):
        if len(trace)!=2 or any(t not in ('plus','minus') for t in trace):raise ValueError('Two explicit one-sided traces required')
        point=np.array([x_in,y_in],float)
        if not np.isfinite(point).all():raise ValueError('Finite reference point required')
        indices=[];natural=[]
        for coordinate,grid,side in zip(point,(self.x,self.y),trace):
            if coordinate<grid[0] or coordinate>grid[-1]:raise ValueError('Reference point outside its physical footprint')
            i=int(np.searchsorted(grid,coordinate,side='right' if side=='plus' else 'left')-1)
            i=min(max(i,0),len(grid)-2);indices.append(i)
            natural.append(2*(coordinate-grid[i])/(grid[i+1]-grid[i])-1)
        # Tensor Lagrange reconstruction at the native +/-1/sqrt(3) points.
        # Negative weights near cell faces reconstruct strain, never stiffness.
        weights=np.prod(1+3*GP*np.array(natural),axis=1)/4
        return dict(strain=(weights@self.strains[tuple(indices)]).tolist(),cell_index=indices,
            natural_coordinates=natural,trace=list(trace),field_identity=self.identity,
            reference_identity=self.reference_identity)
