"""Explicit local-feature mesh pair for numerical assessment, not approval.

Keep the original physical faces, add the actual critical perimeter and resolve
the beam-line/column-face neighbourhood. The second grid retains every point of
the first, doubles the background subdivisions and bisects its near-feature
intervals. No physical face, cut, property or load is moved or smoothed.
"""
from __future__ import annotations
import math
from Design.SMRF_Coupled_Floor_Design import physical_grid
from Model.Floor_Footprint import from_perimeter_beams

METHOD='nested_physical_critical_and_beam_line_feature_mesh_v2_no_background_slivers'


def feature_grid(floor,geometry,*,base_subdivisions_per_bay,slab_thickness_in,
                 critical_depth_in,flange_boundaries,refinement_level):
    h,d=map(float,(slab_thickness_in,critical_depth_in))
    if not all(math.isfinite(v) and v>0 for v in (h,d)) or d>=h:
        raise ValueError('Actual positive slab thickness and smaller critical depth required')
    if refinement_level not in (1,2) or type(refinement_level) is not int:
        raise ValueError('Only the explicitly paired mesh levels1/2 are supported')
    if set(flange_boundaries)!={'x','y'}:raise ValueError('Explicit physical X/Y flange-boundary inventory required')
    fp=from_perimeter_beams(geometry,floor['beams'])
    n=base_subdivisions_per_bay
    if type(n) is not int or n<2:raise ValueError('At least two explicit background subdivisions required')
    axes=[{fp['x_min_in'],fp['x_max_in']},{fp['y_min_in'],fp['y_max_in']}];bands=[[],[]]
    for support in floor['supports']:
        for a,axis,dimension in ((0,'x','h_in'),(1,'y','b_in')):
            center=float(support[axis]);half=float(support[dimension])/2
            axes[a].add(center)
            bands[a].append((center-half-2*d,center+half+2*d))
            for sign in (-1,1):
                for offset in (0.,d/4,d/2,d,2*d):axes[a].add(center+sign*(half+offset))
    for beam in floor['beams']:
        a=1 if beam['axis']=='x' else 0;center=float(beam['line'])
        axes[a].add(center)
        bands[a].append((center-max(beam['b_in']/2,2*h),center+max(beam['b_in']/2,2*h)))
        for sign in (-1,1):
            for offset in (beam['b_in']/2,h/4,h/2,h,2*h):axes[a].add(center+sign*offset)
    # Values identify coordinates, not widths. An X beam's flange endpoints
    # belong to the Y mesh axis, and conversely.
    for axis in ('x','y'):
        a=1 if axis=='x' else 0
        for value in flange_boundaries[axis]:
            if not math.isfinite(float(value)):raise ValueError('Finite physical flange endpoint required')
            axes[a].add(float(value))
    def clean(values,axis):
        low,high=fp[axis+'_min_in'],fp[axis+'_max_in'];out=[]
        for value in sorted(v for v in values if low-1e-9<=v<=high+1e-9):
            if not out or value-out[-1]>1e-8:out.append(value)
        return out
    def background(values,axis,count,minimum_gap):
        # Background sampling points carry no physical geometry. Omit one if
        # it would create a sliver beside an exact retained feature; never
        # snap or delete a physical face/perimeter/flange coordinate.
        length=geometry['lx' if axis=='x' else 'ly'];bays=geometry['nx' if axis=='x' else 'ny']
        for i in range(bays*count+1):
            value=i*length/count
            if min(abs(value-v) for v in values)>=minimum_gap-1e-9:values.add(value)
    for values,axis in zip(axes,('x','y')):background(values,axis,n,min(h,d)/4)
    base=[clean(values,axis) for values,axis in zip(axes,('x','y'))]
    if refinement_level==1:return base
    for a in (0,1):
        for lo,hi in zip(base[a],base[a][1:]):
            mid=(lo+hi)/2
            if any(start<=mid<=end for start,end in bands[a]):axes[a].add(mid)
        background(axes[a],('x','y')[a],2*n,min(h,d)/8)
    return [clean(values,axis) for values,axis in zip(axes,('x','y'))]
