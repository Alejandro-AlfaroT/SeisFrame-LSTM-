"""Diagnostic uniaxial nominal N-M section mechanics with explicit force axes.

Coordinates y run from the physical top downward. N is tension positive;
M is sagging positive about the declared y reference. One concrete strength,
Whitney block, ecu=.003, EPP steel and lumped reinforcement layers are used.
This is not a phi-reduced/code-qualified capacity, biaxial interaction surface,
hinge backbone, steel-development check or experimental material calibration.

Independent verification reference: StructurePoint, May 24 2022, Interaction
Diagram - Tied Reinforced Concrete Column Design Strength (ACI 318-19),
https://structurepoint.org/publication/pdf/Interaction-Diagram-Tied-Reinforced-Concrete-Column-Design-Strength-ACI-318-19.pdf

The legacy centroid displacement mode can create jumps; ambiguous roots are
refused. Generated cages use the explicit circular mode, which subtracts
only the bar area inside the block and transports its actual first moment.
"""
from __future__ import annotations
import math

METHOD_VERSION='diagnostic_uniaxial_section_nm_v2'
ES_KSI=29000.0
ECU=.003


def _number(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if positive and value<=0:raise ValueError(f'{name} must be positive')
    return float(value)


def _section(section):
    if not isinstance(section,dict) or not isinstance(section.get('id'),str) or not section['id'].strip():
        raise ValueError('An explicit section id is required')
    fc=_number(section['fc_ksi'],'fc_ksi',True);fy=_number(section['fy_ksi'],'fy_ksi',True)
    h=_number(section['depth_in'],'depth_in',True)
    bands=[];last=0.
    for band in section['concrete_bands']:
        lo=_number(band['top_in'],'band top');hi=_number(band['bottom_in'],'band bottom')
        width=_number(band['width_in'],'band width',True)
        if abs(lo-last)>1e-10 or hi<=lo or hi>h:raise ValueError('Concrete bands must exactly cover the depth without gaps or overlaps')
        bands.append((lo,hi,width));last=hi
    if not bands or abs(last-h)>1e-10:raise ValueError('Concrete bands must cover the whole depth')
    layers=[];ids=set()
    for layer in section['steel_layers']:
        name=layer['id'];y=_number(layer['depth_in'],'steel depth');area=_number(layer['area_in2'],'steel area',True)
        if not isinstance(name,str) or not name or name in ids:raise ValueError('Steel layer ids must be nonempty and unique')
        if not 0<y<h:raise ValueError('Steel centroids must lie inside the section depth')
        ids.add(name);layers.append((name,y,area))
    if not layers:raise ValueError('At least one steel layer is required')
    if sum(a for _,_,a in layers)>=sum((hi-lo)*b for lo,hi,b in bands):raise ValueError('Steel area cannot consume the gross section')
    return fc,fy,h,bands,layers


def _bar_overlap(layer, low, high, h):
    """Concrete displaced by the part of a round bar inside a stress block.

    Normalize the circular segment to the declared nominal steel area: bar
    tables round that area independently of diameter. A smeared row of equal
    bars uses the same segment fraction. Steel stress remains at its centroid.
    """
    diameter=_number(layer.get('bar_diameter_in'),'bar_diameter_in',True)
    r=diameter/2.;y=layer['depth_in'];area=layer['area_in2']
    if y-r<0 or y+r>h:raise ValueError('Circular bar must fit inside the section depth')
    a=max(-r,min(r,low-y));b=max(-r,min(r,high-y))
    if b<=a:return 0.,None
    def primitive(u):
        root=math.sqrt(max(0.,r*r-u*u))
        return u*root+r*r*math.asin(max(-1.,min(1.,u/r))), -2./3.*root**3
    aa,ma=primitive(a);ab,mb=primitive(b)
    raw=ab-aa
    if raw<=1e-15*r*r:return 0.,None
    return area*raw/(math.pi*r*r),y+(mb-ma)/raw


def _validate_round_displacement(section, bands, h):
    """Conservative bound ensuring removed bar width never exceeds concrete.

    This establishes monotone axial equilibrium for the round-bar branch;
    it is not a cage spacing, development or constructability check.
    """
    bars=[];edges={0.,h}
    for layer in section['steel_layers']:
        r=_number(layer.get('bar_diameter_in'),'bar_diameter_in',True)/2
        y=layer['depth_in'];area=layer['area_in2']
        if y-r<0 or y+r>h:raise ValueError('Circular bar must fit inside the section depth')
        bars.append((y-r,y+r,2*area/(math.pi*r)))
        edges.update((y-r,y+r))
    for lo,hi,_ in bands:edges.update((lo,hi))
    edges=sorted(edges)
    for lo,hi in zip(edges,edges[1:]):
        mid=(lo+hi)/2
        width=next(b for bottom,top,b in bands if bottom<=mid<=top)
        removed_bound=math.fsum(w for a,b,w in bars if a<mid<b)
        if removed_bound>=width:
            raise ValueError('Round-bar displacement width bound exceeds concrete width; use a resolved section geometry')


def section_state(section,neutral_axis_in,*,compression_face,reference_y_in,subtract_displaced_concrete=True,
                  displaced_concrete_mode='centroid'):
    """Evaluate one strain-compatible nominal point; all reference choices explicit."""
    fc,fy,h,bands,layers=_section(section)
    c=_number(neutral_axis_in,'neutral_axis_in',True);ref=_number(reference_y_in,'reference_y_in')
    if compression_face not in ('top','bottom'):raise ValueError('Compression face must be top or bottom')
    if not isinstance(subtract_displaced_concrete,bool):raise ValueError('Concrete-subtraction mode must be boolean')
    if displaced_concrete_mode not in ('centroid','circular'):
        raise ValueError('Concrete displacement must be centroid or circular')
    if subtract_displaced_concrete and displaced_concrete_mode=='circular':
        _validate_round_displacement(section,bands,h)
    beta=max(.65,min(.85,.85-.05*(fc-4.)))
    a=min(beta*c,h);low,high=(0.,a) if compression_face=='top' else (h-a,h)
    concrete=[]
    for lo,hi,b in bands:
        start,end=max(lo,low),min(hi,high)
        if end>start:
            force=.85*fc*b*(end-start);y=(start+end)/2
            concrete.append(dict(force_compression_kip=force,depth_in=y,moment_sagging_kip_in=force*(ref-y)))
    steel=[]
    for (name,y,area),layer in zip(layers,section['steel_layers']):
        d=y if compression_face=='top' else h-y
        strain=ECU*(1.-d/c);stress=max(-fy,min(fy,ES_KSI*strain))
        displaced_area,displaced_y=0.,None
        if subtract_displaced_concrete:
            if displaced_concrete_mode=='circular':
                displaced_area,displaced_y=_bar_overlap(layer,low,high,h)
            elif d<=a:displaced_area,displaced_y=area,y
        displaced=.85*fc*displaced_area
        force=area*stress-displaced
        steel.append(dict(id=name,depth_in=y,area_in2=area,strain_compression_positive=strain,
            stress_compression_ksi=stress,steel_force_compression_kip=area*stress,
            displaced_concrete_force_kip=displaced,net_force_compression_kip=force,
            displaced_concrete_centroid_in=displaced_y,
            moment_sagging_kip_in=area*stress*(ref-y)-(displaced*(ref-displaced_y) if displaced_y is not None else 0.)))
    P=math.fsum(v['force_compression_kip'] for v in concrete)+math.fsum(v['net_force_compression_kip'] for v in steel)
    M=math.fsum(v['moment_sagging_kip_in'] for v in concrete+steel)
    return dict(method=METHOD_VERSION,section_id=section['id'],neutral_axis_in=c,stress_block_depth_in=a,beta1=beta,
        compression_face=compression_face,reference_y_in=ref,axial_tension_kip=-P,moment_sagging_kip_in=M,
        maximum_steel_tension_strain=max(0.,max(-v['strain_compression_positive'] for v in steel)),
        concrete_components=concrete,steel_components=steel,subtract_displaced_concrete=subtract_displaced_concrete,
        displaced_concrete_mode=displaced_concrete_mode,
        strength_basis='Nominal uniaxial section mechanics; no phi, compression cap, development or detailing approval',
        engineering_verified=False,production_enabled=False)


def solve_section_at_axial_force(section,axial_tension_kip,*,compression_face,reference_y_in,
                                 subtract_displaced_concrete=True,force_tolerance_kip=1e-7,
                                 displaced_concrete_mode='centroid'):
    """Find a unique finite-c state; reject unbalanced, out-of-range or ambiguous roots.

    Pure axial endpoints and tension-only strain fields are outside this
    finite-c compression-face branch. Callers must not interpret refusal
    as zero capacity or drop the analysis from an exported dataset.
    """
    fc,fy,h,bands,layers=_section(section)
    target=_number(axial_tension_kip,'axial_tension_kip')
    tol=_number(force_tolerance_kip,'force_tolerance_kip',True)
    if tol>1e-3:raise ValueError('Diagnostic force tolerance must not exceed 0.001 kip')
    if compression_face not in ('top','bottom'):raise ValueError('Compression face must be top or bottom')
    if not isinstance(subtract_displaced_concrete,bool):raise ValueError('Concrete-subtraction mode must be boolean')
    area_steel=math.fsum(a for _,_,a in layers)
    area_gross=math.fsum((hi-lo)*b for lo,hi,b in bands)
    Pmax=.85*fc*(area_gross-(area_steel if subtract_displaced_concrete else 0.))+area_steel*min(fy,ES_KSI*ECU)
    if target<=-Pmax+tol or target>=fy*area_steel-tol:
        raise ValueError('Axial force is at or outside the finite-c branch endpoints; no uniaxial flexural state is returned')
    beta=max(.65,min(.85,.85-.05*(fc-4.)))
    state=lambda c:section_state(section,c,compression_face=compression_face,reference_y_in=reference_y_in,
                                subtract_displaced_concrete=subtract_displaced_concrete,
                                displaced_concrete_mode=displaced_concrete_mode)
    lower=h*1e-12;upper=h*1e10
    edges=[lower,upper]
    if subtract_displaced_concrete and displaced_concrete_mode=='centroid':
        edges.extend((y if compression_face=='top' else h-y)/beta for _,y,_ in layers)
    edges=sorted(set(edges));roots=[]
    def add(c):
        value=state(c)
        if abs(value['axial_tension_kip']-target)>tol:return
        if any(abs(c-v['neutral_axis_in'])<=1e-8*max(1.,c) for v in roots):return
        value.update(requested_axial_tension_kip=target,axial_equilibrium_error_kip=value['axial_tension_kip']-target)
        roots.append(value)
    for k,(left,right) in enumerate(zip(edges,edges[1:])):
        # Do not bisect across a lumped displaced-concrete jump.
        lo=left if k==0 else left+max(1.,left)*1e-11
        hi=right if k==len(edges)-2 else right-max(1.,right)*1e-11
        fl=state(lo)['axial_tension_kip']-target;fh=state(hi)['axial_tension_kip']-target
        if fl*fh>0:continue
        for _ in range(180):
            mid=(lo+hi)/2;fm=state(mid)['axial_tension_kip']-target
            if abs(fm)<tol*.01:break
            if fl*fm<=0:hi=mid
            else:lo=mid;fl=fm
        add(mid)
    for c in edges[1:-1]:add(c)
    if len(roots)!=1:
        raise ValueError(f'Axial equilibrium has {len(roots)} admissible finite-c roots; inspect branch limits or lumped-steel discontinuities')
    return roots[0]


def demand_from_global_cut(force_moment,*,axis,outward_normal_sign,reference_xyz_in,section_top_z_in,section_id):
    """Map an outward global cut wrench to explicit uniaxial section N and M.

    This does not select a flange, repartition force, or suppress other actions.
    Region, material, bar ownership and recovery validity remain caller checks.
    """
    if axis not in ('x','y'):raise ValueError('Beam axis must be x or y')
    if isinstance(outward_normal_sign,bool) or outward_normal_sign not in (-1,1):raise ValueError('Outward sign must be -1 or +1')
    if len(force_moment)!=6 or len(reference_xyz_in)!=3:raise ValueError('Provide six wrench components and a three-coordinate reference')
    f=[_number(v,'wrench') for v in force_moment];ref=[_number(v,'reference') for v in reference_xyz_in]
    top=_number(section_top_z_in,'section_top_z_in')
    if not isinstance(section_id,str) or not section_id.strip():raise ValueError('Explicit section id required')
    N=outward_normal_sign*f[0 if axis=='x' else 1]
    M=outward_normal_sign*(-f[4] if axis=='x' else f[3])
    return dict(section_id=section_id,axis=axis,reference_xyz_in=ref,reference_y_in=top-ref[2],
        axial_tension_kip=N,moment_sagging_kip_in=M,global_wrench=f,outward_normal_sign=outward_normal_sign,
        other_actions_require_assessment=True,engineering_verified=False)
