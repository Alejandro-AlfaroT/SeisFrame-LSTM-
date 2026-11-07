"""Whole-strip integration coordinates bounded by sorted, clipped curves.

These are geometric coordinates, not assertions of constitutive smoothness.
For every z the clipped curves plus the two strip endpoints partition exactly
the same physical interval. Providers must still prove their material branches.
Unproved clipping/order uses interval values only, never guessed derivatives.
"""
from decimal import Decimal
from Design.SMRF_Interval_Material import Interval, Gradient, UnsupportedBox, as_interval, _NEAR
from Design.SMRF_Interval_Taylor import Taylor

METHOD='sorted_clipped_strip_coordinates_v2_centered_rational_unqualified'


def _value(value):
    return value.value if isinstance(value,(Gradient,Taylor)) else value


def _constant(value,like):
    return type(like).constant(value,len(like.derivatives)) if isinstance(like,(Gradient,Taylor)) else as_interval(value)


def _restrict(value,lower,upper):
    """Intersect an already proved geometric range, retaining derivatives."""
    old=_value(value)
    if old.hi<lower or old.lo>upper:raise ValueError('Inconsistent geometric enclosure')
    bounded=Interval(max(old.lo,lower),min(old.hi,upper))
    if isinstance(value,Gradient):return Gradient(bounded,value.derivatives)
    if isinstance(value,Taylor):return Taylor((bounded,)+value.coefficients[1:])
    return bounded


def _clip(value,lower,upper):
    box=_value(value)
    if box.hi<=lower:return _constant(lower,value)
    if box.lo>=upper:return _constant(upper,value)
    if box.lo>=lower and box.hi<=upper:return value
    if isinstance(value,(Gradient,Taylor)):raise UnsupportedBox('coordinate_clipping_branch_not_enclosed')
    return Interval(max(lower,min(upper,box.lo)),max(lower,min(upper,box.hi)))


def clipped_boundaries(curves,z,s_interval):
    """Enclose all ordered pointwise boundaries; ordering is proved for AD."""
    a,b=map(Decimal,s_interval)
    if b<=a:raise ValueError('Positive strip width required')
    values=[_constant(a,z),*[_clip(fn(z),a,b) for fn in curves],_constant(b,z)]
    if isinstance(z,(Gradient,Taylor)):
        values.sort(key=lambda v:(_value(v).lo,_value(v).hi))
        if any(_value(x).hi>_value(y).lo for x,y in zip(values[:-1],values[1:])):
            raise UnsupportedBox('coordinate_order_not_enclosed')
        return values
    # The kth order statistic is monotone in every argument. Sorting lower
    # and upper endpoints separately therefore bounds each sorted curve.
    return [Interval(lo,hi) for lo,hi in zip(sorted(v.lo for v in values),sorted(v.hi for v in values))]


def band_provider(density,curves,index,s_interval):
    """Map t in[0,1],z to one disjoint band, including its exact Jacobian."""
    if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<=len(curves):
        raise ValueError('Valid band index required')
    a,b=map(Decimal,s_interval)
    def provider(t,z):
        if _value(t).lo<0 or _value(t).hi>1:raise ValueError('Reference band coordinate outside[0,1]')
        bounds=clipped_boundaries(curves,z,s_interval)
        left,right=bounds[index:index+2]
        width=_restrict(right-left,Decimal(0),(Interval(b)-a).hi)
        s=_restrict(left+width*t,a,b)
        return [v*width for v in density(s,z)]
    return provider


def material_branch_curves(section,q,region):
    """Auxiliary curves near the existing affine-strain constitutive branches.

    Their rounded threshold coefficients are only coordinate choices. Neither
    their locations nor sorting/clipping confer a material smoothness claim.
    A pole gives the full physical clipped range for values and no derivatives.
    """
    if len(q)!=7 or any(not isinstance(v,Interval) for v in q):raise ValueError('Seven interval strains required')
    if q[6].lo<=0<=q[6].hi:raise UnsupportedBox('zero_weak_curvature_requires_rectangle_coordinates')
    j,other=section.j,section.other
    sa,sb=map(Decimal,region['s'])
    thresholds=(0.,(.2/.34)*section.peak,-section.peak)
    curves=[]
    for threshold in thresholds:
        t=Interval(threshold)
        # Expand the determinant numerator before dividing. Independently
        # enclosing g(z)^2/(4*b(z))-a(z) loses the strong correlation between
        # almost rank-one affine strains and can conceal an interior curve.
        n0=(t-q[j])*(q[other]-t)+q[2].square()/4
        n1=(t-q[j])*q[other+3]-q[j+3]*(q[other]-t)+q[2]*q[5]/2
        n2=-q[j+3]*q[other+3]+q[5].square()/4
        d0=q[6]*(q[other]-t);d1=q[6]*q[other+3]
        def curve(z,threshold=threshold,n0=n0,n1=n1,n2=n2,d0=d0,d1=d1):
            if region['id']=='slab':
                denominator=z*d1+d0
                if _value(denominator).lo<=0<=_value(denominator).hi:
                    if isinstance(z,(Gradient,Taylor)):raise UnsupportedBox('coordinate_curve_pole')
                    return Interval(sa,sb)
                box=_value(z);c=_NEAR.divide(_NEAR.add(box.lo,box.hi),2)
                nc=n0+n1*c+n2*Interval(c).square();dc=d0+d1*c
                derivative_numerator=(n1+2*n2*c)*dc-nc*d1
                delta=z-c
                # Exact identity for quadratic N and affine D:
                # N(c+h)/D(c+h)=N(c)/D(c)
                # +h*((N'(c)D(c)-N(c)D')+N2*D(c)*h)/(D(c)*D(c+h)).
                return delta*(delta*(n2*dc)+derivative_numerator)/(denominator*dc)+nc/dc+section.sref
            a=z*q[j+3]+q[j]
            return section.sref+(_constant(threshold,z)-a)/q[6]
        curves.append(curve)
    return curves
