"""Bounded rectangle integration of verified interval value/gradient providers.

No material method or capacity is qualified here. The caller must supply an
actual interval extension of one function. A value-only cell uses range*area.
On a smooth cell, subtract an arbitrary constant gradient g: its linear term
integrates exactly, and |f' - g| bounds the remainder. This avoids requiring
second derivatives and retains a valid range bound at constitutive branches.
Every active leaf is retained; exhausted refinement returns unknown, not fail.
"""
from __future__ import annotations
from decimal import Decimal
import hashlib
import heapq
import json

from Design.SMRF_Interval_Material import Interval, Gradient, UnsupportedBox, _NEAR, _DOWN, _UP
from Design.SMRF_Interval_Taylor import Taylor

METHOD = 'outward_rectangle_integral_with_optional_gauss_remainder_v2_unqualified'


def _midpoint(interval):
    value = _NEAR.divide(_NEAR.add(interval.lo, interval.hi), Decimal(2))
    if not interval.lo <= value <= interval.hi:
        raise ValueError('Representable cell center required')
    return value


def _intersection(a, b):
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    if lo > hi:
        raise ValueError('Value and gradient providers disagree')
    return Interval(lo, hi)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def integrate_rectangle(value_provider, gradient_provider, rectangle, *,
                        provider_identity, absolute_error, maximum_cells=4096,fourth_provider=None):
    """Enclose each vector component's exact integral on a declared rectangle.

    Uniform parameter intervals may be captured by the providers. Their validity
    is a program premise, not established by the identity string or by samples.
    No claim about compatible strains, material ownership or cut coverage follows.
    """
    if not provider_identity or len(rectangle) != 2:
        raise ValueError('Explicit provider identity and rectangle required')
    if not isinstance(maximum_cells, int) or isinstance(maximum_cells, bool) or not 1 <= maximum_cells <= 100000:
        raise ValueError('Bounded positive cell budget required')
    domain = tuple(Interval(*v) for v in rectangle)
    if any(v.hi <= v.lo for v in domain):
        raise ValueError('Positive rectangle required')
    tolerances = [Interval(v) for v in absolute_error]
    if not tolerances or any(v.lo <= 0 for v in tolerances):
        raise ValueError('Explicit positive integral error per component required')
    sizes = [Interval(v.hi)-Interval(v.lo) for v in domain]
    calls = dict(value=0, gradient=0, fallback_cells=0,fourth=0,gauss_cells=0)
    def value(box):
        calls['value'] += 1
        result = tuple(value_provider(*box))
        if len(result) != len(tolerances) or any(not isinstance(v, Interval) for v in result):
            raise ValueError('Provider must return the declared finite interval vector')
        return result
    def cell(box):
        widths = [Interval(v.hi)-Interval(v.lo) for v in box]
        area = widths[0] * widths[1]
        direct = tuple(v*area for v in value(box))
        center = tuple(_midpoint(v) for v in box)
        smooth = False
        gauss_used=False
        if fourth_provider is not None:
            try:
                fourth=[]
                for axis in range(2):
                    arguments=[Taylor.variable((v.lo,v.hi),4) if i==axis else Taylor.constant(v,4)
                               for i,v in enumerate(box)]
                    calls['fourth']+=1
                    rows=tuple(fourth_provider(*arguments))
                    if len(rows)!=len(tolerances) or any(not isinstance(v,Taylor) or len(v.coefficients)!=5 for v in rows):
                        raise ValueError('Fourth derivative provider must return order4 interval Taylor coefficients')
                    fourth.append([v.coefficients[4] for v in rows])
                means=[(Interval(v.lo)+Interval(v.hi))/2 for v in box]
                offsets=[width/(2*Interval(3).sqrt()) for width in widths]
                quadrature=[Interval(0) for _ in tolerances]
                for signx,signy in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    sample=value((means[0]+signx*offsets[0],means[1]+signy*offsets[1]))
                    quadrature=[a+b*area/4 for a,b in zip(quadrature,sample)]
                # NIST DLMF3.5.19-20, n=2: E=L^5*f''''/4320.
                # Taylor coefficient4=f''''/24. Tensor-product positive
                # weights give E=area*(Lx^4*c4x+Ly^4*c4y)/180, as an interval.
                corrected=[v+area*(widths[0].square().square()*fourth[0][i]
                                   +widths[1].square().square()*fourth[1][i])/180
                           for i,v in enumerate(quadrature)]
                direct=tuple(_intersection(a,b) for a,b in zip(direct,corrected))
                gauss_used=True;calls['gauss_cells']+=1
            except UnsupportedBox:
                pass  # Keep independently valid range/gradient bounds.
        if gradient_provider is not None:
            try:
                calls['gradient'] += 1
                derivatives = tuple(gradient_provider(*[Gradient.variable((v.lo,v.hi),i,2) for i,v in enumerate(box)]))
                if len(derivatives) != len(tolerances) or any(not isinstance(v,Gradient) or len(v.derivatives)!=2 for v in derivatives):
                    raise ValueError('Provider must return two spatial interval derivatives per component')
                center_values = value(tuple(Interval(v) for v in center))
                means = [(Interval(v.lo)+Interval(v.hi))/2 for v in box]
                # These formulae remain valid when the represented center is
                # rounded away from the exact geometric midpoint.
                mean_absolute = [((Interval(c)-v.lo).square()+(Interval(v.hi)-c).square())/(2*w)
                                 for v,c,w in zip(box,center,widths)]
                improved=[]
                for f0, row in zip(center_values, derivatives):
                    linear, error = Interval(0), Interval(0)
                    for d, mean, c, absolute in zip(row.derivatives,means,center,mean_absolute):
                        g = _midpoint(d)
                        linear += g*(mean-c)
                        deviation = d-g
                        radius = max(deviation.lo.copy_abs(),deviation.hi.copy_abs())
                        error += radius*absolute
                    bound = (f0+linear+Interval(error.hi.copy_negate(),error.hi))*area
                    improved.append(bound)
                direct = tuple(_intersection(a,b) for a,b in zip(direct,improved))
                smooth=True
            except UnsupportedBox:
                calls['fallback_cells'] += 1
        return dict(box=box,integral=direct,smooth=smooth,gauss=gauss_used)

    leaves={0:cell(domain)};serial=0;heap=[]
    def priority(leaf):
        return max(float((Interval(v.hi)-v.lo).hi/(2*t.lo)) for v,t in zip(leaf['integral'],tolerances))
    heapq.heappush(heap,(-priority(leaves[0]),0))
    total=list(leaves[0]['integral'])
    def complete():
        return all((Interval(v.hi)-v.lo).hi <= (2*t).lo for v,t in zip(total,tolerances))
    while not complete() and len(leaves)<maximum_cells:
        _,index=heapq.heappop(heap);old=leaves.pop(index)
        axis=max(range(2),key=lambda i:float((Interval(old['box'][i].hi)-old['box'][i].lo).hi/sizes[i].lo))
        middle=_midpoint(old['box'][axis])
        if middle in (old['box'][axis].lo,old['box'][axis].hi):
            leaves[index]=old
            break
        children=[]
        for bounds in ((old['box'][axis].lo,middle),(middle,old['box'][axis].hi)):
            box=list(old['box']);box[axis]=Interval(*bounds)
            new=cell(tuple(box));serial+=1;leaves[serial]=new;children.append(new)
            heapq.heappush(heap,(-priority(new),serial))
        # Replace separately accumulated lower/upper endpoint sums, preserving
        # directed rounding. Subtracting the old INTERVAL would retain its
        # uncertainty and prevent refinement from narrowing the enclosure.
        updated=[]
        for j,v in enumerate(total):
            low=_DOWN.subtract(v.lo,old['integral'][j].lo)
            high=_UP.subtract(v.hi,old['integral'][j].hi)
            for child in children:
                low=_DOWN.add(low,child['integral'][j].lo)
                high=_UP.add(high,child['integral'][j].hi)
            updated.append(Interval(low,high))
        total=updated
    saved_leaves=[dict(rectangle=[v.record() for v in row['box']],
                       integral=[v.record() for v in row['integral']],smooth_gradient_used=row['smooth'],
                       fourth_derivative_used=row['gauss'])
                  for row in leaves.values()]
    result=dict(method=METHOD,provider_identity=provider_identity,rectangle=[v.record() for v in domain],
                integral=[v.record() for v in total],absolute_error=[str(v.lo) for v in tolerances],
                requested_error_met=complete(),status='integral_enclosed' if complete() else 'unknown',
                maximum_cells=maximum_cells,leaf_cells=len(leaves),calls=calls,leaves=saved_leaves,
                fourth_derivative_provider_supplied=fourth_provider is not None,
                provider_enclosure_validity_required=True,continuous_capacity_verified=False,
                engineering_qualified=False,physical_failure_claimed=False)
    result['payload_sha256']=_hash(result)
    return result


def replay_rectangle_integral(record,value_provider,gradient_provider,*,provider_identity,fourth_provider=None):
    if record['provider_identity']!=provider_identity:
        raise ValueError('Integral provider identity changed')
    if record['fourth_derivative_provider_supplied']!=(fourth_provider is not None):
        raise ValueError('Fourth derivative provider changed')
    rebuilt=integrate_rectangle(value_provider,gradient_provider,record['rectangle'],
        provider_identity=provider_identity,absolute_error=record['absolute_error'],maximum_cells=record['maximum_cells'],
        fourth_provider=fourth_provider)
    if rebuilt!=record:
        raise ValueError('Saved interval integral does not reconstruct')
    return rebuilt
