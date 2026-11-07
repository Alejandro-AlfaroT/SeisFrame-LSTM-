"""Whole-interval torsion exemption for the actual stored quadratic forces.

This proves only the strict ACI 318-19 22.7.4.1 threshold inequality. It never
turns finite samples of a nonlinear material reserve into a continuous proof.
The Bernstein basis is nonnegative and sums to one on [0,1]; bounds on all
coefficients therefore bound every point. Rational arithmetic and a bracketed
square root avoid certifying a marginal inequality through floating rounding.
"""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np

METHOD='continuous_quadratic_torsion_threshold_bernstein_v1'


def _f(value):
    if isinstance(value,(bool,np.bool_)) or not math.isfinite(float(value)):
        raise ValueError('Finite numeric physical input required')
    return Fraction(float(value))


def _sqrt_bounds(value):
    root=math.sqrt(float(value));lo=hi=root
    while Fraction(lo)**2>value:lo=math.nextafter(lo,-math.inf)
    while Fraction(hi)**2<value:hi=math.nextafter(hi,math.inf)
    return Fraction(lo),Fraction(hi)


def _outward(value,up):
    result=float(value)
    if (Fraction(result)<value if up else Fraction(result)>value):
        result=math.nextafter(result,math.inf if up else -math.inf)
    return result


def _split(values):
    rows=[values]
    while len(rows[-1])>1:
        rows.append([((a[0]+b[0])/2,(a[1]+b[1])/2) for a,b in zip(rows[-1][:-1],rows[-1][1:])])
    return [r[0] for r in rows],[r[-1] for r in reversed(rows)]


def threshold_interval(section,coefficients,*,phi=.75,maximum_depth=20,maximum_leaves=4096):
    """Prove |T(x)| < phi*Tth(N(x)) for every x in a physical interval.

    coefficients are ascending-power rows [constant,x,x^2], columns [N,T].
    They must come from a same-section, same-case polynomial recovery whose
    degree and physical interval have been established by the caller. No mesh,
    guard inventory, support development or nonlinear strength is inferred.
    """
    c=np.asarray(coefficients,float)
    if c.shape!=(3,2) or not np.isfinite(c).all():raise ValueError('Three finite quadratic N/T coefficient rows required')
    if section.get('status')!='pass' or not 0<phi<=.75:raise ValueError('Actual passing section and phi<=.75 required')
    if type(maximum_depth) is not int or not 0<=maximum_depth<=30 or type(maximum_leaves) is not int or not 1<=maximum_leaves<=65536:
        raise ValueError('Finite positive subdivision bounds required')
    ag,acp,pcp,lam,fc=(_f(section[k]) for k in ('Ag_in2','Acp_in2','pcp_in','lambda','fc_ksi'))
    if min(ag,acp,pcp,lam,fc)<=0:raise ValueError('Positive actual section properties required')
    ph=_f(phi);r2=min(Fraction(10000),fc*1000);rl,rh=_sqrt_bounds(r2)
    # g = (phi*lambda*r*Acp^2/(1000*pcp))^2 * (1-N/k) - T^2,
    # k=4*Ag*lambda*r/1000. g>0 implies the axial radicand is also positive.
    constant=ph**2*lam**2*r2*acp**4/(1000000*pcp**2)
    scale=ph**2*lam*acp**4/(4000*ag*pcp**2);sl,sh=scale*rl,scale*rh
    n=[_f(v) for v in c[:,0]];t=[_f(v) for v in c[:,1]]
    power=[]
    for degree in range(5):
        square=sum((t[i]*t[degree-i] for i in range(3) if 0<=degree-i<3),Fraction())
        low=high=(constant if degree==0 else Fraction())-square
        if degree<3:
            products=(sl*n[degree],sh*n[degree]);low-=max(products);high-=min(products)
        power.append((low,high))
    bernstein=[]
    for k in range(5):
        weights=[Fraction(math.comb(k,i),math.comb(4,i)) for i in range(k+1)]
        bernstein.append(tuple(sum((weights[i]*power[i][j] for i in range(k+1)),Fraction()) for j in (0,1)))
    stack=[(Fraction(),Fraction(1),bernstein,0)];leaves=[];unresolved=[];nonexempt=None;processed=0
    while stack:
        a,b,values,depth=stack.pop();processed+=1
        lower=min(v[0] for v in values);upper=max(v[1] for v in values)
        if lower>0:
            leaves.append(dict(interval=[float(a),float(b)],margin_lower_kip2in2=_outward(lower,False),
                               margin_upper_kip2in2=_outward(upper,True)));continue
        left,right=_split(values);mid=(a+b)/2
        candidates=[(a,values[0]),(mid,left[-1]),(b,values[-1])]
        found=next(((x,v) for x,v in candidates if v[1]<=0),None)
        if found is not None:
            x,v=found;nonexempt=dict(fraction=float(x),margin_upper_kip2in2=_outward(v[1],True));break
        if depth>=maximum_depth or len(stack)+len(leaves)+len(unresolved)+2>maximum_leaves:
            unresolved.append(dict(interval=[float(a),float(b)],reason='declared_subdivision_bound'));continue
        stack.extend([(mid,b,right,depth+1),(a,mid,left,depth+1)])
    passed=nonexempt is None and not unresolved and not stack
    return dict(method=METHOD,status='pass' if passed else 'not_evaluated',
        below_threshold_everywhere=passed,section_id=section.get('section_id'),coefficients=c.tolist(),phi=float(phi),
        basis='strict positive quartic threshold margin; exact rational Bernstein bounds with enclosed sqrt(fc)',
        certified_intervals=leaves,nonexempt_witness=nonexempt,unresolved_intervals=unresolved,
        processed_subintervals=processed,maximum_depth=maximum_depth,maximum_leaves=maximum_leaves,
        nonlinear_material_reserve_certified=False,mesh_qualification_inferred=False)


def continuous_member_coverage(raw):
    """Independently cover every member/case interval, including primary/P5.

    Above-threshold reinforcement remains a separate unresolved continuous
    check. A nonexempt interval is not itself an engineering failure.
    """
    cases=raw.get('analysis_cases',[]);physical=raw.get('polynomial_audits',[])
    guarded=raw.get('guard_polynomial_audits',[]);members={}
    for tag in raw['selected_member_tags']:
        proofs=[];issues=[];section=raw['sections'][str(tag)]
        for case in cases:
            cid=case['analysis_case_id'];records=[r for r in raw['records'] if r['member_tag']==tag and r['case_id']==cid]
            rows=sorted([r for r in physical if r['member_tag']==tag and r['case_id']==cid],key=lambda r:r['interval_in'][0])
            if not records or not rows:
                issues.append('missing_physical_interval_inventory:'+cid);continue
            expected=[min(r['station_in'] for r in records),max(r['station_in'] for r in records)]
            cursor=expected[0]
            for row in rows:
                a,b=map(float,row['interval_in'])
                if a!=cursor or not a<b:issues.append('gap_overlap_or_invalid_physical_interval:'+cid)
                cursor=b
                exact=[g for g in guarded if g['member_tag']==tag and g['case_id']==cid and g['interval_in']==row['interval_in']]
                if case['branch']=='actual_physical_ELF':
                    if len(exact)!=1 or not exact[0]['guards']:
                        issues.append('missing_or_duplicate_primary_P5_interval:'+cid);continue
                    inputs=exact[0]['guards']
                else:inputs=[dict(kind='physical',guard_suffix='',coefficients=row['coefficients'])]
                seen=set()
                for item in inputs:
                    key=(item['guard_suffix'],item['kind'])
                    if key in seen:raise ValueError('Duplicate continuous threshold guard')
                    seen.add(key)
                    coeff=np.asarray(item['coefficients'],float)
                    proof=threshold_interval(section,coeff[:,:2])
                    proofs.append(dict(case_id=cid,interval_in=[a,b],guard_suffix=key[0],kind=key[1],proof=proof))
            if cursor!=expected[1]:issues.append('incomplete_clear_span_interval_inventory:'+cid)
        if not cases:issues.append('missing_analysis_case_inventory')
        passed=bool(proofs) and not issues and all(r['proof']['below_threshold_everywhere'] for r in proofs)
        members[str(tag)]=dict(status='pass' if passed else 'not_evaluated',proofs=proofs,inventory_issues=issues,
            basis='whole-span exact-threshold exemption' if passed else 'continuous nonlinear reserve or interval evidence remains required')
    return dict(method=METHOD,members=members,nonlinear_material_reserve_certified=False)
