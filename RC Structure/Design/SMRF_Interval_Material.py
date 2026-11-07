"""Outward bounds for the existing material law on a smooth strain box.

This is a constitutive building block, not a section capacity certificate.
It does not integrate concrete/voids, establish material ownership, prove
continuous cut coverage, or activate a design profile. Unsupported branch
crossings return unknown. There is no sampling-based enclosure.

The spectral formula is the analytic two-by-two eigensystem of the same
Polak/Thorenfeldt prepeak law as SMRF_Layered_Slab_Section. Forward interval
derivatives enclose every gradient on a box wholly within the selected smooth
branches. Decimal basic operations round outwards; correctly rounded sqrt,
ln and exp are widened to neighboring representable values. A variable power
is exp(n*ln(r)), never the 'almost always correctly rounded' Decimal power.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import hashlib
import json

METHOD = 'actual_prepeak_material_box_enclosure_v3_backbone_domain_unqualified'
PRECISION = 70
_DOWN = Context(prec=PRECISION, rounding=ROUND_FLOOR)
_UP = Context(prec=PRECISION, rounding=ROUND_CEILING)
_NEAR = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)


def _decimal(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a material coefficient')
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError('Finite exact material coefficient required')
    return result


@dataclass(frozen=True)
class Interval:
    lo: Decimal
    hi: Decimal

    def __init__(self, lo, hi=None):
        lo, hi = _decimal(lo), _decimal(lo if hi is None else hi)
        if lo > hi:
            raise ValueError('Ordered enclosure required')
        object.__setattr__(self, 'lo', lo)
        object.__setattr__(self, 'hi', hi)

    def __add__(self, other):
        b = as_interval(other)
        return Interval(_DOWN.add(self.lo, b.lo), _UP.add(self.hi, b.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -as_interval(other)

    def __rsub__(self, other):
        return as_interval(other) + -self

    def __mul__(self, other):
        b = as_interval(other)
        pairs = [(a, v) for a in (self.lo, self.hi) for v in (b.lo, b.hi)]
        return Interval(min(_DOWN.multiply(a, v) for a, v in pairs),
                        max(_UP.multiply(a, v) for a, v in pairs))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ValueError('Reciprocal enclosure crosses zero')
        return Interval(_DOWN.divide(Decimal(1), self.hi),
                        _UP.divide(Decimal(1), self.lo))

    def __truediv__(self, other):
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other):
        return as_interval(other) * self.reciprocal()

    def square(self):
        if self.lo <= 0 <= self.hi:
            return Interval(0, max(_UP.multiply(v, v) for v in (self.lo, self.hi)))
        return Interval(min(_DOWN.multiply(v, v) for v in (self.lo, self.hi)),
                        max(_UP.multiply(v, v) for v in (self.lo, self.hi)))

    def _monotone(self, name):
        operation = getattr(_NEAR, name)
        return Interval(_NEAR.next_minus(operation(self.lo)),
                        _NEAR.next_plus(operation(self.hi)))

    def sqrt(self):
        if self.lo < 0:
            raise ValueError('Negative square-root enclosure')
        value = self._monotone('sqrt')
        return Interval(0 if self.lo == 0 else value.lo, value.hi)

    def log(self):
        if self.lo <= 0:
            raise ValueError('Log enclosure must be strictly positive')
        return self._monotone('ln')

    def exp(self):
        return self._monotone('exp')

    def contains(self, value):
        return self.lo <= _decimal(value) <= self.hi

    def record(self):
        return [str(self.lo), str(self.hi)]


def as_interval(value):
    return value if isinstance(value, Interval) else Interval(value)


@dataclass(frozen=True)
class Gradient:
    value: Interval
    derivatives: tuple

    @classmethod
    def constant(cls, value, dimensions):
        return cls(as_interval(value), (Interval(0),) * dimensions)

    @classmethod
    def variable(cls, bounds, index, dimensions):
        return cls(Interval(*bounds), tuple(Interval(int(i == index)) for i in range(dimensions)))

    def _other(self, other):
        if isinstance(other, Gradient):
            if len(other.derivatives) != len(self.derivatives):
                raise ValueError('Gradient dimension mismatch')
            return other
        return Gradient.constant(other, len(self.derivatives))

    def __add__(self, other):
        other = self._other(other)
        return Gradient(self.value + other.value,
                        tuple(a + b for a, b in zip(self.derivatives, other.derivatives)))

    __radd__ = __add__

    def __neg__(self):
        return Gradient(-self.value, tuple(-v for v in self.derivatives))

    def __sub__(self, other):
        return self + -self._other(other)

    def __rsub__(self, other):
        return self._other(other) + -self

    def __mul__(self, other):
        other = self._other(other)
        return Gradient(self.value * other.value,
                        tuple(a * other.value + self.value * b
                              for a, b in zip(self.derivatives, other.derivatives)))

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = self.value.reciprocal()
        return Gradient(inverse, tuple(-v * inverse.square() for v in self.derivatives))

    def __truediv__(self, other):
        return self * self._other(other).reciprocal()

    def __rtruediv__(self, other):
        return self._other(other) * self.reciprocal()

    def square(self):
        return Gradient(self.value.square(), tuple(2 * self.value * v for v in self.derivatives))

    def sqrt(self):
        value = self.value.sqrt()
        if value.lo <= 0:
            raise UnsupportedBox('spectral_gap_not_strictly_positive')
        return Gradient(value, tuple(v / (2 * value) for v in self.derivatives))

    def log(self):
        return Gradient(self.value.log(), tuple(v / self.value for v in self.derivatives))

    def exp(self):
        value = self.value.exp()
        return Gradient(value, tuple(value * v for v in self.derivatives))


class UnsupportedBox(ValueError):
    """Unproved enclosure domain; never a demonstrated physical failure."""


def _principal_stress(strain, transverse, fc, peak):
    if strain.value.lo > 0:
        # The original law still reports n<=1 as outside its model domain
        # when the stress happens to be zero on a tensile branch.
        _principal_stress_bounds(strain.value, transverse.value, fc, peak)
        return type(strain).constant(0, len(strain.derivatives))
    if strain.value.hi >= 0:
        raise UnsupportedBox('concrete_tension_branch_crossing')
    if strain.value.lo < peak.hi.copy_negate():
        raise UnsupportedBox('concrete_prepeak_domain_not_enclosed')
    if transverse.value.hi <= 0:
        tension = type(strain).constant(0, len(strain.derivatives))
    elif transverse.value.lo > 0:
        tension = transverse
    else:
        raise UnsupportedBox('transverse_tension_branch_crossing')
    softening = .8 + .34 * tension / peak
    if softening.value.hi < 1:
        effective = type(strain).constant(fc, len(strain.derivatives))
    elif softening.value.lo > 1:
        effective = type(strain).constant(fc, len(strain.derivatives)) / softening
    else:
        raise UnsupportedBox('concrete_softening_branch_crossing')
    exponent = .8 + effective * 6.894757293168 / 17
    if exponent.value.lo <= 1:
        raise UnsupportedBox('concrete_backbone_n_not_above_one')
    ratio = -strain / peak
    power = (exponent * ratio.log()).exp()
    return -effective * exponent * ratio / (exponent - 1 + power)


def concrete_stress_gradient(ex, ey, engineering_shear, *, fc_ksi, peak_strain):
    """Three stress components and gradients on a proved smooth box.

    Input Gradients can include affine spatial coordinates as interval constants.
    Repeated compression eigenstrains currently remain unsupported; a strictly
    tensile tensor has identically zero concrete response even at a repeated root.
    """
    fc, peak = Interval(fc_ksi), Interval(peak_strain)
    if fc.lo <= 0 or not 0 < peak.lo <= Decimal('.003'):
        raise ValueError('Positive strength and declared prepeak strain required')
    center = (ex + ey) / 2
    diff = (ex - ey) / 2
    half_shear = engineering_shear / 2
    gap2 = diff.square() + half_shear.square()
    gap_value = gap2.value.sqrt()
    if (center.value - gap_value).lo > 0:
        low, high = center.value-gap_value, center.value+gap_value
        _principal_stress_bounds(low, high, fc, peak)
        _principal_stress_bounds(high, low, fc, peak)
        return (type(ex).constant(0, len(ex.derivatives)),) * 3
    gap = gap2.sqrt()
    low, high = center - gap, center + gap
    low_stress = _principal_stress(low, high, fc, peak)
    high_stress = _principal_stress(high, low, fc, peak)
    mean = (high_stress + low_stress) / 2
    scale = (high_stress - low_stress) / (2 * gap)
    return mean + scale * diff, mean - scale * diff, scale * half_shear


def steel_stress_gradient(strain, *, fy_ksi, es_ksi=29000., strain_limit=.01):
    fy, es, limit = Interval(fy_ksi), Interval(es_ksi), Interval(strain_limit)
    if min(fy.lo, es.lo, limit.lo) <= 0:
        raise ValueError('Positive steel properties and limit required')
    if strain.value.lo < limit.hi.copy_negate() or strain.value.hi > limit.hi:
        raise UnsupportedBox('steel_strain_domain_not_enclosed')
    trial = strain * es
    if trial.value.lo > fy.hi:
        return Gradient.constant(fy, len(strain.derivatives))
    if trial.value.hi < fy.hi.copy_negate():
        return Gradient.constant(-fy, len(strain.derivatives))
    if trial.value.lo > fy.hi.copy_negate() and trial.value.hi < fy.lo:
        return trial
    raise UnsupportedBox('steel_yield_branch_crossing')


def _positive_part(value):
    return Interval(max(Decimal(0), value.lo), max(Decimal(0), value.hi))


def _prepeak_power(ratio, exponent):
    """Whole-box r**n for 0<=r<=1 and n>1, including the tension cutoff."""
    if ratio.lo < 0 or ratio.hi > 1 or exponent.lo <= 1:
        raise UnsupportedBox('power_box_outside_prepeak_domain')
    low = (Interval(ratio.lo).log() * exponent.hi).exp().lo if ratio.lo else Decimal(0)
    high = (Interval(ratio.hi).log() * exponent.lo).exp().hi if ratio.hi else Decimal(0)
    return Interval(low, high)


def _principal_stress_bounds(strain, transverse, fc, peak):
    if strain.lo < peak.hi.copy_negate():
        raise UnsupportedBox('concrete_prepeak_domain_not_enclosed')
    softening = .8 + .34 * _positive_part(transverse) / peak
    effective = fc / Interval(max(Decimal(1), softening.lo), max(Decimal(1), softening.hi))
    exponent = .8 + effective * 6.894757293168 / 17
    if exponent.lo <= 1:
        raise UnsupportedBox('concrete_backbone_n_not_above_one')
    ratio = _positive_part(-strain / peak)
    power = _prepeak_power(ratio, exponent)
    return -effective * exponent * ratio / (exponent - 1 + power)


def concrete_stress_bounds(ex, ey, engineering_shear, *, fc_ksi, peak_strain):
    """Continuous stress enclosure, including branch changes/repeated roots.

    No derivative or C1 claim is made. If a principal gap includes zero, the
    orthogonal eigenprojector diagonal is in [0,1] and off-diagonal in [-.5,.5].
    Using these full ranges encloses any eigenvector orientation. At a repeated
    root the principal stresses coincide, so this expression remains valid.
    The unproved orientation widens the bound; it is never assigned arbitrarily.
    """
    ex, ey, g = map(as_interval, (ex, ey, engineering_shear))
    fc, peak = Interval(fc_ksi), Interval(peak_strain)
    if fc.lo <= 0 or not 0 < peak.lo <= Decimal('.003'):
        raise ValueError('Positive strength and declared prepeak strain required')
    center, diff, half_shear = (ex + ey) / 2, (ex - ey) / 2, g / 2
    gap = (diff.square() + half_shear.square()).sqrt()
    low, high = center - gap, center + gap
    sl = _principal_stress_bounds(low, high, fc, peak)
    sh = _principal_stress_bounds(high, low, fc, peak)
    if gap.lo > 0:
        mean, scale = (sh + sl) / 2, (sh - sl) / (2 * gap)
        return mean + scale * diff, mean - scale * diff, scale * half_shear
    difference = sh - sl
    normal = sl + Interval(0, 1) * difference
    return normal, normal, Interval('-.5', '.5') * difference


def steel_stress_bounds(strain, *, fy_ksi, es_ksi=29000., strain_limit=.01):
    """Exact clipped stress range over the full elastic/yield branch union."""
    strain = as_interval(strain)
    fy, es, limit = Interval(fy_ksi), Interval(es_ksi), Interval(strain_limit)
    if min(fy.lo, es.lo, limit.lo) <= 0:
        raise ValueError('Positive steel properties and limit required')
    if strain.lo < limit.hi.copy_negate() or strain.hi > limit.hi:
        raise UnsupportedBox('steel_strain_domain_not_enclosed')
    trial = strain * es
    low = fy.lo.copy_negate()
    return Interval(min(fy.hi, max(low, trial.lo)), min(fy.hi, max(low, trial.hi)))


def concrete_density_value_enclosure(q_box, s_box, z_box, *, axis, reference_s_in,
                                     fc_ksi, peak_strain, region='slab'):
    """Whole-box nominal density only; branch-union bounds need no smoothness.

    This may support an eventual uniform residual bound for an explicitly
    supplied compatible strain field. No root existence follows from this
    operation, and neither finite quadrature nor cut coverage is certified.
    """
    if axis not in ('x', 'y') or region not in ('slab', 'web') or len(q_box) != 7:
        raise ValueError('Actual axis, slab/web region and seven strain boxes required')
    fc, peak = Interval(fc_ksi), Interval(peak_strain)
    if fc.lo <= 0 or not 0 < peak.lo <= Decimal('.003'):
        raise ValueError('Positive strength and declared prepeak strain required')
    q = [Interval(*v) for v in q_box]
    s, z = Interval(*s_box), Interval(*z_box)
    j = 0 if axis == 'x' else 1
    offset = s - Interval(reference_s_in)
    strains = [q[k] + q[k + 3] * z for k in range(3)]
    strains[j] = strains[j] + q[6] * offset
    inputs = dict(q_box=[v.record() for v in q], s_box=s.record(), z_box=z.record(),
                  axis=axis, reference_s_in=str(_decimal(reference_s_in)),
                  fc_ksi=str(fc.lo), peak_strain=str(peak.lo), region=region)
    result = dict(method=METHOD + '/continuous_values', inputs=inputs, precision=PRECISION,
                  derivative_enclosure_provided=False, continuous_capacity_verified=False,
                  section_integration_verified=False, engineering_qualified=False,
                  physical_failure_claimed=False)
    try:
        if region == 'slab':
            stress = concrete_stress_bounds(*strains, fc_ksi=fc_ksi, peak_strain=peak_strain)
        else:
            normal = _principal_stress_bounds(strains[j], Interval(0), fc, peak)
            zero = Interval(0)
            stress = (normal, zero, zero) if j == 0 else (zero, normal, zero)
        density = list(stress) + [v * z for v in stress] + [stress[j] * offset]
        result.update(status='continuous_material_value_box_enclosed',
                      nominal_density=[v.record() for v in density])
    except UnsupportedBox as exc:
        result.update(status='unknown', reason=str(exc))
    result['payload_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def replay_density_value_enclosure(record):
    rebuilt = concrete_density_value_enclosure(**record['inputs'])
    if rebuilt != record:
        raise ValueError('Saved material value enclosure does not reconstruct')
    return rebuilt


def concrete_density_enclosure(q_box, s_box, z_box, *, axis, reference_s_in,
                               fc_ksi, peak_strain, region='slab'):
    """Nominal seven-resultant concrete density and d/dq, no area integration.

    The q and spatial boxes use the SpatialFloorSection convention. Slab and
    disjoint web are separate regions; no phi, width or void subtraction is
    performed here. Region ownership must be supplied by the eventual integrator.
    Decimal strings preserve outward bounds in the saved record.
    """
    if axis not in ('x', 'y') or region not in ('slab', 'web') or len(q_box) != 7:
        raise ValueError('Actual axis, slab/web region and seven strain boxes required')
    if not 0 < _decimal(fc_ksi) or not 0 < _decimal(peak_strain) <= Decimal('.003'):
        raise ValueError('Positive strength and declared prepeak strain required')
    s, z = Interval(*s_box), Interval(*z_box)
    q = [Gradient.variable(v, i, 7) for i, v in enumerate(q_box)]
    j = 0 if axis == 'x' else 1
    offset = s - Interval(reference_s_in)
    strains = [q[k] + q[k + 3] * z for k in range(3)]
    strains[j] = strains[j] + q[6] * offset
    inputs = dict(q_box=[Interval(*v).record() for v in q_box],
                  s_box=s.record(), z_box=z.record(), axis=axis,
                  reference_s_in=str(_decimal(reference_s_in)),
                  fc_ksi=str(_decimal(fc_ksi)), peak_strain=str(_decimal(peak_strain)), region=region)
    result = dict(method=METHOD, inputs=inputs, precision=PRECISION,
                  continuous_capacity_verified=False, section_integration_verified=False,
                  engineering_qualified=False, physical_failure_claimed=False)
    try:
        if region == 'slab':
            stress = concrete_stress_gradient(*strains, fc_ksi=fc_ksi, peak_strain=peak_strain)
        else:
            zero = Gradient.constant(0, 7)
            normal = _principal_stress(strains[j], zero, Interval(fc_ksi), Interval(peak_strain))
            stress = (normal, zero, zero) if j == 0 else (zero, normal, zero)
        density = list(stress) + [v * z for v in stress] + [stress[j] * offset]
        result.update(status='smooth_material_box_enclosed',
                      nominal_density=[v.value.record() for v in density],
                      jacobian_density=[[d.record() for d in v.derivatives] for v in density])
    except UnsupportedBox as exc:
        result.update(status='unknown', reason=str(exc))
    result['payload_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def replay_density_enclosure(record):
    rebuilt = concrete_density_enclosure(**record['inputs'])
    if rebuilt != record:
        raise ValueError('Saved material enclosure does not reconstruct')
    return rebuilt
