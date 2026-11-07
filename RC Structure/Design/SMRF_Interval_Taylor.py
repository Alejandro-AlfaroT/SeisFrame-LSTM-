"""Outward univariate Taylor coefficients for bounded quadrature derivatives.

Coefficient k encloses f^(k)/k! over every base point in the supplied interval.
No polynomial truncation is used as an approximation to f: the coefficients
bound derivatives, and a separate quadrature remainder theorem is required.
"""
from dataclasses import dataclass
from Design.SMRF_Interval_Material import Interval,as_interval,UnsupportedBox


@dataclass(frozen=True)
class Taylor:
    coefficients: tuple

    def __post_init__(self):
        if not 2<=len(self.coefficients)<=9 or any(not isinstance(v,Interval) for v in self.coefficients):
            raise ValueError('Finite interval Taylor coefficients of order1..8 required')

    @property
    def value(self):return self.coefficients[0]

    @property
    def derivatives(self):
        # Compatibility with the material expression's constant factory.
        # These are Taylor coefficients, NOT raw derivatives.
        return self.coefficients[1:]

    @classmethod
    def constant(cls,value,order):
        if not isinstance(order,int) or isinstance(order,bool) or not 1<=order<=8:
            raise ValueError('Taylor order1..8 required')
        return cls((as_interval(value),)+(Interval(0),)*order)

    @classmethod
    def variable(cls,bounds,order=4):
        if not isinstance(order,int) or isinstance(order,bool) or not 1<=order<=8:
            raise ValueError('Taylor order1..8 required')
        return cls((Interval(*bounds),Interval(1))+(Interval(0),)*(order-1))

    def _other(self,value):
        if isinstance(value,Taylor):
            if len(value.coefficients)!=len(self.coefficients):raise ValueError('Taylor order differs')
            return value
        return Taylor.constant(value,len(self.coefficients)-1)

    def __add__(self,other):
        b=self._other(other);return Taylor(tuple(x+y for x,y in zip(self.coefficients,b.coefficients)))
    __radd__=__add__
    def __neg__(self):return Taylor(tuple(-v for v in self.coefficients))
    def __sub__(self,other):return self+-self._other(other)
    def __rsub__(self,other):return self._other(other)+-self
    def __mul__(self,other):
        b=self._other(other)
        return Taylor(tuple(sum((self.coefficients[k]*b.coefficients[n-k] for k in range(n+1)),Interval(0))
                            for n in range(len(self.coefficients))))
    __rmul__=__mul__
    def reciprocal(self):
        x=self.coefficients;y=[x[0].reciprocal()]
        for n in range(1,len(x)):
            y.append(-y[0]*sum((x[k]*y[n-k] for k in range(1,n+1)),Interval(0)))
        return Taylor(tuple(y))
    def __truediv__(self,other):return self*self._other(other).reciprocal()
    def __rtruediv__(self,other):return self._other(other)*self.reciprocal()
    def square(self):
        product=self*self
        return Taylor((self.value.square(),)+product.coefficients[1:])
    def sqrt(self):
        x=self.coefficients;y=[x[0].sqrt()]
        if y[0].lo<=0:raise UnsupportedBox('Taylor_square_root_not_strictly_positive')
        for n in range(1,len(x)):
            y.append((x[n]-sum((y[k]*y[n-k] for k in range(1,n)),Interval(0)))/(2*y[0]))
        return Taylor(tuple(y))
    def exp(self):
        x=self.coefficients;y=[x[0].exp()]
        for n in range(1,len(x)):
            y.append(sum((k*x[k]*y[n-k] for k in range(1,n+1)),Interval(0))/n)
        return Taylor(tuple(y))
    def log(self):
        x=self.coefficients;inverse=self.reciprocal().coefficients;y=[x[0].log()]
        for n in range(1,len(x)):
            y.append(sum((k*x[k]*inverse[n-k] for k in range(1,n+1)),Interval(0))/n)
        return Taylor(tuple(y))
