import math
import json
from decimal import Decimal,localcontext
import unittest
from Design.SMRF_Spatial_Floor_Section import SpatialFloorSection
from Design.SMRF_Interval_Material import Interval
from Design.SMRF_Interval_Spatial_Section import section_resultant_enclosure,replay_section_resultant_enclosure


def section(*,web=False,credit=True,transverse=False):
    rows=[]
    if transverse:
        rows=[dict(id='rowY',area_in2=.2,fy_ksi=60.,es_ksi=29000.,z_in=2.,db_in=.5,
                   run_interval_in=(-5.,5.),physical_bar_ids=['transverse1'],accounting_length_in=10.)]
    return SpatialFloorSection(10.,4.,ec_ksi=3600.,axis='x',slab_interval_in=(-5.,5.),
        web=dict(id='web',interval_in=(-2.,2.),z_interval_in=(-10.,-5.)) if web else None,
        reference_s_in=0.,longitudinal_bars=[dict(id='barX',area_in2=.79,fy_ksi=60.,es_ksi=29000.,
            z_in=-5. if web else 0.,s_in=0.,db_in=1.,strength_credited=credit)],
        transverse_rows=rows,phi_requirements={'flexure':.65},concrete_peak_strain=.002)


def pi_decimal():
    def atan_inverse(n):
        x=Decimal(1)/n;power=x;answer=x;k=1
        while True:
            power=-power*x*x;term=power/(2*k+1);answer+=term;k+=1
            if abs(term)<Decimal('1e-112'):return answer
    return 16*atan_inverse(5)-4*atan_inverse(239)


def stress_decimal():
    n=Decimal(.8)+Decimal(4.)*Decimal(6.894757293168)/17
    ratio=-Decimal(-.0004)/Decimal(.002)
    return -4*n*ratio/(n-1+(n*ratio.ln()).exp())


class ActualIntervalSectionTests(unittest.TestCase):
    def test_circular_void_and_steel_against_independent_uniform_section(self):
        s=section();q=[(-.0004,-.0004),(.0002,.0002),(0,0)]+[(0,0)]*4
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['.002']*7,maximum_cells_per_term=2048)
        self.assertEqual(result['status'],'strain_box_resultants_enclosed')
        with localcontext() as ctx:
            ctx.prec=120
            area=pi_decimal()*Decimal(s.longitudinal[0]['radius'])**2
            concrete=stress_decimal();steel=Decimal(.79)*Decimal(29000.)*Decimal(-.0004)
            expected=((100-area)*concrete+steel)/10
        self.assertTrue(Interval(*result['nominal_resultants'][0]).contains(expected))
        self.assertTrue(Interval(*result['nominal_resultants'][3]).contains(0))
        self.assertFalse(result['continuous_capacity_verified'])

    def test_straddling_circle_is_subtracted_once_across_slab_and_web(self):
        s=section(web=True);q=[(-.0004,-.0004),(.0002,.0002),(0,0)]+[(0,0)]*4
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['.01']*7,maximum_cells_per_term=2048)
        self.assertEqual(result['status'],'strain_box_resultants_enclosed')
        with localcontext() as ctx:
            ctx.prec=120
            area=pi_decimal()*Decimal(s.longitudinal[0]['radius'])**2
            concrete=stress_decimal();steel=Decimal(.79)*Decimal(29000.)*Decimal(-.0004)
            force=((120-area)*concrete+steel)/10
            moment=((-150+5*area)*concrete-5*steel)/10
        self.assertTrue(Interval(*result['nominal_resultants'][0]).contains(force))
        self.assertTrue(Interval(*result['nominal_resultants'][3]).contains(moment))
        ids=[r['identity'] for r in result['terms']]
        self.assertIn('slab/longitudinal/barX',ids);self.assertIn('web/longitudinal/barX',ids)

    def test_transverse_void_and_credited_length_are_separate(self):
        s=section(transverse=True);q=[(-.0004,-.0004),(.0002,.0002),(0,0)]+[(0,0)]*4
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['.004']*7,maximum_cells_per_term=2048)
        self.assertEqual(result['status'],'strain_box_resultants_enclosed')
        with localcontext() as ctx:
            ctx.prec=120
            bar,row=s.longitudinal[0],s.transverse[0]
            void=pi_decimal()*(Decimal(bar['radius'])**2+Decimal(row['radius'])**2*Decimal(row['count_per_length_in'])*10)
            expected=((100-void)*stress_decimal()+Decimal(.79)*Decimal(29000.)*Decimal(-.0004))/10
            ysteel=Decimal(row['area_in2'])*Decimal(row['count_per_length_in'])*Decimal(29000.)*Decimal(.0002)
        self.assertTrue(Interval(*result['nominal_resultants'][0]).contains(expected))
        self.assertTrue(Interval(*result['nominal_resultants'][1]).contains(ysteel))

    def test_uncredited_bar_void_remains_while_steel_credit_is_zero(self):
        s=section(credit=False);q=[(.001,.001),(.001,.001),(0,0)]+[(0,0)]*4
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['1e-15']*7,maximum_cells_per_term=4)
        self.assertEqual(result['status'],'strain_box_resultants_enclosed')
        self.assertTrue(all(Interval(*v).contains(0) for v in result['nominal_resultants']))
        self.assertEqual(len(result['terms']),2)
        self.assertEqual(Interval(*result['steel'][0]['stress']),Interval(0))
        self.assertEqual(replay_section_resultant_enclosure(s,json.loads(json.dumps(result))),result)
        result['continuous_capacity_verified']=True
        with self.assertRaises(ValueError):replay_section_resultant_enclosure(s,result)

    def test_budget_and_material_unknowns_never_become_capacity_failure(self):
        s=section();q=[(-.0004,-.0004),(.0002,.0002),(0,0)]+[(0,0)]*4
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['1e-12']*7,maximum_cells_per_term=1)
        self.assertEqual(result['status'],'unknown');self.assertFalse(result['physical_failure_claimed'])
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['.01']*7,maximum_terms=1)
        self.assertEqual(result['reason'],'declared_term_budget_exhausted')
        q[0]=(-.004,-.004)
        result=section_resultant_enclosure(s,q,absolute_nominal_error=['.01']*7,maximum_cells_per_term=1)
        self.assertEqual(result['status'],'unknown');self.assertFalse(result['physical_failure_claimed'])


if __name__=='__main__':unittest.main()
