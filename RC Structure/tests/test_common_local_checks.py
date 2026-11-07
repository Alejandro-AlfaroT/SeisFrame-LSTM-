import unittest
from Design.SMRF_Common_Local_Checks import credited_steel,minimum_area_status,support_steel_specs


class CommonLocalTests(unittest.TestCase):
    def test_full_fy_unknown_is_not_turned_into_demand_failure(self):
        bars=[dict(id='a',area_in2=.2),dict(id='b',area_in2=.31)]
        proof=[dict(physical_bar_ids=['a'],status='fail',full_fy_development_domain=dict(status='computed',interval_in=[3.,20.]))]
        at=credited_steel(bars,proof,10.)
        self.assertAlmostEqual(at['provided_developed_in2'],.2);self.assertAlmostEqual(at['installed_area_in2'],.51)
        self.assertEqual(minimum_area_status(.15,at),'pass')
        self.assertEqual(minimum_area_status(.4,at),'unknown')
        self.assertEqual(minimum_area_status(.6,at),'fail')
        end=credited_steel(bars,proof,2.)
        self.assertEqual(end['provided_developed_in2'],0.)
        self.assertEqual(minimum_area_status(.15,end),'unknown')

    def test_edge_free_face_not_fabricated_and_all_actual_local_faces_retained(self):
        fp=dict(x_min_in=-10.,x_max_in=110.,y_min_in=-10.,y_max_in=110.)
        supports=[dict(id=i+1,x=x,y=y,b_in=20.,h_in=20.) for i,(x,y) in enumerate([(0.,0.),(50.,0.),(50.,50.)])]
        definition=dict(floors=[dict(level=1,supports=supports)],columns=[dict(j=s['id']) for s in supports])
        bars=[]
        for axis in ('x','y'):
            for face in ('top','bottom'):
                for v in range(0,101,5):bars.append(dict(id=f'{axis}/{face}/{v}',operator_owner='slab',axis=axis,
                    mat_name=axis+'_'+face,s_in=float(v),db_in=.5,straight_run_interval_in=[-7.,107.],area_in2=.2))
        inventory=dict(slab_thickness_in=8.,floors={'1':dict(footprint=fp,bars=bars)})
        specs,groups=support_steel_specs(definition,inventory)
        self.assertEqual(len(specs),4+6+8)
        self.assertEqual({r['station_in'] for r in groups[0]['axes']['x']},{10.})
        self.assertEqual({r['station_in'] for r in groups[2]['axes']['y']},{40.,60.})
        self.assertTrue(all(r['physical_bar_ids'] for r in specs))
        self.assertEqual(len({r['cut_id'] for r in specs}),18)
