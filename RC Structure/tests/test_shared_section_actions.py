"""Independent equilibrium/section checks, no OpenSees domain or solver."""
import math
import unittest
import numpy as np
from Design.SMRF_Section_Actions import (recover_composite_line,compose_cut,transport_moment,
                                        layer_force_resultants,verify_reinforcement_ownership)
from Design.SMRF_Uniaxial_Section import LayeredSection


class ActionCuts(unittest.TestCase):
    def test_recovery_is_exact_condensation_not_added_membrane(self):
        b,h,t,bf,e=16.,30.,8.,48.,4000.
        aw,af=b*(h-t),bf*t;z=-h/2;iw=b*(h-t)**3/12
        for n,m in [(0.,8000.),(75.,-2400.),(-50.,6000.)]:
            r=recover_composite_line(n_compression_kip=n,m_slab_midplane_kip_in=m,web_width_in=b,
                beam_depth_in=h,slab_thickness_in=t,flange_width_in=bf,web_E_ksi=e,slab_E_ksi=e)
            # Independent full 2x2 stiffness solution, without condensed formula.
            strain=np.linalg.solve(e*np.array([[af+aw,aw*z],[aw*z,iw+aw*z*z]]),[n,m])
            self.assertAlmostEqual(r['n_flange_compression_kip'],e*af*strain[0],places=10)
            self.assertAlmostEqual(r['n_web_compression_kip'],e*aw*(strain[0]+strain[1]*z),places=10)
            self.assertAlmostEqual(r['m_web_centroid_kip_in']+r['m_offset_couple_kip_in'],m,places=10)

    def test_datum_changes_and_duplicate_ownership_are_explicit(self):
        rows=[dict(component_id='web',cut_id='cut7',combination_id='C1',n_compression_kip=-12.,m_kip_in=120.,reference_z_in=-10.),
              dict(component_id='slab',cut_id='cut7',combination_id='C1',n_compression_kip=12.,m_kip_in=35.,reference_z_in=0.)]
        r=compose_cut(rows,combination_id='C1')
        self.assertEqual(r['n_compression_kip'],0.)
        self.assertEqual(r['m_kip_in'],275.)
        self.assertEqual(transport_moment(7.,12.,3.,-1.),40.)
        with self.assertRaises(ValueError):compose_cut(rows+rows[:1],combination_id='C1')
        with self.assertRaises(ValueError):compose_cut(rows,combination_id='C2')

    def test_actual_mat_planes_and_one_owner(self):
        r=layer_force_resultants(-20.,120.,3.,-2.)
        self.assertEqual(r['top_compression_kip_per_ft'],16.)
        self.assertEqual(r['bottom_compression_kip_per_ft'],-36.)
        self.assertEqual(r['moment_residual'],0.)
        bars=[dict(physical_segment_id='top_beam_1',owner='beam_bar_in_flange',area_in2=.79),
              dict(physical_segment_id='mat_x_7',owner='slab_mat',area_in2=.2)]
        self.assertAlmostEqual(verify_reinforcement_ownership(bars)['total_area_in2'],.99)
        with self.assertRaises(ValueError):verify_reinforcement_ownership(bars+bars[:1])


def independent_state(section,c,face,phi_mode):
    """Numerically integrate actual concrete widths excluding circular bars.

    Break at bar edges/region/block boundaries, 128-point Gauss on each part;
    this does not call the production circular-segment or concrete integrals.
    """
    h=section.h;a=section.beta*c
    points={0.,min(a,h),h}
    for y0,y1,_,_ in section.regions:points.update((y0,y1))
    for y,_,abar,_ in section.layers:
        radius=math.sqrt(abar/math.pi);points.update((y-radius,y+radius))
    block=(0.,min(a,h)) if face=='top' else (max(h-a,0.),h)
    points.update(block);points=sorted(x for x in points if block[0]<=x<=block[1])
    xi,weights=np.polynomial.legendre.leggauss(128);force=moment=0.
    for lo,hi in zip(points[:-1],points[1:]):
        yy=(hi+lo)/2+xi*(hi-lo)/2
        width=np.zeros_like(yy)
        for y0,y1,b,fc in section.regions:width+=b*((yy>=y0)&(yy<=y1))
        for y,area,abar,_ in section.layers:
            radius=math.sqrt(abar/math.pi)
            width-=2*np.sqrt(np.maximum(0.,radius*radius-(yy-y)**2))*area/abar
        dforce=.85*section.fc*width*(hi-lo)/2*weights
        force+=float(dforce.sum());moment+=float(np.sum(dforce*(h/2-yy)))
    strains=[]
    for y,area,abar,fy in section.layers:
        depth=y if face=='top' else h-y
        strain=.003*(1-depth/c);stress=min(fy,max(-fy,section.es*strain));f=area*stress
        force+=f;moment+=f*(h/2-y);strains.append(strain)
    phi=1. if phi_mode=='nominal' else .65+.25*min(1.,max(0.,(-min(strains)-max(section.layers[:,3])/section.es)/.003))
    return phi*force,phi*moment


class AxialFlexuralSections(unittest.TestCase):
    def sections(self):
        return [LayeredSection(30.,[(0.,30.,16.,5.)],[(2.5,4*.79,.79,60.),(27.5,4*.79,.79,60.)]),
                LayeredSection(30.,[(0.,8.,48.,5.),(8.,30.,16.,5.)],
                               [(1.,.2*48/8,.2,60.),(2.5,4*.79,.79,60.),(6.5,.2*48/12,.2,60.),(27.5,4*.79,.79,60.)])]

    def test_force_and_moment_match_independent_material_integration(self):
        for s in self.sections():
            for face in ('top','bottom'):
                for c in (1.,2.5/s.beta,7.,13.,25.,60.):
                    for mode in ('nominal','auto'):
                        with self.subTest(h=s.h,areas=s.ast,face=face,c=c,mode=mode):
                            expected=independent_state(s,c,face,mode);actual=s.state(c,face,mode)
                            np.testing.assert_allclose([actual['p'],actual['m']],expected,rtol=1e-7,atol=2e-5)

    def test_inverse_phi_and_tension_reduce_beam_capacity(self):
        s=self.sections()[0]
        no_n=s.at_axial(0.)
        tension=s.at_axial(-137.5412)
        self.assertLess(tension['m'],no_n['m']*.7)
        self.assertAlmostEqual(float(tension['p']),-137.5412,places=8)
        self.assertEqual(float(tension['phi']),.9)
        np.testing.assert_allclose(s.at_axial(100.,'bottom')['m'],-s.at_axial(100.,'top')['m'],rtol=1e-12)

    def test_nominal_probable_and_demand_capacities_cannot_be_confused(self):
        s=self.sections()[1]
        n=s.at_axial(0.,phi_mode='nominal');pr=s.at_axial(0.,phi_mode='nominal',fy_factor=1.25)
        self.assertGreater(pr['m'],n['m']);self.assertEqual(float(pr['phi']),1.)
        with self.assertRaises(ValueError):s.at_axial(0.,fy_factor=1.25)
        with self.assertRaises(ValueError):s.at_axial(-s.steel_force)


if __name__=='__main__':unittest.main()
