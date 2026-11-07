"""Independent physical slab bounds, concrete union and grid mass moments."""
import contextlib
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Structure_Parameters as sp
from Model import Floor_Footprint as fp, Member_Groups as mg, Member_Properties as mp


def perimeter(g, widths):
    nx, ny, lx, ly = (g[k] for k in ('nx','ny','lx','ly'))
    return ([dict(axis='x',line=j*ly,start=i*lx,end=(i+1)*lx,b_in=widths['y_minus' if j==0 else 'y_plus'])
             for j in (0,ny) for i in range(nx)] +
            [dict(axis='y',line=i*lx,start=j*ly,end=(j+1)*ly,b_in=widths['x_minus' if i==0 else 'x_plus'])
             for i in (0,nx) for j in range(ny)])


@contextlib.contextmanager
def installed(grouped=False):
    values=dict(NUM_BAY_X=2,NUM_BAY_Y=3,NUM_FLOOR=4,BAY_X=240.,BAY_Y=216.,STORY_H=168.,
                B_COL=26.,H_COL=30.,B_BEAM=18.,H_BEAM=28.,SLAB_THICKNESS_IN=7.,
                FLOOR_SUPERIMPOSED_DEAD_LOAD_KSF=.05,FLOOR_LIVE_LOAD_KSF=.04,
                CONCRETE_UNIT_WEIGHT_KCF=.15,CONCRETE_UNIT_WEIGHT_KCI=.15/1728.,
                SEISMIC_LIVE_LOAD_FRACTION=.25,ROOF_COLUMN_EXTENSION=False)
    with contextlib.ExitStack() as stack:
        for key,value in values.items():stack.enter_context(mock.patch.object(sp,key,value))
        try:
            state=None
            if grouped:
                state=mg.GroupedDesign.uniform()
                beam=mg.uniform_member_design('beam_y')
                state=state.with_designs({'s01_02__beam_y__edge':replace(beam,b_in=24.,h_in=32.)})
                mg.install(state)
            yield state
        finally:mg.clear()


class FloorFootprintTests(unittest.TestCase):
    def test_asymmetric_outer_faces_area_and_nodal_first_moments(self):
        g=dict(nx=2,ny=1,lx=100.,ly=80.)
        widths=dict(x_minus=10.,x_plus=18.,y_minus=12.,y_plus=20.)
        f=fp.from_perimeter_beams(g,perimeter(g,widths))
        self.assertEqual([f[k] for k in ('x_min_in','x_max_in','y_min_in','y_max_in')],[-5.,209.,-6.,90.])
        self.assertEqual(f['area_in2'],214*96)
        self.assertEqual(f['centroid_in'],[102.,42.])
        areas=fp.consistent_nodal_areas(f,g)
        self.assertAlmostEqual(sum(areas.values()),214*96)
        self.assertAlmostEqual(sum(a*i*100 for (i,j),a in areas.items()),214*96*102)
        self.assertAlmostEqual(sum(a*j*80 for (i,j),a in areas.items()),214*96*42)
        # Extended geometric tributary cells conserve area but not nodal CM.
        cells={(i,j):fp.tributary_cell(f,g,i,j) for i,j in areas}
        self.assertAlmostEqual(sum(c['area_in2'] for c in cells.values()),214*96)
        self.assertNotAlmostEqual(sum(c['area_in2']*i*100 for (i,j),c in cells.items()),214*96*102)

    def test_missing_duplicate_and_stepped_sides_are_not_invented_rectangles(self):
        g=dict(nx=2,ny=2,lx=100.,ly=80.)
        beams=perimeter(g,dict(x_minus=10.,x_plus=10.,y_minus=12.,y_plus=12.))
        for changed in (beams[:-1],beams+[beams[0]], [dict(b,b_in=13.) if k==0 else b for k,b in enumerate(beams)]):
            with self.assertRaises(ValueError):fp.from_perimeter_beams(g,changed)

    def test_column_overlap_retains_projecting_concrete_and_eccentricity(self):
        f=fp.rectangle(-9.,489.,-9.,657.)
        corner=fp.column_overlap(f,0.,0.,30.,26.)
        # Full column [-15,15]x[-13,13]; overlap [-9,15]x[-9,13].
        self.assertEqual(corner['area_in2'],24*22)
        self.assertEqual(corner['centroid_in'],[3.,2.])
        self.assertEqual(fp.column_overlap_area_in2(f,240.,216.,30.,26.),30*26)
        self.assertEqual(fp.column_overlap_area_in2(f,0.,216.,30.,26.),24*26)
        g=dict(nx=2,ny=3,lx=240.,ly=216.)
        result=fp.member_volume_ledger(g,f,[dict(x=0.,y=0.,h_x=30.,b_y=26.)],[],7.,168.)
        self.assertEqual(result['volume_in3'],30*26*168-24*22*7)
        self.assertEqual(result['first_moment_x_in4'],-24*22*7*3)
        self.assertEqual(result['first_moment_y_in4'],-24*22*7*2)
        self.assertAlmostEqual(sum(v*i*240 for (i,j),v in result['nodal_volumes_in3'].items()),-24*22*7*3)
        self.assertAlmostEqual(sum(v*j*216 for (i,j),v in result['nodal_volumes_in3'].items()),-24*22*7*2)

    def test_unequal_joint_faces_beam_volume_and_first_moment(self):
        g=dict(nx=2,ny=1,lx=100.,ly=80.);f=fp.rectangle(-5.,205.,-5.,85.)
        # Unequal end columns put the clear prism centroid at57, not50.
        result=fp.member_volume_ledger(g,f,[],[dict(axis='x',line=0.,start=18.,end=96.,b_in=10.,h_in=20.)],4.,100.)
        volume=78*10*16
        self.assertEqual(result['volume_in3'],volume)
        self.assertEqual(result['first_moment_x_in4'],volume*57)
        self.assertAlmostEqual(sum(v*i*100 for (i,j),v in result['nodal_volumes_in3'].items()),volume*57)
        self.assertAlmostEqual(sum(v*j*80 for (i,j),v in result['nodal_volumes_in3'].items()),0.)

    def test_uniform_and_grouped_actual_volume_mass_and_floor_specific_area(self):
        for grouped in (False,True):
            with self.subTest(grouped=grouped),installed(grouped) as state:
                for k in range(1,5):
                    xover=12. if grouped and k<=2 else 9.
                    xmin,xmax,ymin,ymax=-xover,480+xover,-9.,657.
                    area=(xmax-xmin)*(ymax-ymin)
                    self.assertEqual(mp.floor_area_in2(k),area)
                    f=fp.physical_floor(k,state)
                    self.assertEqual(f['x_min_in'],xmin)
                    # Independent solid-prism arithmetic, with clipping
                    # computed directly instead of using the helper.
                    volume=0.;mx=0.;my=0.
                    for m in mp.members_at_floor(k):
                        d=m.design;x=m.grid_i*240.;y=m.grid_j*216.
                        if m.is_column:
                            ox0=max(x-d.h_in/2,xmin);ox1=min(x+d.h_in/2,xmax)
                            oy0=max(y-d.b_in/2,ymin);oy1=min(y+d.b_in/2,ymax)
                            overlap=(ox1-ox0)*(oy1-oy0)
                            v=d.h_in*d.b_in*168-overlap*7
                            vx=d.h_in*d.b_in*168*x-overlap*7*(ox0+ox1)/2
                            vy=d.h_in*d.b_in*168*y-overlap*7*(oy0+oy1)/2
                            self.assertAlmostEqual(mp.column_self_weight_kip_per_in(m)*168,v*.15/1728)
                            if not grouped:self.assertAlmostEqual(sp.col_self_weight_kip_per_in(m.grid_i,m.grid_j)*168,v*.15/1728)
                        else:
                            axis='x' if m.member_type=='beam_x' else 'y'
                            span=240. if axis=='x' else 216.
                            face=15. if axis=='x' else 13.
                            v=(span-2*face)*d.b_in*(d.h_in-7)
                            vx=v*(x+span/2 if axis=='x' else x)
                            vy=v*(y+span/2 if axis=='y' else y)
                        volume+=v;mx+=vx;my+=vy
                    ledger=mp.floor_member_volume_ledger(k)
                    self.assertAlmostEqual(ledger['volume_in3'],volume)
                    gamma=.15/1728
                    seismic_q=(sp.floor_dead_load_ksf()+.25*.04)/144
                    masses={(i,j):mp.node_seismic_mass(k,i,j)*sp.G for j in range(4) for i in range(3)}
                    self.assertTrue(all(v>0 for v in masses.values()))
                    self.assertAlmostEqual(sum(masses.values()),volume*gamma+area*seismic_q,places=9)
                    self.assertAlmostEqual(sum(v*i*240 for (i,j),v in masses.items()),mx*gamma+area*240*seismic_q,places=7)
                    self.assertAlmostEqual(sum(v*j*216 for (i,j),v in masses.items()),my*gamma+area*324*seismic_q,places=7)
                    self.assertAlmostEqual(sum(masses.values()),mp.floor_seismic_weight_kip(k),places=9)
                    if not grouped:
                        for (i,j),w in masses.items():self.assertAlmostEqual(sp.node_seismic_mass(i,j)*sp.G,w)
                if grouped:
                    with self.assertRaisesRegex(ValueError,'vary by floor'):mp.floor_area_in2()
                    self.assertIsNone(mp.floor_load_metadata()['floor_area_sqft'])
                else:self.assertEqual(mp.floor_area_in2(),498*666)

    def test_no_position_column_and_nonpositive_grid_mass_domain_are_refused(self):
        with installed():
            with self.assertRaisesRegex(ValueError,'grid indices'):sp.col_self_weight_kip_per_in()
        with self.assertRaisesRegex(ValueError,'positive lumped'):
            fp.consistent_nodal_areas(fp.rectangle(-500.,200.,0.,80.),dict(nx=2,ny=1,lx=100.,ly=80.))

    def test_applied_gravity_preserves_actual_force_and_plan_moments(self):
        # Mock load sinks only: no OpenSees model/analyze/eigen. Deliberately
        # unequal outside faces catch the error hidden by mirrored groups.
        from Loads import Gravity_Loads as loads
        from Model.Build_Model import node_tag
        with installed():
            f=fp.rectangle(-7.,492.,-8.,662.)
            with mock.patch.object(fp,'physical_floor',return_value=f):
                for mode in ('nodal','beam_uniform'):
                    with self.subTest(mode=mode),mock.patch.object(sp,'GRAVITY_LOAD_MODEL',mode),\
                         mock.patch.object(loads.ops,'timeSeries'),mock.patch.object(loads.ops,'pattern'),\
                         mock.patch.object(loads.ops,'load') as nodal,mock.patch.object(loads.ops,'eleLoad') as element:
                        loads.apply_gravity_loads(floor_factor=1.3,self_weight_factor=1.2,roof_extension=False)
                        positions={node_tag(k,i,j):(i*240.,j*216.) for k in range(1,5) for j in range(4) for i in range(3)}
                        weight=mx=my=0.
                        for call in nodal.call_args_list:
                            tag,fx,fy,fz,mxi,myi,mzi=call.args
                            x,y=positions[tag]
                            weight-=fz;mx+=mxi+y*fz;my+=myi-x*fz
                        for call in element.call_args_list:
                            args=call.args;m=mg.resolve(args[1]);length=mp.member_length_in(m)
                            w=-(args[5]+args[6])*length
                            x=m.grid_i*240.;y=m.grid_j*216.
                            if m.member_type=='beam_x':x+=120.
                            if m.member_type=='beam_y':y+=108.
                            weight+=w;mx-=w*y;my+=w*x
                        expected_w=expected_x=expected_y=0.
                        for k in range(1,5):
                            v=mp.floor_member_volume_ledger(k)
                            gamma=.15/1728.;q=sp.floor_load_ksi()
                            expected_w+=1.3*q*f['area_in2']+1.2*gamma*v['volume_in3']
                            expected_x+=1.3*q*f['first_moment_x_in3']+1.2*gamma*v['first_moment_x_in4']
                            expected_y+=1.3*q*f['first_moment_y_in3']+1.2*gamma*v['first_moment_y_in4']
                        self.assertAlmostEqual(weight,expected_w,places=8)
                        self.assertAlmostEqual(my,expected_x,places=6)
                        self.assertAlmostEqual(mx,-expected_y,places=6)
                        # Local corner moment oracle: no mirror cancellation.
                        corner=mg.column_at(1,0,0)
                        # overlap[-7,15]x[-8,13], centroid[4,2.5]
                        removed=.15/1728*7*22*21
                        moment=mp.column_weight_eccentricity_kipin(corner)
                        self.assertAlmostEqual(moment[0],removed*2.5)
                        self.assertAlmostEqual(moment[1],-removed*4.)

    def test_accidental_torsion_uses_each_actual_floor_width_and_mass_center_transport(self):
        from Loads import Seismic_ELF as elf
        with installed(True):
            self.assertEqual(elf.accidental_torsion_moment('y',10.,floor=1),.05*504*10)
            self.assertEqual(elf.accidental_torsion_moment('y',10.,floor=3),.05*498*10)
            self.assertEqual(elf.accidental_torsion_moment('x',10.,amplification=2.,floor=1),2*.05*666*10)
            with self.assertRaisesRegex(ValueError,'explicit floor'):elf.accidental_torsion_moment('y',10.)
            with mock.patch.object(elf.ops,'timeSeries'),mock.patch.object(elf.ops,'pattern'),mock.patch.object(elf.ops,'load') as loads,\
                 mock.patch.object(elf,'floor_seismic_mass_center',return_value=[243.,329.]):
                result=elf.apply_elf_loads('x',elf={'story_forces_kip':[10.]*4},accidental_torsion_ratio=.05)
                self.assertEqual(result['mass_center_transport_moment_kip_in'],[-50.]*4)
                for call in loads.call_args_list:self.assertAlmostEqual(call.args[-1],.05*666*10-50.)


if __name__=='__main__':unittest.main()
