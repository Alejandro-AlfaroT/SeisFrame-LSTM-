"""Independent guided-column rigid-floor stiffness and period witnesses."""
import math
import unittest
from unittest.mock import patch
import openseespy.opensees as ops
import Structure_Parameters as sp
from Analysis import Gravity, Modal
from Model.Build_Model import node_tag, floor_master_node


class ExactDesignConstraintTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_exact_rigid_floor_against_closed_form(self):
        # Four guided columns. Integrating Euler-Bernoulli curvature gives
        # k=12EI/L^3; Saint-Venant torsion gives kt=GJ/L. Rigid-floor
        # kinematics then gives Ktheta=sum(kx*y^2+ky*x^2+kt).
        records=[]
        for a,b,height,ix,iy in ((120.,96.,144.,40000.,24000.),
                                  (96.,144.,180.,32000.,52000.),
                                  (180.,120.,120.,60000.,30000.)):
            with patch.multiple(sp, NUM_FLOOR=1, NUM_BAY_X=1, NUM_BAY_Y=1,
                                BAY_X=2*a, BAY_Y=2*b, STORY_H=height,
                                GRAVITY_STEPS=1, GRAVITY_TOL=1e-12,
                                NUM_MODES=2, ELEMENT_FORMULATION='imk'):
                ops.wipe(); ops.model('basic','-ndm',3,'-ndf',6)
                e,g,area,jtors,mass=4000.,1600.,600.,35000.,1.2
                tags=[]
                for jj,y in enumerate((0.,2*b)):
                    for ii,x in enumerate((0.,2*a)):
                        base,top=node_tag(0,ii,jj),node_tag(1,ii,jj)
                        ops.node(base,x,y,0.);ops.fix(base,1,1,1,1,1,1)
                        ops.node(top,x,y,height);ops.fix(top,0,0,1,1,1,0)
                        ops.mass(top,mass,mass,0.,0.,0.,0.);tags.append(top)
                master=floor_master_node(1)
                ops.node(master,a,b,height);ops.fix(master,0,0,1,1,1,0)
                ops.rigidDiaphragm(3,master,*tags)
                # Local x=global z, local y=-global y, local z=global x.
                ops.geomTransf('Linear',1,1.,0.,0.)
                for tag,top in enumerate(tags,1):
                    ops.element('elasticBeamColumn',tag,top-4,top,area,e,g,jtors,ix,iy,1)
                fx,fy,mz=100.,-75.,10000.
                ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
                ops.load(master,fx,fy,0.,0.,0.,mz)
                Gravity.run_gravity_analysis(constraint_handler='Transformation')
                # Global x displacement bends about local y; global y
                # displacement bends about local z (vecxz cross local x).
                kx,ky=12*e*ix/height**3,12*e*iy/height**3
                ktheta=4*(kx*b*b+ky*a*a+g*jtors/height)
                expected=[fx/(4*kx),fy/(4*ky),mz/ktheta]
                actual=[ops.nodeDisp(master,d) for d in (1,2,6)]
                for value,want in zip(actual,expected):
                    self.assertAlmostEqual(value/want,1.,places=9)
                for top in tags:
                    x,y,_=ops.nodeCoord(top)
                    self.assertAlmostEqual(ops.nodeDisp(top,1),actual[0]-(y-b)*actual[2],places=11)
                    self.assertAlmostEqual(ops.nodeDisp(top,2),actual[1]+(x-a)*actual[2],places=11)
                modes=Modal.run_modal_analysis(constraint_handler='Transformation')
                eigen=sorted([4*kx/(4*mass),4*ky/(4*mass),ktheta/(4*mass*(a*a+b*b))])[:2]
                for mode,want in zip(modes,eigen):
                    self.assertAlmostEqual(mode['lambda']/want,1.,places=8)
                records.append({'geometry':[a,b,height],'expected':expected,'actual':actual,
                                'closed_form_eigenvalues':eigen,'actual_eigenvalues':[v['lambda'] for v in modes]})
        print('Independent rigid-floor witnesses:',records)

    def test_unspecified_handler_keeps_nonlinear_selection(self):
        # No solve: nonlinear callers retain the original handler dispatch.
        with patch.object(Gravity,'apply_analysis_constraints') as choose, \
             patch.object(Gravity.ops,'system'),patch.object(Gravity.ops,'numberer'), \
             patch.object(Gravity.ops,'test',side_effect=RuntimeError('before solve')):
            with self.assertRaisesRegex(RuntimeError,'before solve'):
                Gravity.run_gravity_analysis()
            choose.assert_called_once_with()
        with patch.object(Modal,'apply_analysis_constraints') as choose, \
             patch.object(Modal.ops,'wipeAnalysis'), \
             patch.object(Modal.ops,'numberer',side_effect=RuntimeError('before eigen')):
            with self.assertRaisesRegex(RuntimeError,'before eigen'):
                Modal.run_modal_analysis()
            choose.assert_called_once_with()


if __name__=='__main__':
    unittest.main()
