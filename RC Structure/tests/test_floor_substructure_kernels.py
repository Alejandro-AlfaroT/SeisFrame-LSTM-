import math
import unittest
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Floor_Substructure import (native_shell_kernel,native_web_kernel,
    rectangle_torsion_constant,rigid_arm_matrix)


class FloorKernelTests(unittest.TestCase):
    def tearDown(self):ops.wipe()

    def test_shell_constant_membrane_patch_and_rigid_modes(self):
        for dx,dy,t in ((20.,30.,8.),(12.,12.,5.),(48.,10.,12.)):
            k=native_shell_kernel(dx,dy,t,.2)
            xy=k['points'];u=np.zeros((4,6))
            ex,ey,gxy=.0002,-.0001,.0003
            u[:,0]=ex*xy[:,0]+.5*gxy*xy[:,1]
            u[:,1]=ey*xy[:,1]+.5*gxy*xy[:,0]
            force=k['K']@u.ravel()
            expected=t/(1-.2**2)*np.array([ex+.2*ey,ey+.2*ex,.4*gxy])
            result=(k['raw_resultant']@u.ravel()).reshape(4,8)
            np.testing.assert_allclose(result[:,:3],np.tile(expected,(4,1)),rtol=1e-11,atol=1e-12)
            self.assertAlmostEqual(u.ravel()@force,dx*dy*np.dot([ex,ey,gxy],expected),places=10)
            for d in range(6):
                q=np.eye(6)[d];rb=np.concatenate([rigid_arm_matrix(p)@q for p in xy])
                np.testing.assert_allclose(k['K']@rb,np.zeros(24),atol=1e-9)
            for part in k['parts'].values():
                self.assertGreaterEqual(np.linalg.eigvalsh(part).min(),-1e-9)

    def test_web_axial_bending_torsion_and_eccentricity(self):
        for axis in ('x','y'):
            row=native_web_kernel(100.,12.,20.,.2,axis)
            d=0 if axis=='x' else 1
            u=np.zeros(12);u[6+d]=.001
            f=row['K']@u
            self.assertAlmostEqual(f[6+d],240/100*.001,places=12)
            u[:]=0;u[8]=1.
            f=row['K']@u
            self.assertAlmostEqual(f[8],12*(12*20**3/12)/100**3,places=10)
            arm=rigid_arm_matrix([0,0,-16.])
            g=np.array([2.,-3.,4.,5.,-6.,7.])
            du=np.array([.1,.2,-.3,.01,-.02,.03])
            self.assertAlmostEqual((arm@du)@g,du@(arm.T@g),places=12)
        # Independent square torsion value from the converged odd series.
        self.assertAlmostEqual(rectangle_torsion_constant(1.,1.),.14057701496,places=9)
        self.assertAlmostEqual(rectangle_torsion_constant(2.,1.),rectangle_torsion_constant(1.,2.),places=12)

    def test_existing_model_is_preserved(self):
        ops.model('basic','-ndm',3,'-ndf',6);ops.node(987,0.,0.,0.)
        with self.assertRaisesRegex(RuntimeError,'empty isolated'):
            native_shell_kernel(11.123,12.,5.,.2)
        self.assertEqual(ops.getNodeTags(),[987])


if __name__=='__main__':unittest.main()
