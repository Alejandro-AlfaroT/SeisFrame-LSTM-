import unittest
import numpy as np
from Design.SMRF_Reference_Strain_Field import RectangularReferenceStrainField,GP
from Design.SMRF_Material_Floor import common_shell_strain_matrix
from Design.SMRF_Floor_Substructure import native_shell_kernel


class ReferenceStrainFieldTests(unittest.TestCase):
    def test_exact_multiaffine_field_and_one_sided_discontinuity(self):
        def q(x,y):return np.array([1+x,2-y,x*y,2*x-y,3+x*y,x-4*y])*.00001
        grid=[0.,8.,20.];rows={}
        for i in range(2):
            rows[i,0]=[q((grid[i]+grid[i+1])/2+p[0]*(grid[i+1]-grid[i])/2,3+3*p[1])+i*.002 for p in GP]
        field=RectangularReferenceStrainField(x_in=grid,y_in=[0.,6.],cell_strains=rows,reference_identity='independent_polynomial')
        for x in (0.,1.,7.9,8.,13.,20.):
            for y in (0.,1.3,6.):
                np.testing.assert_allclose(field.at(x,y)['strain'],q(x,y)+(0 if x<8 else .002),rtol=1e-13,atol=1e-17)
        np.testing.assert_allclose(field.at(8.,2.,trace=('minus','plus'))['strain'],q(8.,2.),atol=1e-17)
        self.assertNotEqual(field.at(8.,2.)['cell_index'],field.at(8.,2.,trace=('minus','plus'))['cell_index'])

    def test_all_24_native_displacement_basis_fields_survive_nested_refinement(self):
        # Independent native child-element queries and Q4 nodal interpolation.
        # No interpolated D or material-law response serves as the oracle.
        dx,dy=13.,7.;coarse=common_shell_strain_matrix({'native':native_shell_kernel(dx,dy,8.)})
        for xl,xh,yl,yh in ((0.,.3,0.,.4),(.3,1.,.4,1.),(.1,.7,.2,.9)):
            child=common_shell_strain_matrix({'native':native_shell_kernel(dx*(xh-xl),dy*(yh-yl),8.)})
            transfer=np.zeros((24,24))
            for node,(s,t) in enumerate(((xl,yl),(xh,yl),(xh,yh),(xl,yh))):
                shape=np.array([(1-s)*(1-t),s*(1-t),s*t,(1-s)*t])
                for j in range(6):transfer[6*node+j,j::6]=shape
            expected=child@transfer
            for basis in range(24):
                field=RectangularReferenceStrainField(x_in=[0.,dx],y_in=[0.,dy],cell_strains={(0,0):coarse[:,:,basis]},reference_identity='native_basis_'+str(basis))
                actual=[field.at(dx*(xl+xh+(xh-xl)*xi)/2,dy*(yl+yh+(yh-yl)*eta)/2)['strain'] for xi,eta in GP]
                np.testing.assert_allclose(actual,expected[:,:,basis],rtol=2e-12,atol=1e-13)

    def test_missing_invalid_unbound_or_outside_reference_refused(self):
        args=dict(x_in=[0.,1.],y_in=[0.,1.],cell_strains={(0,0):np.zeros((4,6))},reference_identity='state')
        for change in (dict(reference_identity=''),dict(x_in=[0.,0.]),dict(cell_strains={}),dict(cell_strains={(0,0):np.full((4,6),np.nan)})):
            with self.assertRaises(ValueError):RectangularReferenceStrainField(**dict(args,**change))
        field=RectangularReferenceStrainField(**args)
        for point in ((-1e-12,0.),(0.,1.000001),(np.nan,0.)):
            with self.assertRaises(ValueError):field.at(*point)
        with self.assertRaises(ValueError):field.at(0.,0.,trace=('average','plus'))
        changed=RectangularReferenceStrainField(**dict(args,reference_identity='another_state'))
        self.assertNotEqual(changed.identity,field.identity)


if __name__=='__main__':unittest.main()
