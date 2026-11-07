"""Full ABD assembly versus native uncracked shell and independent energies."""
import unittest
import numpy as np
from Design.SMRF_Material_Floor import (MaterialFloorOperator,common_shell_strain_matrix,
    web_normal_strain_matrices,web_operator_from_sections)
from test_floor_generalized_stiffness import floor_fixture


def inputs(floor):
    e,nu,h=floor.ec,floor.nu,floor.thickness
    d=e/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    c=np.zeros((6,6));c[:3,:3]=h*d;c[3:,3:]=h**3/12*d
    slab={r['id']:c for r in floor.elements if r['kind']=='slab'}
    webs={r['id']:r['K'] for r in floor.elements if r['kind']=='web'}
    shear={ident:1. for ident in slab}
    return slab,webs,shear


class MaterialOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.floor=floor_fixture()

    def test_full_elastic_ABD_reproduces_independent_native_floor(self):
        floor=self.floor;slab,webs,shear=inputs(floor)
        model=MaterialFloorOperator(floor,slab_ABD=slab,web_matrices=webs,transverse_shear_factors=shear,
            source_identity='independent_native_operators',material_state_identity='elastic_E_nu_h')
        np.testing.assert_allclose(model.stiffness.toarray(),floor.stiffness.toarray(),rtol=1e-10,atol=1e-8)
        u=np.random.default_rng(621).normal(size=len(floor.dof_ids))*1e-4;loads=floor.loads(-.002)
        actual=model.recover(u,loads);reference=floor.recover(u,loads)
        for a,b in zip(actual['elements'],reference['elements']):
            np.testing.assert_allclose(a['global_resisting_force'],b['global_resisting_force'],rtol=1e-9,atol=1e-8)
            if a['kind']=='slab':np.testing.assert_allclose(a['raw_resultants'],b['raw_resultants'],rtol=1e-9,atol=1e-8)
        self.assertLess(actual['force_partition_relative_error'],1e-10)

    def test_actual_ABD_couplings_and_different_gauss_states_preserve_energy(self):
        floor=self.floor;slab,webs,shear=inputs(floor);rng=np.random.default_rng(448)
        for ident in slab:
            values=[]
            for _ in range(4):
                a=rng.normal(size=(6,6));values.append(a.T@a+np.eye(6))
            slab[ident]=values
        model=MaterialFloorOperator(floor,slab_ABD=slab,web_matrices=webs,transverse_shear_factors=shear,
            source_identity='four_distinct_GP_concrete_steel_fixture',material_state_identity='explicit_matrices')
        u=rng.normal(size=len(floor.dof_ids))*.0001;energy=0.
        strains=model.section_strains(u)
        for row in floor.elements:
            local=row['transform']@u[row['indices']]
            if row['kind']=='web':energy+=local@row['K']@local;continue
            for q,c in zip(strains[row['id']],slab[row['id']]):energy+=row['area_in2']/4*(np.array(q)@c@q)
            energy+=local@(floor.ec*(row['native']['parts']['shear']+row['native']['parts']['drilling']))@local
        self.assertAlmostEqual(float(u@(model.stiffness@u))/energy,1.,places=12)
        recovered=model.recover(u,floor.loads(-.003))
        self.assertLess(recovered['force_partition_relative_error'],1e-10)
        self.assertFalse(model.metadata['material_iteration_converged'])

    def test_missing_or_invalid_material_operators_cannot_be_filled(self):
        floor=self.floor;slab,webs,shear=inputs(floor)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            MaterialFloorOperator(floor,slab_ABD={},web_matrices=webs,transverse_shear_factors=shear,source_identity='s',material_state_identity='m')
        ident=next(iter(slab));slab[ident]=-np.eye(6)
        with self.assertRaisesRegex(ValueError,'negative'):
            MaterialFloorOperator(floor,slab_ABD=slab,web_matrices=webs,transverse_shear_factors=shear,source_identity='s',material_state_identity='m')

    def test_full_web_normal_sections_reproduce_native_both_axes_and_reference_transport(self):
        floor=self.floor;stations=[(1-1/np.sqrt(3))/2,(1+1/np.sqrt(3))/2]
        for row in floor.elements:
            if row['kind']!='web':continue
            section=row['section'];width=section['b_in'];depth=section['h_in']-floor.thickness;e=section['E_ksi']
            area=width*depth;i_strong=width*depth**3/12;i_weak=depth*width**3/12
            z=row['points'][0,2];across=1 if row['axis']=='x' else 0
            # Deliberately translated transverse datum verifies the transported
            # axial reference independently of global-coordinate magnitudes.
            reference=row['points'][0,across]+7.;s=-7.
            c=e*np.array([[area,area*z,area*s],[area*z,i_strong+area*z*z,area*z*s],
                          [area*s,area*z*s,i_weak+area*s*s]])
            k=web_operator_from_sections(row,[c,c],stations=stations,weights=[.5,.5],
                torsion_matrix=row['parts']['web_torsion'],reference_z_in=0.,transverse_reference_in=reference)
            np.testing.assert_allclose(k,row['K'],rtol=1e-10,atol=1e-8)
            local=np.zeros(12);along=0 if row['axis']=='x' else 1;local[6+along]=.001*row['length_in']
            q=web_normal_strain_matrices(row,stations,transverse_reference_in=reference)@local
            np.testing.assert_allclose(q,[[.001,0,0],[.001,0,0]],atol=1e-13)


if __name__=='__main__':unittest.main()
