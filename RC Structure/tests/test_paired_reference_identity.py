import copy,json,unittest
from Design.SMRF_Frozen_Gravity_Design import physical_reference_state_identity


class PairedReferenceIdentityTests(unittest.TestCase):
    def fixture(self):
        return dict(frame=dict(physical_dof_ids=[(1,1)],cases=[dict(physical_displacements=[1.],physical_displacements_lo=[1e-17])]),
            floors=[dict(id='f1',responses=[dict(case_id='DL',dof_ids=[(1,1),('interior',1)],
                retained_displacements=[1.,2.],retained_displacements_lo=[1e-17,-2e-17])])])
    def test_material_reference_binds_interior_and_joint_low_parts_through_json(self):
        original=self.fixture();identity=physical_reference_state_identity(original)
        self.assertEqual(identity,physical_reference_state_identity(json.loads(json.dumps(original))))
        for target,key,index in [('floor','retained_displacements_lo',1),('floor','retained_displacements',1),('frame','physical_displacements_lo',0)]:
            changed=copy.deepcopy(original)
            record=changed['floors'][0]['responses'][0] if target=='floor' else changed['frame']['cases'][0]
            record[key][index]*=1.5
            self.assertNotEqual(identity,physical_reference_state_identity(changed))
        changed=copy.deepcopy(original);changed['floors'][0]['responses'][0]['case_id']='other'
        self.assertNotEqual(identity,physical_reference_state_identity(changed))
    def test_missing_nonfinite_or_incomplete_reference_pair_refused(self):
        for problem in ('missing','nonfinite','incomplete'):
            changed=self.fixture();record=changed['floors'][0]['responses'][0]
            if problem=='missing':record.pop('retained_displacements_lo')
            elif problem=='nonfinite':record['retained_displacements_lo'][1]=float('nan')
            else:record['retained_displacements_lo'].pop()
            with self.assertRaises(ValueError):physical_reference_state_identity(changed)


if __name__=='__main__':unittest.main()
