"""Independent multi-floor loads, full-system coupling and input integrity."""
import copy
import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import openseespy.opensees as ops
from Design.SMRF_Coupled_Floor_Design import CoupledFloorSystem,_hash,physical_grid
from Design.SMRF_Substructure_Frame import ElasticColumn


def fixture():
    # Two unequal-width column stories and two distinct beam floors on2x1 bays.
    nx,ny,lx,ly,nf,sh=2,1,20.,16.,2,10.
    tag=lambda k,i,j:100*k+10*j+i
    nodes=[{'id':tag(k,i,j),'xyz':[i*lx,j*ly,k*sh]} for k in range(3) for j in range(2) for i in range(3)]
    floors=[];columns=[]
    for level in (1,2):
        cb=4. if level==1 else 3.;bh=5. if level==1 else 6.;bw=2. if level==1 else 2.5
        supports=[dict(id=tag(level,i,j),x=i*lx,y=j*ly,b_in=cb,h_in=4.) for j in range(2) for i in range(3)]
        beams=[dict(id=f'{level}:x:{i}:{j}',axis='x',line=j*ly,start=i*lx,end=(i+1)*lx,b_in=bw,h_in=bh,E_ksi=200.) for j in range(2) for i in range(2)]
        beams += [dict(id=f'{level}:y:{i}',axis='y',line=i*lx,start=0.,end=ly,b_in=bw,h_in=bh,E_ksi=200.) for i in range(3)]
        floors.append(dict(id=f'floor_{level}',level=level,z_in=level*sh,supports=supports,beams=beams,fc_ksi=4.,E_ksi=200.))
        columns += [dict(id=f'column:{level}:{i}:{j}',i=tag(level-1,i,j),j=tag(level,i,j),story=level,grid_i=i,grid_j=j,b_in=cb,h_in=4.,E_ksi=200.,fc_ksi=4.) for j in range(2) for i in range(3)]
    definition=dict(method='independent_coupled_test',state_identity='two_stories_actual_cages',policy_id='test',
                    geometry=dict(nx=nx,ny=ny,nf=nf,lx=lx,ly=ly,story_h=sh),slab_thickness_in=1.,gamma_kip_in3=.15/1728,
                    floor_dead_load_ksf=.15,floor_live_load_ksf=.04,nodes=nodes,floors=floors,columns=columns)
    definition['identity_sha256']=_hash(definition)
    return definition


def build(definition,cases):
    return CoupledFloorSystem(definition,cases,source_identity='independent_geometry_fixture',stiffness_basis='explicit algebra sensitivity, not qualified',
        subdivisions_per_bay=2,max_shells_per_floor=1024,max_coupled_dofs=1500,beam_bending_factor=.35,
        slab_bending_factor=.25,web_torsion_factor=.35,column_bending_factor=.7,
        patterns=[{'id':'left','panels':[[0,0]]},{'id':'right','panels':[[1,0]]}])


class CoupledGeometryTests(unittest.TestCase):
    def tearDown(self):ops.wipe()

    def test_frozen_load_rebinding_matches_independent_fresh_system(self):
        original=build(fixture(),[dict(id='DL',dead=1.,live=1.)]);before=original.solve()
        cases=[dict(id='high',dead=1.2,live=1.6),dict(id='pattern',dead=.9,live=1.,live_pattern='left')]
        rebound=original.with_cases(cases,patterns=[dict(id='left',panels=[[0,0]])],case_identity='independent_rebind_test')
        actual=rebound.solve();expected=build(fixture(),cases).solve()
        for a,b in zip(actual['frame']['cases'],expected['frame']['cases']):
            np.testing.assert_allclose(a['physical_displacements'],b['physical_displacements'],rtol=1e-10,atol=1e-11)
            for key in a['columns']:
                np.testing.assert_allclose(a['columns'][key]['global_resisting_force'],b['columns'][key]['global_resisting_force'],rtol=1e-10,atol=1e-9)
        self.assertEqual(original.cases,[dict(id='DL',dead=1.,live=1.)])
        self.assertEqual(before['frame']['cases'],original.solve()['frame']['cases'])
        for a,b in zip(original.floors,rebound.floors):self.assertIs(a['model'],b['model'])

    def test_exact_all_floor_weight_moments_and_pattern_superposition(self):
        d=fixture();cases=[dict(id='DL',dead=1.,live=1.),dict(id='D',dead=1.,live=0.),
                          dict(id='Lleft',dead=0.,live=1.,live_pattern='left'),dict(id='Lright',dead=0.,live=1.,live_pattern='right')]
        system=build(d,cases);result=system.solve(pdelta=False)
        self.assertEqual(result['status'],'equilibrated')
        u=np.array([r['physical_displacements'] for r in result['frame']['cases']])
        np.testing.assert_allclose(u[0],u[1]+u[2]+u[3],rtol=1e-10,atol=1e-10)
        independent_total=0.
        for data in system.floors:
            floor=data['assembly'];level=data['description']['level'];cb=4. if level==1 else 3.;bh=5. if level==1 else 6.;bw=2. if level==1 else 2.5
            # Four X spans, three Y spans; each actual end face subtracts a
            # half column width. No assembly element area/length is used.
            web_volume=(4*(20-4)+3*(16-cb))*bw*(bh-1)
            expected=(40+bw)*(16+bw)/144*(.15+.04)+web_volume*d['gamma_kip_in3'];independent_total+=expected
            f=data['loads'][0]['loads'].reshape(-1,6)
            self.assertAlmostEqual(float(f[:,2].sum()),-expected,places=11)
            xyz=np.array([floor.node_positions[n] for n in floor.node_ids])
            moment=np.sum(f[:,3:]+np.cross(xyz,f[:,:3]),axis=0)
            np.testing.assert_allclose(moment,[-expected*8,expected*20,0.],atol=1e-10)
            left=data['loads'][2]['loads'].reshape(-1,6);left_weight=(20+bw/2)*(16+bw)/144*.04
            self.assertAlmostEqual(float(left[:,2].sum()),-left_weight,places=11)
            left_m=np.sum(left[:,3:]+np.cross(xyz,left[:,:3]),axis=0)
            np.testing.assert_allclose(left_m,[-left_weight*8,left_weight*(10-bw/4),0.],atol=1e-10)
        for c in d['columns']:
            bw=2. if c['story']==1 else 2.5
            x,y=c['grid_i']*20.,c['grid_j']*16.
            overlap=(min(x+2,40+bw/2)-max(x-2,-bw/2))*(min(y+c['b_in']/2,16+bw/2)-max(y-c['b_in']/2,-bw/2))
            independent_total+=(c['b_in']*4*10-overlap)*d['gamma_kip_in3']
        response=result['frame']['cases'][0];ids=result['frame']['physical_dof_ids']
        reactions=np.array(response['physical_residual_or_reaction'])
        base_tags={n['id'] for n in d['nodes'] if n['xyz'][2]==0}
        base_vertical=sum(reactions[i] for i,(tag,dof) in enumerate(ids) if tag in base_tags and dof==3)
        self.assertAlmostEqual(base_vertical,independent_total,places=9)
        for floor in result['floors']:
            self.assertLess(floor['equilibrium']['maximum_interior_force_residual'],1e-8)
            self.assertLess(floor['equilibrium']['maximum_stationary_potential_energy_residual'],1e-8)
            for response in floor['responses']:self.assertLess(response['force_partition_relative_error'],1e-9)

    def test_condensed_multifloor_solution_matches_full_uncondensed_sparse_oracle(self):
        d=fixture();system=build(d,[dict(id='DL',dead=1.,live=1.)]);result=system.solve(pdelta=False)
        # Assemble every retained floor DOF directly, together with all actual
        # columns. This oracle performs no floor condensation and no frame API.
        ids=[]
        for floor in system.floors:ids.extend(floor['assembly'].dof_ids)
        ids += [(n['id'],k) for n in d['nodes'] if n['xyz'][2]==0 for k in range(1,7)]
        self.assertEqual(len(ids),len(set(ids)));index={dof:i for i,dof in enumerate(ids)}
        k=sparse.lil_matrix((len(ids),len(ids)));f=np.zeros(len(ids));positions={n['id']:n['xyz'] for n in d['nodes']}
        for floor in system.floors:
            ix=np.array([index[i] for i in floor['assembly'].dof_ids]);k[np.ix_(ix,ix)]+=floor['model'].stiffness;f[ix]+=floor['loads'][0]['loads']
        for element,_,weights in system.frame.columns:
            c=element.definition;ix=np.array([index[(c[end],j)] for end in ('i','j') for j in range(1,7)])
            k[np.ix_(ix,ix)]+=element.stiffness;f[ix]+=element.equivalent_load(weights[0])
        fixed={index[(n['id'],j)] for n in d['nodes'] if n['xyz'][2]==0 for j in range(1,7)}
        free=np.array([i for i in range(len(ids)) if i not in fixed]);direct=np.zeros(len(ids));direct[free]=spsolve(k.tocsr()[free,:][:,free],f[free])
        frame=result['frame']['cases'][0];actual=dict(zip(map(tuple,result['frame']['physical_dof_ids']),frame['physical_displacements']))
        for dof in actual:self.assertAlmostEqual(actual[dof],direct[index[dof]],places=10)
        for floor in system.floors:
            ub=np.array([[actual[dof]] for dof in floor['assembly'].boundary_dofs]);full=floor['condensed'].recover(ub)[:,0]
            np.testing.assert_allclose(full,[direct[index[dof]] for dof in floor['assembly'].dof_ids],rtol=1e-9,atol=1e-10)

    def test_exact_faces_and_unknown_patterns_are_not_snapped_or_filled(self):
        d=fixture();x,y=physical_grid(d['floors'][1],d['geometry'],subdivisions_per_bay=2)
        for value in (1.25,2.,18.,20.,22.,38.,38.75):self.assertIn(value,x)
        for value in (1.25,1.5,14.5,14.75):self.assertIn(value,y)
        with self.assertRaisesRegex(ValueError,'Unknown actual live-load pattern'):build(d,[dict(id='bad',dead=1.,live=1.,live_pattern='missing')])

    def test_stale_frozen_definition_is_refused(self):
        d=fixture();d['floor_live_load_ksf']*=2
        with self.assertRaisesRegex(ValueError,'identity|hash|changed'):build(d,[dict(id='DL',dead=1.,live=1.)])

    def test_roof_extension_is_once_at_actual_roof_joint(self):
        d=fixture();roof=next(c for c in d['columns'] if c['story']==2 and c['grid_i']==2 and c['grid_j']==1)
        roof['roof_extension_weight_kip']=.123
        d['identity_sha256']=_hash({k:v for k,v in d.items() if k!='identity_sha256'})
        loaded=build(d,[dict(id='D',dead=1.4,live=0.)]);base=build(fixture(),[dict(id='D',dead=1.4,live=0.)])
        self.assertEqual(loaded._columns,base._columns)
        first=loaded.floors[0]['loads'][0]['loads']-base.floors[0]['loads'][0]['loads']
        np.testing.assert_array_equal(first,np.zeros_like(first))
        last=loaded.floors[1];delta=(last['loads'][0]['loads']-base.floors[1]['loads'][0]['loads']).reshape(-1,6)
        self.assertAlmostEqual(-delta[:,2].sum(),1.4*.123)
        xyz=np.array([last['assembly'].node_positions[n] for n in last['assembly'].node_ids])
        np.testing.assert_allclose(np.sum(delta[:,3:]+np.cross(xyz,delta[:,:3]),axis=0),[-1.4*.123*16,1.4*.123*40,0.],atol=1e-12)

    def test_asymmetric_actual_edges_preserve_weight_centroid_and_column_couples(self):
        d=fixture();gamma=d['gamma_kip_in3'];total=mx=my=0.
        for floor in d['floors']:
            for b in floor['beams']:
                if b['axis']=='x':b['b_in']=2. if b['line']==0 else 3.
                elif b['line']==0:b['b_in']=1.5
                elif b['line']==40:b['b_in']=2.5
        d['identity_sha256']=_hash({k:v for k,v in d.items() if k!='identity_sha256'})
        system=build(d,[dict(id='DL',dead=1.,live=1.)]);result=system.solve(pdelta=False)
        self.assertEqual(result['status'],'equilibrated')
        bounds=(-.75,41.25,-1.,17.5)
        for row in system.floors:
            floor=row['description'];level=floor['level'];cb=4. if level==1 else 3.
            # Physical pressure over the asymmetric slab, exact clear webs,
            # and column removed-slice gravity couple, independently summed.
            x0,x1,y0,y1=bounds;w=(x1-x0)*(y1-y0)*.19/144
            floor_mx=-w*(y0+y1)/2;floor_my=w*(x0+x1)/2
            for b in floor['beams']:
                length=b['end']-b['start']-(4 if b['axis']=='x' else cb)
                wb=length*b['b_in']*(b['h_in']-1)*gamma;w+=wb
                x=(b['start']+b['end'])/2 if b['axis']=='x' else b['line']
                y=b['line'] if b['axis']=='x' else (b['start']+b['end'])/2
                floor_mx-=wb*y;floor_my+=wb*x
            for c in (c for c in d['columns'] if c['story']==level):
                x,y=c['grid_i']*20,c['grid_j']*16
                a,b=max(x-2,x0),min(x+2,x1);p,q=max(y-cb/2,y0),min(y+cb/2,y1)
                removed=(b-a)*(q-p)*gamma
                floor_mx+=removed*((p+q)/2-y);floor_my-=removed*((a+b)/2-x)
                wc=4*cb*10*gamma-removed
                total+=wc;mx-=wc*y;my+=wc*x
            f=row['loads'][0]['loads'].reshape(-1,6)
            xyz=np.array([row['assembly'].node_positions[n] for n in row['assembly'].node_ids])
            np.testing.assert_allclose(np.sum(f[:,3:]+np.cross(xyz,f[:,:3]),axis=0),[floor_mx,floor_my,0],atol=1e-10)
            self.assertAlmostEqual(-float(f[:,2].sum()),w,places=10)
            total+=w;mx+=floor_mx;my+=floor_my
        r=np.array(result['frame']['cases'][0]['physical_residual_or_reaction']).reshape(-1,6)
        xyz=np.array([n['xyz'] for n in d['nodes']]);base=xyz[:,2]==0
        self.assertAlmostEqual(float(r[base,2].sum()),total,places=9)
        np.testing.assert_allclose(np.sum(r[base,3:]+np.cross(xyz[base],r[base,:3]),axis=0),[-mx,-my,0],atol=1e-8)


if __name__=='__main__':unittest.main()
