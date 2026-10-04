"""Diagnostic compatible eccentric floor/frame assembly for generated designs.

Explicit inputs only; no Structure_Parameters mutation, file I/O, frozen-case
imports or borrowed reference displacements. Linear small-displacement
gravity mechanics. Native shells plus downstand webs occur once in the floor
matrix; analytical columns occur once in the building matrix.

This API does not supply production design demands: shell/web resultants are
not automatically the effective-flange beam resultants used by existing
strength checks. No P-Delta, seismic/modal solver or cracked composite model
is qualified here. Input-keyed matrices are local to one immutable instance.
"""
from __future__ import annotations
import copy, hashlib, json, math, time
import numpy as np
import openseespy.opensees as ops
from Design.SMRF_Floor_Analysis import _inputs
from Design.SMRF_Floor_Mesh import coupled_floor_mesh
from Design.SMRF_Coupled_Analysis import _rectangle, _frame_element, _integer, _number

METHOD_VERSION='compatible_eccentric_floor_gravity_v6_superlu_responses'

def transport(f,r):
    f=np.array(f,dtype=float).copy();f[3:]+=np.cross(r,f[:3]);return f

def column_stiffness(section,length):
    """Independent Euler-Bernoulli column stiffness in global coordinates."""
    K=np.zeros((12,12));e=section['ec_ksi']
    for ids,k in (([2,8],e*section['area_in2']/length),([5,11],section['g_ksi']*section['j_in4']/length)):
        K[np.ix_(ids,ids)]+=k*np.array([[1.,-1.],[-1.,1.]])
    block=np.array([[12,6*length,-12,6*length],[6*length,4*length**2,-6*length,2*length**2],
                    [-12,-6*length,12,-6*length],[6*length,2*length**2,-6*length,4*length**2]])
    for ids,inertia,sign in (([0,4,6,10],section['iy_in4'],[1,1,1,1]),([1,3,7,9],section['iz_in4'],[1,-1,1,-1])):
        K[np.ix_(ids,ids)]+=e*inertia/length**3*block*np.outer(sign,sign)
    return K

class CompatibleFloor:
    def __init__(self,slab,geometry,sections,mesh_per_bay=4,*,slab_perimeter='centerlines',mesh_spec=None):
        from Design.SMRF_Floor_Sections import require_uniform
        require_uniform(sections,'The compatible-floor diagnostic (SMRF_Floor_Compatibility)')
        self.slab=copy.deepcopy(slab);self.geometry=copy.deepcopy(geometry);self.sections=copy.deepcopy(sections)
        self.mesh=mesh_per_bay
        self.mesh_spec=copy.deepcopy(mesh_spec)
        zero=dict(id='validation',dead_factor=0.,live_factor=0.,live_load_ksf=0.,live_pattern='none')
        nx,ny,lx,ly,hs,fc,nu,mesh,_=_inputs(self.slab,self.geometry,self.sections,zero,mesh_per_bay,allow_zero=True,mesh_spec=self.mesh_spec)
        g,s=self.geometry,self.sections
        nf=_integer(g.get('num_floor'),'num_floor');sh=_number(g.get('story_h_in'),'story_h_in')
        b=_number(s.get('b_beam_in'),'b_beam_in');hb=_number(s.get('h_beam_in'),'h_beam_in')
        bc=_number(s.get('b_col_in'),'b_col_in');hc=_number(s.get('h_col_in'),'h_col_in')
        for key in ('fc_beam_ksi','fc_col_ksi'):_number(s.get(key),key)
        for key in ('beam_stiffness_modifier','column_stiffness_modifier'):_number(s.get(key,1.),key)
        if not hs<hb<sh or hc>=lx or bc>=ly or b>=min(lx,ly):
            raise ValueError('Compatible geometry requires clear spans and a downstand web')
        coupled_floor_mesh(nx,ny,lx,ly,mesh,beam_width_in=b,slab_perimeter=slab_perimeter,mesh_spec=self.mesh_spec)
        self.slab_perimeter=slab_perimeter
        self.mesh=mesh
        g.update(num_floor=nf,num_bay_x=nx,num_bay_y=ny,story_h_in=sh,bay_x_in=lx,bay_y_in=ly)
        self.keys=[(i,j) for j in range(ny+1) for i in range(nx+1)]
        # Bound the dense retained solve, independently of the floor shell cap.
        if 6*len(self.keys)*nf>2000:raise ValueError('Diagnostic retained-DOF budget exceeds 2000')
        self._identity=self._input_identity();self._matrix=None;self._meta=None

    def _input_identity(self):
        payload=dict(method=METHOD_VERSION,slab=self.slab,geometry=self.geometry,sections=self.sections,mesh=self.mesh,
                     slab_perimeter=self.slab_perimeter,mesh_spec=self.mesh_spec)
        return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def _check_identity(self):
        if self._identity!=self._input_identity():raise ValueError('Inputs changed: create a new CompatibleFloor instance')

    def response(self,case,displacements=None,*,capture=False,explicit_mesh=False,_extract_stiffness=False,_reference_solver=False):
        """Isolated floor at z=0 with joint kinematics imposed; no column stiffness."""
        self._check_identity()
        g,s,slab,keys,mesh=self.geometry,self.sections,self.slab,self.keys,self.mesh
        if ops.getNodeTags() or ops.getEleTags():raise RuntimeError('Existing model must be preserved')
        spec=self.mesh_spec
        if explicit_mesh:
            if spec is not None:raise ValueError('Provide either a saved mesh_spec or explicit uniform refinement')
            if isinstance(mesh,bool) or not isinstance(mesh,int) or not 2<=mesh<=60:
                raise ValueError('Diagnostic explicit mesh requires an integer from 2 through 60')
            spec=dict(x_offsets_in=[i*g['bay_x_in']/mesh for i in range(mesh+1)],
                      y_offsets_in=[i*g['bay_y_in']/mesh for i in range(mesh+1)],max_shells=45000)
        nx,ny,lx,ly,hs,fc,nu,_,pressures=_inputs(slab,g,s,case,mesh,allow_zero=True,mesh_spec=spec)
        # Use the existing explicit-grid validator; do not raise its shell budget.
        grid=coupled_floor_mesh(nx,ny,lx,ly,mesh,beam_width_in=s['b_beam_in'],
                                slab_perimeter=self.slab_perimeter,mesh_spec=spec)
        xs=dict(enumerate(grid['x_coordinates_in'],start=grid['index_start']))
        ys=dict(enumerate(grid['y_coordinates_in'],start=grid['index_start']))
        cell_x,cell_y=list(xs)[:-1],list(ys)[:-1]
        mx,my=grid['subdivisions_x_per_bay'],grid['subdivisions_y_per_bay']
        ex,ey=nx*mx,ny*my
        offset=-s['h_beam_in']/2.
        web=_rectangle(s['b_beam_in'],s['h_beam_in']-hs,s['fc_beam_ksi'],s.get('beam_stiffness_modifier',1.),nu)
        gamma=slab['concrete_unit_weight_kcf']/1728.
        u=np.zeros((len(keys),6)) if displacements is None else np.asarray(displacements).reshape(len(keys),6)
        positions={};shell_nodes={};web_nodes={};shells=[];beams=[];loads={}
        def node(p):
            n=len(positions)+1;positions[n]=np.asarray(p,dtype=float);ops.node(n,*p);return n
        try:
            ops.model('basic','-ndm',3,'-ndf',6)
            ops.section('ElasticMembranePlateSection',1,57.*math.sqrt(fc*1000.),nu,hs,0.)
            ops.geomTransf('Linear',2,0.,0.,1.)
            for j,y in ys.items():
                for i,x in xs.items():
                    n=node([x,y,0.]);shell_nodes[i,j]=n
                    if 0<=i<=ex and 0<=j<=ey and (i%mx==0 or j%my==0):
                        nw=node([x,y,offset]);web_nodes[i,j]=nw
                        ops.rigidLink('beam',n,nw)
            tag=0
            for j in cell_y:
                for i in cell_x:
                    tag+=1;nodes=[shell_nodes[i,j],shell_nodes[i+1,j],shell_nodes[i+1,j+1],shell_nodes[i,j+1]]
                    ops.element('ShellMITC4',tag,*nodes,1)
                    pi,pj=max(0,min(nx-1,i//mx)),max(0,min(ny-1,j//my))
                    P=pressures[pi,pj]*(xs[i+1]-xs[i])*(ys[j+1]-ys[j])/144.
                    shells.append(dict(tag=tag,i=i,j=j,nodes=nodes,P=P))
                    for n in nodes:loads[n]=loads.get(n,0.)+P/4.
            for axis,lines,intervals in (('x',ny+1,ex),('y',nx+1,ey)):
                for line in range(lines):
                    for t in range(intervals):
                        i,j=(t,line*my) if axis=='x' else (line*mx,t)
                        ni=web_nodes[i,j];nj=web_nodes[i+int(axis=='x'),j+int(axis=='y')]
                        tag+=1;_frame_element(tag,ni,nj,web,2)
                        length,clear=(lx,lx-s['h_col_in']) if axis=='x' else (ly,ly-s['b_col_in'])
                        w=case['dead_factor']*s['b_beam_in']*(s['h_beam_in']-hs)*gamma*clear/length
                        count=mx if axis=='x' else my
                        segment_length=float(np.linalg.norm(positions[nj]-positions[ni]))
                        beams.append(dict(tag=tag,axis=axis,line_index=line,span_index=t//count,
                            segment_index=t%count,nodes=[ni,nj],w=w,P=w*segment_length))
            if _extract_stiffness:
                if displacements is not None or max(loads.values(),default=0.)!=0 or any(b['w']!=0 for b in beams):
                    raise ValueError('Stiffness extraction requires zero loads and no imposed displacements')
                boundary=[shell_nodes[i*mx,j*my] for i,j in keys]
                return self._prescribed_from_domain(boundary,positions,grid,shell_nodes,web_nodes,shells,beams)
            ops.timeSeries('Linear',1);ops.pattern('Plain',1,1)
            for k,(i,j) in enumerate(keys):
                for dof in range(6):ops.sp(shell_nodes[i*mx,j*my],dof+1,float(u[k,dof]))
            for n,P in loads.items():ops.load(n,0.,0.,-P,0.,0.,0.)
            for item in beams:ops.eleLoad('-ele',item['tag'],'-type','-beamUniform',0.,-item['w'],0.)
            ops.constraints('Transformation');ops.numberer('RCM')
            if _reference_solver:
                ops.system('UmfPack');ops.algorithm('Linear')
            else:
                ops.system('SuperLU');ops.test('FixedNumIter',2)
                ops.algorithm('ModifiedNewton','-factoronce')
            ops.integrator('LoadControl',1.);ops.analysis('Static')
            code=ops.analyze(1)
            if code:raise RuntimeError(f'Floor solve failed: {code}')
            # Explicitly transport web actions to the retained slab node. A raw
            # nodeReaction at that node alone need not include its linked web.
            nodal={n:np.zeros(6) for n in shell_nodes.values()}
            shells_saved=[];beams_saved=[]
            for item in shells:
                f=np.array(ops.eleForce(item['tag'])).reshape(4,6)
                for n,value in zip(item['nodes'],f):nodal[n]+=value
                if capture:
                    shells_saved.append(dict(cell_i=item['i'],cell_j=item['j'],
                        node_positions_in=[positions[n].tolist() for n in item['nodes']],
                        global_nodal_force_kip_kip_in=f.ravel().tolist(),
                        applied_nodal_force_kip_kip_in=[0.,0.,-item['P']/4.,0.,0.,0.]*4,
                        gauss_resultants_raw=list(ops.eleResponse(item['tag'],'stresses'))))
            retained={nw:shell_nodes[key] for key,nw in web_nodes.items()}
            for item in beams:
                f=np.array(ops.eleForce(item['tag'])).reshape(2,6)
                for nw,value in zip(item['nodes'],f):
                    n=retained[nw];nodal[n]+=transport(value,positions[nw]-positions[n])
                if capture:
                    beams_saved.append(dict(axis=item['axis'],line_index=item['line_index'],span_index=item['span_index'],
                        segment_index=item['segment_index'],node_positions_in=[positions[n].tolist() for n in item['nodes']],
                        global_force_kip_kip_in=f.ravel().tolist(),
                        local_force_kip_kip_in=list(ops.eleResponse(item['tag'],'localForce')),
                        body_load_force_kip=[0.,0.,-item['P']],
                        body_load_position_in=((positions[item['nodes'][0]]+positions[item['nodes'][1]])/2).tolist()))
            for n,P in loads.items():nodal[n][2]+=P
            boundary=[shell_nodes[i*mx,j*my] for i,j in keys]
            q=np.array([nodal[n] for n in boundary])
            boundary_set=set(boundary)
            residual=max(np.max(abs(f)) for n,f in nodal.items() if n not in boundary_set)
            relative=float(residual/max(1.,np.max(abs(q))))
            if relative>1e-8:raise RuntimeError(f'Interior force equilibrium failed: {relative}')
            result=dict(boundary_force=q.ravel(),interior_relative_residual=relative,mesh_metadata=grid,
                        response_system='UmfPack' if _reference_solver else 'SuperLU',
                        equilibrium_corrections=1 if _reference_solver else 2)
            if capture:
                result.update(shells=shells_saved,webs=beams_saved,
                    shell_node_displacements=[dict(i=i,j=j,u=list(ops.nodeDisp(n))) for (i,j),n in shell_nodes.items()],
                    boundary_force=q.ravel().tolist())
            return result
        finally:ops.wipe()

    def _prescribed_from_domain(self,boundary,positions,grid,shell_nodes,web_nodes,shells,beams):
        """Reuse one elastic factorization with all retained DOFs prescribed.

        Each retained DOF has one time-varying SP constraint. Unit displacement
        states are independent; native element resisting forces are assembled
        at retained nodes, including the moment of each eccentric web force.
        """
        nb=6*len(boundary);K=np.zeros((nb,nb));max_residual=0.
        for column in range(nb):
            tag=column+1;values=[0.]*(nb+2);values[column+1]=1.
            ops.timeSeries('Path',tag,'-dt',1.,'-values',*values)
            ops.pattern('Plain',tag,tag)
            ops.sp(boundary[column//6],column%6+1,1.)
        # OpenSees 3.8.0's UMFPACK wrapper refactorizes every RHS even when
        # Linear retains the tangent. SuperLU retains its numeric factors.
        ops.constraints('Transformation');ops.numberer('RCM');ops.system('SuperLU')
        # Two corrections with the same tangent and numeric factors. The
        # fixed iteration count is NOT the acceptance test: native interior
        # equilibrium, reciprocity and rigid modes are checked independently.
        ops.test('FixedNumIter',2)
        ops.algorithm('ModifiedNewton','-factoronce')
        ops.integrator('LoadControl',1.);ops.analysis('Static')
        retained={nw:shell_nodes[key] for key,nw in web_nodes.items()}
        boundary_set=set(boundary)
        for column in range(nb):
            if ops.analyze(1):raise RuntimeError('Native prescribed floor extraction failed')
            nodal={n:np.zeros(6) for n in shell_nodes.values()}
            for item in shells:
                for n,value in zip(item['nodes'],np.asarray(ops.eleForce(item['tag'])).reshape(4,6)):
                    nodal[n]+=value
            for item in beams:
                for nw,value in zip(item['nodes'],np.asarray(ops.eleForce(item['tag'])).reshape(2,6)):
                    n=retained[nw];nodal[n]+=transport(value,positions[nw]-positions[n])
            q=np.array([nodal[n] for n in boundary]).ravel();K[:,column]=q
            residual=max((np.max(abs(f)) for n,f in nodal.items() if n not in boundary_set),default=0.)
            max_residual=max(max_residual,float(residual/max(1.,np.max(abs(q)))))
        if max_residual>1e-8:raise RuntimeError(f'Interior extraction equilibrium failed: {max_residual}')
        return dict(boundary_stiffness=K,mesh_metadata=grid,interior_relative_residual=max_residual,
                    method='Native prescribed displacement columns with one elastic factorization')

    def stiffness(self,*,method='prescribed_reuse'):
        self._check_identity()
        g,s,keys,mesh=self.geometry,self.sections,self.keys,self.mesh
        nb=6*len(keys);scale=np.tile([g['bay_x_in']]*3+[1.]*3,len(keys))
        zero=dict(id='unit_boundary_no_gravity',dead_factor=0.,live_factor=0.,live_load_ksf=0.,live_pattern='none')
        start=time.perf_counter();K=np.zeros((nb,nb));max_internal=0.
        if method=='prescribed_reuse':
            extraction=self.response(zero,_extract_stiffness=True)
            K=extraction['boundary_stiffness'];max_internal=extraction['interior_relative_residual']
        elif method=='prescribed':
            for k in range(nb):
                u=np.zeros(nb);u[k]=1.
                result=self.response(zero,u,_reference_solver=True);K[:,k]=result['boundary_force']
                max_internal=max(max_internal,result['interior_relative_residual'])
        else:raise ValueError('Stiffness extraction must be prescribed_reuse or prescribed')
        scaled=K*np.outer(scale,scale)
        asym=float(np.max(abs(scaled-scaled.T))/np.max(abs(scaled)))
        if asym>1e-9:raise RuntimeError(f'Nonreciprocal condensed stiffness: {asym}')
        # Keep the extracted matrix unsymmetrized for assembly. The symmetric part
        # below is only an eigenvalue diagnostic in consistently scaled coordinates.
        eig=np.linalg.eigvalsh((scaled+scaled.T)/2)
        if eig[0]<-1e-8*eig[-1]:raise RuntimeError('Materially negative stiffness eigenvalue')
        rb=np.zeros((nb,6));cx=g['num_bay_x']*g['bay_x_in']/2;cy=g['num_bay_y']*g['bay_y_in']/2
        for j,(i,jj) in enumerate(keys):
            r=np.array([i*g['bay_x_in']-cx,jj*g['bay_y_in']-cy,0.])
            rb[6*j:6*j+3,:3]=np.eye(3)
            for d in range(3):rb[6*j:6*j+3,3+d]=np.cross(np.eye(3)[d],r)
            rb[6*j+3:6*j+6,3:]=np.eye(3)
        rbscaled=rb/scale[:,None]
        rb_error=float(np.max(abs(scaled@rbscaled))/(np.max(abs(scaled))*np.max(abs(rbscaled))))
        if rb_error>1e-9:raise RuntimeError(f'Rigid-body mode did not close: {rb_error}')
        meta=dict(mesh=mesh,mesh_spec=copy.deepcopy(self.mesh_spec),extraction_method=method,
            extraction_system='SuperLU' if method=='prescribed_reuse' else 'UmfPack',
            equilibrium_corrections_per_state=2 if method=='prescribed_reuse' else 1,
            boundary_dofs=nb,relative_scaled_asymmetry=asym,
            rigid_body_relative_residual=rb_error,scaled_eigenvalues=eig.tolist(),
            interior_relative_residual=max_internal,elapsed_seconds=time.perf_counter()-start,
            web_offset_in=-s['h_beam_in']/2,stiffness_symmetrized_for_solution=False,
            production_enabled=False,scope='Static elastic eccentric shell/web floor, six DOFs at every column joint')
        return K,meta

    def solve(self, floor_loadcases, *, inplane_restraint='finite_membrane', capture=True):
        """Assemble every generated floor and column with compatible joint DOFs.

        Cases may differ by floor. Local recovered floor coordinates have z=0;
        joint/column outputs include their story indices. A caller transporting
        cuts to building coordinates must add the story elevation explicitly.
        """
        self._check_identity()
        if ops.getNodeTags() or ops.getEleTags():
            raise RuntimeError('Compatible diagnostic requires an empty OpenSees domain')
        g,s,slab,keys=self.geometry,self.sections,self.slab,self.keys
        nf=g['num_floor'];h=g['story_h_in'];nb=6*len(keys);n=nb*nf
        if not isinstance(floor_loadcases,list) or len(floor_loadcases)!=nf:
            raise ValueError('Provide one explicit gravity case per floor')
        if inplane_restraint not in ('finite_membrane','rigid_joints'):
            raise ValueError('inplane_restraint must be finite_membrane or rigid_joints')
        parsed=[_inputs(slab,g,s,case,self.mesh,allow_zero=True,mesh_spec=self.mesh_spec) for case in floor_loadcases]
        if self._matrix is None:self._matrix,self._meta=self.stiffness()
        col=_rectangle(s['b_col_in'],s['h_col_in'],s['fc_col_ksi'],s.get('column_stiffness_modifier',1.),s.get('slab_poisson_ratio',.2))
        C=column_stiffness(col,h);K=np.zeros((n,n));F=np.zeros(n)
        initial={};column_loads=[]
        for k,case in enumerate(floor_loadcases):
            key=json.dumps(case,sort_keys=True,allow_nan=False)
            if key not in initial:initial[key]=self.response(case)['boundary_force']
            sl=slice(k*nb,(k+1)*nb);K[sl,sl]+=self._matrix;F[sl]-=initial[key]
            P=case['dead_factor']*s['b_col_in']*s['h_col_in']*(h-slab['thickness_in'])*slab['concrete_unit_weight_kcf']/1728.
            fcol=np.zeros(12);fcol[[2,8]]=-P/2.;column_loads.append(fcol)
            for j in range(len(keys)):
                top=list(range(k*nb+6*j,k*nb+6*j+6))
                if k:
                    ids=list(range((k-1)*nb+6*j,(k-1)*nb+6*j+6))+top
                    K[np.ix_(ids,ids)]+=C;F[ids]+=fcol
                else:
                    K[np.ix_(top,top)]+=C[6:,6:];F[top]+=fcol[6:]
        if inplane_restraint=='finite_membrane':T=np.eye(n)
        else:
            per=3+3*len(keys);T=np.zeros((n,per*nf));cx=g['num_bay_x']*g['bay_x_in']/2.;cy=g['num_bay_y']*g['bay_y_in']/2.
            for k in range(nf):
                for j,(i,jj) in enumerate(keys):
                    a=k*nb+6*j;b=k*per
                    T[a,b]=1.;T[a,b+2]=-(jj*g['bay_y_in']-cy)
                    T[a+1,b+1]=1.;T[a+1,b+2]=i*g['bay_x_in']-cx;T[a+5,b+2]=1.
                    for d in range(3):T[a+2+d,b+3+3*j+d]=1.
        Kr=T.T@K@T;Fr=T.T@F;ur=np.linalg.solve(Kr,Fr);u=T@ur
        residual=float(np.max(abs(Kr@ur-Fr))/max(1.,np.max(abs(Fr))))
        if not np.all(np.isfinite(u)) or residual>1e-8:
            raise RuntimeError(f'Compatible assembly equilibrium failed: {residual}')
        columns=[]
        for k in range(nf):
            for j,(i,jj) in enumerate(keys):
                top=u[k*nb+6*j:k*nb+6*j+6]
                bottom=np.zeros(6) if not k else u[(k-1)*nb+6*j:(k-1)*nb+6*j+6]
                force=C@np.r_[bottom,top]-column_loads[k];local=[]
                for a in (0,6):
                    f=force[a:a+6];local.extend([f[2],-f[1],f[0],f[5],-f[4],f[3]])
                columns.append(dict(story=k+1,grid_i=i,grid_j=jj,
                    global_force_kip_kip_in=force.tolist(),local_force_kip_kip_in=local))
        # Independent whole-building gravity ledger and base reaction wrench.
        # Downstand and column self-weight are counted once, excluding slab.
        applied=np.zeros(6);ledger=dict(slab_pressure_kip=0.,beam_drop_weight_kip=0.,column_weight_kip=0.)
        gamma=slab['concrete_unit_weight_kcf']/1728.
        def add_weight(P,x,y,key):
            ledger[key]+=P;applied[2]-=P;applied[3]-=P*y;applied[4]+=P*x
        for k,case in enumerate(floor_loadcases):
            for (i,j),q in parsed[k][-1].items():
                # Independent analytical panel rectangles, not a sum of mesh loads.
                e=s['b_beam_in']/2 if self.slab_perimeter=='beam_outer_faces' else 0.
                x0=i*g['bay_x_in']-(e if i==0 else 0.)
                x1=(i+1)*g['bay_x_in']+(e if i==g['num_bay_x']-1 else 0.)
                y0=j*g['bay_y_in']-(e if j==0 else 0.)
                y1=(j+1)*g['bay_y_in']+(e if j==g['num_bay_y']-1 else 0.)
                add_weight(q*(x1-x0)*(y1-y0)/144.,(x0+x1)/2,(y0+y1)/2,'slab_pressure_kip')
            for axis in ('x','y'):
                lines,spans=(g['num_bay_y']+1,g['num_bay_x']) if axis=='x' else (g['num_bay_x']+1,g['num_bay_y'])
                clear=g['bay_x_in']-s['h_col_in'] if axis=='x' else g['bay_y_in']-s['b_col_in']
                P=case['dead_factor']*s['b_beam_in']*(s['h_beam_in']-slab['thickness_in'])*clear*gamma
                for line in range(lines):
                    for span in range(spans):
                        x,y=((span+.5)*g['bay_x_in'],line*g['bay_y_in']) if axis=='x' else (line*g['bay_x_in'],(span+.5)*g['bay_y_in'])
                        add_weight(P,x,y,'beam_drop_weight_kip')
            for i,j in keys:
                add_weight(-sum(column_loads[k][[2,8]]),i*g['bay_x_in'],j*g['bay_y_in'],'column_weight_kip')
        reaction=np.zeros(6)
        for c in columns:
            if c['story']==1:
                reaction+=transport(c['global_force_kip_kip_in'][:6],[c['grid_i']*g['bay_x_in'],c['grid_j']*g['bay_y_in'],0.])
        force_scale=max(1.,abs(applied[2]));moment_scale=force_scale*max(g['num_bay_x']*g['bay_x_in'],g['num_bay_y']*g['bay_y_in'],nf*h)
        error=reaction+applied
        balance=dict(applied_force_moment=applied.tolist(),base_force_moment=reaction.tolist(),
            force_relative_error=float(max(abs(error[:3]))/force_scale),moment_relative_error=float(max(abs(error[3:]))/moment_scale))
        if max(balance['force_relative_error'],balance['moment_relative_error'])>1e-8:
            raise RuntimeError(f'Compatible whole-building gravity balance failed: {balance}')
        recovery=[]
        if capture:
            for k,case in enumerate(floor_loadcases):
                recovered=self.response(case,u[k*nb:(k+1)*nb],capture=True)
                recovered.update(story=k+1,building_elevation_in=(k+1)*h)
                recovery.append(recovered)
        return dict(method_version=METHOD_VERSION,input_sha256=self._identity,
            inputs=dict(slab=copy.deepcopy(slab),geometry=copy.deepcopy(g),sections=copy.deepcopy(s),
                floor_loadcases=copy.deepcopy(floor_loadcases),mesh_per_bay=self.mesh,slab_perimeter=self.slab_perimeter,
                mesh_spec=copy.deepcopy(self.mesh_spec)),
            inplane_restraint=inplane_restraint,stiffness_audit=copy.deepcopy(self._meta),
            joint_keys=keys,joint_displacements=u.reshape(nf,len(keys),6).tolist(),
            constraint_actions=(K@u-F).reshape(nf,len(keys),6).tolist(),column_actions=columns,
            floors=recovery,reduced_equilibrium_relative_residual=residual,
            weight_ledger=ledger,equilibrium=balance,
            borrowed_reference_displacements=False,production_enabled=False,engineering_verified=False,
            applied_to_design=False,
            design_integration_blockers=['Compatible shell/web cuts must be paired with the same concrete and developed steel in beam/slab strength checks.',
                'The current preliminary frame uses cracked T/L stiffness and P-Delta; this gravity reduction has not replaced its full signed seismic combination model.'])

