"""Finite candidate rectangle packing; continuous infeasibility is never inferred."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree


@dataclass
class Candidate:
    group: int
    center: tuple
    angle: float
    body: Polygon
    clearance: Polygon | None
    wall: bool


def rectangle(center, length, width, angle, front_depth=0):
    a = math.radians(angle)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.array(center)
    body = Polygon([c + x*u + y*v for x,y in [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]])
    clear = None
    if front_depth > 0:
        clear = Polygon([c+x*u+y*v for x,y in [(-length/2,-width/2),(length/2,-width/2),(length/2,-width/2-front_depth),(-length/2,-width/2-front_depth)]])
    return body, clear


class Problem:
    def __init__(self, data):
        self.data = data
        coords = np.asarray(data['boundary'], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3 or not np.isfinite(coords).all():
            raise ValueError('boundary must have >=3 finite 2D points')
        self.origin = coords.min(axis=0)
        coords = coords-self.origin
        raw = Polygon(coords)
        self.warnings = []
        if not raw.is_valid:
            # Only repair duplicate/retraced boundary segments whose noded linework
            # encloses exactly one face. Reject bowties and multi-room guesses.
            faces = list(polygonize(unary_union(LineString(list(coords)+[coords[0]]))))
            if len(faces) != 1:
                raise ValueError('Ambiguous/self-intersecting boundary; provide one simple polygon')
            fixed = faces[0]
            if abs(fixed.area-raw.area) > max(1e-6, fixed.area*1e-10):
                raise ValueError('Boundary repair changes area; fix input explicitly')
            raw = fixed
            self.warnings.append('Normalized duplicate/retraced boundary segments; enclosed area unchanged.')
        if raw.area <= 0 or raw.interiors:
            raise ValueError('Boundary must be a simple positive-area polygon without holes')
        self.room = orient(raw, sign=1)
        self.tol = max(1e-7, math.sqrt(raw.area)*1e-9)
        self.area_tol = self.tol**2
        door = np.asarray(data['door'],dtype=float)
        if door.shape != (2,2) or not np.isfinite(door).all():
            raise ValueError('door must have two finite 2D endpoints')
        self.door = LineString(door-self.origin)
        n = self.door.length
        if n <= self.tol or self.door.difference(self.room.boundary.buffer(self.tol)).length > self.tol:
            raise ValueError('Door must lie along boundary and have positive width')
        if not isinstance(data['isOpenInward'],bool):
            raise ValueError('isOpenInward must be boolean')
        depth = n if data['isOpenInward'] else float(data.get('doorAccessDepth',n))
        if not math.isfinite(depth) or depth < 0:
            raise ValueError('doorAccessDepth must be finite and nonnegative')
        q = np.array(self.door.coords)
        u = (q[1]-q[0])/n
        v = np.array([-u[1],u[0]])
        areas = [Polygon([q[0],q[1],q[1]+s*v*depth,q[0]+s*v*depth]) for s in [1,-1]]
        self.door_zone = max(areas,key=lambda p:p.intersection(self.room).area)
        if data['isOpenInward'] and not self.room.buffer(self.tol).covers(self.door_zone):
            raise ValueError('Full inward N x N door zone does not fit; check input')
        # Outside-opening access zone may extend past concave boundary; obstacles
        # are already forbidden outside room, so only its interior portion matters.
        self.door_zone = self.door_zone.intersection(self.room)
        self.groups=[]
        for name,dims in data['algoToPlace'].items():
            if len(dims)!=2 or any(not math.isfinite(float(x)) or float(x)<=0 for x in dims):
                raise ValueError('All item dimensions must be positive finite numbers')
            length,width = map(float,dims)
            kind = name.split('-')[0]
            depth = float(data.get('fridgeClearance',length/2)) if kind=='fridge' else 0.
            if not math.isfinite(depth) or (kind=='fridge' and depth<=0):
                raise ValueError('fridgeClearance must be finite and positive')
            key = (kind,length,width,depth)
            found = next((g for g in self.groups if g['key']==key),None)
            if found is None:
                found={'key':key,'names':[]}
                self.groups.append(found)
            found['names'].append(name)
        self.edges=[]
        rc = list(self.room.exterior.coords)
        for p,q in zip(rc,rc[1:]):
            p,q=np.array(p),np.array(q)
            d=q-p
            if np.linalg.norm(d)>self.tol:
                self.edges.append((p,q,d/np.linalg.norm(d)))

    def valid(self, body, clear):
        room = self.room.buffer(self.tol)
        if not room.covers(body): return False
        if body.intersection(self.door_zone).area > self.area_tol: return False
        if body.intersection(self.door).length > self.tol: return False
        if clear is not None and not room.covers(clear): return False
        return True

    def overlap(self,a,b):
        # Tiny negative buffers suppress floating point slivers at exact contacts.
        def hit(x,y):
            return x.buffer(-self.tol).intersects(y.buffer(-self.tol))
        return hit(a.body,b.body) or (a.clearance is not None and hit(a.clearance,b.body)) or (b.clearance is not None and hit(b.clearance,a.body))

    def on_wall(self,body):
        return body.boundary.intersection(self.room.boundary.buffer(self.tol)).length > 10*self.tol

    def candidates(self, step, interior=False):
        result=[]
        for gi,g in enumerate(self.groups):
            kind,L,W,D=g['key']; seen=set()
            def add(c,theta):
                theta=theta%360
                if D==0: theta=theta%180
                key=tuple(round(float(x),6) for x in (*c,theta))
                if key in seen: return
                seen.add(key)
                body,clear=rectangle(c,L,W,theta,D)
                if self.valid(body,clear):
                    result.append(Candidate(gi,tuple(map(float,c)),float(theta),body,clear,self.on_wall(body)))
            for p,q,u in self.edges:
                edge_len=np.linalg.norm(q-p); v=np.array([-u[1],u[0]])
                base=math.degrees(math.atan2(u[1],u[0]))
                for turn in range(4):
                    along,deep=(L,W) if turn%2==0 else (W,L)
                    if along>edge_len+self.tol: continue
                    # Both ends plus projected architectural vertices and regular samples.
                    room_vertices=np.asarray(self.room.exterior.coords)
                    anchors=[0.,edge_len-along]
                    for t in (room_vertices-p)@u:
                        anchors.extend([t,t-along])
                    anchors.extend(np.arange(0,max(0,edge_len-along),step))
                    for t in sorted(set(round(float(t),8) for t in anchors if -self.tol<=t<=edge_len-along+self.tol)):
                        add(p+u*(t+along/2)+v*deep/2,base+90*turn)
            if interior:
                angles=sorted(set(round(math.degrees(math.atan2(u[1],u[0]))%90,8) for _,_,u in self.edges))
                for base in angles:
                    a=math.radians(base); u=np.array([math.cos(a),math.sin(a)]); v=np.array([-u[1],u[0]])
                    xy=np.asarray(self.room.exterior.coords)
                    xs,ys=xy@u,xy@v
                    for turn in range(4 if D else 2):
                        along,deep=(L,W) if turn%2==0 else (W,L)
                        for x in np.arange(xs.min()+along/2,xs.max()-along/2+self.tol,step):
                            for y in np.arange(ys.min()+deep/2,ys.max()-deep/2+self.tol,step):
                                add(x*u+y*v,base+90*turn)
        return result


def search(problem,candidates,seconds):
    """Conflict bitsets + MRV backtracking with symmetry breaking for identical items."""
    start=time.monotonic(); n=len(candidates)
    tree=STRtree([c.body for c in candidates])
    conflict=[1<<i for i in range(n)]
    for i,c in enumerate(candidates):
        envelope=c.body if c.clearance is None else c.body.union(c.clearance)
        for j0 in tree.query(envelope):
            j=int(j0)
            if i!=j and problem.overlap(c,candidates[j]):
                conflict[i]|=1<<j; conflict[j]|=1<<i
    masks=[0]*len(problem.groups)
    for i,c in enumerate(candidates): masks[c.group]|=1<<i
    counts=[len(g['names']) for g in problem.groups]
    nodes=0; timed_out=False
    deadline=time.monotonic()+seconds
    def dfs(avail,remaining,chosen):
        nonlocal nodes,timed_out
        nodes+=1
        if time.monotonic()>deadline:
            timed_out=True; return None
        if not any(remaining): return chosen
        options=[]
        for g,k in enumerate(remaining):
            if k:
                count=avail[g].bit_count()
                if count<k: return None
                options.append((count/k,g))
        gi=min(options)[1]
        bits=avail[gi]
        while bits:
            bit=bits&-bits; bits-=bit; idx=bit.bit_length()-1
            new=[m & ~conflict[idx] for m in avail]
            # Identical items use increasing candidate IDs, eliminating permutations.
            new[gi]&=~((1<<(idx+1))-1)
            rem=remaining.copy(); rem[gi]-=1
            answer=dfs(new,rem,chosen+[idx])
            if answer is not None: return answer
            if timed_out: break
        return None
    chosen=dfs(masks,counts,[])
    return chosen,{'candidate_count':n,'nodes':nodes,'search_timed_out':timed_out,'elapsed_seconds':round(time.monotonic()-start,3)}


def validate(problem, placements):
    errors=[]; items=[]
    expected=problem.data['algoToPlace']
    names=[p['name'] for p in placements]
    if len(names)!=len(set(names)) or set(names)!=set(expected): errors.append('Missing/duplicate/unexpected items')
    allowed=[math.degrees(math.atan2(u[1],u[0])) for _,_,u in problem.edges]
    for p in placements:
        if p['name'] not in expected: continue
        L,W=expected[p['name']]
        D=float(problem.data.get('fridgeClearance',L/2)) if p['name'].split('-')[0]=='fridge' else 0
        angle=float(p['rotation_deg'])
        if not math.isfinite(angle) or not np.isfinite(p['center']).all():
            errors.append('Nonfinite pose'); continue
        if not any(abs((angle-a+45)%90-45)<1e-5 for a in allowed): errors.append(p['name']+': invalid angle')
        body,clear=rectangle(np.asarray(p['center'])-problem.origin,L,W,angle,D)
        if not problem.valid(body,clear): errors.append(p['name']+': containment/door/clearance')
        items.append((p['name'],Candidate(0,(),angle,body,clear,False)))
    for i,(na,a) in enumerate(items):
        for nb,b in items[i+1:]:
            if problem.overlap(a,b): errors.append(na+' vs '+nb+': collision')
    return {'passed':not errors,'errors':errors}


def solve(data,step=100.,seconds=15.,interior_step=None):
    if not math.isfinite(step) or step<=0 or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('step and seconds must be positive finite numbers')
    p=Problem(data)
    base={'feasible':None,'status':'UNKNOWN','placements':[],'warnings':p.warnings,
          'model':{'fridge_front':'local -Y edge (length edge)','fridge_default_depth':'length / 2',
                   'door_access_depth':data.get('doorAccessDepth',p.door.length),'all_items_same_floor':True},'stages':[]}
    total=sum(float(L)*float(W) for L,W in data['algoToPlace'].values())
    if total>p.room.area-p.door_zone.area+p.area_tol:
        base.update(feasible=False,status='INFEASIBLE',reason='Item area exceeds usable floor area')
        return base
    for interior,spacing in [(False,step),(True,interior_step or step*2)]:
        candidates=p.candidates(spacing,interior)
        # Prefer wall candidates at every branch, then architectural endpoint order.
        candidates.sort(key=lambda c:not c.wall)
        chosen,stats=search(p,candidates,seconds)
        stats.update(stage='wall_plus_interior' if interior else 'wall_only',step=spacing)
        base['stages'].append(stats)
        if chosen is None: continue
        used=[0]*len(p.groups); placements=[]
        for idx in chosen:
            c=candidates[idx]; g=p.groups[c.group]
            name=g['names'][used[c.group]]; used[c.group]+=1
            placements.append({'name':name,'center':list(np.array(c.center)+p.origin),
                               'rotation_deg':c.angle,'touches_wall':bool(c.wall)})
        placements.sort(key=lambda x:list(data['algoToPlace']).index(x['name']))
        check=validate(p,placements)
        if not check['passed']: raise RuntimeError('Internal validation failed: '+str(check))
        wall_count=sum(x['touches_wall'] for x in placements)
        base.update(feasible=True,status='FEASIBLE',placements=placements,validation=check,
                    wall_count=wall_count,wall_count_optimal=wall_count==len(placements))
        return base
    base['reason']='No solution found in sampled candidates/time budget; continuous infeasibility NOT proved'
    return base


def plot(data,result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    p=Problem(data); fig,ax=plt.subplots(figsize=(8,9))
    coords=np.array(p.room.exterior.coords); ax.plot(*coords.T,color='#26384b',lw=2)
    if not p.door_zone.is_empty and p.door_zone.geom_type=='Polygon':
        ax.add_patch(Patch(p.door_zone.exterior.coords,color='#efb86b',alpha=.35,label='Door reserved zone'))
    ax.plot(*np.array(p.door.coords).T,color='#e08b28',lw=5)
    colors={'fridge':'#709dcf','shelf':'#88bbac','overShelf':'#bfacd8','iceMaker':'#e2b173'}
    for item in result['placements']:
        L,W=data['algoToPlace'][item['name']]; kind=item['name'].split('-')[0]
        c=np.array(item['center'])-p.origin
        b,f=rectangle(c,L,W,item['rotation_deg'],float(data.get('fridgeClearance',L/2)) if kind=='fridge' else 0)
        ax.add_patch(Patch(b.exterior.coords,facecolor=colors.get(kind,'gray'),edgecolor='#26384b',alpha=.85))
        ax.text(*c,item['name'],ha='center',va='center',fontsize=8)
        if f is not None:
            ax.add_patch(Patch(f.exterior.coords,facecolor='#709dcf',alpha=.15,hatch='//',edgecolor='#709dcf'))
            d=np.array(f.centroid.coords[0])-c
            ax.arrow(*c,*(d*.65),width=8,color='#345b8d',length_includes_head=True)
    ax.set_aspect('equal'); ax.margins(.08); ax.grid(alpha=.12)
    ax.set_xlabel(f'X relative to {p.origin[0]:.4f}'); ax.set_ylabel(f'Y relative to {p.origin[1]:.4f}')
    ax.set_title(f"{Path(path).stem} | {result['status']} | Wall: {result.get('wall_count',0)}/{len(data['algoToPlace'])}")
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path); parser.add_argument('--output',type=Path,default=Path('result.json'))
    parser.add_argument('--step',type=float,default=100); parser.add_argument('--seconds',type=float,default=15)
    parser.add_argument('--plot',type=Path); parser.add_argument('--validate',type=Path)
    args=parser.parse_args()
    try:
        data=json.loads(args.input.read_text(encoding='utf-8'))
        if args.validate:
            check=validate(Problem(data),json.loads(args.validate.read_text(encoding='utf-8'))['placements'])
            print(json.dumps(check,ensure_ascii=False)); return 0 if check['passed'] else 1
        result=solve(data,args.step,args.seconds)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if args.plot:
            args.plot.parent.mkdir(parents=True,exist_ok=True); plot(data,result,args.plot)
        print(json.dumps({k:result[k] for k in ('status','feasible','stages')},ensure_ascii=False))
        return 0
    except (ValueError,KeyError,TypeError) as e:
        parser.exit(2,'Invalid input: '+str(e)+'\n')

if __name__=='__main__':
    raise SystemExit(main())
