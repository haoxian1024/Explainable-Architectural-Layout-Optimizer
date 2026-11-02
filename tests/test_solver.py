import copy
import json
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solver import Problem, solve, validate, rectangle


def room():
    return {'boundary':[[0,0],[4000,0],[4000,4000],[0,4000]],'door':[[0,1500],[0,2300]],'isOpenInward':False,'algoToPlace':{'shelf-1':[1000,400]}}

class GeometryTests(unittest.TestCase):
    def test_feasible(self):
        r=solve(room(),step=500,seconds=2)
        self.assertTrue(r['feasible']); self.assertTrue(r['validation']['passed'])
    def test_area_impossible(self):
        d=room(); d['algoToPlace']={'shelf':[9000,9000]}
        self.assertIs(solve(d)['feasible'],False)
    def test_overlap(self):
        d=room(); d['algoToPlace']['shelf-2']=[1000,400]
        items=[{'name':n,'center':[2000,2000],'rotation_deg':0} for n in d['algoToPlace']]
        self.assertFalse(validate(Problem(d),items)['passed'])
    def test_fridge_front(self):
        d=room(); d['algoToPlace']={'fridge':[1000,500],'shelf':[500,100]}
        a=[{'name':'fridge','center':[2000,2000],'rotation_deg':0},{'name':'shelf','center':[2000,1500],'rotation_deg':0}]
        self.assertFalse(validate(Problem(d),a)['passed'])
        a[1]['center']=[2000,2600]
        self.assertTrue(validate(Problem(d),a)['passed'])
    def test_inward_door(self):
        d=room(); d['isOpenInward']=True; p=Problem(d)
        self.assertAlmostEqual(p.door_zone.area,800**2)
        b,c=rectangle([400,1900],300,300,0)
        self.assertFalse(p.valid(b,c))
    def test_edge_contact_allowed(self):
        d=room(); d['algoToPlace']['shelf-2']=[1000,400]
        a=[{'name':'shelf-1','center':[500,200],'rotation_deg':0},{'name':'shelf-2','center':[1500,200],'rotation_deg':0}]
        self.assertTrue(validate(Problem(d),a)['passed'])
    def test_concavity(self):
        d=room(); d['boundary']=[[0,0],[4000,0],[4000,1000],[1000,1000],[1000,4000],[0,4000]]
        p=Problem(d); b,c=rectangle([1500,1500],1800,1800,0)
        self.assertFalse(p.valid(b,c))
    def test_bad_angle(self):
        self.assertFalse(validate(Problem(room()),[{'name':'shelf-1','center':[2000,2000],'rotation_deg':13}])['passed'])
    def test_bad_door(self):
        d=room(); d['door']=[[200,200],[200,1000]]
        with self.assertRaises(ValueError): Problem(d)
    def test_bad_dimensions(self):
        d=room(); d['algoToPlace']['shelf-1']=[-1,400]
        with self.assertRaises(ValueError): Problem(d)
    def test_bowtie_rejected(self):
        d=room(); d['boundary']=[[0,0],[4000,4000],[0,4000],[4000,0]]
        with self.assertRaises(ValueError): Problem(d)
    def test_saved_examples(self):
        root=Path(__file__).resolve().parents[1]
        for i in range(1,5):
            with self.subTest(example=i):
                d=json.loads((root/f'examples/example{i}.json').read_text())
                r=json.loads((root/f'results/example{i}.json').read_text())
                self.assertTrue(r['feasible'])
                self.assertTrue(validate(Problem(d),r['placements'])['passed'])
    def test_unknown_not_false(self):
        d=room(); d['algoToPlace']={'shelf':[5000,10]}
        r=solve(d,step=1000,seconds=1)
        self.assertEqual(r['status'],'UNKNOWN')
        self.assertIsNone(r['feasible'])
    def test_interior_fallback(self):
        d={'boundary':[[0,0],[3,0],[3,3],[0,3]],'door':[[0,1.1],[0,1.9]],
           'isOpenInward':False,'doorAccessDepth':0,
           'algoToPlace':{f'shelf-{i}':[1,1] for i in range(8)}}
        r=solve(d,step=1,interior_step=1,seconds=2)
        self.assertTrue(r['feasible'])
        self.assertEqual(len(r['stages']),2)
        self.assertLess(r['wall_count'],8)
    def test_retraced_boundary(self):
        root=Path(__file__).resolve().parents[1]
        p=Problem(json.loads((root/'examples/example4.json').read_text()))
        self.assertTrue(p.warnings)

if __name__=='__main__': unittest.main()
