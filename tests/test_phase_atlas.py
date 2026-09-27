import unittest
from tristan_bench.phase_atlas import *

class PhaseAtlasTests(unittest.TestCase):
    def r(self,task="code",platform="linux"): return RegimeKey(task,platform,"private","free","lan")
    def t(self,tid,solver,q=.8,lat=100,cost=0,rel=.9,e=.8,regime=None,valid=True):
        return Trial(tid,regime or self.r(),solver,q,lat,cost,rel,e,0,valid)
    def test_phase_specific_winner(self):
        a=self.r("code"); b=self.r("math")
        atlas=build_phase_atlas([self.t("1","A",.9,50,regime=a),self.t("2","B",.8,100,regime=a),self.t("3","B",.95,50,regime=b)])
        self.assertEqual(solver_for(atlas,a),"A"); self.assertEqual(solver_for(atlas,b),"B")
    def test_no_global_winner(self):
        self.assertIsNone(global_winner(build_phase_atlas([])))
    def test_invalid_trial_cannot_win(self):
        atlas=build_phase_atlas([self.t("1","bad",1,1,valid=False),self.t("2","ok",.5,100)])
        self.assertEqual(solver_for(atlas,self.r()),"ok")
    def test_digest_deterministic(self):
        xs=[self.t("1","A"),self.t("2","B",.7)]
        self.assertEqual(build_phase_atlas(xs).atlas_digest,build_phase_atlas(xs).atlas_digest)
    def test_counterfactual_better(self):
        h=HistoricalDecision("d",self.r(),"old",1,10,5)
        r=replay(h,[ReplayCandidate("new",2,5,2,.8)])
        self.assertEqual(r.status,"COUNTERFACTUAL_BETTER"); self.assertEqual(r.work_destroyed,5)
    def test_work_reduction_only(self):
        h=HistoricalDecision("d",self.r(),"old",1,10,5)
        r=replay(h,[ReplayCandidate("new",1,5,5,.8)])
        self.assertEqual(r.status,"WORK_REDUCTION_ONLY")
    def test_coverage(self):
        a=self.r("a"); b=self.r("b")
        atlas=build_phase_atlas([self.t("1","A",regime=a)])
        self.assertEqual(phase_coverage(atlas,[a,b]),.5)

if __name__=="__main__": unittest.main()
