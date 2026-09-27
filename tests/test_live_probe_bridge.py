import unittest
from tristan_bench.live_probe_bridge import *

class LiveProbeBridgeTests(unittest.TestCase):
    def fixture(self):
        raw={"payload_digest":"OUT","node":"N","model":"M","task":"t","wall_ms":10}
        pd=expected_probe_digest(raw)
        receipt={"receipt_digest":"R","status":"VERIFIED","executed":True,"verified":True,"output_digest":"OUT"}
        obs=LiveProbeObservation(
            "o","nid","N","M","opt","t","strict","windows","private","free-local","local",
            "x","x",True,10,"E1","R",pd
        )
        return raw,receipt,obs
    def test_verified_binding(self):
        raw,r,o=self.fixture()
        self.assertTrue(verify_observation_binding(o,raw_probe=raw,worker_receipt=r)[0])
    def test_unverified_receipt_rejected(self):
        raw,r,o=self.fixture(); r=dict(r,status="HOLD_VERIFICATION")
        self.assertFalse(verify_observation_binding(o,raw_probe=raw,worker_receipt=r)[0])
    def test_probe_tamper_rejected(self):
        raw,r,o=self.fixture(); raw=dict(raw,wall_ms=11)
        self.assertFalse(verify_observation_binding(o,raw_probe=raw,worker_receipt=r)[0])
    def test_output_digest_mismatch_rejected(self):
        raw,r,o=self.fixture(); r=dict(r,output_digest="OTHER")
        self.assertFalse(verify_observation_binding(o,raw_probe=raw,worker_receipt=r)[0])
    def test_trial_contains_node_model_regime(self):
        raw,r,o=self.fixture()
        t=verified_probe_to_trial(o,raw_probe=raw,worker_receipt=r)
        self.assertIn("N",t.regime.task_class)
        self.assertIn("M",t.regime.task_class)
        self.assertEqual(t.quality,1.0)
    def test_inexact_trial_quality_zero(self):
        raw,r,o=self.fixture()
        o=LiveProbeObservation(**{**o.__dict__,"exact":False})
        t=verified_probe_to_trial(o,raw_probe=raw,worker_receipt=r)
        self.assertEqual(t.quality,0.0)
    def test_missing_receipt_fails_batch(self):
        raw,r,o=self.fixture()
        with self.assertRaises(ValueError):
            batch_verified_trials([o],{"o":raw},{})
    def test_deterministic_probe_digest(self):
        a={"b":2,"a":1}; b={"a":1,"b":2}
        self.assertEqual(expected_probe_digest(a),expected_probe_digest(b))

if __name__=="__main__": unittest.main()
