from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

from .phase_atlas import RegimeKey, Trial

PROTOCOL="TRISTAN-LIVE-PROBE-PHASE-BRIDGE-R1"

def _canon(v:object)->bytes:
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":"),default=str,allow_nan=False).encode()
def digest(v:object)->str:
    return sha256(_canon(v)).hexdigest()

@dataclass(frozen=True)
class LiveProbeObservation:
    observation_id:str
    node_id:str
    node_name:str
    model:str
    options_digest:str
    task_id:str
    task_class:str
    platform:str
    privacy_class:str
    budget_class:str
    network_class:str
    expected:str
    response:str
    exact:bool
    wall_ms:float
    epoch:str
    worker_receipt_digest:str
    probe_digest:str

    @classmethod
    def from_mapping(cls,m:Mapping[str,object])->"LiveProbeObservation":
        return cls(
            observation_id=str(m["observation_id"]),
            node_id=str(m["node_id"]),
            node_name=str(m["node_name"]),
            model=str(m["model"]),
            options_digest=str(m["options_digest"]),
            task_id=str(m["task_id"]),
            task_class=str(m.get("task_class","strict-format")),
            platform=str(m.get("platform","unknown")),
            privacy_class=str(m.get("privacy_class","private")),
            budget_class=str(m.get("budget_class","free-local")),
            network_class=str(m.get("network_class","local")),
            expected=str(m.get("expected","")),
            response=str(m.get("response","")),
            exact=bool(m.get("exact",False)),
            wall_ms=float(m.get("wall_ms",0.0)),
            epoch=str(m["epoch"]),
            worker_receipt_digest=str(m["worker_receipt_digest"]),
            probe_digest=str(m["probe_digest"]),
        )

    def regime(self)->RegimeKey:
        return RegimeKey(
            task_class=f"{self.task_class}:{self.node_name}:{self.model}:{self.options_digest}:{self.epoch}",
            platform=self.platform,
            privacy_class=self.privacy_class,
            budget_class=self.budget_class,
            network_class=self.network_class
        )

def expected_probe_digest(raw_probe:Mapping[str,object])->str:
    return digest(raw_probe)

def verify_observation_binding(
    observation:LiveProbeObservation,
    *,
    raw_probe:Mapping[str,object],
    worker_receipt:Mapping[str,object],
)->tuple[bool,tuple[str,...]]:
    reasons=[]
    if expected_probe_digest(raw_probe)!=observation.probe_digest:
        reasons.append("PROBE_DIGEST_MISMATCH")
    if str(worker_receipt.get("receipt_digest",""))!=observation.worker_receipt_digest:
        reasons.append("WORKER_RECEIPT_DIGEST_MISMATCH")
    if str(worker_receipt.get("status","")).upper()!="VERIFIED":
        reasons.append("WORKER_NOT_VERIFIED")
    if not bool(worker_receipt.get("executed",False)) or not bool(worker_receipt.get("verified",False)):
        reasons.append("WORKER_FLAGS_INVALID")
    payload_digest=str(raw_probe.get("payload_digest",""))
    if payload_digest and str(worker_receipt.get("output_digest",""))!=payload_digest:
        reasons.append("OUTPUT_DIGEST_MISMATCH")
    return (not reasons,tuple(reasons))

def observation_to_trial(
    observation:LiveProbeObservation,
    *,
    quality_if_exact:float=1.0,
    quality_if_inexact:float=0.0,
    evidence_strength:float=1.0,
    reliability:float=1.0,
)->Trial:
    quality=quality_if_exact if observation.exact else quality_if_inexact
    solver_id=f"{observation.model}@{observation.node_name}#{observation.options_digest}"
    return Trial(
        trial_id=observation.observation_id,
        regime=observation.regime(),
        solver_id=solver_id,
        quality=quality,
        latency_ms=observation.wall_ms,
        monetary_cost=0.0,
        reliability=reliability,
        evidence_strength=evidence_strength,
        future_work_destroyed=0.0,
        valid=True,
    )

def verified_probe_to_trial(
    observation:LiveProbeObservation,
    *,
    raw_probe:Mapping[str,object],
    worker_receipt:Mapping[str,object],
)->Trial:
    ok,reasons=verify_observation_binding(observation,raw_probe=raw_probe,worker_receipt=worker_receipt)
    if not ok:
        raise ValueError("UNVERIFIED_LIVE_PROBE:"+",".join(reasons))
    return observation_to_trial(observation)

def batch_verified_trials(
    observations:Sequence[LiveProbeObservation],
    probes:Mapping[str,Mapping[str,object]],
    receipts:Mapping[str,Mapping[str,object]],
)->tuple[Trial,...]:
    out=[]
    for obs in observations:
        if obs.observation_id not in probes:
            raise ValueError(f"MISSING_PROBE:{obs.observation_id}")
        if obs.worker_receipt_digest not in receipts:
            raise ValueError(f"MISSING_RECEIPT:{obs.worker_receipt_digest}")
        out.append(verified_probe_to_trial(
            obs,
            raw_probe=probes[obs.observation_id],
            worker_receipt=receipts[obs.worker_receipt_digest]
        ))
    return tuple(out)
