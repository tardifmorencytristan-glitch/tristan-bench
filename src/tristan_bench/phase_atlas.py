from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import json
from collections import defaultdict
from typing import Iterable, Sequence

PROTOCOL="TRISTAN-PHASE-ATLAS-REPLAY-R1"

def _canon(v:object)->bytes:
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":"),default=str,allow_nan=False).encode()
def digest(v:object)->str: return sha256(_canon(v)).hexdigest()

@dataclass(frozen=True)
class RegimeKey:
    task_class:str
    platform:str
    privacy_class:str
    budget_class:str
    network_class:str
    def key(self)->str: return "|".join((self.task_class,self.platform,self.privacy_class,self.budget_class,self.network_class))

@dataclass(frozen=True)
class Trial:
    trial_id:str
    regime:RegimeKey
    solver_id:str
    quality:float
    latency_ms:float
    monetary_cost:float
    reliability:float
    evidence_strength:float
    future_work_destroyed:float=0.0
    valid:bool=True
    def utility(self)->float:
        if not self.valid: return float("-inf")
        benefit=max(0.0,self.quality)*max(0.0,self.reliability)*(0.1+max(0.0,self.evidence_strength))+max(0.0,self.future_work_destroyed)*0.2
        burden=1.0+max(0.0,self.latency_ms)/1000+max(0.0,self.monetary_cost)
        return benefit/burden

@dataclass(frozen=True)
class PhaseCell:
    regime_key:str
    winner_solver:str
    winner_utility:float
    runner_up_solver:str|None
    margin:float
    trials:int
    evidence_strength:float

@dataclass(frozen=True)
class PhaseAtlas:
    protocol:str
    cells:tuple[PhaseCell,...]
    atlas_digest:str=""
    def with_digest(self):
        return replace(self,atlas_digest=digest(asdict(replace(self,atlas_digest=""))))

def build_phase_atlas(trials:Iterable[Trial])->PhaseAtlas:
    groups=defaultdict(list)
    for t in trials: groups[t.regime.key()].append(t)
    cells=[]
    for key,items in groups.items():
        valid=[t for t in items if t.valid]
        ranked=sorted(valid,key=lambda t:(t.utility(),t.evidence_strength,t.solver_id),reverse=True)
        if not ranked: continue
        win=ranked[0]; runner=ranked[1] if len(ranked)>1 else None
        margin=win.utility()-(runner.utility() if runner else 0.0)
        cells.append(PhaseCell(key,win.solver_id,win.utility(),runner.solver_id if runner else None,margin,len(items),win.evidence_strength))
    return PhaseAtlas(PROTOCOL,tuple(sorted(cells,key=lambda c:c.regime_key))).with_digest()

def solver_for(atlas:PhaseAtlas,regime:RegimeKey)->str|None:
    for c in atlas.cells:
        if c.regime_key==regime.key(): return c.winner_solver
    return None

@dataclass(frozen=True)
class HistoricalDecision:
    decision_id:str
    regime:RegimeKey
    selected_solver:str
    observed_utility:float
    work_units:float
    proof_units:float

@dataclass(frozen=True)
class ReplayCandidate:
    solver_id:str
    estimated_utility:float
    estimated_work_units:float
    estimated_proof_units:float
    evidence_strength:float

@dataclass(frozen=True)
class ReplayResult:
    decision_id:str
    historical_solver:str
    counterfactual_solver:str
    utility_delta:float
    work_destroyed:float
    proof_work_destroyed:float
    evidence_strength:float
    status:str
    replay_digest:str=""
    def with_digest(self):
        return replace(self,replay_digest=digest(asdict(replace(self,replay_digest=""))))

def replay(history:HistoricalDecision,candidates:Sequence[ReplayCandidate])->ReplayResult:
    if not candidates:
        c=ReplayCandidate(history.selected_solver,history.observed_utility,history.work_units,history.proof_units,1.0)
    else:
        c=max(candidates,key=lambda x:(x.estimated_utility,x.evidence_strength,x.solver_id))
    du=c.estimated_utility-history.observed_utility
    wd=max(0.0,history.work_units-c.estimated_work_units)
    pd=max(0.0,history.proof_units-c.estimated_proof_units)
    status="COUNTERFACTUAL_BETTER" if du>0 else ("WORK_REDUCTION_ONLY" if wd+pd>0 else "NO_IMPROVEMENT")
    return ReplayResult(history.decision_id,history.selected_solver,c.solver_id,du,wd,pd,c.evidence_strength,status).with_digest()

def global_winner(atlas:PhaseAtlas)->None:
    return None

def phase_coverage(atlas:PhaseAtlas,expected_regimes:Sequence[RegimeKey])->float:
    if not expected_regimes: return 1.0
    observed={c.regime_key for c in atlas.cells}
    return sum(1 for r in expected_regimes if r.key() in observed)/len(expected_regimes)
