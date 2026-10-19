"""R16: full Pareto membership verification and asymmetric negative controls.

Standalone, synthetic, offline, deterministic, NumPy needed. No file writes or
network. All loss values are exact Python-scaled integers over a 17x17 grid.
"""
from __future__ import annotations
import hashlib
import json
from itertools import product
import numpy as np

AXIS=tuple(range(-8,9))
GRID=np.array(list(product(AXIS,repeat=2)), dtype=np.int64)
SCALED_TARGETS={
 "R15_symmetric":np.array([
  [[-4,3],[4,-3]],[[-5,-4],[5,4]],[[-2,5],[5,-2]]
 ],dtype=np.int64),
 "R16_asymmetric":np.array([
  [[-6,0],[3,1]],[[2,6],[6,5]],[[4,-4],[7,-2]]
 ],dtype=np.int64),
}
POLICIES=("base_two","base_three","budget_1","x_nonnegative",
 "x_nonpositive","robust_025","robust_050",
 "robust_025_three","robust_050_budget_1","budget_2")
R15_COUNTS=[[21,21,17,14,14,21,21,21,17,21],
 [27,27,17,18,18,27,27,27,17,25],
 [22,26,16,18,12,22,22,26,16,22]]
R15_SHARED=[100,100,49,56,54,100,100,100,49,94]

def scores(targets):
 diff=GRID[None,:,None,:]-targets[:,None,:,:]
 squares=np.sum(diff*diff,axis=-1)
 energy=np.sum(GRID*GRID,axis=1)
 batch=np.zeros((3,10,289,3),dtype=np.int64)
 for j in range(10):
  eps=1 if j in (5,7) else 2 if j in (6,8) else 0
  robust=squares+eps*eps+2*eps*np.max(np.abs(diff),axis=-1)
  batch[:,j,:,:2]=robust
  if j in (1,7):
   batch[:,j,:,2]=energy[None,:]
 return batch

def allowed(policy):
 energy=np.sum(GRID*GRID,axis=1)
 if policy in (2,8):return energy<=16
 if policy==9:return energy<=32
 if policy==3:return GRID[:,0]>=0
 if policy==4:return GRID[:,0]<=0
 return np.ones(289,dtype=bool)

def fast_mask(values,valid):
 # Domination exact, no float tolerance or rounding.
 not_worse=(values[:,None,:]<=values[None,:,:]).all(axis=-1)
 strictly=(values[:,None,:]<values[None,:,:]).any(axis=-1)
 dominates=not_worse & strictly & valid[:,None] & valid[None,:]
 return valid & ~dominates.any(axis=0)

def independent_mask(values,valid):
 # Separate O(N^2) algorithm using Python tuples and explicit comparisons.
 v=[tuple(int(z) for z in row) for row in values]
 active=[i for i in range(len(v)) if bool(valid[i])]
 result=np.zeros(len(v),dtype=bool)
 for candidate in active:
  z=v[candidate]
  defeated=False
  for adversary in active:
   if adversary==candidate:continue
   y=v[adversary]
   if all(k<=j for k,j in zip(y,z)) and any(k<j for k,j in zip(y,z)):
    defeated=True
    break
  result[candidate]=not defeated
 return result

def run_one(label,targets):
 s=scores(targets)
 digest=hashlib.sha256()
 counts=[]
 all_masks=[]
 for problem in range(3):
  cnt=[]
  for policy in range(10):
   f=allowed(policy)
   fast=fast_mask(s[problem,policy],f)
   independent=independent_mask(s[problem,policy],f)
   assert np.array_equal(fast,independent),(label,problem,policy)
   assert fast.any() and np.all(~fast|f)
   digest.update(fast.astype(np.uint8).tobytes())
   all_masks.append(fast)
   cnt.append(int(fast.sum()))
  counts.append(cnt)
 shared=[]
 minima=[]
 for policy in range(10):
  f=allowed(policy)
  full=s[:,policy,:,:2].transpose(1,0,2).reshape(289,6)
  direct=fast_mask(full,f)
  oracle=independent_mask(full,f)
  assert np.array_equal(direct,oracle),(label,"shared",policy)
  digest.update(direct.astype(np.uint8).tobytes())
  shared.append(int(direct.sum()))
  feasible=np.flatnonzero(f)
  # Score each shared candidate by worst objective then aggregate sum.
  best=min(feasible,key=lambda i:(int(np.max(full[i])),int(np.sum(full[i])),int(i)))
  assert direct[best]
  minima.append({"policy":POLICIES[policy],"grid_xy":GRID[best].tolist(),
     "physical_xy":[float(v)/4 for v in GRID[best]],"worst_scaled":int(np.max(full[best]))})
 # Verify the intentionally wrong all-candidates-are-Pareto null hypothesis fails.
 assert any(int(c)<int(np.sum(allowed(j))) for row in counts for j,c in enumerate(row))
 return {"label":label,"counts":counts,"shared":shared,"membership_sha256":digest.hexdigest(),
         "minimax_parameters":minima,
         "distinct_policy_sizes":len(set(map(tuple,counts)))>1 or len(set(shared))>1}

def main():
 outcomes={name:run_one(name,tgt) for name,tgt in SCALED_TARGETS.items()}
 assert outcomes["R15_symmetric"]["counts"]==R15_COUNTS
 assert outcomes["R15_symmetric"]["shared"]==R15_SHARED
 assert outcomes["R15_symmetric"]["membership_sha256"]!=outcomes["R16_asymmetric"]["membership_sha256"]
 chosen={tuple(e["grid_xy"]) for e in outcomes["R16_asymmetric"]["minimax_parameters"]}
 assert any(p!=(0,0) for p in chosen),chosen
 out={"schema":"omega.tristan.r16.full_mask_multifront","status":"PASSED",
      "problem_count":3,"policy_count":10,"candidates":289,
      "evaluated_masks":80,"independent_membership_checks":80,
      "method":"NumPy broadcasting vs independent Python integer tuple dominance on all 80 masks",
      "loss_values":"16x physical loss as nonnegative integers",
      "outcomes":outcomes,
      "limitations":"Two entirely synthetic finite grids. Not continuous global optimality, physics, or a deployed autonomous agent."}
 print(json.dumps(out,sort_keys=True,separators=(",",":")))
if __name__=="__main__":
 main()
