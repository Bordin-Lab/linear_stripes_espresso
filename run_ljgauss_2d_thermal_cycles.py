#!/usr/bin/env python3
"""Run independent density/seed thermal cycles with bounded concurrency."""
import argparse, subprocess, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pypresso",default="pypresso"); p.add_argument("--script",default="simulation/espresso_ljgauss_2d_thermal_cycle.py")
    p.add_argument("--out-root",default="thermal_cycles"); p.add_argument("--nproc",type=int,default=2)
    p.add_argument("--rhos",type=float,nargs="+",required=True); p.add_argument("--seeds",type=int,nargs="+",default=[1])
    p.add_argument("--N",type=int,default=4096); p.add_argument("--T-min",type=float,default=.04); p.add_argument("--T-max",type=float,default=.20); p.add_argument("--dT",type=float,default=.005)
    p.add_argument("--stripe-nx",type=int,
                   help="Particles per stripe row; forwarded to the simulation and must divide N")
    p.add_argument("--stripe-scale-ratio",type=float,default=2.0/1.2,
                   help="Target interstripe/intrastripe spacing ratio")
    p.add_argument("--temperatures",type=float,nargs="+"); p.add_argument("--steps-eq-first",type=int,default=1000000); p.add_argument("--steps-eq",type=int,default=300000); p.add_argument("--steps-prod",type=int,default=500000)
    p.add_argument("--dt",type=float,default=.002); p.add_argument("--sample-every",type=int,default=1000); p.add_argument("--thermo-every",type=int,default=1000); p.add_argument("--traj-every",type=int,default=5000)
    p.add_argument("--initial-config-template",help="e.g. initial/rho_{rho:.4f}_seed_{seed}.npz")
    p.add_argument("--force",action="store_true"); a=p.parse_args()
    root=Path(a.out_root); (root/"logs").mkdir(parents=True,exist_ok=True); pending=[]
    for rho in a.rhos:
      for seed in a.seeds:
        od=root/f"rho_{rho:.5f}"/f"seed_{seed}"; log=root/"logs"/f"rho_{rho:.5f}_seed_{seed}.log"
        if (od/"cycle_summary.dat").exists() and not a.force: continue
        c=[a.pypresso,a.script,"--N",str(a.N),"--rho",str(rho),"--seed",str(seed),"--outdir",str(od),"--T-min",str(a.T_min),"--T-max",str(a.T_max),"--dT",str(a.dT),"--steps-eq-first",str(a.steps_eq_first),"--steps-eq",str(a.steps_eq),"--steps-prod",str(a.steps_prod),"--dt",str(a.dt),"--sample-every",str(a.sample_every),"--thermo-every",str(a.thermo_every),"--traj-every",str(a.traj_every)]
        if a.stripe_nx is not None:
            c += ["--stripe-nx",str(a.stripe_nx),
                  "--stripe-scale-ratio",str(a.stripe_scale_ratio)]
        if a.temperatures: c += ["--temperatures"]+[str(x) for x in a.temperatures]
        if a.initial_config_template: c += ["--initial-config",a.initial_config_template.format(rho=rho,seed=seed)]
        pending.append((rho,seed,od,log,c))
    running=[]
    while pending or running:
      while pending and len(running)<a.nproc:
        job=pending.pop(0); job[2].mkdir(parents=True,exist_ok=True); fh=open(job[3],"w"); fh.write("COMMAND: "+" ".join(job[4])+"\n"); fh.flush()
        proc=subprocess.Popen(job[4],stdout=fh,stderr=subprocess.STDOUT); running.append((job,proc,fh)); print("START",job[0],job[1],proc.pid,flush=True)
      keep=[]
      for job,proc,fh in running:
        rc=proc.poll()
        if rc is None: keep.append((job,proc,fh))
        else: fh.close(); print("DONE",job[0],job[1],"OK" if rc==0 else f"FAILED({rc})",flush=True)
      running=keep
      if pending or running: time.sleep(2)

if __name__=="__main__": main()
