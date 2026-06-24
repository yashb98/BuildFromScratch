"""interp.py core tests — CPU, no model/GPU/network. The load-bearing checks are the
anti-laundering gate (null at the control floor) and the SAE/floor/probe plumbing."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import interp as I

def test_self_test():
    I._self_test()

def test_random_data_yields_null():           # arXiv:2602.14111 reproduced
    torch.manual_seed(0); a=torch.randn(2000,32)
    sae,_=I.train_sae(a,dict_size=256,k=8,steps=80,batch=256,seed=0)
    with torch.no_grad(): recon,_=sae(a.float())
    serr=((a.float()-recon)**2).sum(-1).tolist()
    v=I.ci_disjoint_verdict(serr, I.pca_floor(a,256).tolist(),
                            I.random_sae_floor(a,256,8,seed=1).tolist(), lower_is_better=True)
    assert v["verdict"]=="null"

def test_gate_direction_and_disjointness():
    good=[0.1+0.01*math.sin(i) for i in range(300)]; floor=[1.0+0.01*math.cos(i) for i in range(300)]
    assert I.ci_disjoint_verdict(good,floor,floor,lower_is_better=True)["verdict"]=="win"
    assert I.ci_disjoint_verdict(floor,good,good,lower_is_better=False)["verdict"]=="win"   # higher-better flip
    assert I.ci_disjoint_verdict(floor,floor,floor,lower_is_better=True)["verdict"]=="null" # identical → null

def test_sae_trains_and_l0(tmp_path):
    torch.manual_seed(0); a=torch.randn(1000,16)
    sae,hist=I.train_sae(a,dict_size=128,k=4,steps=60,batch=128,seed=0)
    assert hist[-1][1] < hist[0][1]            # loss went down
    assert 0 < sae.l0(a.float()) <= 16         # sparsity in range

def test_stubs_refuse_until_formula_read():
    for fn in (I.absorption_first_letter, I.synth_recovery_f1_mcc):
        try: fn(); assert False
        except NotImplementedError: pass
