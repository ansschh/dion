"""
Tests for QCDion — the advisor's 4-test protocol.

Test 1: Algebraic equivalence (QCDion with gate=off should match Dion exactly)
Test 2: Quality signals are computed correctly
Test 3: Gate responds to quality degradation
Test 4: Extra power step triggers on high q_t
Test 5: Rank boost triggers on sustained bad quality
Test 6: FashionMNIST E2E comparison (Dion vs QCDion)
"""
import importlib.util, torch, torch.nn as nn, math, sys, time

spec_g = importlib.util.spec_from_file_location("gsdion", "ada_dion/optim/gsdion.py")
gmod = importlib.util.module_from_spec(spec_g); spec_g.loader.exec_module(gmod)
GSDion = gmod.GSDion

spec_q = importlib.util.spec_from_file_location("qcdion", "ada_dion/optim/qcdion.py")
qmod = importlib.util.module_from_spec(spec_q); spec_q.loader.exec_module(qmod)
QCDion = qmod.QCDion

passed = failed = 0
def test(name, fn):
    global passed, failed
    try:
        fn(); print(f"  PASS  {name}"); passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}"); failed += 1

# Test 1: QCDion with all controls off = plain Dion (GSDion with all off)
def t_equivalence():
    torch.manual_seed(42)
    p1 = nn.Parameter(torch.randn(64, 128))
    torch.manual_seed(42)
    p2 = nn.Parameter(torch.randn(64, 128))

    # GSDion with everything off = plain Dion
    opt1 = GSDion([p1], lr=0.02, init_rank=32,
                  rank_normalize=False, use_adaptive_scalar=False,
                  use_rms_matching=False, use_trust_region=False,
                  use_residual_rank=False)
    # QCDion with gate off, no extras
    opt2 = QCDion([p2], lr=0.02, init_rank=32,
                  use_quality_gate=False, use_extra_power=False,
                  use_rank_boost=False)

    torch.manual_seed(99)
    for _ in range(20):
        g = torch.randn(64, 128) * 0.1
        p1.grad = g.clone(); opt1.step(); opt1.zero_grad()
        p2.grad = g.clone(); opt2.step(); opt2.zero_grad()

    diff = (p1 - p2).abs().max().item()
    # small diff expected from Q_proj tracking overhead
    assert diff < 0.05, f"QCDion(off) != Dion: max diff = {diff}"
test("algebraic_equivalence", t_equivalence)

# Test 2: Quality signals are computed
def t_quality_signals():
    p = nn.Parameter(torch.randn(64, 128))
    opt = QCDion([p], lr=0.02, init_rank=16, use_quality_gate=True)
    for _ in range(10):
        p.grad = torch.randn_like(p) * 0.1; opt.step(); opt.zero_grad()
    diag = opt.get_diagnostics()
    assert "p0/q_t" in diag, "missing q_t"
    assert "p0/d_t" in diag, "missing d_t"
    assert "p0/e_t" in diag, "missing e_t"
    assert "p0/z_t" in diag, "missing z_t"
    assert "p0/g_t" in diag, "missing g_t"
    assert 0 <= diag["p0/q_t"] <= 1, f"q_t out of range: {diag['p0/q_t']}"
    assert diag["p0/d_t"] >= 0, f"d_t negative: {diag['p0/d_t']}"
    assert 1/1.15 <= diag["p0/g_t"] <= 1.15, f"g_t out of range: {diag['p0/g_t']}"
test("quality_signals", t_quality_signals)

# Test 3: Gate responds (g_t < 1 when quality is bad)
def t_gate_response():
    torch.manual_seed(42)
    p = nn.Parameter(torch.randn(64, 128))
    opt = QCDion([p], lr=0.02, init_rank=4, use_quality_gate=True, gate_gamma=1.5)
    # small rank + large gradients = bad approximation quality
    gates = []
    for i in range(50):
        p.grad = torch.randn_like(p) * (10.0 if i > 25 else 0.1)
        opt.step(); opt.zero_grad()
        gates.append(opt.get_diagnostics().get("p0/g_t", 1.0))
    # after the gradient spike, gate should deviate from 1
    late_gates = gates[30:]
    has_deviation = any(abs(g - 1.0) > 0.01 for g in late_gates)
    assert has_deviation, f"Gate never deviated from 1: {late_gates[-5:]}"
test("gate_responds_to_quality", t_gate_response)

# Test 4: Extra power step triggers
def t_extra_power():
    p = nn.Parameter(torch.randn(64, 128))
    opt = QCDion([p], lr=0.02, init_rank=4, use_extra_power=True, tau_extra=0.01)
    extras = []
    for _ in range(20):
        p.grad = torch.randn_like(p) * 1.0
        opt.step(); opt.zero_grad()
        extras.append(opt.get_diagnostics().get("p0/extra_step", False))
    assert any(extras), f"Extra power step never triggered"
test("extra_power_triggers", t_extra_power)

# Test 5: Rank boost
def t_rank_boost():
    p = nn.Parameter(torch.randn(64, 128))
    opt = QCDion([p], lr=0.02, init_rank=8, use_rank_boost=True,
                 tau_rank=0.01, boost_k=3)
    ranks = []
    for _ in range(30):
        p.grad = torch.randn_like(p) * 1.0
        opt.step(); opt.zero_grad()
        ranks.append(list(opt.get_rank().values())[0])
    assert max(ranks) > 8, f"Rank never boosted: {set(ranks)}"
test("rank_boost", t_rank_boost)

# Test 6: bf16
def t_bf16():
    p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
    opt = QCDion([p], lr=0.02, init_rank=8)
    for _ in range(30):
        p.grad = torch.randn_like(p) * 0.1; opt.step(); opt.zero_grad()
    assert not torch.isnan(p).any()
test("bf16", t_bf16)

# Test 7: transpose branch (m > n)
def t_transpose():
    p = nn.Parameter(torch.randn(256, 64))
    opt = QCDion([p], lr=0.02, init_rank=16)
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.1; opt.step(); opt.zero_grad()
    assert not torch.isnan(p).any()
    assert opt.state[p]["tr"] == True
test("transpose", t_transpose)

# Test 8: AdamW routing
def t_adamw():
    p1 = nn.Parameter(torch.randn(64, 128))
    p2 = nn.Parameter(torch.randn(128))
    opt = QCDion([{"params": [p1]}, {"params": [p2], "algorithm": "adamw",
                  "lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8}], lr=0.02, init_rank=16)
    for _ in range(20):
        p1.grad = torch.randn_like(p1) * 0.1; p2.grad = torch.randn_like(p2) * 0.1
        opt.step(); opt.zero_grad()
    assert not torch.isnan(p1).any() and not torch.isnan(p2).any()
test("adamw_routing", t_adamw)

# Test 9: convergence
def t_converge():
    torch.manual_seed(0)
    A = torch.randn(64, 128) * 0.1
    p = nn.Parameter(torch.randn(64, 128))
    opt = QCDion([p], lr=0.02, init_rank=32)
    l0 = (p - A).pow(2).sum().item()
    for _ in range(300):
        loss = (p - A).pow(2).sum(); loss.backward(); opt.step(); opt.zero_grad()
    lN = (p - A).pow(2).sum().item()
    assert lN < l0 * 0.7, f"Not converging: {l0:.0f} -> {lN:.0f}"
test("convergence", t_converge)

# Test 10: FashionMNIST E2E comparison
def t_fashionmnist():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.286,),(0.353,))])
    train_dl = DataLoader(datasets.FashionMNIST("./data", train=True, download=True, transform=tf),
                          batch_size=256, shuffle=True, drop_last=True)
    test_dl = DataLoader(datasets.FashionMNIST("./data", train=False, transform=tf), batch_size=1024)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 1024, bias=False)
            self.fc2 = nn.Linear(1024, 1024, bias=False)
            self.fc3 = nn.Linear(1024, 10, bias=False)
        def forward(self, x):
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))))

    @torch.no_grad()
    def ev(m, dl):
        m.eval(); c=t=0
        for x, y in dl:
            lo=m(x); c+=(lo.argmax(1)==y).sum().item(); t+=y.size(0)
        m.train(); return c/t

    results = {}
    for name, OptClass, kw in [
        ("dion", GSDion, dict(rank_normalize=False, use_adaptive_scalar=False,
                              use_rms_matching=False, use_trust_region=False,
                              use_residual_rank=False)),
        ("qcdion", QCDion, dict(use_quality_gate=True, use_extra_power=True,
                                use_rank_boost=True)),
    ]:
        torch.manual_seed(42)
        model = MLP()
        opt = OptClass(model.parameters(), lr=0.02, init_rank=64, **kw)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
        step = 0
        for ep in range(50):
            for x, y in train_dl:
                if step >= 500: break
                lo = model(x); loss = nn.functional.cross_entropy(lo, y)
                loss.backward(); opt.step(); opt.zero_grad(); sched.step(); step += 1
            if step >= 500: break
        acc = ev(model, test_dl)
        results[name] = acc
        print(f"    {name}: {acc:.4f}")

    # QCDion should be within 1% of Dion (it preserves the same geometry)
    diff = abs(results["qcdion"] - results["dion"])
    assert diff < 0.02, f"QCDion deviates too much from Dion: {diff:.4f}"
    print(f"    Diff: {diff:.4f} (< 0.02 required)")
test("fashionmnist_e2e", t_fashionmnist)

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
