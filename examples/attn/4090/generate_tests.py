import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 16 # sequence length?
D = 16
DV = 16

TESTNAME = sys.argv[1]

torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)

if TESTNAME in ['ones']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    #f = (torch.log(0.99*torch.ones((B, H, N), dtype=torch.bfloat16, device='cuda'))).to(torch.float32)
    #f = -0.001*(N-torch.arange(N, device='cuda')[None, None, :].repeat(B, H, 1).to(torch.float32))
    f = -0.001*(torch.arange(N, device='cuda')[None, None, :].repeat(B, H, 1).to(torch.float32))
    # pad f with zero on the right

    f = torch.cat([f, torch.zeros(B, H, 1, dtype=torch.float32, device='cuda')], dim=-1)
    # cut one off the left
    f = f[:, :, 1:]

    #f = 2*torch.ones_like(f) # test: two on the diagonal

    #f = -0.01 * torch.arange(1,N+1, device='cuda')[None, None, :].repeat(B, H, 1).to(torch.float32)
    f = (0.1*torch.arange(1,N+1, device='cuda')[None, None, :].repeat(B, H, 1).to(torch.float32)).log()

    print(f[:,:,:16], 'f')
    #print(f.view(B, H, -1, 16).sum(-1), 'f cumsum')
elif TESTNAME in ['twos']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')*2).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')*2).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
elif TESTNAME in ['arange']:
    q = (torch.ones(B*H*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D)).to(torch.float32)/(D*DV)
    k = (torch.arange(B*H*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D)).to(torch.float32)/(D*DV*2)
    v = (torch.ones(B*H*N*DV, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, DV)).to(torch.float32)
elif TESTNAME in ['randn']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test(Q, K, V, F, TESTNAME='all'):

    def basic(g):
        "layout g"
        B,H,N = g.shape
        l = g.new_zeros(B, H, N, N) + float('-inf')

        for t in range(N): # top to bottom
            l[:, :, t, t] = 0
            for s in range(t-1, -1, -1): # right to left: addition
                l[:, :, t, s] = g[:, :, s]
        return l

    def preforget(g):
        "compute the mask"
        B,H,N = g.shape
        l = g.new_zeros(B, H, N, N) + float('-inf')

        for t in range(N): # top to bottom
            l[:, :, t, t] = 0
            for s in range(t-1, -1, -1): # right to left: addition
                l[:, :, t, s] = l[:, :, t, s+1] + g[:, :, s] # without shifting f: use s+1
        return l

    def loop(f_reg, zero=0):
        B, H, S = f_reg.shape
        T = S
        F = f_reg.new_zeros(B, H, S, T) + float('-inf')
        F[:, :, 0,0] = zero # s0, t0
        for s in range(1, S): # top to bottom
            F[:, :, s, 0] = F[:, :, s-1, 0] + f_reg[:, :, s-1]
            for t in range(1, s+1): # left to right: subtraction
                F[:, :, s, t] = F[:, :, s, t-1] - f_reg[:, :, t-1]
        return F

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    #print(ATT, 'ATT')
    M = torch.exp(preforget(F))
    #M = torch.ones_like(M)
    #M = 2*torch.eye(N, device='cuda').expand(B, H, N, N).to(torch.bfloat16) # test: two on the diagonal
    print(M, 'M')
    out = torch.einsum("bhnm,bhmd->bhnd", ATT * M, V).to(torch.bfloat16)

    print(loop(F), 'loop')
    print(preforget(F), 'preforget')
    print(basic(F)[:, :, :16, :16].exp(), 'basic')
    print(basic(F)[:, :, :16, :16].exp().flip(-1).cumprod(-1).flip(-1), 'basic cumprod')
    assert torch.allclose(loop(F), preforget(F), atol=1e-5) # jeez, this is a lot of error

    K, V = K.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (K * V).cumsum(dim=2)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)
    return out, last_kv_state

o, last_kv_state = pytorch_test(q, k, v, f, TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as file:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    ff = f.exp().to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    kv_statef = last_kv_state.to(torch.float32).flatten().cpu().numpy()

    print(of.reshape(B,H,N,DV)[0,0,:,0])

    for i in trange(B*H*N*D):
        file.write(repr(qf[i]))
        file.write(' ')
    for i in trange(B*H*N*D):
        file.write(repr(kf[i]))
        file.write(' ')
    for i in trange(B*H*N*DV):
        file.write(repr(vf[i]))
        file.write(' ')
    for i in trange(B*H*N):
        file.write(repr(ff[i]))
        file.write(' ')
    for i in trange(B*H*N*DV):
        file.write(repr(of[i]))
        file.write(' ')
    # for i in trange(B*H*D*DV):
    #     file.write(repr(kv_statef[i]))
    #     file.write(' ')


