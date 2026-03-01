import torch , triton
import triton.language as tl

@triton.jit
def layernormkernel(xptr , outptr , eps , gamma , beta , M , K , xstrm , xstrn , outstrm , outstrn ,  gammastry , betastry , MBLOCK :tl.constexpr , KBLOCK :tl.constexpr):
    pid = tl.program_id(axis = 0)

    row = pid * MBLOCK + tl.arange(0 , MBLOCK)
    offsets = tl.arange(0 , KBLOCK)

    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn

    mean = tl.zeros((MBLOCK,) , dtype = tl.float32)
    meansq = tl.zeros((MBLOCK,) , dtype = tl.float32)

    count = tl.zeros((MBLOCK,), dtype=tl.float32) 

    for k in range(0 , K , KBLOCK):
        maskx = (row[: , None] < M) & ((k + offsets)[None , :] < K)
        x = tl.load(ptrx, mask=maskx, other=0.0)

        valid = maskx.to(tl.float32)
        count += tl.sum(valid , axis = 1)

        diff = (x - mean[:, None]) * valid
        delta = tl.sum(diff , axis = 1) 

        mean += delta / count
        meansq = meansq + tl.sum(diff * (x - mean[: , None]) * valid , axis = 1)

        ptrx += KBLOCK * xstrn

    var = meansq / K
    st = tl.rsqrt(var + eps)
    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    gammaptr = gamma + offsets[None, :] * gammastry
    betaptr  = beta  + offsets[None, :] * betastry
    

    out = outptr + row[: , None] * outstrm + offsets[None , :] * outstrn

    for k in range(0 , K , KBLOCK):
        maskx = (row[: , None] < M) & ((k + offsets)[None , :] < K)
        x = tl.load(ptrx , mask = maskx , other = 0.0)

        maskg = (k + offsets)[None, :] < K       
        gama  = tl.load(gammaptr, mask=maskg, other=0.0)
        bet   = tl.load(betaptr,  mask=maskg, other=0.0)

        norm = ((x - mean[: , None]) * st[: , None]) * gama + bet
        tl.store(out , norm , mask = maskx)

        out += KBLOCK * outstrn
        ptrx += KBLOCK * xstrn
        gammaptr += KBLOCK * gammastry
        betaptr  += KBLOCK * betastry