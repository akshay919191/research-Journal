import torch , triton
import triton.language as tl

# we will load q at once but we will add that so we can actual multiply it by K.t at once(k.t will be partial so we have to sum up)
@triton.jit
def attnkernel(
    Qptr , Kptr , Vptr , Outptr,
    d_k ,
    qstrs , qstrd,
    kstrs , kstrd,
    vstrs , vstrd,
    outstrx , outstry,
    seq_len , embed,
    blockSEQ: tl.constexpr , blockEMBED: tl.constexpr , blockK: tl.constexpr
):
    # we have to look up for rows from q and rows from k as columns
    qpid = tl.program_id(axis = 0)
    kpid = tl.program_id(axis = 1)

    row = qpid * blockSEQ +  tl.arange(0 , blockSEQ)
    col = kpid * blockEMBED + tl.arange(0 , blockEMBED)
    # we have made row selection for row and col , like it will cover blockSEQ number of rows from q , and blockEMBED number of rows as cols from k
    # but we need to pick columns too from q and cols from k which will act like rows ( doing transpose without purposing )

    offsets = tl.arange(0 , blockK)
    voffsets = tl.arange(0 , blockEMBED)

    # now we will load q in chunks but we will add the fronting chunks to load a full row, but dims of that should match query's dim 
    # q = tl.zeros((blockSEQ , blockEMBED) , dtype = tl.float32)
    # for k in range(0 , K , blockSEQ):
    #     qptr = Qptr + (row[: , None] * qstrs) + ((k + offsets[None , :]) * qstrd)
    #     Qmask = (row[: , None] < seq_len) & ((k + col[None , :]) < K)

    #     q += tl.load(qptr , mask = Qmask)
    
    # now we gonna evaluate the Q @ K.T for this we need to make a zeros matrix , size will be seq , seq 
    # along with this we need to find the max of the scores we made for softmax for normalizing , so max will be a single number from each rows , total rows are seq as we  are finding in scores which are in dim of seq , seq
    # score = tl.zeros([blockSEQ , blockSEQ] , dtype = tl.float32)
    # rowmax = tl.zeros((blockSEQ,) , dtype = tl.float32)

    # NOW WE ALREADY LOADED THE WHOLE FUCKING QUERY SO TO NOT RAN OUT OF REGISTER WE NEED TO LOAD KEY IN CHUNKS AND THEN MULTIPLY IT , BUT IT WILL NOT MATCH THE SHAPE 
    # so we need to load q , v in chunks then dot it ,so commment out the q 
    # for k in range(0 , K , blockSEQ):
    #     qptr = Qptr + (row[: , None] * qstrs) + ((k + offsets[None , :]) * qstrd) # shape row , k + off
    #     Qmask = (row[: , None] < seq_len) & ((k + offsets[None , :]) < K)

    #     q += tl.load(qptr , mask = Qmask , other = 0.0)

    #     kptr = Kptr + ((k + offsets[: , None]) * kstrd) + (col[None , :] * kstrs) # k + off , col
    #     Kmask = ((k + offsets[: , None]) < K) & (col[None , :] < embed)

    #     k = tl.load(kptr , mask = Kmask , other = 0.0)

    #     score += tl.dot(q , k) # shape row , col
    #     chunk_max = tl.max(score , axis = 1)
    #     rowmax = tl.maximum(chunk_max , rowmax)
    
    # so now we have the Q @ K.T , now we want to scale it with 1 / root(d_k) , so now we have , (Q @ K.T) / root(D_k)
    # score = score * tl.rsqrt(d_k)

    # now calculate the softmax , with logexpsum trick as we on offline softmax
    # its formula is e ^ (x - max) / submission (e ^ (x - max))
    # score -= rowmax[: , None]
    # softmaxed = tl.exp(score) / tl.sum(score , axis = 1)

    # weight = tl.zeros((blockSEQ , blockEMBED) , dtype = tl.float32)
    # for k in range(0 , K , blockK):
    #     vptr = Vptr + (row[: , None] * vstrs) + (offsets[None , :] * vstrd)
    #     maskv = (row[: , None] < seq_len) & ((k + offsets[None , :]) < K)

    #     v = tl.load(vptr , mask = maskv , other = 0.0)
        # here we hit the wall , dims fails to match so we will now go for chunks score for all and use online softmax , will comment out all



    weights = tl.zeros((blockSEQ , blockEMBED) , dtype = tl.float32)

    rowmax = tl.full([blockSEQ], -float("inf"), tl.float32)
    row_sum = tl.zeros([blockSEQ], tl.float32)

    qptr = Qptr + (row[: , None] * qstrs) + (voffsets[None , :] * qstrd) # shape row , k + off
    Qmask = (row[: , None] < seq_len) & (voffsets[None , :] < seq_len)

    q = tl.load(qptr , mask = Qmask , other = 0.0)

    for k in range(0 , seq_len , blockSEQ):
        
        current_k_idx = k + tl.arange(0, blockK)
        kptr = Kptr + (offsets[:, None] * kstrd) + (current_k_idx[None, :] * kstrs) # k + off , col
        Kmask = (voffsets[:, None] < embed) & (current_k_idx[None, :] < seq_len)

        k_ = tl.load(kptr , mask = Kmask , other = 0.0)

        sc = tl.dot(q , k_) # shape row , col
 
        #score
        sc = sc * tl.rsqrt(d_k)

        chunk_max = tl.max(sc , axis = 1)

        new_row_max = tl.maximum(rowmax, chunk_max)
        row_sum = row_sum * tl.exp(rowmax - new_row_max) + + tl.sum(tl.exp(sc - new_row_max[:, None]), axis=1)
        rowmax = new_row_max

    for k in range(0 , seq_len , blockSEQ):
        current_k_idx = k + tl.arange(0, blockSEQ)

        kptr = Kptr + (voffsets[:, None] * kstrd) + (current_k_idx[None, :] * kstrs)  # k + off , col
        Kmask = (voffsets[: , None] < embed) & (current_k_idx[None , :] < seq_len)

        k_ = tl.load(kptr , mask = Kmask , other = 0.0)

        sc = tl.dot(q , k_) # shape row , col
        sc = sc * tl.rsqrt(d_k)

        vptr = Vptr + (current_k_idx[:, None] * vstrs) + (voffsets[None, :] * vstrd) # col , k + offsets
        maskv = (current_k_idx[: , None] < seq_len) & (voffsets[None , :] < seq_len)

        v = tl.load(vptr , mask = maskv , other = 0.0)

        softmaxed = (tl.exp(sc - rowmax[: , None]) / row_sum[: , None]) 
        weights += tl.dot(softmaxed , v)

    out_ptr = Outptr + (row[:, None] * outstrx) + (voffsets[None, :] * outstry)
    tl.store(out_ptr, weights, mask=(row[:, None] < seq_len) & (voffsets[None, :] < embed))