import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
class LearnablePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=16):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        position_encodings = self.position_embeddings(positions) # (seq_len, d_model)
        return x + position_encodings #broadcasting

class Img2Seq(nn.Module):
    def __init__(self, c, h, w, d_model=768):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.d_model = d_model

        self.linear = nn.Linear(c * h * w, d_model)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c * h * w)
        x = self.linear(x)
        x = x.view(b, t, self.d_model)
        return x


class seq2Img(nn.Module):
    def __init__(self, c, h, w, d_model=768):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.d_model = d_model

        self.linear = nn.Linear(d_model, c * h * w)

    def forward(self, x):
        b, t, d = x.shape
        x = x.view(b * t, d)
        x = self.linear(x)
        x = x.view(b, t, self.c, self.h, self.w)
        return x




class Multihead_Attention_origin(nn.Module):
    def __init__(self,emb_size:int = 768,num_heads:int = 8,dropout:float=0):
        super().__init__()
        self.emb_size=emb_size
        self.num_heads=num_heads
        self.queries=nn.Linear(emb_size,emb_size)
        self.keys=nn.Linear(emb_size,emb_size)
        self.values=nn.Linear(emb_size,emb_size)

       # self.qkv=nn.Linear(emb_size,emb_size*3)
        self.att_drop=nn.Dropout(dropout)
        self.projection=nn.Linear(emb_size,emb_size)
    def forward(self,x):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd,bhkd->bhqk',queries,keys)  #[b,heads,q_len,k_len]
        scaling=(self.emb_size//self.num_heads)**(1/2)
        att=F.softmax(energy,dim=-1)/scaling
        att=self.att_drop(att)
        #sum up over third axis
        out=torch.einsum('bhal,bhlv->bhav',att,values)
        out=rearrange(out,'b h n d->b n (h d)')
        out=self.projection(out)
        return out

class Multihead_Attention(nn.Module):
    def __init__(self,input_dim:int,num_heads:int,head_dim:int,output_dim:int,dropout:float=0.):
        super().__init__()
        assert input_dim%num_heads==0,f"input_dim {input_dim} should be divisible by num_heads {num_heads}"

        self.input_dim=input_dim
        self.num_heads=num_heads
        self.head_dim=head_dim
        self.inner_dim=head_dim*num_heads
        self.scale=self.head_dim**-0.5
        self.qkv=nn.Linear(self.input_dim,self.inner_dim*3,bias=False) #
        self.output_dim=output_dim

        self.att_drop=nn.Dropout(dropout)
        self.projection=nn.Linear(self.inner_dim,self.output_dim)
    def forward(self,x,mask=None):
        qkv=rearrange(self.qkv(x),'b n (qkv h d)->(qkv) b h n d',h=self.num_heads,qkv=3)
        q,k,v=qkv[0],qkv[1],qkv[2]
        energy=torch.einsum('bhid,bhjd->bhij',q,k)
        if mask is not None:
            fill_value=torch.finfo(torch.float32).min
            energy=energy.masked_fill(~mask,fill_value)
        att=F.softmax(energy,dim=-1)/self.scale
        att=self.att_drop(att)
        out=torch.einsum('bhij,bhjd->bhid',att,v)
        out=rearrange(out,'b h n d->b n (h d)')
        out=self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn

    def forward(self,x,**kwargs):
        res=x
        x=self.fn(x,**kwargs)
        x+=res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, expansion * input_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * input_dim, input_dim),
        )
    def forward(self, x):
        return self.net(x)

class block(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 1,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                Multihead_Attention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self,c,h,w,d_model=768):
        super().__init__()
        self.embedding = Img2Seq(c,h,w,d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=16)
        self.transformer = block(num_heads=8, head_dim=96,output_dim=768, forward_expansion=4, forward_drop_p=0.)
        self.recover = seq2Img(c,h,w,d_model)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.recover(x)
        return x





