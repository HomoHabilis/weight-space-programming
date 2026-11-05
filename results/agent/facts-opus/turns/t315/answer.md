```python
# Hypothesis: build logit directions d_t=(1+g)*(E[t]-meanE) for all answer tokens (load embedding once in chunks)
print('lnw', getattr(lnw,'shape',None))
from safetensors import safe_open
import glob
fn_=[f for f in glob.glob('/model/*.safetensors')]
for f in fn_:
    with safe_open(f,'pt') as sf:
        if 'model.language_model.embed_tokens.weight' in sf.keys():
            sl=sf.get_slice('model.language_model.embed_tokens.weight'); shp=sl.get_shape(); 
            sumE=torch.zeros(shp[1],dtype=torch.float64)
            for a_ in range(0,shp[0],20000): sumE+=sl[a_:a_+20000].double().sum(0)
            meanE=(sumE/shp[0]).float()
            needed=sorted(set(int(t_) for t_ in NX[ansV]))
            Erows={t_:sl[t_:t_+1][0].float() for t_ in needed}
            fnorm=sf.get_tensor('model.language_model.norm.weight').float() if 'model.language_model.norm.weight' in sf.keys() else None
print(shp, len(Erows), fnorm is not None)
```