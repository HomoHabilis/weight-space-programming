```python
# Hypothesis: address space is near RLIMIT_AS; check usage and free large unused objects
import gc
p_ = psutil.Process(); print('vms GB', p_.memory_info().vms/1e9, 'rss GB', p_.memory_info().rss/1e9)
big = sorted([(k, (v.numel()*v.element_size() if torch.is_tensor(v) else v.nbytes)/1e6) for k,v in globals().items() if (torch.is_tensor(v) or isinstance(v,np.ndarray)) and not k.startswith('_')], key=lambda x:-x[1])[:15]
print([(k, round(m)) for k,m in big])
```