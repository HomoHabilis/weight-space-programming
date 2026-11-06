```python
# Inspect cgroup memory limit/usage
for p in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes']:
    if os.path.exists(p): print(p, open(p).read().strip())
st=open('/sys/fs/cgroup/memory.stat').read() if os.path.exists('/sys/fs/cgroup/memory.stat') else ''
print('\n'.join([l for l in st.split('\n') if l.split(' ')[0] in ('anon','file','file_mapped','shmem','active_file','inactive_file')]))
```