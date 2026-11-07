```python
# Identify line-start (indent) positions inside the first code block in TF1 texts; classify next line comment/code
NL = tok.encode("\n",add_special_tokens=False).ids[0]
def line_starts(b):
    n=int((ids1[b]>=0).sum()); s=starts[b]
    toks=ids1[b,:n]
    text_upto = None
    res=[]  # (pos_of_last_token_before_line_first_content, line_text)
    # decode token by token to find line boundaries
    pieces=[tok.decode([int(x)]) for x in toks[s:n]]
    cur=''; line_start_tok=s
    lines=[]; 
    acc=''
    tok_line_start=[]  # for each line: index of first token of line
    lstart=s
    buf=''
    for k,pc in enumerate(pieces):
        buf+=pc
        if '\n' in pc:
            # line ends within this token; next line starts at next token (approx)
            lines.append((lstart, buf)); buf=''; lstart=s+k+1
    lines.append((lstart,buf))
    return lines, pieces
lines0,_=line_starts(0)
print(len(lines0)); print([ (a, l[:40]) for a,l in lines0[:12]])
```