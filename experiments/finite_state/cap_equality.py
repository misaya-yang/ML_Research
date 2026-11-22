import mpmath as m
m.mp.dps=70
one=m.mpf(1)
p=m.mpf('0.5');ell=p-1/m.sqrt(24);k=1-2*ell
der=(m.mpf('1.5'))*(ell-m.mpf(7)/24)
print('kappa',m.nstr(k,34),'B',m.nstr(1/(1-k),34),'derivative',m.nstr(der,34))
for x in map(m.mpf,('1e-3','1e-4','1e-5','1e-7','1e-10')):
    q=m.matrix([[p-x,p,x]])
    w=[one,one,one+k/((1-k)*x)]
    f=m.matrix([-ell,1-ell,-ell])
    K=m.matrix([[q[j]*min(one,w[j]/w[i]) for j in range(3)] for i in range(3)])
    for i in range(3):K[i,i]+=one-sum(K[i,j] for j in range(3))
    u=f.copy();v=m.matrix([z*z for z in f])
    for _ in range(3):
        ku=K*u
        v=m.matrix([z*z+2*z*y for z,y in zip(f,ku)])+K*v
        u=f+ku
    G=(q*v)[0]/16-ell*(1-ell)/2
    print('x',m.nstr(x,8),'G',m.nstr(G,24),'G/x',m.nstr(G/x,24))
