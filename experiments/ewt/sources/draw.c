/* Generate iid entire conditional HMM paths by backward sampling. Inputs
   are uniform [0,1) innovations and exact forward--backward model tables.
   The log-ratio equals unnormalized target potential minus proposal potential.
   Neither normalizer is used by the audited algorithms. */
#include <math.h>
static inline int pick(const double* c, double u, int d) {
    int lo=0,hi=d-1;
    while(lo<hi) {int mid=(lo+hi)/2;if(c[mid] < u)lo=mid+1;else hi=mid;}
    return lo;
}
void paired(int n,int len,int d,const double *last,const double *back,
    const double *u,const double *ed,const double *ad,int pos,int gold,
    double *logw,unsigned char *h) {
    for(int k=0;k<n;k++) {
        int v=pick(last,*u++,d);
        double score=ed[(len-1)*d+v];
        if(pos==len-1) h[k]=(v==gold);
        for(int j=len-2;j>=0;j--) {
            int prev=pick(back+(j*d+v)*d,*u++,d);
            score+=ed[j*d+prev]+ad[prev*d+v];
            if(pos==j) h[k]=(prev==gold);
            v=prev;
        } logw[k]=score;
    }
}
