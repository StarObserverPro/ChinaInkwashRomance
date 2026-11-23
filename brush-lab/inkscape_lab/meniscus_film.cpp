// Conservative finite-volume surface film. SI units; compile through meniscus.py.
// h is EXCESS liquid above a numerical precursor, whose volume is never counted
// or absorbed. p = -gamma Laplacian(h+z) + dPhi/dh, M = h_total^3/(3 mu).
// Phi = gamma theta(x)^2/2 * [(hp/h_total)^6 - 2(hp/h_total)^3].
// The fixed heterogeneous wetting energy pins the moving contact line; no
// per-frame noise, rendered opacity, or image morphology enters the solver.
#include <algorithm>
#include <cmath>
#include <vector>
extern "C" int film_step(double *h,double *m,double *water,double *mobile,
 const double *z,const double *theta,const double *perm,int H,int W,
 int y0,int y1,int x0,int x1,double dt,double dx,double mu,double gamma,
 double hp,double phi,double thickness,double zreg,double suction,
 double *stats){
 const int nx=x1-x0,ny=y1-y0,n=nx*ny; const double area=dx*dx;
 std::vector<double> p(n),qx(n),qy(n),out(n),fac(n),conc(n);
 double left=dt,absorbed=0.;int steps=0;
 auto idx=[&](int i){return (y0+i/nx)*W+x0+i%nx;};
 while(left>dt*1e-10){
   double maxh=hp,maxdp=0.;
   for(int i=0;i<n;i++)maxh=std::max(maxh,h[idx(i)]+hp);
   // 2-D biharmonic and disjoining-pressure explicit stability bounds.
   const double maxmob=maxh*maxh*maxh/(3*mu);
   double sub=std::min(left,.16*dx*dx*dx*dx/(32*gamma*maxmob));
   sub=std::min(sub,.08*mu*dx*dx/(gamma*.08*hp));
   if(++steps>500000)return -1;
   std::fill(out.begin(),out.end(),0.);
   for(int yy=0;yy<ny;yy++)for(int xx=0;xx<nx;xx++){
     int i=yy*nx+xx,g=idx(i); double v=h[g]+hp,rat=hp/v;
     double center=h[g]+z[g],lap=0.;
     if(xx>0)lap+=h[g-1]+z[g-1]-center;
     if(xx+1<nx)lap+=h[g+1]+z[g+1]-center;
     if(yy>0)lap+=h[g-W]+z[g-W]-center;
     if(yy+1<ny)lap+=h[g+W]+z[g+W]-center;
     p[i]=-gamma*lap/area+3*gamma*theta[g]*theta[g]/hp*(std::pow(rat,4)-std::pow(rat,7));
     conc[i]=h[g]>1e-25?m[g]/(h[g]*area):0.;
   }
   for(int yy=0;yy<ny;yy++)for(int xx=0;xx<nx;xx++){
     int i=yy*nx+xx,g=idx(i);qx[i]=qy[i]=0.;
     if(xx+1<nx){
       double ht=.5*(h[g]+h[g+1])+hp;
       qx[i]=ht*ht*ht/(3*mu)*(p[i]-p[i+1])*sub/area;
       out[qx[i]>=0?i:i+1]+=std::abs(qx[i]);
     }
     if(yy+1<ny){
       double ht=.5*(h[g]+h[g+W])+hp;
       qy[i]=ht*ht*ht/(3*mu)*(p[i]-p[i+nx])*sub/area;
       out[qy[i]>=0?i:i+nx]+=std::abs(qy[i]);
     }
   }
   for(int i=0;i<n;i++)fac[i]=out[i]>0?std::min(1.,std::max(0.,h[idx(i)])*(1-1e-12)/out[i]):1.;
   for(int yy=0;yy<ny;yy++)for(int xx=0;xx<nx;xx++){
     int i=yy*nx+xx,g=idx(i);
     if(xx+1<nx){double q=qx[i]*fac[qx[i]>=0?i:i+1];
       double pigment=q*area*conc[q>=0?i:i+1];
       h[g]-=q;h[g+1]+=q;m[g]-=pigment;m[g+1]+=pigment;}
     if(yy+1<ny){double q=qy[i]*fac[qy[i]>=0?i:i+nx];
       double pigment=q*area*conc[q>=0?i:i+nx];
       h[g]-=q;h[g+W]+=q;m[g]-=pigment;m[g+W]+=pigment;}
   }
   for(int i=0;i<n;i++){
     int g=idx(i);double depth=water[g]/(phi*area);
     double dz=std::sqrt((depth+zreg)*(depth+zreg)+2*perm[g]*suction*sub/(phi*mu))-zreg-depth;
     double take=std::min(std::max(0.,thickness-depth)*phi,std::max(0.,dz)*phi);
     take=std::min(std::max(0.,h[g]),take);
     double pig=h[g]>0?m[g]*take/h[g]:0.;
     h[g]-=take;m[g]-=pig;water[g]+=take*area;mobile[g]+=pig;absorbed+=take*area;
     // Resolve the vanishing-film limit conservatively. Keeping an epsilon
     // height with residual floating-point pigment makes m/h ill-conditioned.
     // Both residual inventories enter the paper; nothing is discarded.
     if(h[g]<1e-12){water[g]+=h[g]*area;absorbed+=h[g]*area;
       mobile[g]+=m[g];h[g]=0.;m[g]=0.;}
   }
   left-=sub;
 }
 stats[0]=steps;stats[1]=absorbed;return 0;
}
