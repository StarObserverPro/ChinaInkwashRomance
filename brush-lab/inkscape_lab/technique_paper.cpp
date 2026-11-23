// v0.15 entry-filtration variant; older paper_transport.cpp remains unchanged.
// Stratified porous paper, conservative finite volumes. All quantities are SI.
// Kirchhoff form of Brooks-Corey capillary transport avoids singular dry-node
// pressure differences. No image kernels or stochastic time-dependent forcing.
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" int paper_step(double* film,double* film_mass,double* water,
 double* fine,double* coarse,double* fixed,const double* capacity,
 const double* conduct,const double* vertical,const double* uptake,
 const double* fiber,int layers,int H,int W,int y0,int y1,int x0,int x1,
 double dt,double dx,double mu,double suction,double phi,double zreg,
 double exponent,double fine_fraction,double fine_length,double coarse_length,
 double fine_rate,double coarse_rate,double maxdt,
 double capture_fine,double capture_coarse,double* stats) {
 const int plane=H*W,n=layers*plane;
 const double area=dx*dx;
 const bool half_power=std::abs(exponent-3.5)<1e-12;
 std::vector<double> s(n),power(n),deriv(n),cf(n),cc(n),row(n),
    dv(n),df(n),dc(n),db(n);
 auto each=[&](auto fn){for(int l=0;l<layers;l++)for(int y=y0;y<y1;y++)
    for(int x=x0;x<x1;x++)fn(l*plane+y*W+x);};
 auto edges=[&](auto fn){
   for(int l=0;l<layers;l++)for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++){
     int p=y*W+x,a=l*plane+p;
     if(x+1<x1)fn(a,a+1,conduct[(l*4)*plane+p],dx);
     if(y+1<y1)fn(a,a+W,conduct[(l*4+1)*plane+p],dx);
     if(x+1<x1&&y+1<y1)fn(a,a+W+1,conduct[(l*4+2)*plane+p],dx*std::sqrt(2.));
     if(x> x0&&y+1<y1)fn(a,a+W-1,conduct[(l*4+3)*plane+p],dx*std::sqrt(2.));
     if(l+1<layers)fn(a,a+plane,vertical[l*plane+p],
       (capacity[a]+capacity[a+plane])/(2*phi*area));
   }
 };
 double left=dt,absorbed=0.,moved=0.;int steps=0;
 while(left>dt*1e-11){
   if(++steps>100000)return -1;
   each([&](int i){
     s[i]=std::clamp(water[i]/capacity[i],0.,1.);
     double forecast=s[i];
     if(i<plane)forecast=std::min(1.,(water[i]+film[i]*area)/capacity[i]);
     deriv[i]=half_power?3.5*forecast*forecast*std::sqrt(forecast):exponent*std::pow(forecast,exponent-1);
     row[i]=0.;
   });
   edges([&](int a,int b,double g,double length){
     double rate=g*std::max(deriv[a],deriv[b]);
     row[a]+=rate/capacity[a];row[b]+=rate/capacity[b];
   });
   double fastest=0.;each([&](int i){fastest=std::max(fastest,row[i]);});
   double sub=std::min({left,maxdt,.38/std::max(fastest,1e-30)});
   // Surface liquid enters the first layer. Saturation and capacity are local
   // to that layer, never diluted across the full paper thickness.
   for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++){
     int p=y*W+x;if(film[p]<=0)continue;
     double local_phi=capacity[p]/(area*fiber[plane+p]); // stored top layer dz
     double depth=water[p]/(local_phi*area);
     double dz=std::sqrt((depth+zreg)*(depth+zreg)+2*uptake[p]*suction*sub/(local_phi*mu))-zreg-depth;
     double take=std::min({film[p]*area,std::max(0.,capacity[p]-water[p]),std::max(0.,dz*local_phi*area)});
     const double available=film[p]*area;
     double pigment=film_mass[p]*take/available;
     if(take==available){pigment=film_mass[p];film[p]=0.;film_mass[p]=0.;}
     else{film[p]-=take/area;film_mass[p]-=pigment;}
     water[p]+=take;
     // Surface interception happens only with actual surface-to-paper uptake.
     // Water enters pores; a material-dependent particle fraction binds to
     // the first fibers instead of being advected through the first voxel.
     double mf=pigment*fine_fraction,mc=pigment*(1-fine_fraction);
     fixed[p]+=mf*capture_fine+mc*capture_coarse;
     fine[p]+=mf*(1-capture_fine);coarse[p]+=mc*(1-capture_coarse);absorbed+=take;
   }
   each([&](int i){
     s[i]=std::clamp(water[i]/capacity[i],0.,1.);
     power[i]=half_power?s[i]*s[i]*s[i]*std::sqrt(s[i]):std::pow(s[i],exponent);
     cf[i]=water[i]>1e-30?fine[i]/water[i]:0.;
     cc[i]=water[i]>1e-30?coarse[i]/water[i]:0.;
     dv[i]=df[i]=dc[i]=db[i]=0.;
   });
   edges([&](int a,int b,double g,double length){
     double q=g*(power[a]-power[b])*sub;
     if(q==0.)return;
     int donor=q>0?a:b,receiver=q>0?b:a;
     double v=std::abs(q),mf=v*cf[donor],mc=v*cc[donor];
     // Filtration deposits particles on fibers along the actual liquid path.
     double catchf=-std::expm1(-length/fine_length*fiber[receiver%plane]);
     double catchc=-std::expm1(-length/coarse_length*fiber[receiver%plane]);
     dv[donor]-=v;dv[receiver]+=v;
     df[donor]-=mf;df[receiver]+=mf*(1-catchf);
     dc[donor]-=mc;dc[receiver]+=mc*(1-catchc);
     db[receiver]+=mf*catchf+mc*catchc;moved+=v;
   });
   each([&](int i){
     water[i]+=dv[i];fine[i]+=df[i];coarse[i]+=dc[i];fixed[i]+=db[i];
     double ff=fine[i]*(-std::expm1(-fine_rate*fiber[i%plane]*sub));
     double fc=coarse[i]*(-std::expm1(-coarse_rate*fiber[i%plane]*sub));
     fine[i]-=ff;coarse[i]-=fc;fixed[i]+=ff+fc;
   });
   left-=sub;
 }
 stats[0]=steps;stats[1]=absorbed;stats[2]=moved;return 0;
}
