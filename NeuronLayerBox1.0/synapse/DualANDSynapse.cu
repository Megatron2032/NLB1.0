#include "cuda_runtime.h"
#include <stdio.h>

__device__ static void dual_and_syna(struct synapse *syn, struct axon *neuro,unsigned char *spike,struct neuron_I *Ix,int number)
{
  float w;
  float v;
  int I;
  float f;
  float g_NMDA;
  float g_AMPA;
  float t_f;
  float t_r;
  float gpeak;
  float timeN;
  float p;             //max g(t)
  float dt;
  float g_scale;
  unsigned long int prespike;
  float T,s,R;
  float t_max,g2_peak;

  w=syn[number].w;
  timeN=syn[number].timeN;
  v=neuro[syn[number].postsynapse_number].v;
  timeN++;

  prespike=(syn[number].axon<<1) | spike[syn[number].persynapse_number];
  if(((prespike>>syn[number].axon_delay)&0x01)==1) //判断突触前神经元是否有信号
  {
    timeN=0;
  }
  syn[number].axon=prespike;

  //NMDA
  t_f=syn[number].t1_f;
  t_r=syn[number].t1_r;
  gpeak=syn[number].g1;
  g_scale=1/(1+expf((-v-25)/12.5));
  s=syn[number].s1;
  if(timeN*tau<t_r){T=1.0/t_r;}
  else{T=0;}
  if(t_r==t_f)
  {
    s=s+tau*(T*(1-s)-s/t_f);
    g_NMDA=g_scale*gpeak*s*g1_peak;
  }
  else
  {
    R=syn[number].R1;
    t_max=(t_f*t_r/(t_f-t_r))*log(t_f/t_r);
    g2_peak=1+(t_r/t_f)/(2*R_max*expf((t_r-t_max)/t_r));
    R=R+tau*((1.0-R)*T-R/t_r);
    s=s+tau*((t_f+t_r)/t_f)*((2.0/t_r)*(1.0-s)*R-(s/t_f));
    g_NMDA=g_scale*gpeak*s*g2_peak;
  }
  syn[number].s1=s;
  syn[number].R1=R;

  //AMPA
  t_f=syn[number].t2_f;
  t_r=syn[number].t2_r;
  gpeak=syn[number].g2;
  s=syn[number].s2;
  if(timeN*tau<t_r){T=1.0/t_r;}
  else{T=0;}
  if(t_r==t_f)
  {
    s=s+tau*(T*(1-s)-s/t_f);
    g_AMPA=gpeak*s*g1_peak;
  }
  else
  {
    R=syn[number].R2;
    t_max=(t_f*t_r/(t_f-t_r))*log(t_f/t_r);
    g2_peak=1+(t_r/t_f)/(2*R_max*expf((t_r-t_max)/t_r));
    R=R+tau*((1.0-R)*T-R/t_r);
    s=s+tau*((t_f+t_r)/t_f)*((2.0/t_r)*(1.0-s)*R-(s/t_f));
    g_AMPA=gpeak*s*g2_peak;
  }
  syn[number].s2=s;
  syn[number].R2=R;

  I=g_NMDA*(0-v)+w*g_AMPA*(0-v);
  atomicAdd(&Ix[syn[number].postsynapse_number].I,I); //  Ix[syn[number].postsynapse_number]+=I_AMPA+I_NMDA;

if(spike[syn[number].postsynapse_number]==1)//判断突触后神经元是否有信号
{
  f=0.5;
}
else if(timeN<=1)
{
  f=-1.5;
}
else if(timeN<=126)
{
  f=(timeN-1)*tau*0.02-0.5;
}
else
{
  f=0;
}
w=w+lr*tau*g_AMPA*f*f*(g_AMPA*f*(-2)+1-w);
//  if(w<0.5){printf("%f\n",w); }

syn[number].w=w;
syn[number].timeN=timeN;
}

__global__ static void Dual_AND_Synapse(struct synapse *syn, struct axon *neuro,unsigned char *spike,struct neuron_I *Ix,int *boxnum,int *THREAD_SYN,int *BLOCK_SYN)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
	int number=(bid * THREAD_SYN[0] + tid)*10;


/*************************突触第一颗虚拟计算单元******************************/
  if(number+0<=boxnum[0])
  {dual_and_syna(syn,neuro,spike,Ix,number+0);}

/********************************************************************************/

/*************************突触第二颗虚拟计算单元******************************/
if(number+1<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+1);}

/********************************************************************************/

/*************************突触第三颗虚拟计算单元******************************/
if(number+2<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+2);}

/********************************************************************************/

/*************************突触第四颗虚拟计算单元******************************/
if(number+3<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+3);}

/********************************************************************************/

/*************************突触第五颗虚拟计算单元******************************/
if(number+4<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+4);}

/********************************************************************************/

/*************************突触第六颗虚拟计算单元******************************/
if(number+5<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+5);}

/********************************************************************************/

/*************************突触第七颗虚拟计算单元******************************/
if(number+6<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+6);}

/********************************************************************************/

/*************************突触第八颗虚拟计算单元******************************/
if(number+7<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+7);}

/********************************************************************************/

/*************************突触第九颗虚拟计算单元******************************/
if(number+8<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+8);}

/********************************************************************************/

/*************************突触第十颗虚拟计算单元******************************/
if(number+9<=boxnum[0])
{dual_and_syna(syn,neuro,spike,Ix,number+9);}

/********************************************************************************/
}
