#include "cuda_runtime.h"
#include <stdio.h>

__device__ static void ex_habi_syna(struct synapse *neuro_syn, struct axon *neuro,unsigned char *spike,struct neuron_I *Ix,int number)
{
  float g_NMDA;
  float g_AMPA;
  float t_f;
  float t_r;
  float gpeak;
  float timeN;
  float v;
  int I;
  float p;             //max g(t)
  float dt;
  float g_scale;
  float z;
  float z_spike;
  unsigned long int prespike;
  float T,s,R;
  float t_max,g2_peak;

  timeN=neuro_syn[number].timeN;
  z=neuro_syn[number].z;
  v=neuro[neuro_syn[number].postsynapse_number].v;
  timeN++;
  if(timeN>1000000)
  {
   timeN=1000000;
  }

  //habituation
  z=z+tau*(1-z)/400;
  //NMDA
  t_f=neuro_syn[number].t1_f;
  t_r=neuro_syn[number].t1_r;
  gpeak=neuro_syn[number].g1;
  g_scale=1/(1+expf((-v-25)/12.5));
  s=neuro_syn[number].s1;
  if(timeN*tau<t_r){T=1.0/t_r;}
  else{T=0;}
  if(t_r==t_f)
  {
    s=s+tau*(T*(1-s)-s/t_f);
    g_NMDA=g_scale*gpeak*s*g1_peak;
  }
  else
  {
    R=neuro_syn[number].R1;
    t_max=(t_f*t_r/(t_f-t_r))*log(t_f/t_r);
    g2_peak=1+(t_r/t_f)/(2*R_max*expf((t_r-t_max)/t_r));
    R=R+tau*((1.0-R)*T-R/t_r);
    s=s+tau*((t_f+t_r)/t_f)*((2.0/t_r)*(1.0-s)*R-(s/t_f));
    g_NMDA=g_scale*gpeak*s*g2_peak;
  }
  neuro_syn[number].s1=s;
  neuro_syn[number].R1=R;

  //AMPA
  t_f=neuro_syn[number].t2_f;
  t_r=neuro_syn[number].t2_r;
  gpeak=neuro_syn[number].g2;
  s=neuro_syn[number].s2;
  if(timeN*tau<t_r){T=1.0/t_r;}
  else{T=0;}
  if(t_r==t_f)
  {
    s=s+tau*(T*(1-s)-s/t_f);
    g_AMPA=gpeak*s*g1_peak;
  }
  else
  {
    R=neuro_syn[number].R2;
    t_max=(t_f*t_r/(t_f-t_r))*log(t_f/t_r);
    g2_peak=1+(t_r/t_f)/(2*R_max*expf((t_r-t_max)/t_r));
    R=R+tau*((1.0-R)*T-R/t_r);
    s=s+tau*((t_f+t_r)/t_f)*((2.0/t_r)*(1.0-s)*R-(s/t_f));
    g_AMPA=gpeak*s*g2_peak;
  }
  neuro_syn[number].s2=s;
  neuro_syn[number].R2=R;

  z_spike=neuro_syn[number].z_spike;
  I=g_NMDA*(0-v)+z_spike*g_AMPA*(0-v);
  atomicAdd(&Ix[neuro_syn[number].postsynapse_number].I,I); //  Ix[neuro_syn[number].postsynapse_number]+=I_AMPA+I_NMDA;

  //if(neuro[neuro_syn[number].persynapse_number].layer==5 && neuro[neuro_syn[number].postsynapse_number].layer==7 && neuro_syn[number].postsynapse_number==600000)
  //{printf("g_AMPA:%f  Ix:%d  timeN:%f  T:%f  g2_peak:%f  s=%f  v=%f  z_spike=%f\n",g_AMPA,Ix[neuro_syn[number].postsynapse_number].I,timeN,T,g2_peak,s,v,z_spike);}

  prespike=(neuro_syn[number].axon<<1) | spike[neuro_syn[number].persynapse_number];
  if(((prespike>>neuro_syn[number].axon_delay)&0x01)==1) //判断突触前神经元是否有信号
  {
    timeN=0;
    neuro_syn[number].z_spike=z;
    z=0;
  }
  neuro_syn[number].axon=prespike;

  neuro_syn[number].timeN=timeN;
  neuro_syn[number].z=z;
}

__global__ static void excit_habituation_Synapse(struct synapse *neuro_syn, struct axon *neuro,unsigned char *spike,struct neuron_I *Ix,int *boxnum,int *THREAD_SYN,int *BLOCK_SYN)
{
   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   int number=(bid * THREAD_SYN[4] + tid)*10;


/*************************突触第一颗虚拟计算单元******************************/
if(number+0<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+0);}

/********************************************************************************/

/*************************突触第二颗虚拟计算单元******************************/
if(number+1<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+1);}

/********************************************************************************/

/*************************突触第三颗虚拟计算单元******************************/
if(number+2<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+2);}

/********************************************************************************/

/*************************突触第四颗虚拟计算单元******************************/
if(number+3<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+3);}

/********************************************************************************/

/*************************突触第五颗虚拟计算单元******************************/
if(number+4<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+4);}

/********************************************************************************/

/*************************突触第六颗虚拟计算单元******************************/
if(number+5<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+5);}

/********************************************************************************/

/*************************突触第七颗虚拟计算单元******************************/
if(number+6<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+6);}

/********************************************************************************/

/*************************突触第八颗虚拟计算单元******************************/
if(number+7<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+7);}

/********************************************************************************/

/*************************突触第九颗虚拟计算单元******************************/
if(number+8<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+8);}

/********************************************************************************/

/*************************突触第十颗虚拟计算单元******************************/
if(number+9<=boxnum[4])
{ex_habi_syna(neuro_syn, neuro, spike, Ix, number+9);}

/********************************************************************************/

}
