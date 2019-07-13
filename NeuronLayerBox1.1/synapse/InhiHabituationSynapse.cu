#include "cuda_runtime.h"
#include <stdio.h>

__device__ static void inhi_habi_syna(struct synapse *neuro_syn, struct axon *neuro, unsigned char *spike, struct neuron_I *Ix, int number)
{
  float ga;
  float gb;
  float t_f;
  float t_r;
  float gpeak;
  float timeN;
  float v;
  int I;
  float p;             //max g(t)
  float dt;
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
  //GABAa
  t_f=neuro_syn[number].t1_f;
  t_r=neuro_syn[number].t1_r;
  gpeak=neuro_syn[number].g1;
  s=neuro_syn[number].s1;
  if(timeN*tau<t_r){T=1.0/t_r;}
  else{T=0;}
  if(t_r==t_f)
  {
    s=s+tau*(T*(1-s)-s/t_f);
    ga=gpeak*s*g1_peak;
  }
  else
  {
    R=neuro_syn[number].R1;
    t_max=(t_f*t_r/(t_f-t_r))*log(t_f/t_r);
    g2_peak=1+(t_r/t_f)/(2*R_max*expf((t_r-t_max)/t_r));
    R=R+tau*((1.0-R)*T-R/t_r);
    s=s+tau*((t_f+t_r)/t_f)*((2.0/t_r)*(1.0-s)*R-(s/t_f));
    ga=gpeak*s*g2_peak;
  }
  neuro_syn[number].s1=s;
  neuro_syn[number].R1=R;

  //GABAb
  t_f=neuro_syn[number].t2_f;
  t_r=neuro_syn[number].t2_r;
  gpeak=neuro_syn[number].g2;
  s=neuro_syn[number].s2;
  if(timeN*tau<t_r){T=1.0/t_r;}
  else{T=0;}
  if(t_r==t_f)
  {
    s=s+tau*(T*(1-s)-s/t_f);
    gb=gpeak*s*g1_peak;
  }
  else
  {
    R=neuro_syn[number].R2;
    t_max=(t_f*t_r/(t_f-t_r))*log(t_f/t_r);
    g2_peak=1+(t_r/t_f)/(2*R_max*expf((t_r-t_max)/t_r));
    R=R+tau*((1.0-R)*T-R/t_r);
    s=s+tau*((t_f+t_r)/t_f)*((2.0/t_r)*(1.0-s)*R-(s/t_f));
    gb=gpeak*s*g2_peak;
  }
  neuro_syn[number].s2=s;
  neuro_syn[number].R2=R;

  z_spike=neuro_syn[number].z_spike;
  I=z_spike*ga*(-70-v)+gb*(-90-v);
  atomicAdd(&Ix[neuro_syn[number].postsynapse_number].I,I);

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

__global__ static void inhi_habituation_Synapse(struct synapse *neuro_syn, struct axon *neuro,unsigned char *spike,struct neuron_I *Ix,int *boxnum,int *THREAD_SYN,int *BLOCK_SYN)
{
   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   int number=(bid * THREAD_SYN[5] + tid)*10;


/*************************突触第一颗虚拟计算单元******************************/
if(number+0<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+0);}
/********************************************************************************/

/*************************突触第二颗虚拟计算单元******************************/
if(number+1<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+1);}
/********************************************************************************/

/*************************突触第三颗虚拟计算单元******************************/
if(number+2<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+2);}
/********************************************************************************/

/*************************突触第四颗虚拟计算单元******************************/
if(number+3<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+3);}
/********************************************************************************/

/*************************突触第五颗虚拟计算单元******************************/
if(number+4<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+4);}
/********************************************************************************/

/*************************突触第六颗虚拟计算单元******************************/
if(number+5<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+5);}
/********************************************************************************/

/*************************突触第七颗虚拟计算单元******************************/
if(number+6<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+6);}
/********************************************************************************/

/*************************突触第八颗虚拟计算单元******************************/
if(number+7<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+7);}
/********************************************************************************/

/*************************突触第九颗虚拟计算单元******************************/
if(number+8<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+8);}
/********************************************************************************/

/*************************突触第十颗虚拟计算单元******************************/
if(number+9<=boxnum[5])
{inhi_habi_syna(neuro_syn,neuro,spike,Ix,number+9);}
/********************************************************************************/


}
