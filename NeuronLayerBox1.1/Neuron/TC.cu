#include "cuda_runtime.h"
#include <stdio.h>

__device__ static void TC(int *input,struct axon *neuro, unsigned char *spike,struct neuron_I *Ix, int number)
{
  //设置神经元计算参数
  float C=200;
  float k=1.6;
  float vr=-60;
  float vt=-50;
  float G_up=2.0;
  float G_down=2.0;
  float a=0.1;
  float b=15;
  float c=-60;
  float d=10;
  float v_peak=40;
  float I;


  float v=neuro[number].v;
  float u=neuro[number].u;
  I=Ix[number].I;


//Izhikevich model
     if(v>-65){b=0;}else{b=15;}
     v=v+tau*(k*(v-vr)*(v-vt)-u+I)/C;
     u=u+tau*a*(b*(v-vr)-u);
     spike[number]=0;
     if(v>v_peak)
     {
       v=c;
       u=u+d;
       spike[number]=1;
     }
  //  if(number==840000)
  //  {printf("I0=%f,I1=%f,I2=%f,I_distal=%f,I_proximal=%f,I=%f,v_distal=%f,v_proximal=%f,v=%f\n",I0,I1,I2,I_distal,I_proximal,I,v_distal,v_proximal,v);}

  neuro[number].v=v;
  neuro[number].u=u;
  Ix[number].I=0;

}

__global__ static void TC_neuron(int *input,struct axon *neuro, unsigned char *spike,struct neuron_I *Ix, int *boxnum, int *THREAD_NUM, int *BLOCK_NUM)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
int number=(THREAD_NUM[0]*BLOCK_NUM[0]+THREAD_NUM[1]*BLOCK_NUM[1])*10+(bid * THREAD_NUM[2] + tid)*10;

/********第一个神经元虚拟计算内核*********/
if((number+0)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+0);}

/********第二个神经元虚拟计算内核********/
if((number+1)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+1);}

/********第三个神经元虚拟计算内核********/
if((number+2)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+2);}

/********第四个神经元虚拟计算内核*********/
if((number+3)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+3);}

/********第五个神经元虚拟计算内核*********/
if((number+4)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+4);}

/********第六个神经元虚拟计算内核*********/
if((number+5)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+5);}

/********第七个神经元虚拟计算内核********/
if((number+6)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+6);}

/********第八个神经元虚拟计算内核*********/
if((number+7)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+7);}

/********第九个神经元虚拟计算内核*********/
if((number+8)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+8);}

/********第十个神经元虚拟计算内核*********/
if((number+9)<=boxnum[2])
{TC(input,neuro,spike,Ix,number+9);}

}
