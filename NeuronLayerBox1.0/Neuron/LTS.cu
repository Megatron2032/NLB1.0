#include "cuda_runtime.h"
#include <stdio.h>

__device__ static void LTS(int *input,struct axon *neuro, unsigned char *spike,struct neuron_I *Ix, int number)
{
  //设置神经元计算参数
  float C=100;
  float k=1;
  float vr=-56;
  float vt=-42;
  float G_up=1;
  float G_down=1;
  float a=0.03;
  float b=8;
  float c=-50;
  float d=20;
  float v_peak=40;
  float sq=0,sp=0;
  float I;


  float v=neuro[number].v;
  float u=neuro[number].u;
  I=Ix[number].I;



  //Izhikevich model
     v=v+tau*(k*(v-vr)*(v-vt)-u+I)/C;
     u=u+tau*a*(b*(v-vr)-u);
     spike[number]=0;
     if(v>v_peak)
     {
       v=c;
       u=u+d;
       spike[number]=1;
     }
     u=fmin(670,u);

  neuro[number].v=v;
  neuro[number].u=u;
  Ix[number].I=0;
}

__global__ static void LTS_neuron(int *input,struct axon *neuro, unsigned char *spike,struct neuron_I *Ix, int *boxnum, int *THREAD_NUM, int *BLOCK_NUM)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
int number=(THREAD_NUM[0]*BLOCK_NUM[0]+THREAD_NUM[1]*BLOCK_NUM[1]+THREAD_NUM[2]*BLOCK_NUM[2]+THREAD_NUM[3]*BLOCK_NUM[3]+THREAD_NUM[4]*BLOCK_NUM[4])*10+(bid * THREAD_NUM[5] + tid)*10;


/********第一个神经元虚拟计算内核*********/
if((number+0)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+0);}

/********第二个神经元虚拟计算内核********/
if((number+1)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+1);}

/********第三个神经元虚拟计算内核********/
if((number+2)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+2);}

/********第四个神经元虚拟计算内核*********/
if((number+3)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+3);}

/********第五个神经元虚拟计算内核*********/
if((number+4)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+4);}

/********第六个神经元虚拟计算内核*********/
if((number+5)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+5);}

/********第七个神经元虚拟计算内核********/
if((number+6)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+6);}

/********第八个神经元虚拟计算内核*********/
if((number+7)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+7);}

/********第九个神经元虚拟计算内核*********/
if((number+8)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+8);}

/********第十个神经元虚拟计算内核*********/
if((number+9)<=boxnum[5])
{LTS(input,neuro,spike,Ix,number+9);}

}
