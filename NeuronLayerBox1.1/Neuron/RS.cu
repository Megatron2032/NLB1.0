#include "cuda_runtime.h"
#include <stdio.h>
//neurontype:RS     example:layer2/3 layer4 layer5 layer6I layer6II


__device__ static void RS(int *input,struct axon *neuro, unsigned char *spike,struct neuron_I *Ix, int number)
{
  //设置神经元计算参数
  float C=100;
  float k=3;
  float vr=-60;
  float vt=-50;
  float G_up=3.0;
  float G_down=5.0;
  float a=0.01;
  float b=5;
  float c=-60;
  float d=400;
  float v_peak=50;
  float I;
  float I_distal;
  float I_proximal;

  float v=neuro[number].v;
  float u=neuro[number].u;
  I=Ix[number].I;


  //if(neuro[number].layer==5 && number==120000){printf("v2=%f,v1=%f,v0=%f,I_proximal=%f,I=%f\n",v_distal,v_proximal,v,I_proximal,I);}

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


  neuro[number].v=v;
  neuro[number].u=u;
  Ix[number].I=0;
}

__global__ static void RS_neuron(int *input,struct axon *neuro,unsigned char *spike,struct neuron_I *Ix, int *boxnum, int *THREAD_NUM, int *BLOCK_NUM)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
int number=(bid * THREAD_NUM[0] + tid)*10;


/********第一个神经元虚拟计算内核*********/
if((number+0)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+0);}

/********第二个神经元虚拟计算内核********/
if((number+1)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+1);}

/********第三个神经元虚拟计算内核********/
if((number+2)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+2);}

/********第四个神经元虚拟计算内核*********/
if((number+3)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+3);}

/********第五个神经元虚拟计算内核*********/
if((number+4)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+4);}

/********第六个神经元虚拟计算内核*********/
if((number+5)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+5);}

/********第七个神经元虚拟计算内核********/
if((number+6)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+6);}

/********第八个神经元虚拟计算内核*********/
if((number+7)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+7);}

/********第九个神经元虚拟计算内核*********/
if((number+8)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+8);}

/********第十个神经元虚拟计算内核*********/
if((number+9)<=boxnum[0])
{RS(input,neuro,spike,Ix,number+9);}


}
