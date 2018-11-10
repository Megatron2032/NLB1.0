#include "cuda_runtime.h"
#include <stdio.h>

__global__ static void post_Synapse(struct  synapse *syn, struct axon *neuro,unsigned char *spike,int *Ix,int *THREAD_SYN,int *BLOCK_SYN)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int number=(bid * THREAD_SYN[1] + tid)*10;

  float w;
	float timeN;
  float v;
	float gsyn;
  float I_AMPA;
  float I_NMDA;
  float f;

/*************************突触第一颗虚拟计算单元******************************/
	w=syn[number].w;
	timeN=syn[number].timeN;
  v=neuro[syn[number].postsynapse_number].v;
  timeN++;

  if(spike[syn[number].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

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
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number].w=w;
  syn[number].timeN=timeN;
/************************************************************************/

/*************************突触第二颗虚拟计算单元******************************/
	w=syn[number+1].w;
	timeN=syn[number+1].timeN;
  v=neuro[syn[number+1].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+1].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+1].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+1].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number].w=w;
  syn[number].timeN=timeN;
/************************************************************************/

/*************************突触第三颗虚拟计算单元******************************/
	w=syn[number+2].w;
	timeN=syn[number+2].timeN;
  v=neuro[syn[number+2].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+2].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+2].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+2].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+2].w=w;
  syn[number+2].timeN=timeN;
/************************************************************************/

/*************************突触第四颗虚拟计算单元******************************/
	w=syn[number+3].w;
	timeN=syn[number+3].timeN;
  v=neuro[syn[number+3].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+3].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+3].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+3].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+3].w=w;
  syn[number+3].timeN=timeN;
/************************************************************************/

/*************************突触第五颗虚拟计算单元******************************/
	w=syn[number+4].w;
	timeN=syn[number+4].timeN;
  v=neuro[syn[number+4].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+4].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+4].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+4].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+4].w=w;
  syn[number+4].timeN=timeN;
/************************************************************************/

/*************************突触第六颗虚拟计算单元******************************/
	w=syn[number+5].w;
	timeN=syn[number+5].timeN;
  v=neuro[syn[number+5].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+5].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+5].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+5].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+5].w=w;
  syn[number+5].timeN=timeN;
/************************************************************************/

/*************************突触第七颗虚拟计算单元******************************/
	w=syn[number+6].w;
	timeN=syn[number+6].timeN;
  v=neuro[syn[number+6].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+6].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+6].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+6].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+6].w=w;
  syn[number+6].timeN=timeN;
/************************************************************************/

/*************************突触第八颗虚拟计算单元******************************/
	w=syn[number+7].w;
	timeN=syn[number+7].timeN;
  v=neuro[syn[number+7].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+7].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+7].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+7].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+7].w=w;
  syn[number+7].timeN=timeN;
/************************************************************************/

/*************************突触第九颗虚拟计算单元******************************/
	w=syn[number+8].w;
	timeN=syn[number+8].timeN;
  v=neuro[syn[number+8].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+8].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+8].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+8].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+8].w=w;
  syn[number+8].timeN=timeN;
/************************************************************************/

/*************************突触第十颗虚拟计算单元******************************/
	w=syn[number+9].w;
	timeN=syn[number+9].timeN;
  v=neuro[syn[number+9].postsynapse_number].v;
  timeN++;

  if(spike[syn[number+9].persynapse_number]==1)//判断突触前神经元是否有信号
  {
  	timeN=0;
  }

  gsyn=w*gmax*A*(expf(-(timeN*tau)/t_1)-expf(-(timeN*tau)/t_2))/(t_1-t_2);
  if(timeN>1000000)
	{
	  timeN=1000000;
	  gsyn=0;
	}
  I_AMPA=gsyn*(Esyn-v);
  I_NMDA=(d1/(d1+d2*d3*expf(d4*v)))*gsyn*(Esyn-v);   //  I_NMDA=(1.0/(1+0.33*2*expf(-0.11*v)))*gsyn*(v-Esyn);
//  if(I_NMDA<0){I_NMDA=0.000001;}
//  if(I_AMPA<0){I_AMPA=0.000001;}

  atomicAdd(&Ix[syn[number+9].postsynapse_number],int(I_AMPA+I_NMDA)); //  Ix[syn[bid * THREAD_NUM + tid].postsynapse_number]+=I_AMPA+I_NMDA;  //ԭ�Ӳ���

  if(spike[syn[number+9].postsynapse_number]==1)//判断突触后神经元是否有信号
  {
  	f=0.5;
  }
  else if(timeN<=1)
  {
  	f=-1.5;
  }
  else if(timeN<=126)
  {
  	f=(timeN-1)*0.2*0.02-0.5;
  }
  else
  {
    f=0;
  }
  w=w+tau*v*v*(gsyn*f*(-2)+1-w);

  syn[number+9].w=w;
  syn[number+9].timeN=timeN;
/************************************************************************/
}
