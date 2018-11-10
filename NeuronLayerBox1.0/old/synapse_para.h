#include "cuda_runtime.h"
#include <stdio.h>
//type:DualAND(1),Post(2),Excit(3),Inhi(4),ExcitHabituation(5),InhiHabituation(6)
#define w_init 0.5
#define z_init 1

__constant__ float gmax=0.00015;
__constant__ float Esyn=0;  //��ת��λ
__constant__ float t_1=2;
__constant__ float t_2=100;



//__constant__ float t_R=55;    //55ms
__constant__ float A=108.311081;   //  A=(t1-t2)/(expf(-t2*(logf(t2/t1))/(t2-t1))-expf(-t1*(logf(t2/t1))/(t2-t1)))

__constant__ float pm=0.59;
__constant__ float t_m=5.6;  //ms
__constant__ float t_d=217;  //ms

__constant__ float d1=1.0;
__constant__ float d2=0.33;
__constant__ float d3=2.0;
__constant__ float d4=-0.11;



struct __align__(16) synapse{
	 float w;
	 float timeN;
	 int persynapse_number;
	 int postsynapse_number;
};

struct __align__(16) neuro_synapse{
	 float z;
	 float timeN;
	 int persynapse_number;
	 int postsynapse_number;
};
