#include "cuda_runtime.h"
#include <stdio.h>
//type:DualAND(1),Post(2),Excit(3),Inhi(4),ExcitHabituation(5),InhiHabituation(6)
#define w_init 0.5
#define z_init 1

__constant__ float lr=0.1;

__constant__ float g1_peak=2.0/(1.0-1.0/(2.71828*2.71828));
__constant__ float R_max=1.0-1.0/(2.71828*2.71828);
struct __align__(16) synapse{
	 float w;
	 float z;
	 float z_spike;
	 float timeN;
	 float g1;          //NMDA
	 float g2;          //AMPA
	 float t1_r;
	 float t1_f;
	 float t2_r;
	 float t2_f;
	 float s1;
	 float R1;
	 float s2;
	 float R2;
	 int persynapse_number;
	 int postsynapse_number;
	 int synaplayer;
	 int connect_point;              //0:neuron 1:proximal 2:distal
	 unsigned int axon_delay;
	 unsigned long int axon;
};

struct __align__(16) neuro_synapse{
	 float w;
	 float z;
	 float z_spike;
	 float timeN;
	 float g1;          //NMDA or GABAa
	 float g2;          //AMPA or GABAb
	 float t1_r;
	 float t1_f;
	 float t2_r;
	 float t2_f;
	 float s1;
	 float R1;
	 float s2;
	 float R2;
	 int persynapse_number;
	 int postsynapse_number;
	 int synaplayer;
	 int connect_point;              //0:neuron 1:proximal 2:distal
	 unsigned int axon_delay;
	 unsigned long int axon;
};

struct __align__(16) gap_junction{
	 float timeN;
	 float g;
	 int persynapse_number;
	 int postsynapse_number;
};
