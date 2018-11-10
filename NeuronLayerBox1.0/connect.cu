#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>

struct syn_loaddata
{
  int *type;
  int *Prelayer;
  int *Postlayer;
  int *inter;
  int *outer;
  int *connect_point;
  float *g1;
  float *g2;
  float *t1r;
  float *t1f;
  float *t2r;
  float *t2f;
  float *axon_delay;
};

int compute_num(int *X,int *Y,struct syn_loaddata *syn_data,int n);

/********************************************read***************************************/
int read_synapse_func(char restore_file[100],int *_THREAD_SYN,int *_BLOCK_SYN,struct synapse *SYN[6],int *N_SYN)
{
  int i=0;
  int syn_num=0;
  char StrLine;
  FILE *fp_restore=fopen(restore_file,"r");
  printf("load from ");
  for(i=0;i<100;i++)
  {
    if(restore_file[i]=='\0')
    {break;}
    printf("%c",restore_file[i]);
  }
  printf("\n");
  if(!fp_restore)
  {
   printf("can't open restore_file\n");
   return -1;
  }

  while (!feof(fp_restore))
  {
    fscanf(fp_restore, "%c", &StrLine);
    if(StrLine=='#')
    {
      fscanf(fp_restore, "%d", &syn_num);
      syn_num-=1;
      fscanf(fp_restore, "%d", &(_THREAD_SYN[syn_num]));
      fscanf(fp_restore, "%d", &(_BLOCK_SYN[syn_num]));
      fscanf(fp_restore, "%d", &(N_SYN[syn_num]));
      printf("THREAD_SYN[%d]=%d,BLOCK_SYN[%d]=%d,N_SYN[%d]=%d\n",syn_num,_THREAD_SYN[syn_num],syn_num,_BLOCK_SYN[syn_num],syn_num,N_SYN[syn_num]);
      SYN[syn_num]=(struct synapse*)malloc(sizeof(struct synapse)*(_THREAD_SYN[syn_num]*_BLOCK_SYN[syn_num]*10));
      if(_THREAD_SYN[syn_num]*_BLOCK_SYN[syn_num]>0)
      {
        for(i=0;i<_THREAD_SYN[syn_num]*_BLOCK_SYN[syn_num]*10;i++)
        {
          fscanf(fp_restore, "%d", &(SYN[syn_num][i].synaplayer));
          fscanf(fp_restore, "%d", &(SYN[syn_num][i].persynapse_number));
          fscanf(fp_restore, "%d", &(SYN[syn_num][i].postsynapse_number));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].w));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].z));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].timeN));
          fscanf(fp_restore, "%d", &(SYN[syn_num][i].connect_point));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].g1));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].g2));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].s1));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].s2));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].R1));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].R2));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].t1_r));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].t1_f));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].t2_r));
          fscanf(fp_restore, "%f", &(SYN[syn_num][i].t2_f));
          fscanf(fp_restore, "%u", &(SYN[syn_num][i].axon_delay));
        }
        //printf("synaplayer=%d,persynapse_number=%d,postsynapse_number=%d,axon_delay=%u\n",SYN[syn_num][0].synaplayer,SYN[syn_num][0].persynapse_number,SYN[syn_num][0].postsynapse_number,SYN[syn_num][0].axon_delay);
      }
    }
 }
 fclose(fp_restore);
 return 0;
}



/********************************************analysis***************************************/
int analysis_synapse_func(struct syn_loaddata *syn_data,int *X_max,int *Y_max,int maxline,int box,int *_THREAD_SYN,int *_BLOCK_SYN)
{
  int i;
  int num=0;
  for(i=0;i<maxline;i++)
  {
    if(syn_data->type[i]==box)
    {
        if(X_max[syn_data->Prelayer[i]]<=0 || Y_max[syn_data->Prelayer[i]]<=0 || X_max[syn_data->Postlayer[i]]<=0 || Y_max[syn_data->Postlayer[i]]<=0)
        {
          printf("error:xmax and ymax must >0\n");
          return -1;
        }
        num+=compute_num(X_max,Y_max,syn_data,i);
    }
  }
  if(num==0)
  {
    *_THREAD_SYN=0;
    *_BLOCK_SYN=0;
  }
  else if(num%1280==0)
  {
    *_BLOCK_SYN=num/1280;
    *_THREAD_SYN=128;
  }
  else if(num%960==0)
  {
    *_BLOCK_SYN=num/960;
    *_THREAD_SYN=96;
  }
  else if(num%640==0)
  {
    *_BLOCK_SYN=num/640;
    *_THREAD_SYN=64;
  }
  else if(num%320==0)
  {
    *_BLOCK_SYN=num/320;
    *_THREAD_SYN=32;
  }
  else
  {
    *_BLOCK_SYN=ceil(num/320.0);
    *_THREAD_SYN=32;
  }
  printf("num=%d\n",num);
  return 0;
}

int compute_num(int *X,int *Y,struct syn_loaddata *syn_data,int n)
{
  int all=0;
  FILE *fp=fopen("load_data/data.txt","ab");
  int x1,x2,y1,y2;
  x1=X[syn_data->Prelayer[n]];   //persynapse xmax
  y1=Y[syn_data->Prelayer[n]];   //persynapse ymax
  x2=X[syn_data->Postlayer[n]];   //postsynapse xmax
  y2=Y[syn_data->Postlayer[n]];   //postsynapse ymax
  float length1=syn_data->inter[n];
  float length2=syn_data->outer[n];
  for(int x_pre=0;x_pre<x1;x_pre++)
  {
    for(int y_pre=0;y_pre<y1;y_pre++)
    {
      int x_on=(int)(x_pre*x2*1.0/x1+0.5);
      int y_on=(int)(y_pre*y2*1.0/y1+0.5);
      for(int l_x=-length2;l_x<=length2;l_x++)
      {
        for(int l_y=-length2;l_y<=length2;l_y++)
        {
          float l=sqrt(l_x*l_x+l_y*l_y);
          if(l<=length2 && l>=length1)
          {
            if((x_on+l_x)>=0 && (x_on+l_x)<x2 && (y_on+l_y)>=0 && (y_on+l_y)<y2)
            {
              all+=1;
              int dataPtr[8];
              dataPtr[0]=syn_data->type[n];
              dataPtr[1]=n;
              dataPtr[2]=syn_data->Prelayer[n];
              dataPtr[3]=syn_data->Postlayer[n];
              dataPtr[4]=x_pre;
              dataPtr[5]=y_pre;
              dataPtr[6]=x_on+l_x;
              dataPtr[7]=y_on+l_y;
              for(int p=0;p<8;p++)
              {fprintf(fp, "%d ", dataPtr[p]);}
              fprintf(fp, "%f", l);
              fprintf(fp, "\n");
            }
          }
        }
      }
    }
  }
  fclose(fp);
  return all;
}


/**********************************connetc*************************************/
int connect_error(struct axon *neuron_copy,int *max,int pre_addr,int post_addr,int j,int k,int aj,int ak,int Prelayer,int Postlayer,int numbers)
{
  if(neuron_copy[pre_addr+j*max[Prelayer]+k].layer!=Prelayer || neuron_copy[post_addr+aj*max[Postlayer]+ak].layer!=Postlayer)
  {
    printf("synapse_cpoy_numbers=%d\n",numbers);
    printf("per=%d\n",pre_addr+(j)*max[Prelayer]+k);
    printf("post=%d\n",post_addr+(aj)*max[Postlayer]+ak);
    printf("per_layer=%d\n",neuron_copy[pre_addr+(j)*max[Prelayer]+k].layer);
    printf("post_layer=%d\n",neuron_copy[post_addr+(aj)*max[Postlayer]+ak].layer);
    printf("Prelayer=%d,Postlayer=%d\n",Prelayer,Postlayer);
    return -1;
  }
  return 0;
}

int addr_compute(struct axon *neuron_copy,int layer)
{
  int j=0;
  int layer_addr=-1;
  while(1)
  {
    if(neuron_copy[j].layer==layer)             //寻找突触前神经元对应neuronbox首地址
    {layer_addr=j;j=0;break;}
    j++;
  }
  return layer_addr;
}

int connect_synapse_func(struct syn_loaddata *syn_data,int *X_max,int *Y_max,int box,int *_THREAD_SYN,int *_BLOCK_SYN,struct synapse *synapse_cpoy,struct axon *neuron_copy,int *N_NUM)
{
  //syn_type:突触box类型
  //Prelayer:突触前层
  //Postlayer:突触后层
  //connect_point:突触连接点,[0,1,2]
  //X_max:各层神经元x最大尺寸
  //Y_max:各层神经元y最大尺寸
  //maxline:synapse.txt中总共层间连接数,文件中的每行相当于一个连接
  //box:突触的指定box类型(1,2,3,4,5,6)
  //synapse_cpoy:突触计算数据
  //neuron_copy:神经元计算数据
  //_THREAD_SYN,_BLOCK_SYN:cuda计算突触线程数相关


  int i;
  int j,k;
  int numbers=0;
  int all_num=0;
  int per_addr=0;
  int post_addr=0;

  int error=0;
  FILE *fp=fopen("load_data/data.txt","r");
  int m;
  int type,synlayer,Perlayer,Postlayer,x_per,y_per,x_post,y_post;
  int old_Perlayer=-1,old_Postlayer=-1;
  float l;
  int first_flag=0;
  while (!feof(fp))
  {
    fscanf(fp, "%d", &m);
    type=m;
    fscanf(fp, "%d", &m);
    synlayer=m;
    fscanf(fp, "%d", &m);
    Perlayer=m;
    fscanf(fp, "%d", &m);
    Postlayer=m;
    fscanf(fp, "%d", &m);
    x_per=m;
    fscanf(fp, "%d", &m);
    y_per=m;
    fscanf(fp, "%d", &m);
    x_post=m;
    fscanf(fp, "%d", &m);
    y_post=m;
    fscanf(fp, "%f", &l);
    all_num++;
    if(type==box)
    {
      if(first_flag==0){first_flag=1;}  //判断是否第一次进入相应的synapse box

      if(Perlayer!=old_Perlayer)
      {
        //printf("type=%d,synlayer=%d,perlayer=%d,postlayer=%d,x_per=%d,y_per=%d,x_post=%d,y_post=%d,l=%f\n",type,synlayer,Perlayer,Postlayer,x_per,y_per,x_post,y_post,l);
        printf("start per addr search\n");
        per_addr=addr_compute(neuron_copy,Perlayer);
        printf("per_addr=%d , Perlayer=%d\n",per_addr,Perlayer);
        printf("search per addr over\n");
        old_Perlayer=Perlayer;
      }
      if(Postlayer!=old_Postlayer)
      {
        printf("start post addr search\n");
        post_addr=addr_compute(neuron_copy,Postlayer);
        printf("post_addr=%d, Postlayer=%d\n",post_addr,Postlayer);
        printf("search post addr over\n");
        old_Postlayer=Postlayer;
      }


      //connect_point:突触连接点 0:神经元上 1:神经元树突附近  2:树突顶端
      //g1:NMDA or GABAa w*gpeak
      //g2:AMPA or GABAb w*gpeak
      //t1r:NMDA or GABAa计算参数
      //t1f:NMDA or GABAa计算参数
      //t2r:AMPA or GABAb计算参数
      //t2f:AMPA or GABAb计算参数
      //axon_delay:轴突延迟 tau的整数倍
      float w_ij;
      if(numbers>=(*_THREAD_SYN) *(*_BLOCK_SYN)*10){printf("error:number=%d,THREAD_SYN=%d,BLOCK_SYN=%d,box=%d\n",numbers,*_THREAD_SYN,*_BLOCK_SYN,box);error=-1;break;}
      if(syn_data->outer[synapse_cpoy[numbers].synaplayer]==0){w_ij=1;}
      else{w_ij=exp(-pow(l/syn_data->outer[synapse_cpoy[numbers].synaplayer],2));}
      synapse_cpoy[numbers].synaplayer=synlayer;
      synapse_cpoy[numbers].persynapse_number=per_addr+y_per*X_max[Perlayer]+x_per;
      synapse_cpoy[numbers].postsynapse_number=post_addr+y_post*X_max[Postlayer]+x_post;
      if(connect_error(neuron_copy,X_max,per_addr,post_addr,y_per,x_per,y_post,x_post,Perlayer,Postlayer,numbers)!=0){printf("connect_error\n");error=-1;break;}
      synapse_cpoy[numbers].w=w_init;
      synapse_cpoy[numbers].z=z_init;
      synapse_cpoy[numbers].timeN=0;
      synapse_cpoy[numbers].connect_point=syn_data->connect_point[synapse_cpoy[numbers].synaplayer];
      synapse_cpoy[numbers].g1=syn_data->g1[synapse_cpoy[numbers].synaplayer]*w_ij;
      synapse_cpoy[numbers].g2=syn_data->g2[synapse_cpoy[numbers].synaplayer]*w_ij;
      synapse_cpoy[numbers].s1=0;
      synapse_cpoy[numbers].R1=0;
      synapse_cpoy[numbers].s2=0;
      synapse_cpoy[numbers].R2=0;
      synapse_cpoy[numbers].t1_r=syn_data->t1r[synapse_cpoy[numbers].synaplayer];
      synapse_cpoy[numbers].t1_f=syn_data->t1f[synapse_cpoy[numbers].synaplayer];
      synapse_cpoy[numbers].t2_r=syn_data->t2r[synapse_cpoy[numbers].synaplayer];
      synapse_cpoy[numbers].t2_f=syn_data->t2f[synapse_cpoy[numbers].synaplayer];
      synapse_cpoy[numbers].axon_delay=floor(syn_data->axon_delay[synapse_cpoy[numbers].synaplayer]/tau_cpu);
      numbers++;
    }
    else
    {
      if(first_flag==1){break;}//已经进入相应的synapse box中途断开再出来后面没有相应的box，相同box是连续的。
    }
  }
  if(error<0){return -1;}
  N_NUM[box-1]=numbers-1;
  fclose(fp);
  printf("all_search=%d,number=%d\n",all_num,numbers);
  return 0;
}


void save_synapse_func(char save_file_name[100],int _THREAD_SYN,int _BLOCK_SYN,struct synapse *synapse_copy, int synapse_num, int N_SYN)
{
  FILE *fp_save=fopen(save_file_name,"ab");
  fprintf(fp_save, "#\n%d\n", synapse_num);
  fprintf(fp_save, "%d %d %d\n", _THREAD_SYN,_BLOCK_SYN,N_SYN);
  for(int i=0;i<_THREAD_SYN*_BLOCK_SYN*10;i++)
  {
    fprintf(fp_save, "%d ", synapse_copy[i].synaplayer);
    fprintf(fp_save, "%d ", synapse_copy[i].persynapse_number);
    fprintf(fp_save, "%d ", synapse_copy[i].postsynapse_number);
    fprintf(fp_save, "%f ", synapse_copy[i].w);
    fprintf(fp_save, "%f ", synapse_copy[i].z);
    fprintf(fp_save, "%f ", synapse_copy[i].timeN);
    fprintf(fp_save, "%d ", synapse_copy[i].connect_point);
    fprintf(fp_save, "%f ", synapse_copy[i].g1);
    fprintf(fp_save, "%f ", synapse_copy[i].g2);
    fprintf(fp_save, "%f ", synapse_copy[i].s1);
    fprintf(fp_save, "%f ", synapse_copy[i].s2);
    fprintf(fp_save, "%f ", synapse_copy[i].R1);
    fprintf(fp_save, "%f ", synapse_copy[i].R2);
    fprintf(fp_save, "%f ", synapse_copy[i].t1_r);
    fprintf(fp_save, "%f ", synapse_copy[i].t1_f);
    fprintf(fp_save, "%f ", synapse_copy[i].t2_r);
    fprintf(fp_save, "%f ", synapse_copy[i].t2_f);
    fprintf(fp_save, "%u\n", synapse_copy[i].axon_delay);
  }
  fclose(fp_save);
}
