#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <termios.h>

struct shared_iamge
{
    int written_flag;//作为一个标志，非0：表示可读，1表示可写
    int read_flag;
    int length;
    int layer;
    int data[1280*720];
    int width;
    int height;
    int next_frame;
};


struct shared_iamge* output_init(int outputNlines)
{
  /*****************共享内存设置output***************************/
  void *shm_image = NULL;//分配的共享内存的原始首地址
  struct shared_iamge *shared_image;//指向shm_image
  int shmid_image;//共享内存标识符
  //创建共享内存
  shmid_image = shmget((key_t)1031, sizeof(struct shared_iamge)*outputNlines, 0666|IPC_CREAT);
  if(shmid_image == -1 )
  {
      fprintf(stderr, "output shmget_image failed\n");
      exit(EXIT_FAILURE);
  }
  else
  {
    printf( "output shmid_image=%d\n",shmid_image);
  }
  //将共享内存连接到当前进程的地址空间
  shm_image = shmat(shmid_image, 0, 0);
  if(shm_image == (void*)-1)
  {
      fprintf(stderr, "output shmat_image failed\n");
      exit(EXIT_FAILURE);
  }
  printf("\nimage Memory attached at %ld\n", (long)shm_image);
  //设置共享内存
  shared_image = (struct shared_iamge*)shm_image;
  /******************************************************/

 return shared_image;
}
