//gcc -D_REENTRANT qt-show.c -o show -lpthread -lglut -lGLU -lGL
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <termios.h>
#include <curses.h>    //sudo apt-get install libncurses5-dev
#include <term.h>
#include <GL/glut.h>
#include <pthread.h>
#define FileName "load_data/input.bmp"
static GLint imagewidth;
static GLint imageheight;
static GLint pixelwidth;
static GLint pixelheight;
static GLint pixellength;
static GLint imagelength;
static GLubyte* pixeldata;
static GLubyte* imagedata;

#define OUTPUT_LAYER 9

static struct termios initial_settings, new_settings;
static int peek_character = -1;
void init_keyboard();
void close_keyboard();
int kbhit();
int readch();
void *thread_function(void *arg);



struct shared_use_st
{
    int written;//作为一个标志，非0：表示可读，0表示可写
    int Xmax;
    int Ymax;
    int _switch_;
    int output_layer;
    int output_layer_xmax;
    int output_layer_ymax;
    int timen;
    int Nlines;
    int step;
};

struct shared_iamge
{
    int written_flag;//作为一个标志，非0：表示可读，1表示可写
    int read_flag;
    int length;
    int image[1280*720];//记录写入和读取的文本
};

void display(void)
{

 glClear(GL_COLOR_BUFFER_BIT);
 //---------------------------------
 glDrawPixels(pixelwidth,pixelheight,GL_BGR_EXT,GL_UNSIGNED_BYTE,pixeldata);
 //---------------------------------
 glFlush();
 glutSwapBuffers();
}

int time_flag=0;
void timeFunc(int value){
  time_flag++;
    display();
    // Present frame every 40 ms
    glutTimerFunc(40, timeFunc, 0);
}

int main(int argc,char* argv[])
{
  /*****************共享内存设置***************************/
   void *shm = NULL;
   struct shared_use_st *shared = NULL;
   int shmid;
   //创建共享内存
   shmid = shmget((key_t)2886, sizeof(struct shared_use_st), 0666|IPC_CREAT);
   if(shmid == -1)
   {
       fprintf(stderr, "shmget failed\n");
       exit(EXIT_FAILURE);
   }
   //将共享内存连接到当前进程的地址空间
   shm = shmat(shmid, (void*)0, 0);
   if(shm == (void*)-1)
   {
       fprintf(stderr, "shmat failed\n");
       exit(EXIT_FAILURE);
   }
   printf("Memory attached at %ld\n", (long)shm);
   //设置共享内存
   shared = (struct shared_use_st*)shm;
  /***********************************************************/
  /*****************共享内存设置output***************************/
   void *shm_image = NULL;//分配的共享内存的原始首地址
   struct shared_iamge *shared_image;//指向shm_image
   int shmid_image;//共享内存标识符
   //创建共享内存
   shmid_image = shmget((key_t)1235, sizeof(struct shared_iamge), 0666|IPC_CREAT);
   if(shmid_image == -1 )
   {
       fprintf(stderr, "shmget_image failed\n");
       exit(EXIT_FAILURE);
   }
   //将共享内存连接到当前进程的地址空间
   shm_image = shmat(shmid_image, 0, 0);
   if(shm_image == (void*)-1)
   {
       fprintf(stderr, "shmat_image failed\n");
       exit(EXIT_FAILURE);
   }
   printf("\nimage Memory attached at %ld\n", (long)shm_image);
   //设置共享内存
   shared_image = (struct shared_iamge*)shm_image;
   shared_image->read_flag = 1;
   /***********************************************************/
   /*****************共享内存设置input***************************/
   void *shm_input_image = NULL;//分配的共享内存的原始首地址
   struct shared_iamge *shared_input_image;//指向shm_image
   int shmid_input_image;//共享内存标识符
   //创建共享内存
   shmid_input_image = shmget((key_t)1236, sizeof(struct shared_iamge), 0666|IPC_CREAT);
   if(shmid_input_image == -1 )
   {
       fprintf(stderr, "shmget_input_image failed\n");
       exit(EXIT_FAILURE);
   }
   //将共享内存连接到当前进程的地址空间
   shm_input_image = shmat(shmid_input_image, 0, 0);
   if(shm_input_image == (void*)-1)
   {
       fprintf(stderr, "shmat_input_image failed\n");
       exit(EXIT_FAILURE);
   }
   printf("\nimage Memory attached at %ld\n", (long)shm_input_image);
   //设置共享内存
   shared_input_image = (struct shared_iamge*)shm_input_image;
   /******************************************************/
   printf("wait signal\n");
   while(shared->written!=1);
   printf("signal come\n");
   int c;
   int i=0,j=0;
   int n;
   int Nlines=shared->Nlines;
   FILE *fout = fopen("load_data/NeuronLayerAddr.txt", "r");
   if(!fout)
   {
    printf("can't open load_data/NeuronLayerAddr.txt\n");
    return -1;
   }
   printf("Nlines=%d\n",Nlines);
   int layeraddr[Nlines];
   i=0;
   while (!feof(fout))
   {
       fscanf(fout,"%d",&n);
       layeraddr[i]=n;
       if(i<Nlines)
       {
         printf("layeraddr[%d]=%d\n",i,layeraddr[i]);
       }
       i++;
   }
   fclose(fout);
   /***************读取图片****************/
   FILE* pfile=fopen(FileName,"rb");
   if(pfile == 0)
   {
     printf("no picture\n");
     return -1;
   }
   fseek(pfile,0x0012,SEEK_SET);
   fread(&imagewidth,sizeof(imagewidth),1,pfile);
   fread(&imageheight,sizeof(imageheight),1,pfile);
   printf("imagewidth=%d imageheight=%d\n",imagewidth,imageheight);
   if(imagewidth!=shared->Xmax || imageheight!=shared->Ymax)
   {
     printf("imagewidth or imageheight did not match input.txt\n");
     return -1;
   }
   imagelength=imagewidth*3;
   while(imagelength%4 != 0)imagelength++;
   imagelength *= imageheight;

   imagedata = (GLubyte*)malloc(imagelength);
   if(imagedata == 0) exit(0);
   fseek(pfile,54,SEEK_SET);
   fread(imagedata,imagelength,1,pfile);
   fclose(pfile);
   shared_input_image->length=imageheight*imagewidth;

   for(i=0;i<imageheight;i++)//imageheight
   {
     for(j=0;j<imagewidth;j++)//imagewidth
     {
        shared_input_image->image[i*imagewidth+j]=0.114*imagedata[(i*imagewidth+j)*3+0]+0.587*imagedata[(i*imagewidth+j)*3+1]+0.299*imagedata[(i*imagewidth+j)*3+2];
        //if(shared_input_image->image[i*imagewidth+j]>0){printf("%d=%d\n",i*imagewidth+j,(int)shared_input_image->image[i*imagewidth+j]);}
     }
   }
   /*******************************************/

   /***************************************/
   shared->written=0;
   shared->output_layer=OUTPUT_LAYER;
   shared->_switch_=1;
   printf("switch ON\n");
   printf("wait signal\n");
   while(shared->written!=1);
   printf("signal come\n");
   /*******************output******************/
   pixelwidth=shared->output_layer_xmax;
   pixelheight=shared->output_layer_ymax;
   pixellength=pixelwidth*3;
   while(pixellength%4 != 0)pixellength++;
   pixellength *= pixelheight;
   pixeldata = (GLubyte*)malloc(pixellength);
   if(pixeldata == 0) exit(0);
   /*****************************************/
   printf("output_width=%d output_height=%d\n",pixelwidth,pixelheight);

   int res;
   pthread_t a_thread;
   void *thread_result;
   res = pthread_create(&a_thread, NULL, thread_function, NULL);
   if (res != 0)
   {
      perror("Thread creation failed!");
      exit(EXIT_FAILURE);
   }

  //  for(i=0;i<pixelheight;i++)
  //  {
  //    for(j=0;j<pixelwidth;j++)
  //    {
  //     //  pixeldata[(i*pixelwidth+j)*3]=0;//imagedata[(i*imagewidth+j)*3];  //blue
  //     //  pixeldata[(i*pixelwidth+j)*3+1]=0;//imagedata[(i*pixelwidth+j)*3+1];  //green
  //     //  pixeldata[(i*pixelwidth+j)*3+2]=imagedata[(i*pixelwidth+j)*3+2];    //read
  //      pixeldata[(i*pixelwidth+j)*3]=shared_image->image[i*pixelwidth+j];
  //      pixeldata[(i*pixelwidth+j)*3+1]=pixeldata[(i*pixelwidth+j)*3];
  //      pixeldata[(i*pixelwidth+j)*3+2]=pixeldata[(i*pixelwidth+j)*3];
  //    }
  //  }

  int ch = 0;
  init_keyboard();
  int num=0;
  while(ch != 'q')
  {
    if(shared_image->written_flag > num)
   {
    num=shared_image->written_flag;
    for(i=0;i<pixelheight;i++)
    {
      for(j=0;j<pixelwidth;j++)
      {
        pixeldata[(i*pixelwidth+j)*3+2]=shared_image->image[i*pixelwidth+j];
        pixeldata[(i*pixelwidth+j)*3+1]=0;//pixeldata[(i*imagewidth+j)*3+2];
        pixeldata[(i*pixelwidth+j)*3+0]=0;//pixeldata[(i*imagewidth+j)*3+2];
      }
    }
    shared_image->read_flag=shared_image->read_flag+1;
   }
   if(kbhit()){ch = readch();}
  }

  shared->_switch_=0;
  printf("switch OFF\n");
  close_keyboard();

  res = pthread_join(a_thread, &thread_result);
   if (res != 0)
   {
       perror("Thread cancel failed!\n");
       exit(EXIT_FAILURE);
   }
  printf("Thread joined, it returned %s\n", (char *)thread_result);
  sleep(1);
  if(shmdt(shm) == -1)
  {
    fprintf(stderr, "shmdt failed\n");
    exit(EXIT_FAILURE);
  }
  if(shmdt(shm_image) == -1)
  {
   fprintf(stderr, "shm_image failed\n");
   exit(EXIT_FAILURE);
  }
  free(pixeldata);
  free(imagedata);
  exit(0);
}

void *thread_function(void *arg)
{
    int argc=0;
    char** argv=0;
    printf("thread_function is running.\n");
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
    glutInitWindowPosition(300,300);
    glutInitWindowSize(pixelwidth,pixelheight);
    glutCreateWindow(FileName);
    glutDisplayFunc(&display);
    glutTimerFunc(100, timeFunc, 0);
    glutMainLoop();
}

void init_keyboard()
{
    tcgetattr(0, &initial_settings);
    new_settings = initial_settings;
    new_settings.c_lflag &= ~ICANON;
    new_settings.c_lflag &= ~ECHO;
    new_settings.c_lflag &= ~ISIG;
    new_settings.c_cc[VMIN] = 1;
    new_settings.c_cc[VTIME] = 0;
    tcsetattr(0,TCSANOW, &new_settings);
}

void close_keyboard()
{
    tcsetattr(0,TCSANOW, &initial_settings);
}

int kbhit()
{
    char ch;
    int nread;

    if(peek_character != -1)
    {
        return -1;
    }
    new_settings.c_cc[VMIN] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
    nread = read(0, &ch, 1);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0,TCSANOW, &new_settings);

    if(nread == 1)
    {
        peek_character = ch;
        return 1;
    }
    return 0;
}

int readch()
{
    char ch;
    if(peek_character != -1)
    {
        ch = peek_character;
        peek_character = -1;
        return ch;
    }
    read (0, &ch, 1);
    return ch;
}
