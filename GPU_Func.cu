#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. �Լ��� Colab ȯ�濡�� �����ؾ� �մϴ�.
// 2. �����Ӱ� �����ϼŵ� ������ ��� �Լ����� GPU�� Ȱ���ؾ� �մϴ�.
// 3. CPU_Func.cu�� �ִ� Image_Check�Լ����� True�� Return�Ǿ�� �ϸ�, CPU�ڵ忡 ���� �ӵ��� ����� �մϴ�.
/////////////////////////////////////////////////////////////////////////


//---------- constant memory 선언 ----------
__constant__ int d_WIDTH;       // 이미지 너비
__constant__ int d_HEIGHT;      // 이미지 높이
__constant__ float filter[25];  // Noise Reduction에 쓰이는 필터
__constant__ int filter_x[9];   // Intensity Gradient에 쓰이는 x 필터
__constant__ int filter_y[9];   // Intensity Gradient에 쓰이는 y 필터
__constant__ uint8_t low_t;     // Hysteresis_Thresholding에 쓰임
__constant__ uint8_t high_t;    // Hysteresis_Thresholding에 쓰임

//---------- 전역 변수 선언 ----------
uint8_t * d_gray;               // GPU에서 grayscale한 이미지 참조
uint8_t * d_gaussian;           // GPU에서 gaussian한 이미지 참조
uint8_t * d_sobel;              // GPU에서 soble 적용한 이미지 참조
uint8_t * d_angle;              // GPU에서 angle 정보 참조
uint8_t * d_suppression_pixel;  // GPU에서 suppression_pixcel 적용한 이미지 참조

__global__ void grayScaleKernel(uint8_t* d_gray){

    // 스레드 인덱스    
    int thread_idx = blockIdx.x* 1024+threadIdx.x;
    // 배열의 인덱스, 스레드가 배열의 요소3개씩 처리한다. 
    int arr_idx = (thread_idx*3);
    if(arr_idx >2400051)
        return;
    
    //BGR. d_buf[i]= B, d_buf[i+1] = G, d_buf[i+2]=R
    float B = d_gray[arr_idx] * 0.114;
    float G = d_gray[arr_idx +1] * 0.587;
    float R = d_gray[arr_idx +2] * 0.299;

    int tmp = B+G+R;

    d_gray[arr_idx] =tmp;
    d_gray[arr_idx+1] = tmp;
    d_gray[arr_idx+2] = tmp;
}

void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {

    //buf를 GPU에 올릴려면 +2해줘서 빠지는게 없이 올라가야한다.
    int realLen = len+2; 
    // 메모리 할당할 size
    int size = realLen-start_add;

    // 과제에서 result 부분, gray scale한거 저장할 메모리 할당
    cudaMalloc((void**)&d_gray,size); 
    
    // 원본 사진을 d_gray에 옮긴다. 원본으로 gray를 만들 수 있다.
    cudaMemcpy(d_gray,buf+start_add,size,cudaMemcpyHostToDevice); 
  
    // launch kernel function
    grayScaleKernel <<< 782, 1024>>>(d_gray); 
     
    //GPU에서 작업한 gray 호스트로 전달.
    cudaMemcpy(gray+start_add,d_gray,size,cudaMemcpyDeviceToHost);
}

__global__ void Noise_Reduction_Kernel(uint8_t *gaussian,uint8_t *gray){
    
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    //global index
    int global_idx = row * d_WIDTH + col;

 
    ///---------- 블럭 안의 스레드들이 공유할 픽셀 정보----------
   	extern __shared__ uint8_t pixcel[];

    pixcel[threadIdx.y*blockDim.x + threadIdx.x] = gray[global_idx*3];
    __syncthreads(); // 모든 스레드들이 픽셀 배열에 값을 넣을 때까지 대기

    float v =0;
    
    // shared memory로만 계산이 가능한 경우
    if( (threadIdx.x >= 2 && threadIdx.x <=blockDim.x - 3) && (threadIdx.y>=2 && threadIdx.y <= blockDim.y -3) )
    {
        //GaussianBlur
        //25번 반복정도는 스레드가 혼자 하기.
        for(int i =-2; i<=2; i++ ){
            for(int j= -2;j<=2 ;j++){
               v+= pixcel[ ((i+threadIdx.y)*(blockDim.x)+(j+threadIdx.x))] * filter[(i+2)*5+(j+2)];    
            }
        }
        gaussian[global_idx* 3] = v;
        gaussian[global_idx* 3 +1] = v;
        gaussian[global_idx* 3 +2] = v;
    }
    
    // global memory를 써야하는 경우와 제로 패딩한 값이 필요할 때,
    else{
         for(int i =-2; i<=2; i++ ){
            for(int j= -2;j<=2 ;j++){
                if( i+row < 0 || i+row >=d_HEIGHT || j+col <0 || j+col>=d_WIDTH){
                     //제로 패딩인 경우 0을 더함
                     v+=0.0; 
                }
                else if(i+threadIdx.y < 0 || i+threadIdx.y>=blockDim.y || j+threadIdx.x <0 || j+threadIdx.x>=blockDim.x){
                     //쉐어드 메모리로 접근 불가한 경우, 글로벌 메모리에서 해결
                     v+= gray[ ((i+row)*(d_WIDTH)+(j+col)) *3] * filter[(i+2)*5+(j+2)];    
                }
                else{
                    //쉐어드 메모리에서 해결 가능한 경우
                    v+= pixcel[ ((i+threadIdx.y)*(blockDim.x)+(j+threadIdx.x))] * filter[(i+2)*5+(j+2)];    
                }
            }
        }
          gaussian[global_idx* 3] = v;
          gaussian[global_idx* 3 +1] = v;
          gaussian[global_idx* 3 +2] = v;
    }
}

void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian) {

    //---------- width와 height를 컨스턴트 메모리에 올린다.---------- 
    cudaMemcpyToSymbol(d_WIDTH,&width, sizeof(int));
    cudaMemcpyToSymbol(d_HEIGHT,&height, sizeof(int));
    
    
   //---------- filter를 계산.  25번이니깐 cpu에서 돌렸다.----------  
    float h_filter[25] = {0}; 
	float sigma = 1.0;
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			h_filter[(i + 2) * 5 + j + 2]
				= (1 / (2 * 3.14* sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
		}
	}

    //---------- 필터를 컨스턴트 메모리에 올린다.---------- 
    cudaMemcpyToSymbol(filter,&h_filter, sizeof(float)*25);
    
    //----------  디바이스 메모리에 할당.----------  
    cudaMalloc((void**)&d_gaussian,width*height*3); //RGB값도 있어서 픽셀 수 * 3만큼 해준다!

    //----------  Grid, Block 차원 결정 ---------- 
    const int TILE_WIDTH  = 50;
    const int TILE_HEIGHT = 20;
    dim3 dimGrid(width/TILE_WIDTH,height/TILE_HEIGHT,1); //2차원, Block 개수 (20,40), 800개
    dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,1);  // 2차원, Block안의 스레드 개수 (50,20) 1000개

    //---------- 커널 함수 실행 ----------
    Noise_Reduction_Kernel <<<dimGrid, dimBlock,TILE_HEIGHT*TILE_WIDTH*sizeof(uint8_t) >>>(d_gaussian,d_gray); // launch test function
 
    ///---------- GPU에서 작업한 가우시안 호스트로 전달.----------
    cudaMemcpy(gaussian,d_gaussian,width*height*3,cudaMemcpyDeviceToHost);
 
    //---------- 메모리해제.----------
    cudaFree(d_gray);
    cudaFree(filter);
}


__global__ void Intensity_Gradient_Kernel(uint8_t* gaussian, uint8_t* sobel, uint8_t* angle){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    //global index
    int global_idx = row * d_WIDTH + col;

    
    ///---------- 블럭 안의 스레드들이 공유할 픽셀 정보----------
    extern __shared__ uint8_t pixcel[];
    pixcel[threadIdx.y* blockDim.x + threadIdx.x] = gaussian[global_idx*3];
     __syncthreads(); // 모든 스레드들이 픽셀 배열에 값을 넣을 때까지 대기
 
	int gx = 0;
	int gy = 0;
    
    // shared memory로만 계산이 가능한 경우
    if( (threadIdx.x >= 1 && threadIdx.x <=blockDim.x -2) && (threadIdx.y>=1 && threadIdx.y <=blockDim.y-2) ){
        //9번 반복정도는 스레드가 혼자 하기.
        for(int i =-1; i<=1; i++ ){
            for(int j= -1;j<=1 ;j++){
                gy += pixcel[ ((i+threadIdx.y)*(blockDim.x)+(j+threadIdx.x))] * filter_y[(i+1)*3+(j+1)];
                gx += pixcel[ ((i+threadIdx.y)*(blockDim.x)+(j+threadIdx.x))] * filter_x[(i+1)*3+(j+1)];    
            }
        }
    }
    else{
        //9번 반복정도는 스레드가 혼자 하기.
        for(int i =-1; i<=1; i++ ){
            for(int j= -1;j<=1 ;j++){
                if( i+row < 0 || i+row >=d_HEIGHT || j+col <0 || j+col>=d_WIDTH){
                    //제로 패딩인 경우 0을 더함
                    gy+=0;
                    gx+=0;
                }
                else if(i+threadIdx.y < 0 || i+threadIdx.y>=blockDim.y || j+threadIdx.x <0 || j+threadIdx.x>=blockDim.x){
                    //쉐어드 메모리로 접근 불가한 경우, 글로벌 메모리에서 해결
                    float g =gaussian[ ( (row + i) * (d_WIDTH) + col + j)*3];
                    gy += (int)g * filter_y[(i+1) * 3 + (j+1)];
                    gx += (int)g * filter_x[(i+1) * 3 + (j+1)];
                }
                else{
                    //쉐어드 메모리에서 해결 가능한 경우
                    gy += pixcel[ ((i+threadIdx.y)*(blockDim.x)+(j+threadIdx.x))] * filter_y[(i+1)*3+(j+1)];
                    gx += pixcel[ ((i+threadIdx.y)*(blockDim.x)+(j+threadIdx.x))] * filter_x[(i+1)*3+(j+1)];    
                }
            }
        }
    }
   
    int t = sqrt(gx * gx + gy * gy);

	uint8_t  v = 0;
	if (t > 255) {
		v = 255;
	}
	else
		v = t;
    
    sobel[global_idx * 3] = v;
	sobel[global_idx * 3 + 1] = v;
	sobel[global_idx * 3 + 2] = v;
	
    float t_angle = 0;
    if(gy != 0 || gx != 0) 
        t_angle= (float)atan2(gy, gx) * 180.0 / 3.14;
    if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
        angle[global_idx] = 0;
    else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
        angle[global_idx] = 45;
    else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
        angle[global_idx] = 90;
    else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
        angle[global_idx] = 135;
}

void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle){
    //---------- filter를 정의 ----------  
    int c_filter_x[9] = {-1,0,1
						,-2,0,2
						,-1,0,1};
	int c_filter_y[9] = {1,2,1
						,0,0,0
						,-1,-2,-1};

    //---------- 필터를 컨스턴트 메모리에 올린다.---------- 
    cudaMemcpyToSymbol(filter_x,&c_filter_x, sizeof(int)*9);
    cudaMemcpyToSymbol(filter_y,&c_filter_y, sizeof(int)*9);                    
    

    //----------  디바이스 메모리에 할당.----------    
    cudaMalloc((void**)&d_sobel,width*height*3); // *과제에서 source 부분, 원본 이미지 할당.
    cudaMalloc((void**)&d_angle,width*height);

    //---------- 디바이스에 gaussian 이미지 복사 ----------
    cudaMemcpy(d_gaussian,gaussian,width*height*3,cudaMemcpyHostToDevice); // source 부분을 GPU에 옮긴다.
    
    
    //----------  Grid, Block 차원 결정 ---------- 
    const int TILE_WIDTH  = 50;
    const int TILE_HEIGHT = 20;
    dim3 dimGrid(width/TILE_WIDTH,height/TILE_HEIGHT,1); //2차원, Block 개수 (20,40), 800개
    dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,1);  // 2차원, Block안의 스레드 개수 (50,20) 1000개

    //---------- 커널 함수 실행 ----------
    Intensity_Gradient_Kernel<<< dimGrid,dimBlock,TILE_HEIGHT*TILE_WIDTH*sizeof(uint8_t) >>>(d_gaussian,d_sobel,d_angle);

    ///---------- GPU에서 작업한 sobel과 angle 호스트로 전달.----------
    cudaMemcpy(sobel,d_sobel,width*height*3,cudaMemcpyDeviceToHost);
    cudaMemcpy(angle,d_angle,width*height,cudaMemcpyDeviceToHost);
    
    //---------- 메모리해제.----------
    cudaFree(d_gaussian);
    cudaFree(filter_x);
    cudaFree(filter_y);
    
}


__global__ void Non_maximum_Suppression_Kernel(uint8_t* angle,uint8_t* sobel, uint8_t* suppression_pixel,int * min, int * max){
    
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // 맨 위, 맨 아래, 맨 왼쪽, 맨 오른쪽인 경우에는 연산하지 않는다.
    if(row==0 || row == d_HEIGHT-1||col==0 || col==d_WIDTH-1){
        return;
    }
    
    //---------- 블럭 안의 스레드들이 공유할 v값 min, max를 결정하기 위함----------
    extern __shared__ uint8_t value_v[];
    
    //global index
    int global_idx = row * d_WIDTH + col;

	uint8_t p1 = 0;
	uint8_t p2 = 0;

    // 조금이라도 글로벌 메모리 접근을 줄이고자
    // 한번 접근해서 레지스터에 저장. 이 값으로 if문 조건 판별
    uint8_t tmp_angle = angle[global_idx]; 

    if (tmp_angle == 0) {
            p1 = sobel[((row+1) * d_WIDTH + col)*3];
            p2 = sobel[((row-1) * d_WIDTH + col) * 3];
    }
    else if (tmp_angle == 45) {
        p1 = sobel[((row + 1) * d_WIDTH + col-1) * 3];
        p2 = sobel[((row - 1) * d_WIDTH + col+1) * 3];
    }
    else if (tmp_angle == 90) {
        p1 = sobel[((row) * d_WIDTH + col+1) * 3];
        p2 = sobel[((row) * d_WIDTH + col-1) * 3];
    }
    else {
        p1 = sobel[((row + 1) * d_WIDTH + col+1) * 3];
        p2 = sobel[((row - 1) * d_WIDTH + col-1) * 3];
    }
    
    uint8_t v = sobel[(row * d_WIDTH + col) * 3];
    
    if ((v >= p1) && (v >= p2)) {
        suppression_pixel[(row * d_WIDTH + col) * 3] = v;
        suppression_pixel[(row * d_WIDTH + col) * 3 + 1] = v;
        suppression_pixel[(row * d_WIDTH + col) * 3 + 2] = v;
    }
      
    value_v[threadIdx.y*blockDim.y+threadIdx.x] = v;
    __syncthreads(); // 모든 스레드들이 v값을 채울 때 까지 대기
   

    //첫번째 스레드에서 한 블럭의 min, max 결정
    if(threadIdx.x == 0){
        int t_min = 255;
        int t_max = 0;
        for (int i = 0; i<blockDim.x*blockDim.y; i++){
            if(t_min > value_v[i])
                t_min=v;
            if(t_max < value_v[i])
                t_max =v;
        }
        //모든 블럭에 대해서 min, max 결정
        atomicMin(min,t_min);
        atomicMax(max,t_max);
    }
}

void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max){
    
    // 디바이스에서 min, max를 가리키는 포인터
    int * d_min;
    int * d_max;

    //----------  디바이스 메모리에 할당.----------    
    cudaMalloc((void**)&d_suppression_pixel,width*height*3);
    cudaMalloc((void**)&d_min, sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(int));
    
    //전달 받은 min max값 임시로 저장.
    int g_min = min;
    int g_max = max;
    
    //----------  Grid, Block 차원 결정 ----------
    // Block안의 스레드 개수를 적게 하는 것이 시간이 더 짧게 걸렸다.
    // 아무래도 min, max를 비교할 때 요구되는 반복문의 횟수가 짧아져서 그런듯 깊다. 
    const int TILE_WIDTH  = 10;
    const int TILE_HEIGHT = 10;

    dim3 dimGrid(width/TILE_WIDTH,height/TILE_HEIGHT,1); //2차원, Block 개수 1000개
    dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,1);  // 2차원, Block안의 스레드 개수 800개
    
    //---------- 커널 함수 실행 ----------
    Non_maximum_Suppression_Kernel<<< dimGrid,dimBlock,TILE_HEIGHT*TILE_WIDTH*sizeof(uint8_t) >>>(d_angle,d_sobel,d_suppression_pixel,d_min,d_max);

    //---------- GPU에서 작업한 것 호스트로 전달.----------
    cudaMemcpy(suppression_pixel,d_suppression_pixel,width*height*3,cudaMemcpyDeviceToHost);
    cudaMemcpy(&g_min,d_min,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&g_max,d_max,sizeof(int),cudaMemcpyDeviceToHost);
   
    //min, max값 업데이트
     min =g_min;
     max =g_max;
    
    //---------- 메모리해제.----------
    cudaFree(d_sobel);
    cudaFree(d_angle);
}


__global__ void Save_Temp_Hystersis_Kernel(uint8_t* suppression_pixel,uint8_t* tmp_hysteresis){
 
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
   
    //global index
    int global_idx = row *d_WIDTH + col;
   
    uint8_t v = suppression_pixel[global_idx*3];
    if (v < low_t) { // 버려
            tmp_hysteresis[global_idx]=0;
        }
        else if (v < high_t) { //어중간~
            tmp_hysteresis[global_idx]=123;
        }
        else { //무조건 edge로 판별.
            tmp_hysteresis[global_idx]=255;
        }
}

__global__ void Hysteresis_Thresholding_Kernel(uint8_t* hysteresis, uint8_t* tmp_hysteresis){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
   
    //global index
    int global_idx = row * d_WIDTH + col;

    //반복문을 빠져나오기 위함
    bool loop_out =false;
    
    // 255인곳은 255로 저장
    if (tmp_hysteresis[global_idx]==255){
        hysteresis[global_idx*3] = 255;
		hysteresis[global_idx * 3+1] = 255;
		hysteresis[global_idx * 3+2] = 255;
    }
    // 123인 곳은 판별
    else if(tmp_hysteresis[global_idx] == 123){   
        for (int i = row-1; i < row+2; i++) {
		    for (int j = col-1; j < col+2; j++) {
			    if ((i < d_HEIGHT && j < d_WIDTH) && (i >= 0 && j >= 0)) {
				    if (tmp_hysteresis[(i * d_WIDTH + j)] == 255) {
					    hysteresis[global_idx*3] = 255;
					    hysteresis[global_idx * 3+1] = 255;
					    hysteresis[global_idx * 3+2] = 255;
                        loop_out=true; // 255인 곳 한번 발견하면 바로 반복문을 나온다.
                        break;
				    }
			    }
            if(loop_out)
                break;
		    }
	    }
    }
    // 255가 아닌 곳은 0으로
    if (hysteresis[global_idx* 3] != 255) {
				hysteresis[global_idx * 3] = 0;
				hysteresis[global_idx* 3+1] = 0;
				hysteresis[global_idx * 3+2] = 0;
	}
}

void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max) {
    //----------- 전달 받은 min과 max로 low_t와 hiht_t 구하기 ----------- 
    uint8_t diff = max - min;
	uint8_t c_low_t = min + diff * 0.01;
	uint8_t c_high_t = min + diff * 0.2;

    //---------- 필터를 컨스턴트 메모리에 올린다.---------- 
    cudaMemcpyToSymbol(low_t,&c_low_t, sizeof(uint8_t));
    cudaMemcpyToSymbol(high_t,&c_high_t, sizeof(uint8_t));
    

    //----------- 디바이스 메모리를 가리킬 변수 선언 ----------- 
    uint8_t * d_hysteresis; //hysteresis, 최종 결과 이미지
    uint8_t * d_tmp_hysteresis; //중간 저장하는 이미지

   //----------  디바이스 메모리에 할당.----------    
    cudaMalloc((void**)&d_hysteresis,width*height*3); // RGB값이라 *3 해줘야함.
    cudaMalloc((void**)&d_tmp_hysteresis,width*height); // RGB값 필요 없어서 weight랑 height만
    
    //----------  Grid, Block 차원 결정 ---------- 
    const int TILE_WIDTH  = 50;
    const int TILE_HEIGHT = 20;
    dim3 dimGrid(width/TILE_WIDTH,height/TILE_HEIGHT,1); 
    dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,1);  
    
    //---------- 커널 함수 실행 ---------- 
    Save_Temp_Hystersis_Kernel<<< dimGrid,dimBlock>>>(d_suppression_pixel,d_tmp_hysteresis);
    //---------- 커널 함수 실행 ----------
    Hysteresis_Thresholding_Kernel<<< dimGrid,dimBlock >>>(d_hysteresis,d_tmp_hysteresis);
    
    //---------- GPU에서 작업한 것 호스트로 전달.----------
    cudaMemcpy(hysteresis,d_hysteresis,width*height*3, cudaMemcpyDeviceToHost);
    
    //---------- 메모리해제.----------
    cudaFree(d_suppression_pixel);
    cudaFree(d_hysteresis);
    cudaFree(d_tmp_hysteresis);
}