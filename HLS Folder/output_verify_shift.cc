
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include "net_hls_shift.h"

#define EPSILON	1e-04
//#define EPSILON	0.5f

#define conv1_3 3 //filter number for first conv 3x3 kernel
#define conv2_3 32 //filter number for second conv 3x3 kernel
#define conv3_3 64 //filter number for third conv 3x3 kernel
#define conv4_3 128 //filter number for forth conv 3x3 kernel
#define conv1_1 32 //filter number for first conv 1x1 kernel
#define conv2_1 64 //filter number for second conv 1x1 kernel
#define conv3_1 128 //filter number for third conv 1x1 kernel
#define conv4_1 256 //filter number for forth conv 1x1 kernel
#define conv5_1 10 //filter number for last conv 1x1 kernel

#define img_row 176
#define img_col 320



extern float image[3][img_row][img_col]; //define image variable

///define weight variables////

//first dw conv block --> 3x3x3--32x1x1
extern  float conv1_wt_3x3[conv1_3][3][3];
extern  float conv1_bias_3x3[conv1_3];

extern  float conv1_wt_1x1[conv1_1][conv1_3];

//second dw conv block --> 32x3x3--64x1x1
extern  float conv2_wt_3x3[conv2_3][3][3];

extern  float conv2_wt_1x1[conv2_1][conv2_3];

//third dw conv block --> 64x3x3--128x1x1
extern  float conv3_wt_3x3[conv3_3][3][3];

extern  float conv3_wt_1x1[conv3_1][conv3_3];

//forth dw conv block --> 128x3x3--256x1x1

extern  float conv4_wt_1x1[conv4_1][conv4_3];

//final block --> 256x1x1--10x1x1
extern  float conv5_wt_1x1[conv5_1][conv4_1];


/// define feature map variable to PL --> cnn output ///

float conv1_3x3_out_PL[conv1_3][img_row][img_col];
float conv1_1x1_out_PL[conv1_1][img_row][img_col];
float pool1_out_PL[conv1_1][img_row/2][img_col/2];

float conv2_3x3_out_PL[conv2_3][img_row/2][img_col/2];
float conv2_1x1_out_PL[conv2_1][img_row/2][img_col/2];
float pool2_out_PL[conv2_1][img_row/4][img_col/4];

float conv3_3x3_out_PL[conv3_3][img_row/4][img_col/4];
float conv3_1x1_out_PL[conv3_1][img_row/4][img_col/4];
float pool3_out_PL[conv3_1][img_row/8][img_col/8];

float shift_out_PL[conv4_3][img_row/8][img_col/8];
float conv4_1x1_out_PL[conv4_1][img_row/8][img_col/8];
float conv5_1x1_out_PL[conv5_1][img_row/8][img_col/8];


/// define feature map variable for verification --> cnn output ///

float conv1_3x3_out[conv1_3][img_row][img_col];
float conv1_1x1_out[conv1_1][img_row][img_col];
float pool1_out[conv1_1][img_row/2][img_col/2];

float conv2_3x3_out[conv2_3][img_row/2][img_col/2];
float conv2_1x1_out[conv2_1][img_row/2][img_col/2];
float pool2_out[conv2_1][img_row/4][img_col/4];

float conv3_3x3_out[conv3_3][img_row/4][img_col/4];
float conv3_1x1_out[conv3_1][img_row/4][img_col/4];
float pool3_out[conv3_1][img_row/8][img_col/8];

float shift_out[conv4_3][img_row/8][img_col/8];
float conv4_1x1_out[conv4_1][img_row/8][img_col/8];
float conv5_1x1_out[conv5_1][img_row/8][img_col/8];


using namespace std;

FILE* fout;

//finds max value from 4 inputs
float max_2x2(float a1, float a2, float a3, float a4)
{
  float temp1, temp2;
  temp1 = (a1>a2)? a1:a2;
  temp2 = (a3>a4)? a3:a4;
  return (temp1>temp2)? temp1:temp2;
}

// first dw conv block --> conv3x3+relu-->conv1x1+relu-->maxpool//
void conv1_3x3(
              float input[conv1_3][img_row][img_col],
              float weight[conv1_3][3][3],
              float bias[conv1_3],
              float output[conv1_3][img_row][img_col]
              )
{
  // input wxhxch, output wxhxch, kernel mxn
  for (int ch = 0; ch < conv1_3; ch++){
    for (int h = 0; h < img_row; h++){
      for (int w = 0;  w < img_col; w++){
        float sum = 0;
        for (int m = 0; m < 3; m++){
          for (int n = 0; n < 3; n++){
            sum+=weight[ch][m][n]* ((h+m-1>=0 && w+n-1>=0 && h+m-1<img_row && w+n-1<img_col) ? input[ch][h+m-1][w+n-1] : 0);
          }
        }
        float result = sum + bias[ch];
        output[ch][h][w] = (result>0)? result : 0.0f; // relu activation
      }
    }
  }

  //write result
  fout = fopen("conv1_3_out","w");
  for (int i=0; i<conv1_3; i++){
    for (int j=0; j<img_row; j++){
      for (int k=0; k<img_col; k++){
        fprintf(fout, "conv1_3_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
      }
    }
  }
  fclose(fout);

}



void conv1_1x1(
            float input[conv1_3][img_row][img_col],
            float weight[conv1_1][conv1_3],
            float output[conv1_1][img_row][img_col]
            )
{
  // input wxhxchi, output wxhxcho, kernel mxn
    for(int cho = 0; cho < conv1_1; cho++) {
        for(int h = 0; h < img_row; h++) {
            for(int w = 0; w < img_col; w++) {
                float sum = 0;
                for(int chi = 0; chi < conv1_3; chi++ ) {
                    sum += weight[cho][chi] * input[chi][h][w];
                }
                output[cho][h][w] = (sum > 0)? sum : 0.0f;
            }
        }
    }

    fout = fopen("conv1_1x1_out", "w");
    for(int i = 0; i < conv1_1; i++) {
        for(int j = 0; j < img_row; j++) {
            for(int k = 0; k < img_col; k ++) {
                fprintf(fout, "conv1_1x1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);
}

void max_pool_1(
                   float input[conv1_1][img_row][img_col],
                   float output[conv1_1][img_row/2][img_col/2]
                )
{
    //cout << "max_pool_3..." << endl;

    for(int ch = 0; ch < conv1_1; ch++) {
        for(int h = 0; h < img_row/2; h++) {
            for(int w = 0; w < img_col/2; w++) {

                output[ch][h][w] = max_2x2(
                                        input[ch][h*2][w*2],
                                        input[ch][h*2+1][w*2],
                                        input[ch][h*2][w*2+1],
                                        input[ch][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fout = fopen("max_pool_1_out", "w");
    for(int i = 0; i < conv1_1; i++) {
        for(int j = 0; j < img_row/2; j++) {
            for(int k = 0; k < img_col/2; k ++) {
                fprintf(fout, "max_pool_1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);

}

// second dw conv block --> conv3x3+relu-->conv1x1+relu-->maxpool//
void conv2_3x3(
              float input[conv2_3][img_row/2][img_col/2],
              float weight[conv2_3][3][3],
              float output[conv2_3][img_row/2][img_col/2]
              )
{
  // input wxhxch, output wxhxch, kernel mxn
  for (int ch = 0; ch < conv2_3; ch++){
    for (int h = 0; h < img_row/2; h++){
      for (int w = 0;  w < img_col/2; w++){
        float sum = 0;
        for (int m = 0; m < 3; m++){
          for (int n = 0; n < 3; n++){
            sum+=weight[ch][m][n]* ((h+m-1>=0 && w+n-1>=0 && h+m-1<img_row/2 && w+n-1<img_col/2) ? input[ch][h+m-1][w+n-1] : 0);
          }
        }
        output[ch][h][w] = (sum>0)? sum : 0.0f; // relu activation
      }
    }
  }

  //write result
  fout = fopen("conv2_3_out","w");
  for (int i=0; i<conv2_3; i++){
    for (int j=0; j<img_row/2; j++){
      for (int k=0; k<img_col/2; k++){
        fprintf(fout, "conv2_3_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
      }
    }
  }
  fclose(fout);

}



void conv2_1x1(
            float input[conv2_3][img_row/2][img_col/2],
            float weight[conv2_1][conv2_3],
            float output[conv2_1][img_row/2][img_col/2]
            )
{
  // input wxhxchi, output wxhxcho, kernel mxn
    for(int cho = 0; cho < conv2_1; cho++) {
        for(int h = 0; h < img_row/2; h++) {
            for(int w = 0; w < img_col/2; w++) {
                float sum = 0;
                for(int chi = 0; chi < conv2_3; chi++ ) {
                    sum += weight[cho][chi] * input[chi][h][w];
                }
                output[cho][h][w] = (sum > 0)? sum : 0.0f;
            }
        }
    }

    fout = fopen("conv2_1x1_out", "w");
    for(int i = 0; i < conv2_1; i++) {
        for(int j = 0; j < img_row/2; j++) {
            for(int k = 0; k < img_col/2; k ++) {
                fprintf(fout, "conv2_1x1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);
}

void max_pool_2(
                   float input[conv2_1][img_row/2][img_col/2],
                   float output[conv2_1][img_row/4][img_col/4]
                )
{
    //cout << "max_pool_3..." << endl;

    for(int ch = 0; ch < conv2_1; ch++) {
        for(int h = 0; h < img_row/4; h++) {
            for(int w = 0; w < img_col/4; w++) {

                output[ch][h][w] = max_2x2(
                                        input[ch][h*2][w*2],
                                        input[ch][h*2+1][w*2],
                                        input[ch][h*2][w*2+1],
                                        input[ch][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fout = fopen("max_pool_2_out", "w");
    for(int i = 0; i < conv2_1; i++) {
        for(int j = 0; j < img_row/4; j++) {
            for(int k = 0; k < img_col/4; k ++) {
                fprintf(fout, "max_pool_2_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);

}


// third dw conv block --> conv3x3+relu-->conv1x1+relu-->maxpool//
void conv3_3x3(
              float input[conv3_3][img_row/4][img_col/4],
              float weight[conv3_3][3][3],
              float output[conv3_3][img_row/4][img_col/4]
              )
{
  // input wxhxch, output wxhxch, kernel mxn
  for (int ch = 0; ch < conv3_3; ch++){
    for (int h = 0; h < img_row/4; h++){
      for (int w = 0;  w < img_col/4; w++){
        float sum = 0;
        for (int m = 0; m < 3; m++){
          for (int n = 0; n < 3; n++){
            sum+=weight[ch][m][n]* ((h+m-1>=0 && w+n-1>=0 && h+m-1<img_row/4 && w+n-1<img_col/4) ? input[ch][h+m-1][w+n-1] : 0);
          }
        }
        output[ch][h][w] = (sum>0)? sum : 0.0f; // relu activation
      }
    }
  }

  //write result
  fout = fopen("conv3_3_out","w");
  for (int i=0; i<conv3_3; i++){
    for (int j=0; j<img_row/4; j++){
      for (int k=0; k<img_col/4; k++){
        fprintf(fout, "conv3_3_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
      }
    }
  }
  fclose(fout);

}



void conv3_1x1(
            float input[conv3_3][img_row/4][img_col/4],
            float weight[conv3_1][conv3_3],
            float output[conv3_1][img_row/4][img_col/4]
            )
{
  // input wxhxchi, output wxhxcho, kernel mxn
    for(int cho = 0; cho < conv3_1; cho++) {
        for(int h = 0; h < img_row/4; h++) {
            for(int w = 0; w < img_col/4; w++) {
                float sum = 0;
                for(int chi = 0; chi < conv3_3; chi++ ) {
                    sum += weight[cho][chi] * input[chi][h][w];
                }
                output[cho][h][w] = (sum > 0)? sum : 0.0f;
            }
        }
    }

    fout = fopen("conv3_1x1_out", "w");
    for(int i = 0; i < conv3_1; i++) {
        for(int j = 0; j < img_row/4; j++) {
            for(int k = 0; k < img_col/4; k ++) {
                fprintf(fout, "conv3_1x1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);
}

void max_pool_3(
                   float input[conv3_1][img_row/4][img_col/4],
                   float output[conv3_1][img_row/8][img_col/8]
                )
{
    //cout << "max_pool_3..." << endl;

    for(int ch = 0; ch < conv3_1; ch++) {
        for(int h = 0; h < img_row/8; h++) {
            for(int w = 0; w < img_col/8; w++) {

                output[ch][h][w] = max_2x2(
                                        input[ch][h*2][w*2],
                                        input[ch][h*2+1][w*2],
                                        input[ch][h*2][w*2+1],
                                        input[ch][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fout = fopen("max_pool_3_out", "w");
    for(int i = 0; i < conv3_1; i++) {
        for(int j = 0; j < img_row/8; j++) {
            for(int k = 0; k < img_col/8; k ++) {
                fprintf(fout, "max_pool_3_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);

}



// shift block
void shift(
              float input[conv4_3][img_row/8][img_col/8],
              float output[conv4_3][img_row/8][img_col/8]
              )
{
  // input wxhxch, output wxhxch, kernel mxn
  int grp = 14; //(conv4_3/9)*9
  for (int g=0; g<9; g++){
    if (g==8) grp = 16;
    for (int ch = 0; ch < grp; ch++){
      for (int h = 0; h < img_row/8; h++){
        for (int w = 0;  w < img_col/8; w++){
          switch(g){
            case 2:
            output[g*14+ch][h][w] = (w>0)? input[g*14+ch][h][w-1] : 0;
            break;
            case 3:
            output[g*14+ch][h][w] = (w!=39)? input[g*14+ch][h][w+1] : 0;
            break;
            case 0:
            output[g*14+ch][h][w] = (h>0)? input[g*14+ch][h-1][w] : 0;
            break;
            case 1:
            output[g*14+ch][h][w] = (h!=21)? input[g*14+ch][h+1][w] : 0;
            break;
            case 4:
            output[g*14+ch][h][w] = (w>0 && h>0)? input[g*14+ch][h-1][w-1] : 0;
            break;
            case 6:
            output[g*14+ch][h][w] = (w>0 && h!=21)? input[g*14+ch][h+1][w-1] : 0;
            break;
            case 5:
            output[g*14+ch][h][w] = (w!=39 && h>0)? input[g*14+ch][h-1][w+1] : 0;
            break;
            case 7:
            output[g*14+ch][h][w] = (w!=39 && h!=21)? input[g*14+ch][h+1][w+1] : 0;
            break;
            case 8:
            output[g*14+ch][h][w] = input[g*14+ch][h][w];
            break;
          }
        }
      }
    }
  }

  //write result
  fout = fopen("shift_out","w");
  for (int i=0; i<conv4_3; i++){
    for (int j=0; j<img_row/8; j++){
      for (int k=0; k<img_col/8; k++){
        fprintf(fout, "shift_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
      }
    }
  }
  fclose(fout);

}



void conv4_1x1(
            float input[conv4_3][img_row/8][img_col/8],
            float weight[conv4_1][conv4_3],
            float output[conv4_1][img_row/8][img_col/8]
            )
{
  // input wxhxchi, output wxhxcho, kernel mxn
    for(int cho = 0; cho < conv4_1; cho++) {
        for(int h = 0; h < img_row/8; h++) {
            for(int w = 0; w < img_col/8; w++) {
                float sum = 0;
                for(int chi = 0; chi < conv4_3; chi++ ) {
                    sum += weight[cho][chi] * input[chi][h][w];
                }
                output[cho][h][w] = (sum > 0)? sum : 0.0f;
            }
        }
    }

    fout = fopen("conv4_1x1_out", "w");
    for(int i = 0; i < conv4_1; i++) {
        for(int j = 0; j < img_row/8; j++) {
            for(int k = 0; k < img_col/8; k ++) {
                fprintf(fout, "conv4_1x1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);
}

//last conv block, no relu
void conv5_1x1(
            float input[conv4_1][img_row/8][img_col/8],
            float weight[conv5_1][conv4_1],
            float output[conv5_1][img_row/8][img_col/8]
            )
{
  // input wxhxchi, output wxhxcho, kernel mxn
    for(int cho = 0; cho < conv5_1; cho++) {
        for(int h = 0; h < img_row/8; h++) {
            for(int w = 0; w < img_col/8; w++) {
                float sum = 0;
                for(int chi = 0; chi < conv4_1; chi++ ) {
                    sum += weight[cho][chi] * input[chi][h][w];
                }
                output[cho][h][w] = sum;
            }
        }
    }

    fout = fopen("conv5_1x1_out", "w");
    for(int i = 0; i < conv5_1; i++) {
        for(int j = 0; j < img_row/8; j++) {
            for(int k = 0; k < img_col/8; k ++) {
                fprintf(fout, "conv5_1x1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fout);
}

// calculate bounding box
void compute_bounding_box( float input[conv5_1][img_row/8][img_col/8])
{
	int batch_size = 1;
	int num_anchors = 2;
	int h = img_row/8;
	int w = img_col/8;

	float output[conv5_1][img_row/8][img_col/8];

    float box[4] = {1, 1.06357021727,1, 2.65376815391};

	float conf_thresh = 0.0;
	int conf_j = 0;
	int conf_m = 0;
	int conf_n = 0;

	//preprocessing anchor boxes xs and ys
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5][m][n] = 1/(1+exp(-input[j*5][m][n]))+n;
			}
		}
	}

	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
        output[j*5+1][m][n] = 1/(1+exp(-input[j*5+1][m][n]))+m;
			}
		}
	}
	//preprocessing anchor boxes ws and hs
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5+2][m][n] = exp(input[j*5+2][m][n])*box[j*2];
			}
		}
	}
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5+3][m][n] = exp(input[j*5+3][m][n])*box[j*2+1];
			}
		}
	}
	//preprocessing anchor boxes det_conf
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5+4][m][n] = 1/(1+exp(-input[j*5+4][m][n]));
			}
		}
	}

	//find the maximum num
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				if(output[j*5+4][m][n] > conf_thresh){
					conf_thresh = output[j*5+4][m][n];
					conf_j = j;
					conf_m = m;
					conf_n = n;
				}
			}
		}
	}

	//calculate the output
	float predict_box[5] = {output[conf_j*5+0][conf_m][conf_n]/w,\
		output[conf_j*5+1][conf_m][conf_n]/h,\
		output[conf_j*5+2][conf_m][conf_n]/w,\
		output[conf_j*5+3][conf_m][conf_n]/h,\
		output[conf_j*5+4][conf_m][conf_n]};

	printf("Golden Model:\n");
    printf("conf_thresh: %f, conf_j: %d, conf_m: %d, conf_n: %d\n", conf_thresh, conf_j, conf_m, conf_n);
	for(int i = 0; i < 5; i++){
		printf("%f\n",predict_box[i]);
	}

	int x1, y1, x2, y2;

	x1 = (unsigned int)(((predict_box[0] - predict_box[2]/2.0) * 640));
	y1 = (unsigned int)(((predict_box[1] - predict_box[3]/2.0) * 360));
	x2 = (unsigned int)(((predict_box[0] + predict_box[2]/2.0) * 640));
	y2 = (unsigned int)(((predict_box[1] + predict_box[3]/2.0) * 360));

	FILE* fcim;
	char* filename = "golden_model_output";
	fcim = fopen(filename, "w");
	fprintf(fcim, "%d\n", x1);
	fprintf(fcim, "%d\n", y1);
	fprintf(fcim, "%d\n", x2);
	fprintf(fcim, "%d\n", y2);
	fclose(fcim);


	printf("%d %d %d %d\n", x1, y1, x2, y2);
}



//initialize Model
void test_model()
{
	conv1_3x3(image, conv1_wt_3x3, conv1_bias_3x3, conv1_3x3_out);
	conv1_1x1(conv1_3x3_out, conv1_wt_1x1, conv1_1x1_out);
	max_pool_1(conv1_1x1_out, pool1_out);

  conv2_3x3(pool1_out, conv2_wt_3x3, conv2_3x3_out);
	conv2_1x1(conv2_3x3_out, conv2_wt_1x1, conv2_1x1_out);
	max_pool_2(conv2_1x1_out, pool2_out);

  conv3_3x3(pool2_out, conv3_wt_3x3, conv3_3x3_out);
  conv3_1x1(conv3_3x3_out, conv3_wt_1x1, conv3_1x1_out);
  max_pool_3(conv3_1x1_out, pool3_out);

  shift(pool3_out, shift_out);
  conv4_1x1(shift_out, conv4_wt_1x1, conv4_1x1_out);
  conv5_1x1(conv4_1x1_out, conv5_wt_1x1, conv5_1x1_out);

	compute_bounding_box(conv5_1x1_out);
}


void fill_output( int layer, float buf[16][24][42], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 1; j <= 22; j++) {
			for(int k = 1; k <= 40; k++) {
				switch (layer)
				{
				case 1:
					conv1_3x3_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 2:
					conv1_1x1_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 4:
					conv2_3x3_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 5:
					conv2_1x1_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 7:
					conv3_3x3_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 8:
					conv3_1x1_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 10:
					shift_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 11:
					conv4_1x1_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 12:
					conv5_1x1_out_PL[ch*16+i][col*22+j-1][row*40+k-1] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}




void fill_output_pool( int layer, float buf[16][11][20], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 11; j++) {
			for(int k = 0; k < 20; k++) {
				switch (layer)
				{
				case 3:
					pool1_out_PL[i + ch*16][j + col*11][k + row*20] = buf[i][j][k];
					break;
				case 6:
					pool2_out_PL[i + ch*16][j + col*11][k + row*20] = buf[i][j][k];
					break;
				case 9:
					pool3_out_PL[i + ch*16][j + col*11][k + row*20] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}


int PL_test_compare_layer_1()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv1_3; ch++) {
			for(int w = 0; w < img_col; w++) {
				for(int h = 0; h < img_row; h++) {
				if( abs(conv1_3x3_out_PL[ch][h][w] - conv1_3x3_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_2()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_2";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv1_1; ch++) {
		for(int w = 0; w < img_col; w++) {
			for(int h = 0; h < img_row; h++) {

				if( abs(conv1_1x1_out_PL[ch][h][w] - conv1_1x1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_3()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv1_1; ch++) {
			for(int w = 0; w < img_col/2; w++) {
				for(int h = 0; h < img_row/2; h++) {
				if( abs(pool1_out_PL[ch][h][w] - pool1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_4()
{
	FILE* fo;
	FILE* fo2;
	int pass = 1;

	char* filename = "Comp_layer_4";
	char* filename2 = "layer4_rtl_out";
	fo = fopen(filename, "w");
	fo2 = fopen(filename2, "w");

	for(int ch = 0; ch < conv2_3; ch++) {
			for(int w = 0; w < img_col/2; w++) {
				for(int h = 0; h < img_row/2; h++) {
				if( abs(conv2_3x3_out_PL[ch][h][w] - conv2_3x3_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}


	return pass;
}


int PL_test_compare_layer_5()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_5";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv2_1; ch++) {
			for(int w = 0; w < img_col/2; w++) {
				for(int h = 0; h < img_row/2; h++) {
				if( abs(conv2_1x1_out_PL[ch][h][w] - conv2_1x1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}



int PL_test_compare_layer_6()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_6";

	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv2_1; ch++) {
			for(int w = 0; w < img_col/4; w++) {
				for(int h = 0; h < img_row/4; h++) {
				if( abs(pool2_out_PL[ch][h][w] - pool2_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_7()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_7";


	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv3_3; ch++) {
			for(int w = 0; w < img_col/4; w++) {
				for(int h = 0; h < img_row/4; h++) {
				if( abs(conv3_3x3_out_PL[ch][h][w] - conv3_3x3_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_8()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_8");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv3_1; ch++) {
			for(int w = 0; w < img_col/4; w++) {
				for(int h = 0; h < img_row/4; h++) {
				if( abs(conv3_1x1_out_PL[ch][h][w] - conv3_1x1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_9()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_9");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv3_1; ch++) {
			for(int w = 0; w < img_col/8; w++) {
				for(int h = 0; h < img_row/8; h++) {
					if( abs(pool3_out_PL[ch][h][w] - pool3_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_10()
{
	FILE* fo;
	FILE* fo2;

	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_10");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv4_3; ch++) {
    for(int w = 0; w < img_col/8; w++) {
      for(int h = 0; h < img_row/8; h++) {
				if( abs(shift_out_PL[ch][h][w] - shift_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}

		fprintf(fo, "%d\n\n",ch);
	}

	  //write result
	  fo2 = fopen("shift_out_rtl","w");
	  for (int i=0; i<conv4_3; i++){
	    for (int j=0; j<img_row/8; j++){
	      for (int k=0; k<img_col/8; k++){
	        fprintf(fo2, "shift_output[%d][%d][%d] = %f\n", i, j, k, shift_out_PL[i][j][k]);
	      }
	    }
	  }
	  fclose(fo2);

	return pass;
}


int PL_test_compare_layer_11()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_11");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv4_1; ch++) {
    for(int w = 0; w < img_col/8; w++) {
      for(int h = 0; h < img_row/8; h++) {
				if( abs(conv4_1x1_out_PL[ch][h][w] - conv4_1x1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_test_compare_layer_12()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_12");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < conv5_1; ch++) {
    for(int w = 0; w < img_col/8; w++) {
      for(int h = 0; h < img_row/8; h++) {
				if( abs(conv5_1x1_out_PL[ch][h][w] - conv5_1x1_out[ch][h][w]) < EPSILON ) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}
