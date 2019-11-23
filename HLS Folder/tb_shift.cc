
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include "net_hls_shift.h"


float image[3][176][320];

//first dw conv block --> 3x3x3--32x1x1
float conv1_wt_3x3_in[16][3][3];
float conv1_bias_3x3_in[16];
float conv1_wt_3x3[3][3][3];
float conv1_bias_3x3[3];

float conv1_wt_1x1_in[32][16];
float conv1_wt_1x1_reorder[2][16][16];
float conv1_wt_1x1[32][3];

//second dw conv block --> 32x3x3--64x1x1
float conv2_wt_3x3[32][3][3];

float conv2_wt_1x1[64][32];

//third dw conv block --> 64x3x3--128x1x1
float conv3_wt_3x3[64][3][3];

float conv3_wt_1x1[128][64];
float conv3_wt_1x1_reorder[32][16][16];


float conv4_wt_1x1[256][128];
float conv4_wt_1x1_reorder[128][16][16];

//final block --> 256x1x1--10x1x1
float conv5_wt_1x1[10][256];
float conv5_wt_1x1_in[16][256];
float conv5_wt_1x1_reorder[16][16][16];


//fixed point weight

FIX_WT fix_conv_1_weight_in[16][3][3];
FIX_WT fix_conv_1_bias_in[16];

FIX_WT fix_conv_2_weight_in[32][16];
FIX_WT fix_conv_2_weight_reorder[2][16][16];

FIX_WT fix_conv_4_weight_in[32][3][3];

FIX_WT fix_conv_5_weight_in[64][32];
FIX_WT fix_conv_5_weight_reorder[8][16][16];

FIX_WT fix_conv_7_weight_in[64][3][3];

FIX_WT fix_conv_8_weight_in[128][64];
FIX_WT fix_conv_8_weight_reorder[32][16][16];

FIX_WT fix_conv_11_weight_in[256][128];
FIX_WT fix_conv_11_weight_reorder[128][16][16];

FIX_WT fix_conv_12_weight_tmp[10][256];
FIX_WT fix_conv_12_weight_in[16][256];
FIX_WT fix_conv_12_weight_reorder[16][16][16];


//2+8+32+128+16=186
FIX_16_1 fix_conv_weight_1x1_all[186][16][16];

//1+2+4=7
FIX_16_1 fix_conv_weight_3x3_all[7][16][3][3];

//1
FIX_16_1 fix_bias_all[16];


FIX_FM DDR_pool_3_out_PL[32][90][162];		/// DDR storage for pool3 output with padding
FIX_FM DDR_pool_6_out_PL[64][46][82];		/// DDR storage for pool6 output with padding

FIX_FM DDR_pool_out_PL[64][90][162];	/// DDR Storage for pooling layers' output


uint8  fix_image_raw[3][176][320];	// 0~255 RGB raw data
uint8  fix_image_raw_pad[3][178][322];	// 0~255 RGB raw data


FIX_FM DDR_buf[36][16][24][42];


FIX_32_25 my_exp_fix(FIX_FM input);

void test_model();
void reorder_weight_fix();



int test_one_frame( char* filename )
{
    std::ifstream ifs_param("weights_shift.bin", std::ios::in | std::ios::binary);

    ///////////// Prepare Image //////////////////////
    std::ifstream ifs_image_raw(filename, std::ios::in | std::ios::binary);
    ifs_image_raw.read((char*)(**fix_image_raw), 3*176*320*sizeof(uint8));


    ///////////////// PADDING FOR RAW IMAGE ///////////
    for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 178; j++) {
			for(int k = 0; k < 322; k++) {
				if(j==0 || k==0 || j==177 || k==321) {
					fix_image_raw_pad[i][j][k] = 127;
				}
				else {
					fix_image_raw_pad[i][j][k] = fix_image_raw[i][j-1][k-1];
				}
			}
		}
    }

    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 176; j++) {
		for(int k = 0; k < 320; k++) {
			image[0][j][k] = (((fix_image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((fix_image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((fix_image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}


    //std::cout << image[0][0][0] << " " << image[1][0][0] << " " << image[2][0][0] << std::endl;

    ///////////// Read Weights ///////////////////////
    ifs_param.read((char*)(**conv1_wt_3x3), 3*3*3*sizeof(float));
    ifs_param.read((char*)conv1_bias_3x3, 3*sizeof(float));
    ifs_param.read((char*)(*conv1_wt_1x1), 32*3*sizeof(float));

    ifs_param.read((char*)(**conv2_wt_3x3), 32*3*3*sizeof(float));
    ifs_param.read((char*)(*conv2_wt_1x1), 64*32*sizeof(float));

    ifs_param.read((char*)(**conv3_wt_3x3), 64*3*3*sizeof(float));
    ifs_param.read((char*)(*conv3_wt_1x1), 128*64*sizeof(float));

    ifs_param.read((char*)(*conv4_wt_1x1), 256*128*sizeof(float));

    ifs_param.read((char*)(*conv5_wt_1x1), 10*256*sizeof(float));
    ifs_param.close();


    /////// GOLDEN MODEL ///////////
    printf("Computing test Model...\n");
    test_model();

    reorder_weight_fix();


    float predict_box[5];

    model_shift(fix_image_raw_pad,

    		fix_conv_weight_1x1_all,
    		fix_conv_weight_3x3_all,
			  fix_bias_all,

			  DDR_pool_3_out_PL,
			  DDR_pool_6_out_PL,

			  DDR_buf,

			  predict_box

			);


    FILE* fp;
    fp=fopen("golden_model_output","r");
    if (fp == NULL)
        {
            printf("no such file.");
            return 1;
        }
    int tmp;
    int golden_output[4];
    for (int i=0; i<4; i++){
    	fscanf(fp, "%d", &tmp);
    	golden_output[i] = tmp;
    }
    fclose(fp);

	int coord[4];
	predict_box[0] = predict_box[0] / 40;
	predict_box[1] = predict_box[1] / 22;
	predict_box[2] = predict_box[2] / 40;
	predict_box[3] = predict_box[3] / 22;

	coord[0] = (unsigned int)(((predict_box[0] - predict_box[2]/2.0) * 640));
	coord[1] = (unsigned int)(((predict_box[1] - predict_box[3]/2.0) * 360));
	coord[2] = (unsigned int)(((predict_box[0] + predict_box[2]/2.0) * 640));
	coord[3] = (unsigned int)(((predict_box[1] + predict_box[3]/2.0) * 360));

	int retval=0;

    for (int i=0; i<4; i++){
    	printf("%d\n", coord[i]);
    }
	for(int i = 0; i < 4; i++){
		if (abs(golden_output[i]-coord[i])>10){
			retval = 1;
			printf("does not match, expected %d, got %d\n", golden_output[i], coord[i]);
			return retval;
		}

	}

	printf("all results are expected");
    return retval;
}



int main()
{

	int new_ret = 0;
	printf("1.bin\n");
	new_ret = test_one_frame("1.bin");

  // printf("2.bin\n");
	// new_ret = test_one_frame("2.bin");

	return new_ret;

}
