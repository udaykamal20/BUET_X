

#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include "net_hls_shift.h"


//first dw conv block --> 3x3x3--32x1x1
extern float conv1_wt_3x3_in[16][3][3];
extern float conv1_bias_3x3_in[16];
extern  float conv1_wt_3x3[3][3][3];
extern  float conv1_bias_3x3[3];

extern float conv1_wt_1x1_in[32][16];
extern float conv1_wt_1x1_reorder[2][16][16];
extern  float conv1_wt_1x1[32][3];

//second dw conv block --> 32x3x3--64x1x1
extern  float conv2_wt_3x3[32][3][3];

extern  float conv2_wt_1x1[64][32];

//third dw conv block --> 64x3x3--128x1x1
extern  float conv3_wt_3x3[64][3][3];

extern  float conv3_wt_1x1[128][64];
extern float conv3_wt_1x1_reorder[32][16][16];


extern  float conv4_wt_1x1[256][128];
extern float conv4_wt_1x1_reorder[128][16][16];

//final block --> 256x1x1--10x1x1
extern  float conv5_wt_1x1[10][256];
extern float conv5_wt_1x1_in[16][256];
extern float conv5_wt_1x1_reorder[16][16][16];


//fixed point weight

extern FIX_WT fix_conv_1_weight_in[16][3][3];
extern FIX_WT fix_conv_1_bias_in[16];

extern FIX_WT fix_conv_2_weight_in[32][16];
extern FIX_WT fix_conv_2_weight_reorder[2][16][16];

extern FIX_WT fix_conv_4_weight_in[32][3][3];

extern FIX_WT fix_conv_5_weight_in[64][32];
extern FIX_WT fix_conv_5_weight_reorder[8][16][16];

extern FIX_WT fix_conv_7_weight_in[64][3][3];

extern FIX_WT fix_conv_8_weight_in[128][64];
extern FIX_WT fix_conv_8_weight_reorder[32][16][16];


extern FIX_WT fix_conv_11_weight_in[256][128];
extern FIX_WT fix_conv_11_weight_reorder[128][16][16];

extern FIX_WT fix_conv_12_weight_tmp[10][256];
extern FIX_WT fix_conv_12_weight_in[16][256];
extern FIX_WT fix_conv_12_weight_reorder[16][16][16];


//2+8+32+128+16=186
FIX_WT fix_conv_weight_1x1_all_8[186][16][16];

//1+2+4=7
FIX_WT fix_conv_weight_3x3_all_8[7][16][3][3];

//1
FIX_WT fix_bias_all_8[16];

//2+8+32+128+16=186
extern FIX_16_1 fix_conv_weight_1x1_all[186][16][16];

//1+2+4=7
extern FIX_16_1 fix_conv_weight_3x3_all[7][16][3][3];

//1
extern FIX_16_1 fix_bias_all[16];


void reorder_weight_fix()
{

    std::ofstream ofs_param_write("params_shift_fix.bin", std::ios::out | std::ios::binary);



    //for conv1
    for(int j = 0; j < 3; j++) {
    	for(int k = 0; k < 3; k++) {
    		for(int i = 0; i < 16; i++) {
    			if(i < 3) {
    				conv1_wt_3x3_in[i][j][k] = conv1_wt_3x3[i][j][k];
    				conv1_bias_3x3_in[i] = conv1_bias_3x3[i];

    				//for fixed-point data
    				fix_conv_1_weight_in[i][j][k] = (FIX_WT)conv1_wt_3x3[i][j][k];
    				fix_conv_1_bias_in[i] = (FIX_WT)conv1_bias_3x3[i];


    			}
    			else {
    				conv1_wt_3x3_in[i][j][k] = 0;
    				conv1_bias_3x3_in[i] = 0;
    				fix_conv_1_weight_in[i][j][k] = 0;
    				fix_conv_1_bias_in[i] = 0;
    			}
    		}
    	}
    }




    //for conv2
    for(int i = 0; i < 32; i++) {
    	for(int j = 0; j < 16; j++) {
    		if(j < 3) {
    			conv1_wt_1x1_in[i][j] = conv1_wt_1x1[i][j];
    			fix_conv_2_weight_in[i][j] = (FIX_WT)conv1_wt_1x1[i][j];
    		}
    		else {
    			conv1_wt_1x1_in[i][j] = 0.0f;
    			fix_conv_2_weight_in[i][j] = 0;
    		}
    	}
    }

    // reorder conv2
    for(int col = 0; col < 2; col++) {
    	for(int row = 0; row < 1; row++) {
    		for(int i = 0; i < 16; i++) {
    			for(int j = 0; j < 16; j++) {
    				fix_conv_2_weight_reorder[col][i][j] = fix_conv_2_weight_in[i + 16*col][j + 16*row];
    			}
    		}
    	}
    }

    //for conv4
    for(int j = 0; j < 32; j++) {
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_4_weight_in[j][k][i] = (FIX_WT)conv2_wt_3x3[j][k][i];
			}
		}
	}

    //for conv5
    for(int j = 0; j < 64; j++) {
		for(int k = 0; k < 32; k++) {
			fix_conv_5_weight_in[j][k] = (FIX_WT)conv2_wt_1x1[j][k];

		}
	}

    // reorder conv5
    for(int col = 0; col < 4; col++) {
    	for(int row = 0; row < 2; row++) {
    		for(int i = 0; i < 16; i++) {
    			for(int j = 0; j < 16; j++) {
    				fix_conv_5_weight_reorder[col*2 + row][i][j] = fix_conv_5_weight_in[i + col*16][j + row*16];
    			}
    		}
    	}
    }


    //for conv7
	for(int j = 0; j < 64; j++) {
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_7_weight_in[j][k][i] = (FIX_WT)conv3_wt_3x3[j][k][i];
			}
		}
	}

    //for conv8
	for(int i = 0; i < 128; i++) {
		for(int j = 0; j < 64; j++) {
			fix_conv_8_weight_in[i][j] = (FIX_WT)conv3_wt_1x1[i][j];
		}
	}

	//reorder conv8
	for(int col = 0; col < 8; col++ ) {
		for(int row = 0; row < 4; row++) {
			for(int i = 0; i < 16; i++) {
				for(int j = 0; j < 16; j++) {
					fix_conv_8_weight_reorder[col*4 + row][i][j] = fix_conv_8_weight_in[i + col*16][j + row*16];
				}
			}
		}
	}


	//for conv11
	for(int i = 0; i < 256; i++) {
		for(int j = 0; j < 128; j++) {
			fix_conv_11_weight_in[i][j] = (FIX_WT)conv4_wt_1x1[i][j];
		}
	}

	//// reorder conv_11_weight
	for(int col = 0; col < 16; col++ ) {
		for(int row = 0; row < 8; row++) {
			for(int i = 0; i < 16; i++) {
				for(int j = 0; j < 16; j++) {
					fix_conv_11_weight_reorder[col*8 + row][i][j] = fix_conv_11_weight_in[i + col*16][j + row*16];
				}
			}
		}
	}

	//for conv12
	for(int ch = 0; ch < 16; ch++ ) {
		for(int i = 0; i < 256; i++) {
			if( ch < 10 ) {
				fix_conv_12_weight_in[ch][i] = (FIX_WT)conv5_wt_1x1[ch][i];
			}
			else
				fix_conv_12_weight_in[ch][i] = (FIX_WT)0.0;
		}
	}


	//reorder conv12
	for(int row = 0; row < 16; row++) {
		for(int i = 0; i < 16; i++) {
			for(int j = 0; j < 16; j++) {
				fix_conv_12_weight_reorder[row][i][j] = fix_conv_12_weight_in[i][j + row*16];
			}
		}
	}


	//////////// put all reordered weights together
//1+2+4=7
	// copy conv_1 to conv_weight_3x3_all
	int index_3x3 = 0;
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++)
				fix_conv_weight_3x3_all[index_3x3][i][j][k] = fix_conv_1_weight_in[i][j][k];
		}
	}

	// copy conv_4 to conv_weight_3x3_all
	for(int i = 0; i < 32; i++) {
		if( i % 16 == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%16][j][k] = fix_conv_4_weight_in[i][j][k];
			}
		}
	}


	// copy conv_7 to conv_weight_3x3_all
	for(int i = 0; i < 64; i++) {
		if( i % 16 == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%16][j][k] = fix_conv_7_weight_in[i][j][k];
			}
		}
	}


	// copy conv_2_reorder to conv_weight_1x1_all
	int index_1x1 = -1;
	for(int i = 0; i < 2; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_2_weight_reorder[i][j][k];
			}
		}
	}


	// copy conv_5_reorder to conv_weight_1x1_all
	for(int i = 0; i < 8; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_5_weight_reorder[i][j][k];
			}
		}
	}


	// copy conv_8_reorder to conv_weight_1x1_all
	for(int i = 0; i < 32; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_8_weight_reorder[i][j][k];
			}
		}
	}

	// copy conv_11_reorder to conv_weight_1x1_all
	for(int i = 0; i < 128; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_11_weight_reorder[i][j][k];
			}
		}
	}

	// copy conv_12_reorder to conv_weight_1x1_all
	for(int i = 0; i < 16; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_12_weight_reorder[i][j][k];
			}
		}
	}


//copy bias
	int index_bias = 0;
	for(int i = 0; i < 16; i++) {
		fix_bias_all[i] = fix_conv_1_bias_in[i];
	}


	printf("index_1x1: %d\n", index_1x1);
	printf("index_3x3: %d\n", index_3x3);
	printf("index_bias: %d\n", index_bias);


    // write conv_1x1 weights into params_shift_fix.bin
    ofs_param_write.write((char*)fix_conv_weight_1x1_all, 186*16*16*sizeof(FIX_16_1));

    // write conv_3x3 into params_shift_fix.bin
    ofs_param_write.write((char*)fix_conv_weight_3x3_all, 7*16*3*3*sizeof(FIX_16_1));

    // write bias_all into params_shift_fix.bin
    ofs_param_write.write((char*)fix_bias_all, 16*sizeof(FIX_16_1));

    ofs_param_write.close();

}
