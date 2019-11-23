#include "net_hls_shift.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>




// feature map buffers
FIX_FM FM_buf1[16][24][42];
FIX_FM FM_buf2[16][24][42];
FIX_FM FM_buf3[16][24][42];
FIX_FM FM_buf4[16][24][42];
FIX_FM FM_buf5[16][24][42];
FIX_FM FM_buf6[16][24][42];
FIX_FM FM_buf7[16][24][42];
FIX_FM FM_buf8[16][24][42];
FIX_FM FM_buf9[16][24][42];
FIX_FM FM_buf10[16][24][42];
FIX_FM FM_buf11[16][24][42];
FIX_FM FM_buf12[16][24][42];
FIX_FM FM_buf13[16][24][42];
FIX_FM FM_buf14[16][24][42];
FIX_FM FM_buf15[16][24][42];

FIX_FM FM_buf_pool[16][11][20];


FIX_WT weight_buf_1x1[4][16][16];
FIX_WT weight_buf_3x3[4][16][3][3];

FIX_WT bias_buf[16];


void fill_output( int layer, float buf[16][24][42], int ch, int col, int row);
void fill_output_pool( int layer, float buf[16][11][20], int ch, int col, int row);

int PL_test_compare_layer_1();
int PL_test_compare_layer_2();
int PL_test_compare_layer_3();
int PL_test_compare_layer_4();
int PL_test_compare_layer_5();
int PL_test_compare_layer_6();
int PL_test_compare_layer_7();
int PL_test_compare_layer_8();
int PL_test_compare_layer_9();
int PL_test_compare_layer_10();
int PL_test_compare_layer_11();
int PL_test_compare_layer_12();



FIX_32_25 my_exp_fix(FIX_FM input)
{
#pragma HLS latency min=2 max=20
	FIX_32_25 output;

	output = (FIX_32_25)hls::exp((float)input);
	return output;
}


void compute_bounding_box(float predict_box[5])
{
    int batch_size = 1;
    int num_anchors = 2;
    int h = 22;
    int w = 40;

    FIX_32_4 box[4] = {1, 1.06357021727,1, 2.65376815391};

    FIX_32_4 conf_thresh = 0.0;
    int conf_j = 0;
    int conf_m = 0;
    int conf_n = 0;

    FIX_32_4 conf_box1 = 0.0;
    FIX_32_4 conf_box2 = 0.0;

    for(int m = 1; m <= h; m++){
        for(int n = 1 ;n <= w; n++){
#pragma HLS pipeline
            conf_box1 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf2[4][m][n]));
            if(conf_box1 > conf_thresh){
								conf_thresh = conf_box1;
								conf_j = 0;
								conf_m = m;
								conf_n = n;

            }
        }
    }

    for(int m = 1; m <= h; m++){
        for(int n = 1; n <= w; n++){
#pragma HLS pipeline
            conf_box2 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf2[9][m][n]));
            if(conf_box2 > conf_thresh){
                conf_thresh = conf_box2;
                conf_j = 1;
                conf_m = m;
                conf_n = n;
            }
        }
    }

    if( conf_j == 0 ) {
        // first bounding box
        predict_box[0] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf2[0][conf_m][conf_n])) + (FIX_32_25)(conf_n-1);
        predict_box[1] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf2[1][conf_m][conf_n])) + (FIX_32_25)(conf_m-1);
        predict_box[2] = my_exp_fix(FM_buf2[2][conf_m][conf_n]) * box[0];
        predict_box[3] = my_exp_fix(FM_buf2[3][conf_m][conf_n]) * box[1];
        predict_box[4] = conf_thresh;
    }
    else if( conf_j == 1 ) {
        // second bounding box
        predict_box[0] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf2[5][conf_m][conf_n])) + (FIX_32_25)(conf_n-1);
        predict_box[1] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf2[6][conf_m][conf_n])) + (FIX_32_25)(conf_m-1);
        predict_box[2] = my_exp_fix(FM_buf2[7][conf_m][conf_n]) * box[2];
        predict_box[3] = my_exp_fix(FM_buf2[8][conf_m][conf_n]) * box[3];
        predict_box[4] = conf_thresh;
    }


#ifdef CSIM_DEBUG
    printf("PL output:\n");
    printf("conf_j: %d, conf_m: %d, conf_n:%d\n\n", conf_j, conf_m-1, conf_n-1);


	printf("%f\n", predict_box[0] / w);
	printf("%f\n", predict_box[1] / h);
	printf("%f\n", predict_box[2] / w);
	printf("%f\n", predict_box[3] / h);
	printf("%f\n", predict_box[4]);


	int x1, y1, x2, y2;
	float predict_box_temp[4];
	predict_box_temp[0] = predict_box[0] / w;
	predict_box_temp[1] = predict_box[1] / h;
	predict_box_temp[2] = predict_box[2] / w;
	predict_box_temp[3] = predict_box[3] / h;

	x1 = (unsigned int)(((predict_box_temp[0] - predict_box_temp[2]/2.0) * 640));
	y1 = (unsigned int)(((predict_box_temp[1] - predict_box_temp[3]/2.0) * 360));
	x2 = (unsigned int)(((predict_box_temp[0] + predict_box_temp[2]/2.0) * 640));
	y2 = (unsigned int)(((predict_box_temp[1] + predict_box_temp[3]/2.0) * 360));

	printf("%d %d %d %d\n", x1, y1, x2, y2);
#endif


}


void buffer_copy_to_axi( FIX_FM dest[16][24][42], FIX_FM src[16][24][42])
{
	//memcpy(dest, src, sizeof(FIX_FM)*16*22*42);
	for(int i = 0; i < 16; i++)
		for(int j = 1; j <= 22; j++)
			for(int k = 1; k <=40; k++)
#pragma HLS pipeline
				dest[i][j][k] = src[i][j][k];
}

void buffer_copy_from_axi( FIX_FM dest[16][24][42], FIX_FM src[16][24][42])
{
	//memcpy(dest, src, sizeof(FIX_FM)*16*22*42);
	for(int i = 0; i < 16; i++)
		for(int j = 0; j < 24; j++)
			for(int k = 0; k < 42; k++)
#pragma HLS pipeline
				dest[i][j][k] = src[i][j][k];
}


void load_weight_2D_from_axi( FIX_WT dest[16][16], FIX_16_1 src[16][16])
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 16; j++) {
#pragma HLS pipeline
			dest[i][j] = (FIX_WT)src[i][j];
		}
	}
}

void load_weight_3D_from_axi( FIX_WT dest[16][3][3], FIX_16_1 src[16][3][3])
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
#pragma HLS pipeline
				dest[i][j][k] = (FIX_WT)src[i][j][k];
			}
		}
	}
}


void load_bias_from_axi(FIX_WT dest[16], FIX_16_1 src[16])
{
	for(int i = 0; i < 16; i++) {
		dest[i] = (FIX_WT)src[i];
	}
}

void set_bias( FIX_FM buf[16][24][42], FIX_WT bias[16])
{
#pragma HLS ARRAY_PARTITION variable=buf  dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias dim=1 complete

	for(int j = 1; j <= 22; j++) {
		for(int k = 1; k <= 40; k++) {
#pragma HLS pipeline

			for(int i = 0; i < 16; i++)
				buf[i][j][k] = bias[i];

		}
	}
}


void copy_to_DDR_pool9( FIX_FM dest[16][24][42], FIX_FM buf[16][11][20], int b_col, int b_row )
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 11; j++) {
			for(int k = 0; k < 20; k++) {
#pragma HLS pipeline
				dest[i][j+1 + b_col*11][k+1 + b_row*20] = buf[i][j][k];
			}
		}
	}
}


void copy_to_DDR_pool3( FIX_FM ddr_pool3[32][90][162], FIX_FM buf[16][11][20], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 11; j++) {
			for(int k = 0; k < 20; k++) {
#pragma HLS pipeline
				ddr_pool3[i + ch*16][j+1 + col*11][k+1 + row*20] = buf[i][j][k];
			}
		}
	}
}


void copy_to_DDR_pool6( FIX_FM ddr_pool6[64][46][82], FIX_FM buf[16][11][20], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 11; j++) {
			for(int k = 0; k < 20; k++) {
#pragma HLS pipeline
				ddr_pool6[i + ch*16][j+1 + col*11][k+1 + row*20] = buf[i][j][k];
			}
		}
	}
}



void load_pool3_from_axi(FIX_FM buf[16][24][42], FIX_FM DDR_pool3[32][90][162],
							int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int h = 0; h < 24; h++) {
			for(int w = 0; w < 42; w++ ) {
#pragma HLS pipeline
				buf[i][h][w] = DDR_pool3[i + ch*16][h + col*22][w + row*40];
			}
		}
	}
}


void load_pool6_from_axi(FIX_FM buf[16][24][42], FIX_FM DDR_pool6[64][46][82],
							int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int h = 0; h < 24; h++) {
			for(int w = 0; w < 42; w++ ) {
#pragma HLS pipeline
				buf[i][h][w] = DDR_pool6[i + ch*16][h + col*22][w + row*40];
			}
		}
	}
}


FIX_FM img_norm_ch[256] = {
		-2.000000, -1.984314, -1.968627, -1.952941, -1.937255, -1.921569, -1.905882, -1.890196, -1.874510, -1.858824, -1.843137, -1.827451, -1.811765, -1.796078, -1.780392, -1.764706, -1.749020,
		-1.733333, -1.717647, -1.701961, -1.686275, -1.670588, -1.654902, -1.639216, -1.623529, -1.607843, -1.592157, -1.576471, -1.560784, -1.545098, -1.529412, -1.513725, -1.498039,
		-1.482353, -1.466667, -1.450980, -1.435294, -1.419608, -1.403922, -1.388235, -1.372549, -1.356863, -1.341176, -1.325490, -1.309804, -1.294118, -1.278431, -1.262745, -1.247059,
		-1.231373, -1.215686, -1.200000, -1.184314, -1.168627, -1.152941, -1.137255, -1.121569, -1.105882, -1.090196, -1.074510, -1.058824, -1.043137, -1.027451, -1.011765, -0.996078,
		-0.980392, -0.964706, -0.949020, -0.933333, -0.917647, -0.901961, -0.886275, -0.870588, -0.854902, -0.839216, -0.823529, -0.807843, -0.792157, -0.776471, -0.760784, -0.745098,
		-0.729412, -0.713725, -0.698039, -0.682353, -0.666667, -0.650980, -0.635294, -0.619608, -0.603922, -0.588235, -0.572549, -0.556863, -0.541176, -0.525490, -0.509804, -0.494118,
		-0.478431, -0.462745, -0.447059, -0.431373, -0.415686, -0.400000, -0.384314, -0.368627, -0.352941, -0.337255, -0.321569, -0.305882, -0.290196, -0.274510, -0.258824, -0.243137,
		-0.227451, -0.211765, -0.196078, -0.180392, -0.164706, -0.149020, -0.133333, -0.117647, -0.101961, -0.086275, -0.070588, -0.054902, -0.039216, -0.023529, -0.007843, 0.007843,
		0.023529, 0.039216, 0.054902, 0.070588, 0.086275, 0.101961, 0.117647, 0.133333, 0.149020, 0.164706, 0.180392, 0.196078, 0.211765, 0.227451, 0.243137, 0.258824,
		0.274510, 0.290196, 0.305882, 0.321569, 0.337255, 0.352941, 0.368627, 0.384314, 0.400000, 0.415686, 0.431373, 0.447059, 0.462745, 0.478431, 0.494118, 0.509804,
		0.525490, 0.541176, 0.556863, 0.572549, 0.588235, 0.603922, 0.619608, 0.635294, 0.650980, 0.666667, 0.682353, 0.698039, 0.713725, 0.729412, 0.745098, 0.760784,
		0.776471, 0.792157, 0.807843, 0.823529, 0.839216, 0.854902, 0.870588, 0.886275, 0.901961, 0.917647, 0.933333, 0.949020, 0.964706, 0.980392, 0.996078, 1.011765,
		1.027451, 1.043137, 1.058824, 1.074510, 1.090196, 1.105882, 1.121569, 1.137255, 1.152941, 1.168627, 1.184314, 1.200000, 1.215686, 1.231373, 1.247059, 1.262745,
		1.278431, 1.294118, 1.309804, 1.325490, 1.341176, 1.356863, 1.372549, 1.388235, 1.403922, 1.419608, 1.435294, 1.450980, 1.466667, 1.482353, 1.498039, 1.513725,
		1.529412, 1.545098, 1.560784, 1.576471, 1.592157, 1.607843, 1.623529, 1.639216, 1.654902, 1.670588, 1.686275, 1.701961, 1.717647, 1.733333, 1.749020, 1.764706,
		1.780392, 1.796078, 1.811765, 1.827451, 1.843137, 1.858824, 1.874510, 1.890196, 1.905882, 1.921569, 1.937255, 1.952941, 1.968627, 1.984314, 2.000000
};

void load_image_chunk_norm(FIX_FM img_buf[16][24][42], uint8 image_in_raw_pad[3][178][322],
							int col, int row)
{
	for(int itr = 0, i = 0, j = 0; itr < 24 * 42; itr++,j++) {
#pragma HLS PIPELINE II=1
		if(j == 42) {j = 0; i++;}
#ifdef CSIM_DEBUG
			if(i + col*22 == 0 || i + col*22 == 177 || j + row*40 == 0 || j + row*40 == 321 )
				img_buf[0][i][j] = 0.0;
			else
#endif
				img_buf[0][i][j] = img_norm_ch[(image_in_raw_pad[0][i + col*22][j + row*40])];
		}

	for(int itr = 0, i = 0, j = 0; itr < 24 * 42; itr++,j++) {
	#pragma HLS PIPELINE II=1
		if(j == 42) {j = 0; i++;}
	#ifdef CSIM_DEBUG
			if(i + col*22 == 0 || i + col*22 == 177 || j + row*40 == 0 || j + row*40 == 321 )
				img_buf[0][i][j] = 0.0;
			else
	#endif
				img_buf[1][i][j] = img_norm_ch[(image_in_raw_pad[1][i + col*22][j + row*40])];
		}

	for(int itr = 0, i = 0, j = 0; itr < 24 * 42; itr++,j++) {
#pragma HLS PIPELINE II=1
		if(j == 42) {j = 0; i++;}
#ifdef CSIM_DEBUG
			if(i + col*22 == 0 || i + col*22 == 177 || j + row*40 == 0 || j + row*40 == 321 )
				img_buf[0][i][j] = 0.0;
			else
#endif
				img_buf[2][i][j] = img_norm_ch[(image_in_raw_pad[2][i + col*22][j + row*40])];
		}
}



inline FIX_FM max(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d)
{
	FIX_FM t1, t2;

	if(a > b) t1 = a;
	else t1 = b;

	if(c > d) t2 = c;
	else t2 = d;

	if(t1 > t2) return t1;
	else return t2;
}


void relu_with_max_pooling(FIX_FM buf_in[16][24][42], FIX_FM buf_out[16][11][20])
{
#pragma HLS ARRAY_PARTITION variable=buf_in cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=buf_out cyclic dim=1 factor=16

	for(int i = 0; i < 11; i++) {
		for(int j = 0; j < 20; j++) {
#pragma HLS pipeline
			for(int ch = 0; ch < 16; ch++) {
#pragma HLS unroll
				buf_out[ch][i][j] = max(buf_in[ch][i*2+1][j*2+1], buf_in[ch][i*2+1][j*2+2],
								     	  buf_in[ch][i*2+2][j*2+1], buf_in[ch][i*2+2][j*2+2]);
				if (buf_out[ch][i][j]<0){
					buf_out[ch][i][j] = 0;
				}
			}
		}
	}
}


void clear_buf( FIX_FM buf[16][24][42])
{

	for(int j = 0; j < 24; j++) {
		for(int k = 0; k < 42; k++) {
#pragma HLS pipeline
			for(int i = 0; i < 16; i++) {
				buf[i][j][k] = 0;
			}
		}
	}
}



void Relu( FIX_FM buf[16][24][42] )
{
	for(int j = 1; j <= 22; j++) {
		for(int k = 1; k <= 40; k++) {
#pragma HLS pipeline
			for(int i = 0; i < 16; i++) {
#pragma HLS unroll
				if( buf[i][j][k] < 0 ) {
					buf[i][j][k] = 0;
				}
			}
		}
	}
}



void model_shift(uint8 image_in_raw_pad[3][178][322],

				FIX_16_1 conv_weight_1x1_all[186][16][16],
				FIX_16_1 conv_weight_3x3_all[7][16][3][3],
				FIX_16_1 bias_all[16],

				FIX_FM DDR_pool3_out_PL[32][90][162],
				FIX_FM DDR_pool6_out_PL[64][46][82],

				FIX_FM DDR_buf[36][16][24][42],

				float predict_box[5]
)
{

#pragma HLS INTERFACE m_axi depth=3*178*322 	port=image_in_raw_pad			  offset=slave	bundle=IMG
#pragma HLS INTERFACE m_axi depth=186*16*16		port=conv_weight_1x1_all		offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=7*16*3*3		port=conv_weight_3x3_all		offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=16			port=bias_all					      offset=slave	bundle=INPUT

#pragma HLS INTERFACE m_axi depth=32*90*162		port=DDR_pool3_out_PL			offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=64*46*82		port=DDR_pool6_out_PL			offset=slave	bundle=INPUT

#pragma HLS INTERFACE m_axi depth=36*16*24*42	port=DDR_buf					offset=slave	bundle=INPUT

#pragma HLS INTERFACE m_axi depth=5				port=predict_box				offset=slave	bundle=OUTPUT

#pragma HLS INTERFACE s_axilite register	port=return



#pragma HLS ALLOCATION instances=CONV_1x1			 		limit=1 function
#pragma HLS ALLOCATION instances=CONV_3x3_group     		limit=1 function
#pragma HLS ALLOCATION instances=relu_with_max_pooling		limit=1 function
#pragma HLS ALLOCATION instances=load_image_chunk_norm		limit=1 function
#pragma HLS ALLOCATION instances=my_exp_fix					limit=1 function
#pragma HLS ALLOCATION instances=set_bias					limit=1 function
#pragma HLS ALLOCATION instances=Relu						limit=1 function
#pragma HLS ALLOCATION instances=load_weight_3D_from_axi	limit=1 function
#pragma HLS ALLOCATION instances=load_weight_2D_from_axi	limit=1 function


#pragma HLS ARRAY_PARTITION variable=weight_buf_3x3 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight_buf_1x1 dim=1 complete

	/////////////////////////////// CONV_1 to POOL_3 ////////////////////////////

	load_weight_3D_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[0]);
	load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[0]);
	load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[1]);

	for(int row = 0; row < 8; row++) {
		for(int col = 0; col < 8; col++) {
#pragma HLS unroll

			///// CONV_1 (3x3)  <---  IMG ch:0 col:{{_col}} row:{{_row}}
			load_image_chunk_norm(FM_buf1, image_in_raw_pad, col, row);
			load_bias_from_axi(bias_buf, bias_all);
			set_bias(FM_buf3, bias_buf);
			CONV_3x3_group(FM_buf1, FM_buf3, weight_buf_3x3[0], 1);

#ifdef CSIM_DEBUG
fill_output(1, FM_buf3, 0, col, row);
#endif


			for(int ch_conv2 = 0; ch_conv2 < 2; ch_conv2++) {
#pragma HLS unroll

				int weight_3x3_index = 0;
				int weight_1x1_index = 0;


				///// CONV_2 (1x1)  <---  CONV_1 ch:{{_ch_conv2}} col:{{_col}} row:{{_row}}
				clear_buf(FM_buf15);
				CONV_1x1(FM_buf3, FM_buf15, weight_buf_1x1[ch_conv2]);

				///// POOL_3  <---  CONV_2 ch:{{_ch_conv2}} col:{{_col}} row:{{_row}}
				relu_with_max_pooling(FM_buf15, FM_buf_pool);
				copy_to_DDR_pool3( DDR_pool3_out_PL, FM_buf_pool, ch_conv2, col, row);

#ifdef CSIM_DEBUG
	fill_output(2, FM_buf15, ch_conv2, col, row);
	fill_output_pool(3, FM_buf_pool, ch_conv2, col, row);
#endif

			}
		}
	}



	/////////////////////////////// CONV_4 to POOL_6  ////////////////////////////

	load_weight_3D_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[1]);
	load_weight_3D_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[2]);

	for(int row = 0; row < 4; row++) {
		for(int col = 0; col < 4; col++) {
#pragma HLS unroll

			///// CONV_4  <---  POOL_3 ch:0 col:{{_col}} row:{{_row}}

			// load from DDR_pool_3_out_PL
			load_pool3_from_axi(FM_buf1, DDR_pool3_out_PL, 0, col, row);
			CONV_3x3_group(FM_buf1, FM_buf3, weight_buf_3x3[0], 0);

#ifdef CSIM_DEBUG
fill_output(4, FM_buf3, 0, col, row);
#endif

			///// CONV_4  <---  POOL_3 ch:1 col:{{_col}} row:{{_row}}

			// load from DDR_pool_3_out_PL
			load_pool3_from_axi(FM_buf2, DDR_pool3_out_PL, 1, col, row);
			CONV_3x3_group(FM_buf2, FM_buf4, weight_buf_3x3[1], 0);

#ifdef CSIM_DEBUG
fill_output(4, FM_buf4, 1, col, row);
#endif


			for(int ch_conv5 = 0; ch_conv5 < 4; ch_conv5++) {
#pragma HLS unroll

				int weight_1x1 = 2 + ch_conv5*2;
				int weight_3x3 = 1;

				///// CONV_5  <---  CONV_4 ch:{{_ch_conv5}} col:{{_col}} row:{{_row}}

				// load weight
				clear_buf(FM_buf15);
				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1]);
				CONV_1x1(FM_buf3, FM_buf15, weight_buf_1x1[0]);

				// load weight
				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+1]);
				CONV_1x1(FM_buf4, FM_buf15, weight_buf_1x1[1]);


				///// POOL_6  <--- CONV_5 ch:{{_ch_conv5}} col:{{_col}} row:{{_row}}
				//max_pooling(FM_buf15, FM_buf_pool);
				relu_with_max_pooling(FM_buf15, FM_buf_pool);
				copy_to_DDR_pool6( DDR_pool6_out_PL, FM_buf_pool, ch_conv5, col, row);

			#ifdef CSIM_DEBUG
				fill_output(5, FM_buf15, ch_conv5, col, row);
				fill_output_pool(6, FM_buf_pool, ch_conv5, col, row);
			#endif

			}
		}
	}



	/////////////////////////////// CONV_7 to POOL_9  ////////////////////////////


	for(int col = 0; col < 2; col++) {
		for(int row = 0; row < 2; row++) {
			///// CONV_7  <--- POOL_6 ch:{{_ch_conv7}} col:{{_col}} row:{{_row}}
			for(int ch_conv7 = 0; ch_conv7 < 4; ch_conv7++) {

#pragma HLS unroll

				int weight_3x3 = 3 + ch_conv7;

				switch (ch_conv7) {
				case 0:
					load_weight_3D_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3]);
					load_pool6_from_axi(FM_buf1, DDR_pool6_out_PL, ch_conv7, col, row);
					CONV_3x3_group(FM_buf1, FM_buf3, weight_buf_3x3[0], 0);

#ifdef CSIM_DEBUG
	fill_output(7, FM_buf3, ch_conv7, col, row);
#endif
					break;
				case 1:
					load_weight_3D_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3]);
					load_pool6_from_axi(FM_buf2, DDR_pool6_out_PL, ch_conv7, col, row);
					CONV_3x3_group(FM_buf2, FM_buf4, weight_buf_3x3[1], 0);

#ifdef CSIM_DEBUG
	fill_output(7, FM_buf4, ch_conv7, col, row);
#endif
					break;
				case 2:
					load_weight_3D_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3]);
					load_pool6_from_axi(FM_buf1, DDR_pool6_out_PL, ch_conv7, col, row);
					CONV_3x3_group(FM_buf1, FM_buf5, weight_buf_3x3[0], 0);

#ifdef CSIM_DEBUG
	fill_output(7, FM_buf5, ch_conv7, col, row);
#endif
					break;
				case 3:
					load_weight_3D_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3]);
					load_pool6_from_axi(FM_buf2, DDR_pool6_out_PL, ch_conv7, col, row);
					CONV_3x3_group(FM_buf2, FM_buf6, weight_buf_3x3[1], 0);

#ifdef CSIM_DEBUG
	fill_output(7, FM_buf6, ch_conv7, col, row);
#endif
					break;

				}
			}

			for(int ch_conv8 = 0; ch_conv8 < 8; ch_conv8++ ) {
#pragma HLS unroll

				int b_row = row % 2;
				int b_col = col % 2;
				int weight_1x1 = 10 + ch_conv8 * 4;

				///// CONV_8  <--- CONV_7 ch:{{_ch_conv8}} col:{{_col}} row:{{_row}}
				clear_buf(FM_buf15);

				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1]);
				CONV_1x1(FM_buf3, FM_buf15, weight_buf_1x1[0]);

				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+1]);
				CONV_1x1(FM_buf4, FM_buf15, weight_buf_1x1[1]);

				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1+2]);
				CONV_1x1(FM_buf5, FM_buf15, weight_buf_1x1[0]);

				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+3]);
				CONV_1x1(FM_buf6, FM_buf15, weight_buf_1x1[1]);


				///// POOL_9  <--- CONV_8 ch:{{_ch_conv8}} col:{{_col}} row:{{_row}}
				relu_with_max_pooling(FM_buf15, FM_buf_pool);
				copy_to_DDR_pool9( DDR_buf[ch_conv8], FM_buf_pool, b_col, b_row);

			#ifdef CSIM_DEBUG
				fill_output(8, FM_buf15, ch_conv8, col, row);
				fill_output_pool(9, FM_buf_pool, ch_conv8, col, row);
			#endif


			}
		}
	}



	/////////////////////////////// CONV_10 to CONV_12  //////////////////////////

	clear_buf(FM_buf2);
	for(int col = 0; col < 1; col++) {
		for(int row = 0; row < 1; row++) {
			for(int ch_conv10 = 0; ch_conv10 < 8; ch_conv10++ ) {
#pragma HLS unroll

				///// CONV_10  <--- POOL_9 ch:{{_ch_conv10}} col:{{_col}} row:{{_row}}

				buffer_copy_from_axi(FM_buf1, DDR_buf[ch_conv10]);


				switch (ch_conv10) {
				case 0:
					shift(FM_buf1, FM_buf3, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf3, ch_conv10, col, row);
#endif
					break;

				case 1:
					shift(FM_buf1, FM_buf4, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf4, ch_conv10, col, row);
#endif
					break;

				case 2:
					shift(FM_buf1, FM_buf5, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf5, ch_conv10, col, row);
#endif
					break;

				case 3:
					shift(FM_buf1, FM_buf6, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf6, ch_conv10, col, row);
#endif
					break;

				case 4:
					shift(FM_buf1, FM_buf7, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf7, ch_conv10, col, row);
#endif
					break;

				case 5:
					shift(FM_buf1, FM_buf8, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf8, ch_conv10, col, row);
#endif
					break;

				case 6:
					shift(FM_buf1, FM_buf9, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf9, ch_conv10, col, row);
#endif
					break;

				case 7:
				    //shift(FM_buf1, FM_buf10, ch_conv10);

#ifdef CSIM_DEBUG
	fill_output(10, FM_buf1, ch_conv10, col, row);
#endif
					break;

				}
			}

			for(int ch_conv11 = 0; ch_conv11 < 16; ch_conv11++ ) {
#pragma HLS unroll

				int weight_1x1 = 42 + ch_conv11*8;
				int weight_1x1_conv12 = 170 + ch_conv11;
				int b_row = row%2;
				int b_col = col%2;

				///// CONV_11  <--- CONV_10 ch:{{_ch_conv11}} col:{{_col}} row:{{_row}}
				clear_buf(FM_buf15);

				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1]);
				CONV_1x1(FM_buf3, FM_buf15, weight_buf_1x1[0]);

				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+1]);
				CONV_1x1(FM_buf4, FM_buf15, weight_buf_1x1[1]);

				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1+2]);
				CONV_1x1(FM_buf5, FM_buf15, weight_buf_1x1[0]);

				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+3]);
				CONV_1x1(FM_buf6, FM_buf15, weight_buf_1x1[1]);

				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1+4]);
				CONV_1x1(FM_buf7, FM_buf15, weight_buf_1x1[0]);

				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+5]);
				CONV_1x1(FM_buf8, FM_buf15, weight_buf_1x1[1]);

				load_weight_2D_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1+6]);
				CONV_1x1(FM_buf9, FM_buf15, weight_buf_1x1[0]);

				load_weight_2D_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1+7]);
				CONV_1x1(FM_buf1, FM_buf15, weight_buf_1x1[1]);

				Relu(FM_buf15);


			#ifdef CSIM_DEBUG
				fill_output(11, FM_buf15, ch_conv11, col, row);
			#endif

				/////////////////// CONV_12 + ch{{_ch_conv11}} of CONV_11
				load_weight_2D_from_axi(weight_buf_1x1[2], conv_weight_1x1_all[weight_1x1_conv12]);
				CONV_1x1(FM_buf15, FM_buf2, weight_buf_1x1[2]);

			}
		}
	}


#ifdef CSIM_DEBUG
	fill_output(12, FM_buf2, 0, 0, 0);
#endif


#ifdef CSIM_DEBUG
	PL_test_compare_layer_1();
	PL_test_compare_layer_2();
	PL_test_compare_layer_3();
	PL_test_compare_layer_4();
	PL_test_compare_layer_5();
	PL_test_compare_layer_6();
	PL_test_compare_layer_7();
	PL_test_compare_layer_8();
	PL_test_compare_layer_9();
	PL_test_compare_layer_10();
	PL_test_compare_layer_11();
	PL_test_compare_layer_12();
#endif

	compute_bounding_box(predict_box);

	return;


}
