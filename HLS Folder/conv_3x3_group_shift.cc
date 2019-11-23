

// conv 3x3 for group (depth-wise convolutions)

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls_shift.h"

void CONV_3x3_group(
					FIX_FM bottom[16][24][42],
					FIX_FM top[16][24][42],
					FIX_WT weights3x3[16][3][3], int bias)
{

	FIX_FM fm[16];
	FIX_FM zero = 0;
#pragma HLS ARRAY_PARTITION variable=bottom cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=top cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=weights3x3 dim=1 factor=16

#pragma HLS ARRAY_PARTITION variable=fm complete
//#pragma HLS ARRAY_PARTITION variable=fm_buf complete


	for(int h = 1; h <= 22; h++){
		for(int w = 1; w <= 40; w++){
#pragma HLS pipeline
			for(int co = 0; co < 16; co++){
				if (bias==1) fm[co] = top[co][h][w];
				else fm[co] = 0;
				for(int i=0, j=0, iter=0; iter<9;iter++,j++){
					if(j == 3) {j = 0; i++;}
					fm[co]+=weights3x3[co][i][j]*bottom[co][h+i-1][w+j-1];
				}
				top[co][h][w] = (fm[co]<0)? zero:fm[co];

			}
		}
	}
}
