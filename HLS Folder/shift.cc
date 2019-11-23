

// shift module

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls_shift.h"


void shift( FIX_FM bottom[16][24][42],
					  FIX_FM top[16][24][42], int ch)
{
	#pragma HLS ARRAY_PARTITION variable=bottom cyclic dim=1 factor=16
	#pragma HLS ARRAY_PARTITION variable=top cyclic dim=1 factor=16

  for (int h = 1; h <= 22; h++){
    for (int w = 1;  w <= 40; w++){
      switch(ch){
        	case 0:
#pragma HLS pipeline
				for (int i=0; i<14; i++)
						top[i][h][w] = bottom[i][h-1][w];
				for (int j=0; j<2; j++)
						top[j+14][h][w] = bottom[j+14][h+1][w];
				break;

			case 1:
#pragma HLS pipeline
				for (int i=0; i<12; i++)
						top[i][h][w] = bottom[i][h+1][w];
				for (int j=0; j<4; j++)
						top[j+12][h][w] = bottom[j+12][h][w-1];
				break;

			case 2:
#pragma HLS pipeline
				for (int i=0; i<10; i++)
						top[i][h][w] = bottom[i][h][w-1];
				for (int j=0; j<6; j++)
						top[j+10][h][w] = bottom[j+10][h][w+1];
				break;

			case 3:
#pragma HLS pipeline
				for (int i=0; i<8; i++)
						top[i][h][w] = bottom[i][h][w+1];
				for (int j=0; j<8; j++)
						top[j+8][h][w] = bottom[j+8][h-1][w-1];
				break;

			case 4:
#pragma HLS pipeline
				for (int i=0; i<6; i++)
						top[i][h][w] = bottom[i][h-1][w-1];
				for (int j=0; j<10; j++)
						top[j+6][h][w] = bottom[j+6][h-1][w+1];
				break;

			case 5:
#pragma HLS pipeline
				for (int i=0; i<4; i++)
						top[i][h][w] = bottom[i][h-1][w+1];
				for (int j=0; j<12; j++)
						top[j+4][h][w] = bottom[j+4][h+1][w-1];
				break;

			case 6:
#pragma HLS pipeline
				for (int i=0; i<2; i++)
						top[i][h][w] = bottom[i][h+1][w-1];
				for (int j=0; j<14; j++)
						top[j+2][h][w] = bottom[j+2][h+1][w+1];
				break;

			}
		}
	}
}
