//    # Without Pragma (Latency=545,LUT's=171)

//# define N 16
//
//void pragma (int A[N][N],int x[N],int y[N]){
//	for(int i=0;i<N;i++){
//        # pragma HLS pipeline off
//		int acc=0;
//		for (int j=0;j<N;j++){
//			acc += A[i][j]*x[j];
//
//		}
//		y[i]=acc;
//	}
//}


// # Unroll inner loop (Latency=169,LUT's=1298)

//# define N 16
//
//void pragma (int A[N][N],int x[N],int y[N]){
//	for(int i=0;i<N;i++){
//        # pragma HLS pipeline off
//		int acc=0;
//		for (int j=0;j<N;j++){
//            # pragma HLS unroll
//			acc += A[i][j]*x[j];
//
//		}
//		y[i]=acc;
//	}
//}

// # Pipeline outer loop (Latency =139,LUT's=1352)

//# define N 16
//
//void pragma (int A[N][N],int x[N],int y[N]){
//	for(int i=0;i<N;i++){
//        # pragma HLS pipeline II=1
//		int acc=0;
//		for (int j=0;j<N;j++){
//            # pragma HLS unroll
//			acc += A[i][j]*x[j];
//
//		}
//		y[i]=acc;
//	}
//}
// Inner loop automatically unrolled when pipeline the outer loop.


// # Array partitioning (Latency=27,LUT's=10125)

//# define N 16
//
//void pragma (int A[N][N],int x[N],int y[N]){
//    # pragma HLS array_partition variable = A
//    # pragma HLS array_parition variable = x
//	for(int i=0;i<N;i++){
//        # pragma HLS pipeline
//		int acc=0;
//		for (int j=0;j<N;j++){
//			acc += A[i][j]*x[j];
//
//		}
//		y[i]=acc;
//	}
//}

// # loop flattening / Loop merge (No of transitions decreases)

//void pragma (int A,int x,int y){
//	int acc=0;
//    for(int i=3;i>=0;i--){
//    	y=A;
//    }
//	for(int i=3;i>=0;i--){
//		for (int j=3;j>=0;j--){
//            # pragma HLS loop_flattten
//			acc += A*x;
//		}
//		y=acc;
//	}
//
//	for(int i=3;i>=0;i--){
//		y=x;
//	}


// # Array Reshaping (latency=141,LUT's=5717)
// # Array Mapping (latency=139,LUT's=1352)

//# define N 16
//
//void pragma (int A[N][N],int x[N],int y[N]){
//    //# pragma HLS array_reshape variable = A type=block factor=2 // vertical mapping (longer words)
//    //# pragma HLS array_reshape variable = x type=cycle factor=2
//    # pragma HLS array_map variable=A instance = pragma horizontal  // horizontal mapping (more elements)
//    # pragma HLS array_map variable=x instance = pragma horizontal
//	for(int i=0;i<N;i++){
//        # pragma HLS pipeline
//		int acc=0;
//		for (int j=0;j<N;j++){
//			acc += A[i][j]*x[j];
//
//		}
//		y[i]=acc;
//	}
//}


// # HLS aggregate (Create single wide bus for all structure elements)

//typedef struct {
//	unsigned char A,B,C;
//} pixel;
//void pragma(pixel *a){
//# pragma HLS aggregate variable = a compact=bit
//
//}

// # HLS allocation (restrict no of RTL instances & hardware resources) (Latency=545,LUT's=171)

//# define N 16
//void pragma (int A[N][N],int x[N],int y[N]){
//	# pragma HLS allocation instances = pragma limit=2 function
//	for(int i=0;i<N;i++){
//        # pragma HLS pipeline off
//		int acc=0;
//		for (int j=0;j<N;j++){
//			acc += A[i][j]*x[j];
//
//		}
//		y[i]=acc;
//	}
//}


// # HLS dataflow & HLS stream & HLS trip_counts::(Latency=12,LUT's=189)

//void pragma(int *a,int*b,int*c) {
//     # pragma HLS dataflow
//	for(int i=0;i<10;i++){
//		a[i]=b[i] + c[i];
//	}
//}
// task-level parallelism and allow fn and loops overlaps in their operation.

// # HLS stream ()
//#include <hls_stream.h>
//
//void pragma(hls::stream<int> &in,hls::stream<int> out) {
//#pragma HLS stream variable=in depth=2
//#pragma HLS stream variable=out depth=2
//
//int a = in.read();
//int b = in.read();
//out.write(a + b);
//}
// # HLS inline (Latency=12,LUT's=87)
//
//void pragma(int *a,int*b,int*c) {
//     # pragma HLS inline
//	for(int i=0;i<10;i++){
//		a[i]=b[i] + c[i];
//	}
//}
// remove function as separate entity and also decrease area.
