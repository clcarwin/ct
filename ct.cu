extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>
#include "cublas_v2.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#define TB 128

cublasHandle_t handle;

/* operations */
struct opPlus {
public:
	static const float base_value = 0.0;
	__device__ float operator()(float x, float y)
	{
		return x + y;
	}
};

struct opMinus {
public:
	static const float base_value = 0.0;
	__device__ float operator()(float x, float y)
	{
		return x - y;
	}
};

struct opMult {
public:
	static const float base_value = 1.0;
	__device__ float operator()(float x, float y)
	{
		return x * y;
	}
};

struct opDiv {
public:
	static const float base_value = 1.0;
	__device__ float operator()(float x, float y)
	{
		return x / y;
	}
};

struct opMax {
public:
	static const float base_value = -2e38;
	__device__ float operator()(float x, float y)
	{
		return fmaxf(x, y);
	}
};

struct opExp {
public:
	__device__ float operator()(float x)
	{
		return exp(x);
	}
};

struct opSigmoid {
public:
	__device__ float operator()(float x)
	{
		return 1 / (1 + exp(-x));
	}
};

struct opSigmoidDeriv {
public:
	__device__ float operator()(float x, float y)
	{
		return x * y * (1 - y);
	}
};

struct opTanh {
public:
	__device__ float operator()(float x)
	{
		return 2 / (1 + exp(-2 * x)) - 1;
	}
};

struct opTanhDeriv {
public:
	__device__ float operator()(float x, float y)
	{
		return x * (1 - y * y);
	}
};

struct opCCE {
public:
	__device__ float operator()(float input, float target)
	{
		return target > 0 ? target * log(input) : 0;
	}
};

/* Is A in column major format? */
int is_cm(THCudaTensor *A)
{
	return A->stride[0] == 1;
}

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}

int cublas_init(lua_State *L)
{
	assert(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
	return 0;
}

int dot(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int trans_A = luaL_optint(L, 4, 0);
	int trans_B = luaL_optint(L, 5, 0);
	float alpha = luaL_optnumber(L, 6, 1.0);
	float beta = luaL_optnumber(L, 7, 0.0);

	assert(trans_A == 0 || trans_A == 1);
	assert(trans_B == 0 || trans_B == 1);

	if (!(A->nDimension == 2 && B->nDimension == 2 && C->nDimension == 2)) {
		luaL_error(L, "Matrices expected");
	}

	if (!(is_cm(A) && is_cm(B) && is_cm(C))) {
		luaL_error(L, "Matrices not in column major order");
	}

	int a = A->size[trans_A];
	int b = A->size[1 - trans_A];
	int c = B->size[trans_B];
	int d = B->size[1 - trans_B];

	if (b != c || a != C->size[0] || d != C->size[1]) {
		luaL_error(L, "Size mismatch");
	}

	assert(cublasSgemm(handle,
		trans_A ? CUBLAS_OP_T : CUBLAS_OP_N,
		trans_B ? CUBLAS_OP_T : CUBLAS_OP_N,
		a, d, c, &alpha,
		THCudaTensor_data(A), A->size[0],
		THCudaTensor_data(B), B->size[0], &beta, 
		THCudaTensor_data(C), C->size[0]) == CUBLAS_STATUS_SUCCESS);
	//assert(cudaDeviceSynchronize() == CUBLAS_STATUS_SUCCESS);
	return 0;
}

template<class Op>
int transform1(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int lenA = THCudaTensor_nElement(A);
	int lenB = THCudaTensor_nElement(B);

	if (!is_cm(A) || !is_cm(B)) {
		luaL_error(L, "Matrices not in column major order");
	}

	if (lenA != lenB) {
		luaL_error(L, "Size mismatch");
	}

	thrust::device_ptr<float> pA(THCudaTensor_data(A));
	thrust::device_ptr<float> pB(THCudaTensor_data(B));
	thrust::transform(pA, pA + lenA, pB, op);
	return 0;
}

template<class Op>
int transform2(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int lenA = THCudaTensor_nElement(A);
	int lenB = THCudaTensor_nElement(B);
	int lenC = THCudaTensor_nElement(C);

	if (!is_cm(A) || !is_cm(B) || !is_cm(C)) {
		luaL_error(L, "Matrices not in column major order");
	}

	if (lenA != lenB || lenA != lenC) {
		luaL_error(L, "Size mismatch");
	}

	thrust::device_ptr<float> pA(THCudaTensor_data(A));
	thrust::device_ptr<float> pB(THCudaTensor_data(B));
	thrust::device_ptr<float> pC(THCudaTensor_data(C));
	thrust::transform(pA, pA + lenA, pB, pC, op);
	return 0;
}

int sigmoid(lua_State *L)
{
	return transform1(opSigmoid(), L);
}

int mult_by_sigmoid_deriv(lua_State *L)
{
	return transform2(opSigmoidDeriv(), L);
}

int tanh(lua_State *L)
{
	return transform1(opTanh(), L);
}

int mult_by_tanh_deriv(lua_State *L)
{
	return transform2(opTanhDeriv(), L);
}

int cce(lua_State *L)
{
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	transform2(opCCE(), L);
	thrust::device_ptr<float> pC(THCudaTensor_data(C));
	float sum = thrust::reduce(pC, pC + THCudaTensor_nElement(C));
	
	lua_pushnumber(L, -sum);
	return 1;
}

int _exp(lua_State *L)
{
	return transform1(opExp(), L);
}

/* What a crazy bug!
 *
 *
 *
 *
 *
 */
template <class Op, int axis>
__global__ void kMatVect(Op op, float *A, float *x, int len, int size0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		if (axis == 0) A[i] = op(A[i], x[i / size0]);
		if (axis == 1) A[i] = op(A[i], x[i % size0]);
	}
}

template <class Op>
int mat_vect(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int axis = luaL_checkint(L, 3);

	if (!is_cm(A)) {
		luaL_error(L, "Matrix not in column major order");
	}
	
	int len = THCudaTensor_nElement(A);
	if (axis == 0) {
		if (A->size[1] != THCudaTensor_nElement(x)) {
			luaL_error(L, "Size mismatch");
		}
		kMatVect<Op, 0><<<(len - 1) / TB + 1, TB>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), len, A->size[0]);
	} else if (axis == 1) {
		if (A->size[0] != THCudaTensor_nElement(x)) {
			luaL_error(L, "Size mismatch");
		}
		kMatVect<Op, 1><<<(len - 1) / TB + 1, TB>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), len, A->size[0]);
	}
	
	checkCudaError(L);
	return 0;
}

int add_mat_vect(lua_State *L)
{
	return mat_vect(opPlus(), L);
}

int sub_mat_vect(lua_State *L)
{
	return mat_vect(opMinus(), L);
}

int mult_mat_vect(lua_State *L)
{
	return mat_vect(opMult(), L);
}

int div_mat_vect(lua_State *L)
{
	return mat_vect(opDiv(), L);
}

__global__ void kAdd(float *A, float *B, float *C, float alpha, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) C[i] = A[i] + alpha * B[i];
}

/* C = A + alpha * B */
int add(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	float alpha = luaL_optnumber(L, 4, 1.0);

	if (!(is_cm(A) && is_cm(B) && is_cm(C))) {
		luaL_error(L, "Matrices not in column major order");
	}

	if (!(A->size[0] == B->size[0] && A->size[1] == B->size[1] && A->size[0] == C->size[0] && A->size[1] == C->size[1])) {
		luaL_error(L, "Size mismatch");
	}

	int len = THCudaTensor_nElement(A);
	kAdd<<<(len - 1) / TB + 1, TB>>>(THCudaTensor_data(A), THCudaTensor_data(B), THCudaTensor_data(C), alpha, len);
	checkCudaError(L);
	return 0;
}

/* What a crazy bug!
 *
 *
 *
 *
 *
 */
template <class Op>
__global__ void kReduce(Op op, float *A, float *x, int n, int axis)
{
	extern __shared__ float sdata[];

	int i = threadIdx.x;

	sdata[i] = op.base_value;
	if (i < n) {
		if (axis == 0) {
			sdata[i] = A[threadIdx.x + n * blockIdx.x];
		} else if (axis == 1) {
			sdata[i] = A[gridDim.x * threadIdx.x + blockIdx.x];
		}
	}
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (i < s) {
			sdata[i] = op(sdata[i], sdata[i + s]);
		}
		__syncthreads();
	}

	if (i == 0) {
		x[blockIdx.x] = sdata[0];
	}
}

template <class Op>
int reduce(Op op, lua_State *L)
{
	int reduce_dim, other_dim;

	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int axis = luaL_checkint(L, 3);

	if (!is_cm(A)) {
		luaL_error(L, "Matrix not in column major order");
	}

	assert(axis == 0 || axis == 1);
	if (axis == 0) {
		reduce_dim = A->size[0];
		other_dim = A->size[1];
	} else if (axis == 1) {
		reduce_dim = A->size[1];
		other_dim = A->size[0];
	}

	assert(reduce_dim <= 1024);
	if (other_dim != THCudaTensor_nElement(x)) {
		luaL_error(L, "Size mismatch"); 
	}

	int threads = 1;
	while(threads < reduce_dim) {
		threads = threads << 1;
	}

	kReduce<Op><<<other_dim, threads, threads * sizeof(float)>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), reduce_dim, axis);
	checkCudaError(L);
	return 0;
}

int sum(lua_State *L)
{
	return reduce(opPlus(), L);
}

int _max(lua_State *L)
{
	return reduce(opMax(), L);
}

static const struct luaL_Reg funcs[] = {
	{"add", add},
	{"add_mat_vect", add_mat_vect},
	{"cce", cce},
	{"cublas_init", cublas_init},
	{"div_mat_vect", div_mat_vect},
	{"dot", dot},
	{"exp", _exp},
	{"max", _max},
	{"mult_by_sigmoid_deriv", mult_by_sigmoid_deriv},
	{"mult_by_tanh_deriv", mult_by_tanh_deriv},
	{"mult_mat_vect", mult_mat_vect},
	{"sigmoid", sigmoid},
	{"sub_mat_vect", sub_mat_vect},
	{"sum", sum},
	{"tanh", tanh},

	{NULL, NULL}
};

extern "C" int luaopen_libct(lua_State *L) {
	luaL_openlib(L, "ct", funcs, 0);
	return 1;
}
