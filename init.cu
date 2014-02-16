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

#define TB 128

cublasHandle_t handle;

/* Is A in column major format? */
int is_cm(THCudaTensor *A)
{
	return A->stride[0] == 1;
}

extern "C" int cublas_init(lua_State *L)
{
	assert(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
	return 0;
}

extern "C" int sgemm(lua_State *L)
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
		luaL_error(L, "Incorrect matrix size");
	}

	assert(cublasSgemm(handle,
		trans_A ? CUBLAS_OP_T : CUBLAS_OP_N,
		trans_B ? CUBLAS_OP_T : CUBLAS_OP_N,
		a, d, c, &alpha,
		THCudaTensor_data(A), A->size[0],
		THCudaTensor_data(B), B->size[0], &beta, 
		THCudaTensor_data(C), C->size[0]) == CUBLAS_STATUS_SUCCESS);
	assert(cudaDeviceSynchronize() == CUBLAS_STATUS_SUCCESS);

	return 0;
}

static const struct luaL_Reg funcs[] = {
	{"cublas_init", cublas_init},
	{"sgemm", sgemm},
	{NULL, NULL}
};

extern "C" int luaopen_ct(lua_State *L) {
	luaL_openlib(L, "ct", funcs, 0);
	return 1;
}
