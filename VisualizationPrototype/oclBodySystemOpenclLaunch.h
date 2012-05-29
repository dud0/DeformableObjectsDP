/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __CL_BODYSYSTEMOPENCL_LAUNCH_H
#define __CL_BODYSYSTEMOPENCL_LAUNCH_H

#ifdef __cplusplus
    extern "C"
    {
#endif

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

#include "oclBodySystemOpencl.h"

    class dim3 {
    	public:
    		size_t x;
    		size_t y;
    		size_t z;

    		dim3(size_t _x=1, size_t _y=1, size_t _z=1) {x = _x; y = _y; z = _z;}
    	};

int  CreateProgramAndKernel(cl_context ctx, cl_device_id* cdDevices, const char* kernel_name, cl_kernel* kernel, bool bDouble);
void AllocateNBodyArrays(cl_context ctx, cl_mem* vel, int numBodies, int dFlag);
void DeleteNBodyArrays(cl_mem* vel);
void allocateArray(cl_context cxGPUContext, cl_mem*memObj, size_t size);
void freeArray(cl_mem memObj);


void IntegrateNbodySystem(cl_command_queue cqCommandQueue,
                          cl_kernel MT_kernel, cl_kernel noMT_kernel,
                          cl_mem newPos, cl_mem newVel, cl_mem newF, cl_mem newEdge, cl_mem newForces,
                          cl_mem oldPos, cl_mem oldVel, cl_mem oldF, cl_mem oldEdge, cl_mem oldForces,
                          cl_mem pboCLOldPos, cl_mem pboCLNewPos,
                          float deltaTime, float damping, float softSq,
                          int numBodies, int numEdges, int p, int q,
                          int bUsePBO, bool bDouble);

void computeExternalForces(cl_command_queue cqCommandQueue,
		cl_kernel k,
		cl_mem newForces,
		cl_mem newFc,
		cl_mem oldForces,
		cl_mem oldFc,
		cl_mem oldVelocities,
		int numBodies, int p, int q,
		bool bDouble);
void computeSpringsForces(cl_command_queue cqCommandQueue,
   		cl_kernel k,
   		cl_mem newForces,
   		cl_mem newEdges,
   		cl_mem newPositions,
   		cl_mem oldForces,
   		cl_mem oldPositions,
   		cl_mem oldEdges,
   		int numEdges, int p, int q,
   		bool bDouble);
void integrateSystem(cl_command_queue cqCommandQueue,
    		cl_kernel k,
    		cl_mem newPositions,
    		cl_mem newVelocities,
    		cl_mem newEdges,
    		cl_mem newForces,
    		cl_mem oldPositions,
    		cl_mem oldVelocities,
    		cl_mem oldEdges,
    		cl_mem oldForces,
    		float deltaTime, float damping,
    		int numBodies, int p, int q,
    		bool bDouble);

void integrateSystemVerlet(cl_command_queue cqCommandQueue,
        		cl_kernel k,
        		cl_mem newPositions,
        		cl_mem oldPositions,
        		cl_mem newBeforePos,
        		cl_mem oldBeforePos,
        		cl_mem oldForces,
        		float deltaTime,
        		int numBodies, int p, int q,
        		bool bDouble);

void calcHash(cl_command_queue cqCommandQueue,
    		cl_kernel k,
        cl_mem d_Hash,
        cl_mem d_Index,
        cl_mem d_Pos,
        int numParticles
    );

void findCellBoundsAndReorder(cl_command_queue cqCommandQueue,
    		cl_kernel k,
    		cl_kernel memSet,
        cl_mem d_CellStart,
        cl_mem d_CellEnd,
        cl_mem d_ReorderedPos,
        cl_mem d_ReorderedVel,
        cl_mem d_ReorderedForce,
        cl_mem d_Hash,
        cl_mem d_Index,
        cl_mem d_Pos,
        cl_mem d_Vel,
        cl_mem d_Forces,
        uint numParticles,
        uint numCells
    );
void collide(cl_command_queue cqCommandQueue,
    		cl_kernel k,
        cl_mem d_Vel,
        cl_mem d_Forces,
        cl_mem d_ReorderedPos,
        cl_mem d_ReorderedVel,
        cl_mem d_ReorderedForce,
        cl_mem d_Index,
        cl_mem d_CellStart,
        cl_mem d_CellEnd,
        uint   numParticles,
        uint   numCells
    );

void bitonicSort(
        cl_command_queue cqCommandQueue,
        cl_kernel k_g,
        cl_kernel k_l,
        cl_mem d_DstKey,
        cl_mem d_DstVal,
        cl_mem d_SrcKey,
        cl_mem d_SrcVal,
        unsigned int batch,
        unsigned int arrayLength,
        unsigned int dir
    );

void CopyArrayFromDevice(int __size, cl_command_queue cmdq, float *host, cl_mem device, cl_mem pboCL, int numBodies, bool bDouble);
void CopyArrayToDevice(int __size, cl_command_queue cmdq, cl_mem device, const float *host, int numBodies, bool bDouble);
cl_mem RegisterGLBufferObject(cl_context ctx, unsigned int pboGL);
void UnregisterGLBufferObject(cl_mem pboCL);
void ThreadSync(cl_command_queue cmdq);

void _copyArrayFromDevice(cl_command_queue cqCommandQueue, void *hostPtr, cl_mem memObj, unsigned int vbo, size_t size);
void _copyArrayToDevice(cl_command_queue cqCommandQueue, cl_mem memObj, const void *hostPtr, size_t offset, size_t size);

#ifdef __cplusplus
    }
#endif

#endif
