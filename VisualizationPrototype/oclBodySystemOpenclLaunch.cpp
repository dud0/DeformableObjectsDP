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

#include "oclBodySystemOpenclLaunch.h"
#include <oclUtils.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
// var to hold path to executable
extern const char* cExecutablePath;

extern "C"
{
    char* clSourcefile = "oclNbodyKernel.cl";

    static size_t uSnap(size_t a, size_t b){
        return ((a % b) == 0) ? a : (a - (a % b) + b);
    }

    //GPU buffer allocation
    void allocateArray(cl_context cxGPUContext, cl_mem *memObj, size_t size){
        cl_int ciErrNum;
        shrLog(" clCreateBuffer (GPU GMEM, %u bytes)...\n\n", size);
        *memObj = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void freeArray(cl_mem memObj){
        cl_int ciErrNum;
        ciErrNum = clReleaseMemObject(memObj);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void AllocateNBodyArrays(cl_context cxGPUContext, cl_mem* vel, int numBodies, int dFlag)
    {
        // 4 floats each for alignment reasons
        unsigned int memSize;
		
		if (dFlag == 0)
		{
			memSize = sizeof( float) * 4 * numBodies;
		}
		else
		{
			memSize = sizeof( double) * 4 * numBodies;
		}
		vel[0] = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, NULL, NULL);
		vel[1] = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, NULL, NULL);
    }

    void DeleteNBodyArrays(cl_mem vel[2])
    {
        clReleaseMemObject(vel[0]);
        clReleaseMemObject(vel[1]);
    }

    //host<->device memcopies
    void _copyArrayFromDevice(cl_command_queue cqCommandQueue, void *hostPtr, cl_mem memObj, unsigned int vbo, size_t size){
        cl_int ciErrNum;
        assert( vbo == 0 );
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void _copyArrayToDevice(cl_command_queue cqCommandQueue, cl_mem memObj, const void *hostPtr, size_t offset, size_t size){
        cl_int ciErrNum;
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void CopyArrayFromDevice(int __size, cl_command_queue cqCommandQueue, float *host, cl_mem device, cl_mem pboCL, int numBodies, bool bDouble)
    {   
        cl_int ciErrNum;
        unsigned int size;

        if (pboCL) 
        {
            ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &pboCL, 0, NULL, NULL);
            oclCheckError(ciErrNum, CL_SUCCESS);
        }

		if (bDouble)
		{

			size = numBodies * 4 * sizeof(double);

			double *dHost = (double *)malloc(size);
			ciErrNum = clEnqueueReadBuffer(cqCommandQueue, device, CL_TRUE, 0, size, dHost, 0, NULL, NULL);

			for (int i = 0; i < numBodies * 4; i++)
			{
				host[i] = (float)(dHost[i]);
			}

			free(dHost);
		}
		else
		{

			size = numBodies * 4 * sizeof(float);

        	ciErrNum = clEnqueueReadBuffer(cqCommandQueue, device, CL_TRUE, 0, size, host, 0, NULL, NULL);
        }
		oclCheckError(ciErrNum, CL_SUCCESS);
		
        if (pboCL) 
        {
            ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &pboCL, 0, NULL, NULL);
            oclCheckError(ciErrNum, CL_SUCCESS);
        }
    }

    void CopyArrayToDevice(int __size, cl_command_queue cqCommandQueue, cl_mem device, const float* host, int numBodies, bool bDouble)
    {
        cl_int ciErrNum;
        unsigned int size;
		if (bDouble)
		{
			size = numBodies * 4 * sizeof(double);

			double *cdHost = (double *)malloc(size);

			for (int i = 0; i < numBodies * 4; i++)
			{
				cdHost[i] = (double)host[i];
			}

			ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, device, CL_TRUE, 0, size, cdHost, 0, NULL, NULL);
			free(cdHost);
		}
		else
		{
			size = numBodies*4*sizeof(float);

			ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, device, CL_TRUE, 0, size, host, 0, NULL, NULL);
		}
		oclCheckError(ciErrNum, CL_SUCCESS);
    }

    cl_mem RegisterGLBufferObject(cl_context cxGPUContext, unsigned int pboGL)
    {
        return clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, pboGL, NULL);
    }

    void UnregisterGLBufferObject(cl_mem pboCL)
    {
        clReleaseMemObject(pboCL);
    }

    void ThreadSync(cl_command_queue cqCommandQueue) 
    { 
        clFinish(cqCommandQueue); 
    }

    /*
     *
     *
     *
     *
     *
     * 	   DEFORMABLE OBJECTS SIMULATION
     *
     *
     *
     *
     *
     *
     */

    void computeExternalForces(cl_command_queue cqCommandQueue,
    		cl_kernel k,
    		cl_mem newForces,
    		cl_mem newFc,
    		cl_mem oldForces,
    		cl_mem oldFc,
    		cl_mem oldVelocities,
    		int numBodies, int p, int q,
    		bool bDouble)
    {
    	int sharedMemSize;

    	//for double precision
    	if (bDouble)
    	{
    		sharedMemSize = p * q * sizeof(cl_double4); // 4 doubles for pos
    	}
    	else
    	{
    		sharedMemSize = p * q * sizeof(cl_float4); // 4 floats for pos
    	}

    	size_t global_work_size[2];
    	size_t local_work_size[2];
    	cl_int ciErrNum = CL_SUCCESS;
    	cl_kernel kernel;

    	kernel = k;

    	ciErrNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&newForces);
    	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&newFc);
    	ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&oldForces);
    	ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oldFc);
    	ciErrNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oldVelocities);
    	ciErrNum |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&numBodies);

    	oclCheckError(ciErrNum, CL_SUCCESS);

    	// set work-item dimensions
    	local_work_size[0] = 256;
    	local_work_size[1] = q;
    	global_work_size[0]= numBodies;
    	global_work_size[1]= q;

    	// execute the kernel:
    	ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    	oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void computeSpringsForces(cl_command_queue cqCommandQueue,
    		cl_kernel k,
    		cl_mem newForces,
    		cl_mem newEdges,
    		cl_mem newPositions,
    		cl_mem oldForces,
    		cl_mem oldPositions,
    		cl_mem oldEdges,
    		int numEdges, int p, int q,
    		bool bDouble)
    {
    	int sharedMemSize;

    	p = 2048;

    	//for double precision
    	if (bDouble)
    	{
    		sharedMemSize = p * q * sizeof(cl_double4); // 4 doubles for pos
    	}
    	else
    	{
    		sharedMemSize = p * q * sizeof(cl_float4); // 4 floats for pos
    	}

    	size_t global_work_size[2];
    	size_t local_work_size[2];
    	cl_int ciErrNum = CL_SUCCESS;
    	cl_kernel kernel;

    	kernel = k;

    	ciErrNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&newForces);
    	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&newEdges);
    	ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&newPositions);
    	ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oldForces);
    	ciErrNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oldPositions);
    	ciErrNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&oldEdges);
    	ciErrNum |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&numEdges);

    	oclCheckError(ciErrNum, CL_SUCCESS);

    	//dim3 grid(numEdges/128, 1 ,1);

    	// set work-item dimensions
    	local_work_size[0] = 256;
    	local_work_size[1] = q;
    	global_work_size[0]= numEdges;
    	global_work_size[1]= q;

    	// execute the kernel:
    	ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    	oclCheckError(ciErrNum, CL_SUCCESS);
    }

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
    		bool bDouble)
    {
    	int sharedMemSize;

    	//for double precision
    	if (bDouble)
    	{
    		sharedMemSize = p * q * sizeof(cl_double4); // 4 doubles for pos
    	}
    	else
    	{
    		sharedMemSize = p * q * sizeof(cl_float4); // 4 floats for pos
    	}

    	size_t global_work_size[2];
    	size_t local_work_size[2];
    	cl_int ciErrNum = CL_SUCCESS;
    	cl_kernel kernel;

    	kernel = k;

    	ciErrNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&newPositions);
    	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&newVelocities);
    	ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&newEdges);
    	ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&newForces);
    	ciErrNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oldPositions);
    	ciErrNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&oldVelocities);
    	ciErrNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&oldEdges);
    	ciErrNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&oldForces);
    	ciErrNum |= clSetKernelArg(kernel, 8, sizeof(cl_float), (void *)&deltaTime);
    	ciErrNum |= clSetKernelArg(kernel, 9, sizeof(cl_float), (void *)&damping);

    	oclCheckError(ciErrNum, CL_SUCCESS);

    	// set work-item dimensions
    	local_work_size[0] = 256;
    	local_work_size[1] = q;
    	global_work_size[0]= numBodies;
    	global_work_size[1]= q;

    	// execute the kernel:
    	ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    	oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void integrateSystemVerlet(cl_command_queue cqCommandQueue,
        		cl_kernel k,
        		cl_mem newPositions,
        		cl_mem oldPositions,
        		cl_mem newBeforePos,
        		cl_mem oldBeforePos,
        		cl_mem oldForces,
        		float deltaTime,
        		int numBodies, int p, int q,
        		bool bDouble)
        {
        	int sharedMemSize;

        	//for double precision
        	if (bDouble)
        	{
        		sharedMemSize = p * q * sizeof(cl_double4); // 4 doubles for pos
        	}
        	else
        	{
        		sharedMemSize = p * q * sizeof(cl_float4); // 4 floats for pos
        	}

        	size_t global_work_size[2];
        	size_t local_work_size[2];
        	cl_int ciErrNum = CL_SUCCESS;
        	cl_kernel kernel;

        	kernel = k;

        	ciErrNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&newPositions);
        	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&oldPositions);
        	ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&newBeforePos);
        	ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oldBeforePos);
        	ciErrNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oldForces);
        	ciErrNum |= clSetKernelArg(kernel, 5, sizeof(cl_float), (void *)&deltaTime);

        	oclCheckError(ciErrNum, CL_SUCCESS);

        	// set work-item dimensions
        	local_work_size[0] = 256;
        	local_work_size[1] = q;
        	global_work_size[0]= numBodies;
        	global_work_size[1]= q;

        	// execute the kernel:
        	ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

        	oclCheckError(ciErrNum, CL_SUCCESS);
        }

    void calcHash(cl_command_queue cqCommandQueue,
    		cl_kernel k,
        cl_mem d_Hash,
        cl_mem d_Index,
        cl_mem d_Pos,
        int numParticles
    ){
    	static size_t wgSize = 64;
        cl_int ciErrNum;
        size_t globalWorkSize = uSnap(numParticles, wgSize);

        ciErrNum  = clSetKernelArg(k, 0, sizeof(cl_mem), (void *)&d_Hash);
        ciErrNum |= clSetKernelArg(k, 1, sizeof(cl_mem), (void *)&d_Index);
        ciErrNum |= clSetKernelArg(k, 2, sizeof(cl_mem), (void *)&d_Pos);
        ciErrNum |= clSetKernelArg(k, 3,  sizeof(uint), (void *)&numParticles);
        oclCheckError(ciErrNum, CL_SUCCESS);

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, k, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    static void memsetOCL(cl_command_queue cqCommandQueue,
    		cl_kernel k,
        cl_mem d_Data,
        uint val,
        uint N
    ){
    	static size_t wgSize = 64;
        cl_int ciErrNum;
        size_t globalWorkSize = uSnap(N, wgSize);

        ciErrNum  = clSetKernelArg(k, 0, sizeof(cl_mem), (void *)&d_Data);
        ciErrNum |= clSetKernelArg(k, 1, sizeof(cl_uint), (void *)&val);
        ciErrNum |= clSetKernelArg(k, 2, sizeof(cl_uint), (void *)&N);
        oclCheckError(ciErrNum, CL_SUCCESS);

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, k, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

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
    ){
    	static size_t wgSize = 64;
        cl_int ciErrNum;
        memsetOCL(cqCommandQueue, memSet, d_CellStart, 0xFFFFFFFFU, numCells);
        //memsetOCL(d_CellEnd, 0xFFFFFFFFU, numCells);
        size_t globalWorkSize = uSnap(numParticles, wgSize);

        ciErrNum  = clSetKernelArg(k, 0, sizeof(cl_mem), (void *)&d_CellStart);
        ciErrNum |= clSetKernelArg(k, 1, sizeof(cl_mem), (void *)&d_CellEnd);
        ciErrNum |= clSetKernelArg(k, 2, sizeof(cl_mem), (void *)&d_ReorderedPos);
        ciErrNum |= clSetKernelArg(k, 3, sizeof(cl_mem), (void *)&d_ReorderedVel);
        ciErrNum |= clSetKernelArg(k, 4, sizeof(cl_mem), (void *)&d_ReorderedForce);
        ciErrNum |= clSetKernelArg(k, 5, sizeof(cl_mem), (void *)&d_Hash);
        ciErrNum |= clSetKernelArg(k, 6, sizeof(cl_mem), (void *)&d_Index);
        ciErrNum |= clSetKernelArg(k, 7, sizeof(cl_mem), (void *)&d_Pos);
        ciErrNum |= clSetKernelArg(k, 8, sizeof(cl_mem), (void *)&d_Vel);
        ciErrNum |= clSetKernelArg(k, 9, sizeof(cl_mem), (void *)&d_Forces);
        ciErrNum |= clSetKernelArg(k, 10, (wgSize + 1) * sizeof(cl_uint), NULL);
        ciErrNum |= clSetKernelArg(k, 11, sizeof(cl_uint), (void *)&numParticles);
        oclCheckError(ciErrNum, CL_SUCCESS);

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, k, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

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
    ){
    	static size_t wgSize = 64;
        cl_int ciErrNum;
        size_t globalWorkSize = uSnap(numParticles, wgSize);

        ciErrNum  = clSetKernelArg(k, 0, sizeof(cl_mem), (void *)&d_Vel);
        ciErrNum  = clSetKernelArg(k, 1, sizeof(cl_mem), (void *)&d_Forces);
        ciErrNum |= clSetKernelArg(k, 2, sizeof(cl_mem), (void *)&d_ReorderedPos);
        ciErrNum |= clSetKernelArg(k, 3, sizeof(cl_mem), (void *)&d_ReorderedVel);
        ciErrNum |= clSetKernelArg(k, 4, sizeof(cl_mem), (void *)&d_ReorderedForce);
        ciErrNum |= clSetKernelArg(k, 5, sizeof(cl_mem), (void *)&d_Index);
        ciErrNum |= clSetKernelArg(k, 6, sizeof(cl_mem), (void *)&d_CellStart);
        ciErrNum |= clSetKernelArg(k, 7, sizeof(cl_mem), (void *)&d_CellEnd);
        ciErrNum |= clSetKernelArg(k, 8, sizeof(uint),   (void *)&numParticles);
        oclCheckError(ciErrNum, CL_SUCCESS);

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, k, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    static cl_uint factorRadix2(cl_uint& log2L, cl_uint L){
        if(!L){
            log2L = 0;
            return 0;
        }else{
            for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
            return L;
        }
    }

    static const unsigned int LOCAL_SIZE_LIMIT = 512U;

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
    ){
        if(arrayLength < 2)
            return;

        //Only power-of-two array lengths are supported so far
        cl_uint log2L;
        cl_uint factorizationRemainder = factorRadix2(log2L, arrayLength);
        oclCheckError( factorizationRemainder == 1, shrTRUE );

        dir = (dir != 0);

        cl_int ciErrNum;
        size_t localWorkSize, globalWorkSize;

        //Launch bitonicSortLocal1
        ciErrNum  = clSetKernelArg(k_l, 0,  sizeof(cl_mem), (void *)&d_DstKey);
        ciErrNum |= clSetKernelArg(k_l, 1,  sizeof(cl_mem), (void *)&d_DstVal);
        ciErrNum |= clSetKernelArg(k_l, 2,  sizeof(cl_mem), (void *)&d_SrcKey);
        ciErrNum |= clSetKernelArg(k_l, 3,  sizeof(cl_mem), (void *)&d_SrcVal);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize  = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, k_l, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        for(unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        {
        	for(unsigned stride = size / 2; stride > 0; stride >>= 1)
        	{
        		//Launch bitonicMergeGlobal
        		ciErrNum  = clSetKernelArg(k_g, 0,  sizeof(cl_mem), (void *)&d_DstKey);
        		ciErrNum |= clSetKernelArg(k_g, 1,  sizeof(cl_mem), (void *)&d_DstVal);
        		ciErrNum |= clSetKernelArg(k_g, 2,  sizeof(cl_mem), (void *)&d_DstKey);
        		ciErrNum |= clSetKernelArg(k_g, 3,  sizeof(cl_mem), (void *)&d_DstVal);
        		ciErrNum |= clSetKernelArg(k_g, 4, sizeof(cl_uint), (void *)&arrayLength);
        		ciErrNum |= clSetKernelArg(k_g, 5, sizeof(cl_uint), (void *)&size);
        		ciErrNum |= clSetKernelArg(k_g, 6, sizeof(cl_uint), (void *)&stride);
        		ciErrNum |= clSetKernelArg(k_g, 7, sizeof(cl_uint), (void *)&dir);
        		oclCheckError(ciErrNum, CL_SUCCESS);

        		localWorkSize  = LOCAL_SIZE_LIMIT / 4;
        		globalWorkSize = batch * arrayLength / 2;

        		ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, k_g, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        		oclCheckError(ciErrNum, CL_SUCCESS);
        	}
        }
    }

    /*
     *
     *
     *
     *
     *
     * 	   DEFORMABLE OBJECTS SIMULATION
     *
     *
     *
     *
     *
     *
     */


    // Function to read in kernel from uncompiled source, create the OCL program and build the OCL program 
    // **************************************************************************************************
    int CreateProgramAndKernel(cl_context cxGPUContext, cl_device_id* cdDevices, const char *kernel_name, cl_kernel *kernel, bool bDouble)
    {
        cl_program cpProgram;
        size_t szSourceLen;
        cl_int ciErrNum = CL_SUCCESS; 

        // Read the kernel in from file
        shrLog("\nLoading Uncompiled kernel from .cl file, using %s\n", clSourcefile);
        char* cPathAndFile = shrFindFilePath(clSourcefile, NULL);
        oclCheckError(cPathAndFile != NULL, shrTRUE);
        char* pcSource = oclLoadProgSource(cPathAndFile, "", &szSourceLen);
        oclCheckError(pcSource != NULL, shrTRUE);

	// Check OpenCL version -> vec3 types are supported only from version 1.1 and above
	char cOCLVersion[32];
	clGetDeviceInfo(cdDevices[0], CL_DEVICE_VERSION, sizeof(cOCLVersion), &cOCLVersion, 0);

	int iVec3Length = 3;
	if( strncmp("OpenCL 1.0", cOCLVersion, 10) == 0 ) {
		iVec3Length = 4;
	}


		//for double precision
		char *pcSourceForDouble;
		std::stringstream header;
		if (bDouble)
		{
			header << "#define REAL double";
			header << std::endl;
			header << "#define REAL4 double4";
			header << std::endl;
			header << "#define REAL3 double" << iVec3Length;
			header << std::endl;
			header << "#define ZERO3 {0.0, 0.0, 0.0" << ((iVec3Length == 4) ? ", 0.0}" : "}");
			header << std::endl;
		}
		else
		{
			header << "#define REAL float";
			header << std::endl;
			header << "#define REAL4 float4";
			header << std::endl;
			header << "#define REAL3 float" << iVec3Length;
			header << std::endl;
			header << "#define ZERO3 {0.0f, 0.0f, 0.0f" << ((iVec3Length == 4) ? ", 0.0f}" : "}");
			header << std::endl;
		}
		
		header << pcSource;
		pcSourceForDouble = (char *)malloc(header.str().size() + 1);
		szSourceLen = header.str().size();
#ifdef WIN32
        strcpy_s(pcSourceForDouble, szSourceLen + 1, header.str().c_str());
#else
        strcpy(pcSourceForDouble, header.str().c_str());
#endif

        // create the program 
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&pcSourceForDouble, &szSourceLen, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        shrLog("clCreateProgramWithSource\n"); 

        // Build the program with 'mad' Optimization option
#ifdef MAC
	char *flags = "-cl-fast-relaxed-math -DMAC";
#else
	char *flags = "-cl-fast-relaxed-math";
#endif
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, flags, NULL, NULL);
        if (ciErrNum != CL_SUCCESS)
        {
            // write out standard error, Build Log and PTX, then cleanup and exit
            shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
            oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
            oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclNbody.ptx");
            oclCheckError(ciErrNum, CL_SUCCESS); 
        }
        shrLog("clBuildProgram\n"); 

        // create the kernel
        *kernel = clCreateKernel(cpProgram, kernel_name, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS); 
        shrLog("clCreateKernel\n"); 

		size_t wgSize;
		ciErrNum = clGetKernelWorkGroupInfo(*kernel, cdDevices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
		if (wgSize == 64) {
		  shrLog(
			 "ERROR: Minimum work-group size 256 required by this application is not supported on this device.\n");
		  exit(0);
		}
	
		free(pcSourceForDouble);

        return 0;
    }
}
