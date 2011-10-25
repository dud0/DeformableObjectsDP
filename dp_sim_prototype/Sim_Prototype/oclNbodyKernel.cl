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

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define BLOCKDIM 256
#define LOOP_UNROLL 4
#define ZERO4 0

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i + mul24(get_local_size(0), get_local_id(1))]

// This macro is only used the multithreadBodies (MT) versions of kernel code below
#define SX_SUM(i,j) sharedPos[i + mul24((uint)get_local_size(0), (uint)j)]    // i + blockDimx * j

REAL3 bodyBodyInteraction(REAL3 ai, REAL4 bi, REAL4 bj, REAL softeningSquared) 
{
    REAL3 r;

    // r_ij  [3 FLOPS]
    r.x = bi.x - bj.x;
    r.y = bi.y - bj.y;
    r.z = bi.z - bj.z;
    //r.w = 0;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    REAL distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    REAL invDist = rsqrt((float)distSqr);
	REAL invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    REAL s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

// This is the "tile_calculation" function from the GPUG3 article.
REAL3 gravitation(REAL4 myPos, REAL3 accel, REAL softeningSquared, __local REAL4* sharedPos)
{
    // The CUDA 1.1 compiler cannot determine that i is not going to 
    // overflow in the loop below.  Therefore if int is used on 64-bit linux 
    // or windows (or long instead of long long on win64), the compiler
    // generates suboptimal code.  Therefore we use long long on win64 and
    // long on everything else. (Workaround for Bug ID 347697)
#ifdef _Win64
    unsigned long long i = 0;
#else
    unsigned long i = 0;
#endif

    // Here we unroll the loop

    // Note that having an unsigned int loop counter and an unsigned
    // long index helps the compiler generate efficient code on 64-bit
    // OSes.  The compiler can't assume the 64-bit index won't overflow
    // so it incurs extra integer operations.  This is a standard issue
    // in porting 32-bit code to 64-bit OSes.
    int blockDimx = get_local_size(0);
    for (unsigned int counter = 0; counter < blockDimx; ) 
    {
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter++;
#if LOOP_UNROLL > 1
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter++;
#endif
#if LOOP_UNROLL > 2
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter += 2;
#endif
#if LOOP_UNROLL > 4
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos, softeningSquared); 
	counter += 4;
#endif
    }

    return accel;
}

// WRAP is used to force each block to start working on a different 
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at 
// once.
#define WRAP(x,m) (((x)<m)?(x):(x-m))  // Mod without divide, works on values from 0 up to 2m

REAL3 computeBodyAccel_MT(REAL4 bodyPos, 
                           __global REAL4* positions, 
                           int numBodies, 
                           REAL softeningSquared, 
                           __local REAL4* sharedPos)
{

    REAL3 acc = ZERO3;
    
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);

    for (unsigned int tile = blockIdxy; tile < numTiles + blockIdxy; tile++) 
    {
        sharedPos[threadIdxx + blockDimx * threadIdxy] = 
            positions[WRAP(blockIdxx + mul24(blockDimy, tile) + threadIdxy, gridDimx) * blockDimx
                      + threadIdxx];
       
        // __syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(bodyPos, acc, softeningSquared, sharedPos);
        
        // __syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is 
    // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only 
    // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple 
    // threads per body.  We still can use blocks of 256 threads, but they are arranged in blockDimy rows 
    // of blockDimx threads each.  Each thread processes 1/blockDimy of the forces that affect each body, and then 
    // 1/blockDimy of the threads (those with threadIdx.y==0) add up the partial sums from the other 
    // threads for that body.  To enable this, use the "--blockDimx=" and "--blockDimy=" command line options to 
    // this example. e.g.: "nbody.exe --numBodies=1024 --blockDimx=64 --blockDimy=4" will use 4 threads per body and 256 
    // threads per block. There will be numBodies/blockDimx = 16 blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads per body is greater 
    // than one, so that when it is not we don't have to execute the more complex code required!
        SX_SUM(threadIdxx, threadIdxy).x = acc.x;
        SX_SUM(threadIdxx, threadIdxy).y = acc.y;
        SX_SUM(threadIdxx, threadIdxy).z = acc.z;

        barrier(CLK_LOCAL_MEM_FENCE);//__syncthreads();

        // Save the result in global memory for the integration step
        if (get_local_id(0) == 0) 
        {
            for (unsigned int i = 1; i < blockDimy; i++) 
            {
                acc.x += SX_SUM(threadIdxx, i).x;
                acc.y += SX_SUM(threadIdxx, i).y;
                acc.z += SX_SUM(threadIdxx, i).z;
            }
        }

    return acc;
}

REAL3 computeBodyAccel_noMT(REAL4 bodyPos, 
                             __global REAL4* positions, 
                             int numBodies, 
                             REAL softeningSquared, 
                             __local REAL4* sharedPos)
{
    REAL3 acc = ZERO3;
    
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);

    for (unsigned int tile = blockIdxy; tile < numTiles + blockIdxy; tile++) 
    {
        sharedPos[threadIdxx + mul24(blockDimx, threadIdxy)] = 
            positions[WRAP(blockIdxx + tile, gridDimx) * blockDimx + threadIdxx];
       
        barrier(CLK_LOCAL_MEM_FENCE);// __syncthreads();

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(bodyPos, acc, softeningSquared, sharedPos);
        
        barrier(CLK_LOCAL_MEM_FENCE);// __syncthreads();
    }

    // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is 
    // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only 
    // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple 
    // threads per body.  We still can use blocks of 256 threads, but they are arranged in blockDimy rows 
    // of blockDimx threads each.  Each thread processes 1/blockDimy of the forces that affect each body, and then 
    // 1/blockDimy of the threads (those with threadIdx.y==0) add up the partial sums from the other 
    // threads for that body.  To enable this, use the "--blockDimx=" and "--blockDimy=" command line options to 
    // this example. e.g.: "nbody.exe --numBodies=1024 --blockDimx=64 --blockDimy=4" will use 4 threads per body and 256 
    // threads per block. There will be numBodies/blockDimx = 16 blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads per body is greater 
    // than one, so that when it is not we don't have to execute the more complex code required!

    return acc;
}

REAL3 computeFg(int numBodies) {
	REAL3 Fg = ZERO3;
	
	unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
    // predpokladame, ze mass castice je 1, G je 9.823
    Fg.x = 0;
    Fg.y = -9.823;
    Fg.z = 0;
	
	return Fg;
}

REAL3 computeFd(int numBodies, REAL4 bodyVel) {

	REAL3 Fd = ZERO3;
	
	unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
    //koeficient odporu vzduchu som si urcil ako 0.3
    Fd.x = -0.3*bodyVel.x;
    Fd.y = -0.3*bodyVel.y;
    Fd.z = -0.3*bodyVel.z;
	
	return Fd;
}

REAL3 computeFc(int numBodies, REAL4 Fc) {

    REAL3 k_Fc = ZERO3;
	
	unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
    
    k_Fc.x = Fc.x;
    k_Fc.y = Fc.y;
    k_Fc.z = Fc.z;
	
	return k_Fc;
}

REAL3 computeFp(int numBodies, __global REAL4* positions, __global REAL4* velocities, __global REAL3 *edges, int numEdges) {

	REAL3 Fp = ZERO3;
	REAL3 Fs = ZERO3;
	REAL3 Fd = ZERO3;
	
	unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
    unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    
    REAL4 actPos = positions[index];
    REAL4 actVel = positions[index];
    
    int indN = 0;
    
    float Ks = 0.0001;
    float Kd = 0.0001;
    REAL4 tmpVec = ZERO4;
 	REAL4 tmpVecVel = ZERO4;
    REAL vectorSize = 1.0;
    
    for (unsigned int i = 0; i < numEdges; i++) {
    	
    	barrier(CLK_LOCAL_MEM_FENCE);
    	
    	tmpVec = ZERO4;
    	tmpVecVel = ZERO4;
    	Fs.x = 0;
    	Fs.y = 0;
    	Fs.z = 0;
    	Fd.x = 0;
    	Fd.y = 0;
    	Fd.z = 0;
    	
    	vectorSize = 1.0;
    	
    	if (index == edges[i].x) {
    		indN = (int)edges[i].y;
    	}
    	else if (index == edges[i].y) {
    		indN = (int)edges[i].x;
    	}
    	else {
    		continue;
    	}
    	
    	tmpVec.x = actPos.x - positions[indN].x;
    	tmpVec.y = actPos.y - positions[indN].y;
    	tmpVec.z = actPos.z - positions[indN].z;
    	tmpVec.w = 1.0;
    	
    	tmpVecVel.x = actVel.x - velocities[indN].x;
    	tmpVecVel.y = actVel.y - velocities[indN].y;
    	tmpVecVel.z = actVel.z - velocities[indN].z;
    	tmpVecVel.w = 1.0;
    	
    	vectorSize = rsqrt(tmpVec.x*tmpVec.x + tmpVec.y*tmpVec.y + tmpVec.z*tmpVec.z);
    	
    	// vypocitame Fs
    	Fs.x = Ks*(tmpVec.x/vectorSize)*(vectorSize-edges[i].z);
    	Fs.y = Ks*(tmpVec.y/vectorSize)*(vectorSize-edges[i].z);
    	Fs.z = Ks*(tmpVec.z/vectorSize)*(vectorSize-edges[i].z);
    	
    	// vypocitame Fd
    	Fd.x = Kd*(tmpVecVel.x)*(tmpVec.x/vectorSize);
    	Fd.y = Kd*(tmpVecVel.y)*(tmpVec.y/vectorSize);
    	Fd.z = Kd*(tmpVecVel.z)*(tmpVec.z/vectorSize);
    	
    	// dostaneme Fp
    	Fp.x = (Fs.x+Fd.x);
    	Fp.y = (Fs.y+Fd.y);
    	Fp.z = (Fs.z+Fd.z);
    	
    	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    return Fp;
}

REAL3 computeFp_new(int numBodies, int numEdges, __global REAL3 *edges, __global REAL4* oldForces) {

	REAL3 Fp = ZERO3;

	unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
    unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    
    int indN = 0;
    
    for (unsigned int i = 0; i < numEdges; i++) {
    	
    	barrier(CLK_LOCAL_MEM_FENCE);
    	
    	if (index == edges[i].x) {
    		indN = (int)edges[i].y;
    	}
    	else if (index == edges[i].y) {
    		indN = (int)edges[i].x;
    	}
    	else {
    		continue;
    	}
    	
    	Fp.x += oldForces[indN].x;
    	Fp.y += oldForces[indN].y;
    	Fp.z += oldForces[indN].z;
    	
    	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    return Fp;
}

REAL3 computeAcceleration(__global REAL4* newForces, __global REAL4* oldForces, REAL4 bodyPos, 
                           __global REAL4* positions, 
                           int numBodies, 
                           REAL softeningSquared, 
                           __local REAL4* sharedPos,
                           REAL4 bodyVel, REAL4 Fc_a, __global REAL4* velocities, __global REAL3 *edges, int numEdges)
{

    REAL3 acc = ZERO3;
    
    REAL3 Fg = ZERO3; // gravitacna
    REAL3 Fd = ZERO3; // odpor prostredia
    REAL3 Fc = ZERO3; // impulz od pouzivatela
    REAL3 Fp = ZERO3; // pruzna (medzi bodmi)
    
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
	unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    
    REAL4 Fsum = oldForces[index];
    
    // todo: in this function will solve all forces
    
    Fg = computeFg(numBodies);
    Fd = computeFd(numBodies, bodyVel);
    Fc = computeFc(numBodies, Fc_a);
    //Fp = computeFp(numBodies, positions, velocities, edges, numEdges);
    //Fp = computeFp_new(numBodies, numEdges, edges, oldForces);
    
    Fsum.x = Fg.x + Fd.x + Fc.x + Fp.x;
    Fsum.y = Fg.y + Fd.y + Fc.y + Fp.y;
    Fsum.z = Fg.z + Fd.z + Fc.z + Fp.z;
    newForces[index] = Fsum;
    
    //acc = (Fg+Fd+Fc+Fp)/1; //mass is 1
  	acc.x = (Fg.x + Fd.x + Fc.x + Fp.x) / 1;
  	acc.y = (Fg.y + Fd.y + Fc.y + Fp.y) / 1;
  	acc.z = (Fg.z + Fd.z + Fc.z + Fp.z) / 1;
  
  
    return acc;
}

__kernel void integrateBodies_MT(
            __global REAL4* newPos,
            __global REAL4* newVel, 
            __global REAL4* newF,
            __global REAL3* newEdge,
            __global REAL4* newForces,
            __global REAL4* oldPos,
            __global REAL4* oldVel,
            __global REAL4* oldF,
            __global REAL3* oldEdge,
            __global REAL4* oldForces,
            REAL deltaTime,
            REAL damping,
            REAL softeningSquared,
            int numBodies,
            __local REAL4* sharedPos, int numEdges)
{
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);

    unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    REAL4 pos = oldPos[index];
    REAL4 vel = oldVel[index];  
    REAL4 Fc = oldF[index];
    REAL3 accel = computeAcceleration(newForces, oldForces, pos, oldPos, numBodies, softeningSquared, sharedPos, vel, Fc, oldVel, oldEdge, numEdges);

    // acceleration = force \ mass; 
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
    // (because they cancel out).  Thus here force == acceleration
       
    vel.x += accel.x * deltaTime;
    vel.y += accel.y * deltaTime;
    vel.z += accel.z * deltaTime;  

    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;
        
    // new position = old position + velocity * deltaTime
    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

    // store new position and velocity
    newPos[index] = pos;
    newVel[index] = vel;
    if((Fc.x > 0) || (Fc.y > 0) || (Fc.z > 0)) {
    	Fc.x = 0;
    	Fc.y = 0;
    	Fc.z = 0;
    }
    newF[index] = Fc;
    for (int i = 0; i < numEdges; i++) {
    	newEdge[i] = oldEdge[i];
    }
}

__kernel void integrateBodies_noMT(
            __global REAL4* newPos,
            __global REAL4* newVel, 
            __global REAL4* newF,
            __global REAL3* newEdge,
            __global REAL4* newForces,
            __global REAL4* oldPos,
            __global REAL4* oldVel,
            __global REAL4* oldF,
            __global REAL3* oldEdge,
            __global REAL4* oldForces,
            REAL deltaTime,
            REAL damping,
            REAL softeningSquared,
            int numBodies,
            __local REAL4* sharedPos, int numEdges)
{
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);

    unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    REAL4 pos = oldPos[index];   
    REAL3 accel = computeBodyAccel_noMT(pos, oldPos, numBodies, softeningSquared, sharedPos);

    // acceleration = force \ mass; 
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
    // (because they cancel out).  Thus here force == acceleration
    REAL4 vel = oldVel[index];
       
    vel.x += accel.x * deltaTime;
    vel.y += accel.y * deltaTime;
    vel.z += accel.z * deltaTime;  

    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;
        
    // new position = old position + velocity * deltaTime
    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

    // store new position and velocity
    newPos[index] = pos;
    newVel[index] = vel;
}

