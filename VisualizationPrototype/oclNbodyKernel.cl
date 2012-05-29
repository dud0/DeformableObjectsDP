

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


/************************************************************************************************
 ************************************************************************************************
 ************************************************************************************************
 *** D e f o r m a b l e   o b j e c t s   S I M U L A T I O N **********************************
 ************************************************************************************************
 ************************************************************************************************
 ************************************************************************************************
 *                                                                                              *
 * 			N  E  W     I  M  P  L  E  M  E  N  T  A  T  I  O  N                                *
 *                                                                                              *
 ************************************************************************************************/
 
 /*********************************** P H Y S I C S *********************************************/
 
 REAL3 getFg(int numBodies) {
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
    Fg.y = 0;
    Fg.z = 0;
	
	return Fg;
}

REAL3 getFd(int numBodies, REAL4 bodyVel) {

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
   /* Fd.x = -0.3*bodyVel.x;
    Fd.y = -0.3*bodyVel.y;
    Fd.z = -0.3*bodyVel.z;
	*/
	return Fd;
}

REAL3 getFc(int numBodies, REAL4 Fc) {

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
 
 /*********************************** K E R N E L S *********************************************/
 
 __kernel void externForces(__global REAL4* newForces,   
 							__global REAL4* newFc,
 							__global REAL4* oldForces,
                           __global REAL4* Fc_a, 
                           __global REAL4* velocities,
                           int numBodies) {
 	
 	REAL3 Fg = ZERO3; // gravitacna
    REAL3 Fd = ZERO3; // odpor prostredia
    REAL3 Fc = ZERO3; // impulz od pouzivatela
    REAL4 Fsum = ZERO4; // celkova sila
   
    
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numBodies / mul24(blockDimx, blockDimy);
    
	unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    
    if (oldForces[index].w == 0) {
    	return;
    } 
    else {
    	Fg = getFg(numBodies);
    	Fd = getFd(numBodies, velocities[index]);
   		Fc = getFc(numBodies, Fc_a[index]);
   
    	Fsum.x = Fg.x + Fd.x + Fc.x;
    	Fsum.y = Fg.y + Fd.y + Fc.y;
    	Fsum.z = Fg.z + Fd.z + Fc.z;
    	Fsum.w = 1;
   
    	newForces[index] = Fsum;
    	newFc[index] = Fc_a[index];
 	
 		//oldForces[index] = newForces[index];
 	}
 }
 
 __kernel void springsForces(__global REAL4* newForces,
 			__global REAL4* newEdges,
 			__global REAL4* newPositions,
 			__global REAL4* oldForces,
 			__global REAL4* oldPositions, 
            __global REAL4* oldEdges,
            int numEdges) {
 	
	unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int blockIdxy = get_group_id(1);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int numTiles = numEdges / mul24(blockDimx, blockDimy);
    
    unsigned int index = mul24(blockIdxx, blockDimx) + threadIdxx;
    
    if (oldEdges[index].w == 0) {
    	// mrtve vazby - neriesime
    	return;
    }
    else {
    
    	REAL Ks = 50;
    	REAL TRHANIE = 2.5;
   
   		REAL restL = oldEdges[index].z;
   
   		//int P1 = 0, P2 = 0; 
		int P1 = oldEdges[index].x;
		int P2 = oldEdges[index].y;
	
		REAL3 tmpVec = ZERO3;
		tmpVec.x = oldPositions[P1].x - oldPositions[P2].x;
		tmpVec.y = oldPositions[P1].y - oldPositions[P2].y;
		tmpVec.z = oldPositions[P1].z - oldPositions[P2].z;
   
  		REAL vectorLength;
  		vectorLength = sqrt(tmpVec.x*tmpVec.x + tmpVec.y*tmpVec.y + tmpVec.z*tmpVec.z);
   
   		/*if(oldPositions[P1].w ==3) {
   			Ks = 100;
   			TRHANIE = 4.0;
   			if (vectorLength >= restL*2) {
    			oldEdges[index].z = restL*2;
    		}
   		}*/
    	
    	// trhanie
    	if (vectorLength >= restL*TRHANIE) {
    		oldEdges[index].w = 0;
    	}
    	
    	// plasticita
    	/*if (vectorLength >= restL*2) {
    		oldEdges[index].z = restL*2;
    	}*/
    	
    	// vypocitame Fs
    	REAL3 Fs = ZERO3;
    	Fs.x = Ks * (restL - vectorLength) * (tmpVec.x / vectorLength);
    	Fs.y = Ks * (restL - vectorLength) * (tmpVec.y / vectorLength);
    	Fs.z = Ks * (restL - vectorLength) * (tmpVec.z / vectorLength);    
   
   barrier(CLK_LOCAL_MEM_FENCE);
   
		newForces[P1].x += Fs.x;
		newForces[P1].y += Fs.y;
		newForces[P1].z += Fs.z;

		newForces[P2].x -= Fs.x;
		newForces[P2].y -= Fs.y;
		newForces[P2].z -= Fs.z;
	
		newEdges[index] = oldEdges[index];
	}
 }
 
 __kernel void integrateBodies(__global REAL4* newPos,
            __global REAL4* newVel, 
            __global REAL4* newEdg,
            __global REAL4* newForces,
            __global REAL4* oldPos,
            __global REAL4* oldVel,
            __global REAL4* oldEdg,
            __global REAL4* oldForces,
            REAL deltaTime,
            REAL damping) {
    
    unsigned int index = get_global_id(0);   
    
    REAL4 force = oldForces[index];
    REAL4 pos = oldPos[index];
    REAL4 vel = oldVel[index];  
    REAL3 accel = ZERO3;

	if (pos.w == 0) {
		return;
	}
	else {
    	// acceleration = force \ mass;
    	// mass is 1
    	REAL mass;
    	mass = 1;
    
    	accel.x = oldForces[index].x / mass;
    	accel.y = oldForces[index].y / mass;
    	accel.z = oldForces[index].z / mass;
       
    	vel.x += accel.x * deltaTime;
    	vel.y += accel.y * deltaTime;
    	vel.z += accel.z * deltaTime;  

   		vel.x *= damping;
    	vel.y *= damping;
    	vel.z *= damping;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
    	// new position = old position + velocity * deltaTime
    	pos.x += vel.x * deltaTime;
    	pos.y += vel.y * deltaTime;
    	pos.z += vel.z * deltaTime;

		//Collide with cube
    	if(pos.x < 1.0f + 0.5){
        	pos.x = 1.0f + 0.5;
        	//force.x += 100;
        	vel.x *= -0.5;
        	newForces[index] = force;
    	}
    	if(pos.x > 31.0f - 0.5){
        	pos.x = 31.0f - 0.5;
        	//force.x -= 100.0f;
        	vel.x *= -0.5;
        	newForces[index] = force;
    	}

    	if(pos.y < 1.0f + 0.5){
        	pos.y = 1.0f + 0.5;
        	//force.y += 100;
        	vel.y *= -0.5; 
        	newForces[index] = force;
   	 	}
    	if(pos.y > 31.0f - 0.5){
        	pos.y = 31.0f - 0.5;
        	//force.y -= 100;
        	vel.y *= -0.5;
        	newForces[index] = force;
    	}

    	if(pos.z < 1.0f + 0.5){
        	pos.z = 1.0f + 0.5;
        	//force.z += 100;
        	vel.z *= -0.5;
        	newForces[index] = force;
    	}
    	if(pos.z > 31.0f - 0.5){
        	pos.z = 31.0f - 0.5;
        	//force.z -= 100;
        	vel.z *= -0.5;
        	newForces[index] = force;
    	}

    	// store new position and velocity
    	newPos[index] = pos;
    	newVel[index] = vel;
    	//newForces[index] = force;
    }
 }
 
 __kernel void integrateVerlet(__global REAL4* newPos, __global REAL4* oldPos, __global REAL4* newBeforePos, __global REAL4* oldBeforePos, __global REAL4* oldForces, REAL deltaTime) {
 
 
	unsigned int index = get_global_id(0);
	
	REAL4 xOld = oldBeforePos[index];
	REAL4 xNew = oldPos[index];
	REAL3 accel = ZERO3;
	REAL mass = 1;
 
 	if (xOld.w == 0) {
 		return;
 	}
 	else {
 		accel.x = oldForces[index].x / mass;
 		accel.y = oldForces[index].y / mass;
 		accel.z = oldForces[index].z / mass;
 	
 		xNew.x = (2 * oldPos[index].x) - oldBeforePos[index].x + (accel.x * (deltaTime*deltaTime));
 		xNew.y = (2 * oldPos[index].y) - oldBeforePos[index].y + (accel.y * (deltaTime*deltaTime));
 		xNew.z = (2 * oldPos[index].z) - oldBeforePos[index].z + (accel.z * (deltaTime*deltaTime));
 	
 		xOld.x = oldPos[index].x;
 		xOld.y = oldPos[index].y;
 		xOld.z = oldPos[index].z;
 	
 		newPos[index] = xNew;
 		newBeforePos[index] = xOld;
 	}
 }
 
 
 /*
  *
  *  C O L L I S I O N     D E T E C T I O N
  *
  */
 
#define UMAD(a, b, c)  ( (a) * (b) + (c) ) 
 
////////////////////////////////////////////////////////////////////////////////
// Save particle grid cell hashes and indices
////////////////////////////////////////////////////////////////////////////////
int4 getGridPos(float4 p){
    int4 gridPos;
    gridPos.x = (int)floor((p.x) / 0.5);
    gridPos.y = (int)floor((p.y) / 0.5);
    gridPos.z = (int)floor((p.z) / 0.5);
    gridPos.w = 0;
    return gridPos;
}

//Calculate address in grid from position (clamping to edges)
uint getGridHash(int4 gridPos){
    //Wrap addressing, assume power-of-two grid dimensions
    gridPos.x = gridPos.x & (64 - 1);
    gridPos.y = gridPos.y & (64 - 1);
    gridPos.z = gridPos.z & (64 - 1);
    return UMAD( UMAD(gridPos.z, 64, gridPos.y), 64, gridPos.x );
}


//Calculate grid hash value for each particle
__kernel void calcHash(
    __global REAL4       *d_Hash, //output
    __global REAL4       *d_Index, //output
    __global const REAL4 *d_Pos, //input: positions
    uint numParticles
){
    const uint index = get_global_id(0);
    if(index >= numParticles)
        return;

    float4 p = d_Pos[index];

    //Get address in grid
    int4  gridPos = getGridPos(p);
    uint gridHash = getGridHash(gridPos);

    //Store grid hash and particle index
    d_Hash[index].x = gridHash;
    d_Hash[index].y = 0;
    d_Hash[index].z = 0;
    d_Hash[index].w = 0;
    
    d_Index[index].x = index;
    d_Index[index].y = 0;
    d_Index[index].z = 0;
    d_Index[index].w = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Find cell bounds and reorder positions+velocities by sorted indices
////////////////////////////////////////////////////////////////////////////////
__kernel void Memset(
    __global REAL4 *d_Data,
    uint val,
    uint N
){
    if(get_global_id(0) < N) {
        d_Data[get_global_id(0)].x = val;
        d_Data[get_global_id(0)].y = 0;
        d_Data[get_global_id(0)].z = 0;
        d_Data[get_global_id(0)].w = 0;
    }
}

__kernel void findCellBoundsAndReorder(
    __global REAL4   *d_CellStart,     //output: cell start index
    __global REAL4   *d_CellEnd,       //output: cell end index
    __global float4 *d_ReorderedPos,  //output: reordered by cell hash positions
    __global float4 *d_ReorderedVel,  //output: reordered by cell hash velocities
	__global float4 *d_ReorderedForce, //output: reordered by cell hash forces

    __global REAL4   *d_Hash,    //input: sorted grid hashes
    __global REAL4   *d_Index,   //input: particle indices sorted by hash
    __global REAL4 	 *d_Pos,     //input: positions array sorted by hash
    __global REAL4   *d_Vel,     //input: velocity array sorted by hash
    __global REAL4   *d_Forces,
    __local uint *localHash,          //get_group_size(0) + 1 elements
    uint    numParticles
){
    uint hash;
    const uint index = get_global_id(0);

    //Handle case when no. of particles not multiple of block size
    if(index < numParticles){
        hash = d_Hash[index].x;

        //Load hash data into local memory so that we can look 
        //at neighboring particle's hash value without loading
        //two hash values per thread
        localHash[get_local_id(0) + 1] = hash;

        //First thread in block must load neighbor particle hash
        if(index > 0 && get_local_id(0) == 0)
            localHash[0] = d_Hash[index - 1].x;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(index < numParticles){
        //Border case
        if(index == 0)
            d_CellStart[hash].x = 0;

        //Main case
        else{
            if(hash != localHash[get_local_id(0)])
                d_CellEnd[localHash[get_local_id(0)]].x  = d_CellStart[hash].x = index;
        };

        //Another border case
        if(index == numParticles - 1)
            d_CellEnd[hash].x = numParticles;


        //Now use the sorted index to reorder the pos and vel arrays
        uint sortedIndex = d_Index[index].x;
        float4 pos = d_Pos[sortedIndex];
        float4 vel = d_Vel[sortedIndex];
        float4 force = d_Forces[sortedIndex];

        d_ReorderedPos[index] = pos;
        d_ReorderedVel[index] = vel;
        d_ReorderedForce[index] = force;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Process collisions (calculate accelerations)
////////////////////////////////////////////////////////////////////////////////
float4 collideSpheres(
    float4 posA,
    float4 posB,
    float4 velA,
    float4 velB,
    float radiusA,
    float radiusB,
    float spring,
    float damping,
    float shear,
    float attraction
){
    //Calculate relative position
    float4     relPos = (float4)(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z, 0);
    float        dist = sqrt(relPos.x * relPos.x + relPos.y * relPos.y + relPos.z * relPos.z);
    float collideDist = radiusA + radiusB;

	// v pripade ze sa jedna o to iste teleso
	if (posA.w == posB.w) {
		return 0;
	}

    float4 force = (float4)(0, 0, 0, 0);
    if(dist < collideDist){
        float4 norm = (float4)(relPos.x / dist, relPos.y / dist, relPos.z / dist, 0);

        //Relative velocity
        float4 relVel = (float4)(velB.x - velA.x, velB.y - velA.y, velB.z - velA.z, 0);

        //Relative tangential velocity
        float relVelDotNorm = relVel.x * norm.x + relVel.y * norm.y + relVel.z * norm.z;
        float4 tanVel = (float4)(relVel.x - relVelDotNorm * norm.x, relVel.y - relVelDotNorm * norm.y, relVel.z - relVelDotNorm * norm.z, 0);

        //Spring force (potential)
        float springFactor = -spring * (collideDist - dist);
        force = (float4)(
            springFactor * norm.x + damping * relVel.x + shear * tanVel.x + attraction * relPos.x,
            springFactor * norm.y + damping * relVel.y + shear * tanVel.y + attraction * relPos.y,
            springFactor * norm.z + damping * relVel.z + shear * tanVel.z + attraction * relPos.z,
            0
        );
    }

    return force;
}



__kernel void collide(
    __global REAL4       *d_Vel,          //output: new velocity
    __global REAL4       *d_Forces,
    __global const float4 *d_ReorderedPos, //input: reordered positions
    __global const float4 *d_ReorderedVel, //input: reordered velocities
    __global const float4 *d_ReorderedForce,
    __global const float4   *d_Index,        //input: reordered particle indices
    __global const float4   *d_CellStart,    //input: cell boundaries
    __global const float4   *d_CellEnd,
    uint    numParticles
){
    uint index = get_global_id(0);
    if(index >= numParticles)
        return;

    float4   pos = d_ReorderedPos[index];
    float4   vel = d_ReorderedVel[index];
    float4   forces = d_ReorderedForce[index];
    float4 force = (float4)(0, 0, 0, 0);

    //Get address in grid
    int4 gridPos = getGridPos(pos);

    //Accumulate surrounding cells
    for(int z = -1; z <= 1; z++)
        for(int y = -1; y <= 1; y++)
            for(int x = -1; x <= 1; x++){
                //Get start particle index for this cell
                uint   hash = getGridHash(gridPos + (int4)(x, y, z, 0));
                uint startI = d_CellStart[hash].x;

                //Skip empty cell
                if(startI == 0xFFFFFFFFU)
                    continue;

                //Iterate over particles in this cell
                uint endI = d_CellEnd[hash].x;
                for(uint j = startI; j < endI; j++){
                    if(j == index)
                        continue;

                    float4 pos2 = d_ReorderedPos[j];
                    float4 vel2 = d_ReorderedVel[j];

                    //Collide two spheres
                    force += collideSpheres(
                        pos, pos2,
                        vel, vel2,
                        0.16, 0.16, 
                        0.9, 0.8, 0.12, 0.012
                    );
                }
            }

    //Write new velocity back to original unsorted location
    int ind;
    ind = (int)d_Index[index].x;
    d_Vel[ind] = vel + force;
    //d_Forces[ind] += (force*100);
}

////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into local memory)
////////////////////////////////////////////////////////////////////////////////

#define LOCAL_SIZE_LIMIT 512U

inline void ComparatorPrivate(
    uint *keyA,
    uint *valA,
    uint *keyB,
    uint *valB,
    uint dir
){
    if( (*keyA > *keyB) == dir ){
        uint t;
        t = *keyA; *keyA = *keyB; *keyB = t;
        t = *valA; *valA = *valB; *valB = t;
    }
}

inline void ComparatorLocal(
    __local uint *keyA,
    __local uint *valA,
    __local uint *keyB,
    __local uint *valB,
    uint dir
){
    if( (*keyA > *keyB) == dir ){
        uint t;
        t = *keyA; *keyA = *keyB; *keyB = t;
        t = *valA; *valA = *valB; *valB = t;
    }
}

//Bottom-level bitonic sort
//Almost the same as bitonicSortLocal with the only exception
//of even / odd subarrays (of LOCAL_SIZE_LIMIT points) being
//sorted in opposite directions
__kernel void bitonicSortLocal1(
    __global REAL4 *d_DstKey,
    __global REAL4 *d_DstVal,
    __global REAL4 *d_SrcKey,
    __global REAL4 *d_SrcVal
){
    __local uint l_key[LOCAL_SIZE_LIMIT];
    __local uint l_val[LOCAL_SIZE_LIMIT];

    //Offset to the beginning of subarray and load data
    d_SrcKey += get_group_id(0) * LOCAL_SIZE_LIMIT + get_local_id(0);
    d_SrcVal += get_group_id(0) * LOCAL_SIZE_LIMIT + get_local_id(0);
    d_DstKey += get_group_id(0) * LOCAL_SIZE_LIMIT + get_local_id(0);
    d_DstVal += get_group_id(0) * LOCAL_SIZE_LIMIT + get_local_id(0);
    l_key[get_local_id(0) +                      0] = d_SrcKey[                     0].x;
    l_val[get_local_id(0) +                      0] = d_SrcVal[                     0].x;
    l_key[get_local_id(0) + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)].x;
    l_val[get_local_id(0) + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)].x;

    uint comparatorI = get_global_id(0) & ((LOCAL_SIZE_LIMIT / 2) - 1);

    for(uint size = 2; size < LOCAL_SIZE_LIMIT; size <<= 1){
        //Bitonic merge
        uint ddd = (comparatorI & (size / 2)) != 0;
        for(uint stride = size / 2; stride > 0; stride >>= 1){
            barrier(CLK_LOCAL_MEM_FENCE);
            uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
            ComparatorLocal(
                &l_key[pos +      0], &l_val[pos +      0],
                &l_key[pos + stride], &l_val[pos + stride],
                ddd
            );
        }
    }

    //Odd / even arrays of LOCAL_SIZE_LIMIT elements
    //sorted in opposite directions
    {
        uint ddd = (get_group_id(0) & 1);
        for(uint stride = LOCAL_SIZE_LIMIT / 2; stride > 0; stride >>= 1){
            barrier(CLK_LOCAL_MEM_FENCE);
            uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
            ComparatorLocal(
                &l_key[pos +      0], &l_val[pos +      0],
                &l_key[pos + stride], &l_val[pos + stride],
               ddd
            );
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    d_DstKey[                     0].x = l_key[get_local_id(0) +                      0];
    d_DstVal[                     0].x = l_val[get_local_id(0) +                      0];
    d_DstKey[(LOCAL_SIZE_LIMIT / 2)].x = l_key[get_local_id(0) + (LOCAL_SIZE_LIMIT / 2)];
    d_DstVal[(LOCAL_SIZE_LIMIT / 2)].x = l_val[get_local_id(0) + (LOCAL_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for 'stride' >= LOCAL_SIZE_LIMIT
__kernel void bitonicMergeGlobal(
    __global REAL4 *d_DstKey,
    __global REAL4 *d_DstVal,
    __global REAL4 *d_SrcKey,
    __global REAL4 *d_SrcVal,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir
){
    uint global_comparatorI = get_global_id(0);
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    //Bitonic merge
    uint ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    uint keyA = d_SrcKey[pos +      0].x;
    uint valA = d_SrcVal[pos +      0].x;
    uint keyB = d_SrcKey[pos + stride].x;
    uint valB = d_SrcVal[pos + stride].x;

    ComparatorPrivate(
        &keyA, &valA,
        &keyB, &valB,
        ddd
    );

    d_DstKey[pos +      0].x = keyA;
    d_DstVal[pos +      0].x = valA;
    d_DstKey[pos + stride].x = keyB;
    d_DstVal[pos + stride].x = valB;
}