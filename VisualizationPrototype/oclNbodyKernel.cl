

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
    Fg.y = /*-9.82300*/0;
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
    Fd.x = -0.3*bodyVel.x;
    Fd.y = -0.3*bodyVel.y;
    Fd.z = -0.3*bodyVel.z;
	
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
    
    Fg = getFg(numBodies);
    Fd = getFd(numBodies, velocities[index]);
    Fc = getFc(numBodies, Fc_a[index]);
   
    Fsum.x = Fg.x + Fd.x + Fc.x;
    Fsum.y = Fg.y + Fd.y + Fc.y;
    Fsum.z = Fg.z + Fd.z + Fc.z;
    Fsum.w = 1;
   
    newForces[index] = Fsum;
    newFc[index] = Fc_a[index];
 }
 
 __kernel void springsForces(__global REAL4* newForces,
 			__global REAL4* newEdges,
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
    
    REAL Ks = 20;
   
   	REAL restL = oldEdges[index].z;
   
   	int P1 = 0, P2 = 0; 
	P1 = oldEdges[index].x;
	P2 = oldEdges[index].y;
	
	REAL3 tmpVec = ZERO3;
	tmpVec.x = oldPositions[P1].x - oldPositions[P2].x;
	tmpVec.y = oldPositions[P1].y - oldPositions[P2].y;
	tmpVec.z = oldPositions[P1].z - oldPositions[P2].z;
   
  	REAL vectorLength;
  	vectorLength = sqrt(tmpVec.x*tmpVec.x + tmpVec.y*tmpVec.y + tmpVec.z*tmpVec.z);
    	
    // vypocitame Fs
    REAL3 Fs = ZERO3;
    Fs.x = Ks * (restL - vectorLength) * (tmpVec.x / vectorLength);
    Fs.y = Ks * (restL - vectorLength) * (tmpVec.y / vectorLength);
    Fs.z = Ks * (restL - vectorLength) * (tmpVec.z / vectorLength);    
   
	newForces[P1].x += Fs.x;
	newForces[P1].y += Fs.y;
	newForces[P1].z += Fs.z;
	
	newForces[P2].x -= Fs.x;
	newForces[P2].y -= Fs.y;
	newForces[P2].z -= Fs.z;
	
	newEdges[index] = oldEdges[index];
 
 }
 
 __kernel void integrateBodies(__global REAL4* newPos,
            __global REAL4* newVel, 
            __global REAL4* newEdg,
            __global REAL4* oldPos,
            __global REAL4* oldVel,
            __global REAL4* oldEdg,
            __global REAL4* oldForces,
            REAL deltaTime,
            REAL damping) {
 
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
    REAL3 accel = ZERO3;

    // acceleration = force \ mass;
    // mass is 1
    REAL mass;
    mass = 1;
    
    accel.x = oldForces[index].x / mass;
    accel.y = oldForces[index].y / mass;
    accel.z = oldForces[index].z / mass;
     
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
 }
 