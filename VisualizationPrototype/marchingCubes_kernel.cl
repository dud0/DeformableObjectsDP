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

// #include "defines.h"
// #include "tables.h"


// The number of threads to use for triangle generation (limited by shared memory size)
#define NTHREADS 32

// volume data
sampler_t volumeSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
sampler_t tableSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


// compute position in 3d grid from 1d index
// only works for power of 2 sizes
int4 calcGridPos(uint i, uint4 gridSizeShift, uint4 gridSizeMask)
{
    int4 gridPos;
    gridPos.x = (i & gridSizeMask.x);
    gridPos.y = ((i >> gridSizeShift.y) & gridSizeMask.y);
    gridPos.z = ((i >> gridSizeShift.z) & gridSizeMask.z);
    return gridPos;
}

float calcPointContribution(float4 point, int4 gridPos, float radius)
{
	float distSqr;

	distSqr = (pow((float)(point.x-gridPos.x), 2.0f) + pow((float)(point.y-gridPos.y), 2.0f) + pow((float)(point.z-gridPos.z), 2.0f));

	if (distSqr>pow(radius, 2.0f)) 
	{
		return 0.0f;
	}
	else
	{
		return (- 0.444444f * pow(distSqr,3.0f) / pow(radius, 6.0f)) + (1.888889f * pow(distSqr,2.0f) / pow(radius, 4.0f)) - (2.444444f * distSqr / pow(radius, 2.0f)) + 1;
	}	
}

__kernel
void
calcFieldValue(__global float *volumeData, __global float4 *points, uint count, float radius, uint4 gridSizeShift, uint4 gridSizeMask)
{
	int index = get_global_id(0);

	int4 gridPos = calcGridPos(index, gridSizeShift, gridSizeMask);

	int i;
	float sum=0;

	float4 minPoint, maxPoint;

	minPoint.x = gridPos.x - radius;
	minPoint.y = gridPos.y - radius;
	minPoint.z = gridPos.z - radius;

	maxPoint.x = gridPos.x + radius;
	maxPoint.y = gridPos.y + radius;
	maxPoint.z = gridPos.z + radius;	

	for (i=0;i<count;i++)
	{
		if (points[i].x>=minPoint.x && points[i].y>=minPoint.y && points[i].z>=minPoint.z && points[i].x<=maxPoint.x && points[i].y<=maxPoint.y && points[i].z<=maxPoint.z) {
			sum+=calcPointContribution(points[i], gridPos, radius);
		}		
	}

	volumeData[index]=sum;
}

float getFieldValue(__global float *volumeData, int4 gridPos, uint4 gridSizeShift) {
	int index = ((gridPos.x | (gridPos.y << gridSizeShift.y)) | (gridPos.z << gridSizeShift.z));
	return volumeData[index];
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
__kernel
void
classifyVoxel(__global uint* voxelVerts, __global uint *voxelOccupied, __global float *volumeData,
              uint4 gridSize, uint4 gridSizeShift, uint4 gridSizeMask, uint numVoxels,
              float4 voxelSize, float isoValue,  __read_only image2d_t numVertsTex)
{
    uint blockId = get_group_id(0);
    uint i = get_global_id(0);

    int4 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

    // read field values at neighbouring grid vertices
    float field[8];
    field[0] = getFieldValue(volumeData, gridPos, gridSizeShift);
    field[1] = getFieldValue(volumeData, gridPos + (int4)(1, 0, 0 ,0), gridSizeShift);
    field[2] = getFieldValue(volumeData, gridPos + (int4)(1, 1, 0 ,0), gridSizeShift);
    field[3] = getFieldValue(volumeData, gridPos + (int4)(0, 1, 0 ,0), gridSizeShift);
    field[4] = getFieldValue(volumeData, gridPos + (int4)(0, 0, 1 ,0), gridSizeShift);
    field[5] = getFieldValue(volumeData, gridPos + (int4)(1, 0, 1 ,0), gridSizeShift);
    field[6] = getFieldValue(volumeData, gridPos + (int4)(1, 1, 1 ,0), gridSizeShift);
    field[7] = getFieldValue(volumeData, gridPos + (int4)(0, 1, 1 ,0), gridSizeShift);

    // calculate flag indicating if each vertex is inside or outside isosurface
    int cubeindex;
	cubeindex =  (field[0] < isoValue); 
	cubeindex += (field[1] < isoValue)*2; 
	cubeindex += (field[2] < isoValue)*4; 
	cubeindex += (field[3] < isoValue)*8; 
	cubeindex += (field[4] < isoValue)*16; 
	cubeindex += (field[5] < isoValue)*32; 
	cubeindex += (field[6] < isoValue)*64; 
	cubeindex += (field[7] < isoValue)*128;

    // read number of vertices from texture
    uint numVerts = read_imageui(numVertsTex, tableSampler, (int2)(cubeindex,0)).x;

    if (i < numVoxels) {
        voxelVerts[i] = numVerts;
        voxelOccupied[i] = (numVerts > 0);
    }
}
     

// compact voxel array
__kernel
void
compactVoxels(__global uint *compactedVoxelArray, __global uint *voxelOccupied, __global uint *voxelOccupiedScan, uint numVoxels)
{
    uint i = get_global_id(0);

    if (voxelOccupied[i] && (i < numVoxels)) {
        compactedVoxelArray[ voxelOccupiedScan[i] ] = i;
    }
}



// compute interpolated vertex along an edge
float4 vertexInterp(float isolevel, float4 p0, float4 p1, float f0, float f1)
{
    float t = (isolevel - f0) / (f1 - f0);
	return mix(p0, p1, t);
} 

// compute interpolated vertex position and normal along an edge
void vertexInterp2(float isolevel, float4 p0, float4 p1, float4 f0, float4 f1, float4* p, float4* n)
{
    float t = (isolevel - f0.w) / (f1.w - f0.w);
	*p = mix(p0, p1, t);
    (*n).x = mix(f0.x, f1.x, t);
    (*n).y = mix(f0.y, f1.y, t);
    (*n).z = mix(f0.z, f1.z, t);
//    n = normalize(n);
} 



// calculate triangle normal
float4 calcNormal(float4 v0, float4 v1, float4 v2)
{
    float4 edge0 = v1 - v0;
    float4 edge1 = v2 - v0;
    // note - it's faster to perform normalization in vertex shader rather than here
    return cross(edge0, edge1);
}

// version that calculates flat surface normal for each triangle
__kernel
void
generateTriangles2(__global float4 *pos, __global float *norm, __global uint *compactedVoxelArray, __global uint *numVertsScanned, 
                   __global float *volumeData,
                   uint4 gridSize, uint4 gridSizeShift, uint4 gridSizeMask,
                   float4 voxelSize, float isoValue, uint activeVoxels, uint maxVerts, 
                   __read_only image2d_t numVertsTex, __read_only image2d_t triTex)
{
    uint i = get_global_id(0);
    uint tid = get_local_id(0);

    if (i > activeVoxels - 1) {
        i = activeVoxels - 1;
    }

    uint voxel = compactedVoxelArray[i];

    // compute position in 3d grid
    int4 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

    float4 p;
    p.x = -1.0f + (gridPos.x * voxelSize.x);
    p.y = -1.0f + (gridPos.y * voxelSize.y);
    p.z = -1.0f + (gridPos.z * voxelSize.z);
    p.w = 1.0f;

    // calculate cell vertex positions
    float4 v[8];
    v[0] = p;
    v[1] = p + (float4)(voxelSize.x, 0, 0,0);
    v[2] = p + (float4)(voxelSize.x, voxelSize.y, 0,0);
    v[3] = p + (float4)(0, voxelSize.y, 0,0);
    v[4] = p + (float4)(0, 0, voxelSize.z,0);
    v[5] = p + (float4)(voxelSize.x, 0, voxelSize.z,0);
    v[6] = p + (float4)(voxelSize.x, voxelSize.y, voxelSize.z,0);
    v[7] = p + (float4)(0, voxelSize.y, voxelSize.z,0);

    float4 field[8];
    field[0].w = getFieldValue(volumeData, gridPos, gridSizeShift);
    field[1].w = getFieldValue(volumeData, gridPos + (int4)(1, 0, 0 ,0), gridSizeShift);
    field[2].w = getFieldValue(volumeData, gridPos + (int4)(1, 1, 0 ,0), gridSizeShift);
    field[3].w = getFieldValue(volumeData, gridPos + (int4)(0, 1, 0 ,0), gridSizeShift);
    field[4].w = getFieldValue(volumeData, gridPos + (int4)(0, 0, 1 ,0), gridSizeShift);
    field[5].w = getFieldValue(volumeData, gridPos + (int4)(1, 0, 1 ,0), gridSizeShift);
    field[6].w = getFieldValue(volumeData, gridPos + (int4)(1, 1, 1 ,0), gridSizeShift);
    field[7].w = getFieldValue(volumeData, gridPos + (int4)(0, 1, 1 ,0), gridSizeShift);

    field[0].x = field[0].w - field[1].w;
    field[0].y = field[0].w - field[3].w;
    field[0].z = field[0].w - field[4].w;

    field[1].x = -field[1].w + field[0].w;
    field[1].y =  field[1].w - field[2].w;
    field[1].z =  field[1].w - field[5].w;

    field[2].x = -field[2].w + field[3].w;
    field[2].y = -field[2].w + field[1].w;
    field[2].z =  field[2].w - field[6].w;

    field[3].x =  field[3].w - field[2].w;
    field[3].y = -field[3].w + field[0].w;
    field[3].z =  field[3].w - field[7].w;

    field[4].x =  field[4].w - field[5].w;
    field[4].y =  field[4].w - field[7].w;
    field[4].z = -field[4].w + field[0].w;

    field[5].x = -field[5].w + field[4].w;
    field[5].y =  field[5].w - field[6].w;
    field[5].z = -field[5].w + field[1].w;

    field[6].x = -field[6].w + field[7].w;
    field[6].y = -field[6].w + field[5].w;
    field[6].z = -field[6].w + field[2].w;

    field[7].x =  field[7].w - field[6].w;
    field[7].y = -field[7].w + field[4].w;
    field[7].z = -field[7].w + field[3].w;

    // recalculate flag
    int cubeindex;
	cubeindex =  (field[0].w < isoValue); 
	cubeindex += (field[1].w < isoValue)*2; 
	cubeindex += (field[2].w < isoValue)*4; 
	cubeindex += (field[3].w < isoValue)*8; 
	cubeindex += (field[4].w < isoValue)*16; 
	cubeindex += (field[5].w < isoValue)*32; 
	cubeindex += (field[6].w < isoValue)*64; 
	cubeindex += (field[7].w < isoValue)*128;

	// find the vertices where the surface intersects the cube 
	float4 vertlist[12];
	float4 normlist[12];

	vertexInterp2(isoValue, v[0], v[1], field[0], field[1], &vertlist[0], &normlist[0]);
	vertexInterp2(isoValue, v[1], v[2], field[1], field[2], &vertlist[1], &normlist[1]);
	vertexInterp2(isoValue, v[2], v[3], field[2], field[3], &vertlist[2], &normlist[2]);
	vertexInterp2(isoValue, v[3], v[0], field[3], field[0], &vertlist[3], &normlist[3]);
	vertexInterp2(isoValue, v[4], v[5], field[4], field[5], &vertlist[4], &normlist[4]);
	vertexInterp2(isoValue, v[5], v[6], field[5], field[6], &vertlist[5], &normlist[5]);
	vertexInterp2(isoValue, v[6], v[7], field[6], field[7], &vertlist[6], &normlist[6]);
	vertexInterp2(isoValue, v[7], v[4], field[7], field[4], &vertlist[7], &normlist[7]);
	vertexInterp2(isoValue, v[0], v[4], field[0], field[4], &vertlist[8], &normlist[8]);
	vertexInterp2(isoValue, v[1], v[5], field[1], field[5], &vertlist[9], &normlist[9]);
	vertexInterp2(isoValue, v[2], v[6], field[2], field[6], &vertlist[10], &normlist[10]);
	vertexInterp2(isoValue, v[3], v[7], field[3], field[7], &vertlist[11], &normlist[11]);

    // output triangle vertices
    uint numVerts = read_imageui(numVertsTex, tableSampler, (int2)(cubeindex,0)).x;

    for(int i=0; i<numVerts; i+=3) {
        uint index = numVertsScanned[voxel] + i;

        float4 v[3];
	float4 n[3];
        uint edge;
        edge = read_imageui(triTex, tableSampler, (int2)(i,cubeindex)).x;
        v[0] = vertlist[edge];
	n[0] = normlist[edge];

        edge = read_imageui(triTex, tableSampler, (int2)(i+1,cubeindex)).x;
        v[1] = vertlist[edge];
	n[1] = normlist[edge];

        edge = read_imageui(triTex, tableSampler, (int2)(i+2,cubeindex)).x;
        v[2] = vertlist[edge];
	n[2] = normlist[edge];

        // calculate triangle surface normal
        //float4 n = calcNormal(v[0], v[1], v[2]);

        if (index < (maxVerts - 3)) {
		pos[index] = v[0];
        	norm[index*3] = n[0].x;
		norm[index*3 + 1] = n[0].y;
		norm[index*3 + 2] = n[0].z;

	        pos[index+1] = v[1];
        	norm[index*3 + 3] = n[1].x;
		norm[index*3 + 4] = n[1].y;
		norm[index*3 + 5] = n[1].z;

	        pos[index+2] = v[2];
        	norm[index*3 + 6] = n[2].x;
		norm[index*3 + 7] = n[2].y;
		norm[index*3 + 8] = n[2].z;
        }
    }
}
