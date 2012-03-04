/**************************************************************************************/
/*                                                                                    */
/*  Visualization Library                                                             */
/*  http://www.visualizationlibrary.org                                               */
/*                                                                                    */
/*  Copyright (c) 2005-2010, Michele Bosi                                             */
/*  All rights reserved.                                                              */
/*                                                                                    */
/*  Redistribution and use in source and binary forms, with or without modification,  */
/*  are permitted provided that the following conditions are met:                     */
/*                                                                                    */
/*  - Redistributions of source code must retain the above copyright notice, this     */
/*  list of conditions and the following disclaimer.                                  */
/*                                                                                    */
/*  - Redistributions in binary form must reproduce the above copyright notice, this  */
/*  list of conditions and the following disclaimer in the documentation and/or       */
/*  other materials provided with the distribution.                                   */
/*                                                                                    */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            */
/*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR  */
/*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    */
/*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      */
/*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON    */
/*  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     */
/*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      */
/*                                                                                    */
/**************************************************************************************/

#ifndef MCApplet_INCLUDE_ONCE
#define MCApplet_INCLUDE_ONCE

//simulation headers
#include "oclBodySystem.h"
#include "oclBodySystemOpencl.h"
#include "oclBodySystemOpenclLaunch.h"

//VL
#include <vlGraphics/Applet.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/SceneManagerActorTree.hpp>
#include <vlGraphics/Geometry.hpp>
#include <vlGraphics/Rendering.hpp>
#include <vlGraphics/Actor.hpp>
#include <vlCore/Time.hpp>
#include <vlGraphics/Effect.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/GLSL.hpp>

#include <math.h>
#include <memory>
#include <cassert>

#include "defines.h"
#include "tables.h"

#include <CL/cl_gl.h>

#include <GL/glx.h>

#define SCALE 10

#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"

#define REFRESH_DELAY	  10 //ms

#include "oclScan_common.h"

class MCAnimator: public vl::ActorEventCallback
{
public:
	MCAnimator()
	{
		cPathAndName = NULL;

		g_glInterop = true;
		pArgc = NULL;
		pArgv = NULL;

		gridSizeLog2[0]=5;
		gridSizeLog2[1]=5;
		gridSizeLog2[2]=5;
		gridSizeLog2[3]=0;

		numVoxels    = 0;
		maxVerts     = 0;
		activeVoxels = 0;
		totalVerts   = 0;


		radius = 5.5f;

		isoValue = 0.5f;
		dIsoValue = 0.005f;

		d_pos = 0;
		d_normal = 0;

		pointCnt = 1000;

		d_voxelVerts = 0;
		d_voxelVertsScan = 0;
		d_voxelOccupied = 0;
		d_voxelOccupiedScan = 0;

		// tables
		d_numVertsTable = 0;
		d_triTable = 0;
		totalTime = 0;

		frameCheckNumber=4;

		// Auto-Verification Code

		fpsLimit = 100;        // FPS limit for sampling
		g_Index = 0;
		frameCount = 0;
		g_TotalErrors = 0;
		g_bNoprompt = false;
		bQATest = false;

		numBodies = 1000;
		int p = 256;
		int q = 1;

		//simulation init
		hPos =0;
		hVel = 0;
		hF = 0; // Fc
		hForces = 0; //all Forces
		hColor = 0;
		hEdge = 0; // array of all edges
		nEdges =0;

		m_timestep=0.003f;
		m_clusterScale=1.54f;
		m_velocityScale=1.0f;
		m_softening=0.1f;
		m_damping=0.8f;
		m_pointSize=5.0f;
		m_x=0;
		m_y=-2;
		m_z=-100;

		initCL();

		if ((q * p) > 256)
		{
			p = 256 / q;
		}

		if ((q == 1) && (numBodies < p))
		{
			p = numBodies;
		}



		InitNbody(cdDevices[uiDeviceUsed], cxGPUContext, cqCommandQueue, numBodies, p, q, false, false, NBODY_CONFIG_SHELL);
		ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, false);

		initMC();
	}

	vl::Geometry * getNewGeometry() {
		vl::Geometry *geom = new vl::Geometry;

		vl::ref<vl::ArrayFloat4> vert4 = new vl::ArrayFloat4;

		vert4->setBufferObjectDirty(false);
		vert4->bufferObject()->setHandle(posVbo);

		geom->setVertexArray(vert4.get());

		vl::ref<vl::ArrayFloat3> norm3 = new vl::ArrayFloat3;

		norm3->setBufferObjectDirty(false);
		norm3->bufferObject()->setHandle(normalVbo);

		geom->setNormalArray(norm3.get());

		polys = new vl::DrawArrays(vl::PT_TRIANGLES, 0, 0);

		geom->drawCalls()->push_back(polys.get());

		return geom;
	}

	virtual void onActorRenderStarted(vl::Actor*, vl::real frame_clock, const vl::Camera*, vl::Renderable* renderable, const vl::Shader*, int pass)
	{

		if (pass>0)
			return;

		//run kernels to update particle positions


		nbody->update(m_timestep);


		// run kernels to generate geometry
		computeIsosurface();

		polys->setCount(totalVerts);

		vl::ref<vl::Geometry> geom = vl::cast<vl::Geometry>( renderable );

		geom->setBoundsDirty(true);
	}

	virtual void onActorDelete(vl::Actor*) {
		Cleanup(0);
	}

protected:
	//simulation fields
	int numBodies;
	BodySystemOpenCL * nbody;
	float* hPos;
	float* hVel;
	float* hF; // Fc
	float* hForces; //all Forces
	float* hColor;
	float* hEdge; // array of all edges
	int nEdges;

	float m_timestep;
	float m_clusterScale;
	float m_velocityScale;
	float m_softening;
	float m_damping;
	float m_pointSize;
	float m_x, m_y, m_z;

	//--

	cl_platform_id cpPlatform;
	cl_uint uiNumDevices;
	cl_device_id* cdDevices;
	cl_uint uiDeviceUsed;
	cl_uint uiDevCount;
	cl_context cxGPUContext;
	cl_device_id device;
	cl_command_queue cqCommandQueue;
	cl_program cpProgram;
	cl_kernel calcFieldValueKernel;
	cl_kernel classifyVoxelKernel;
	cl_kernel compactVoxelsKernel;
	cl_kernel generateTriangles2Kernel;
	cl_int ciErrNum;

	char* cPathAndName;          		// var for full paths to data, src, etc.
	char* cSourceCL;                    // Buffer to hold source for compilation
	cl_bool g_glInterop;

	int *pArgc;
	char **pArgv;

	class dim3 {
	public:
		size_t x;
		size_t y;
		size_t z;

		dim3(size_t _x=1, size_t _y=1, size_t _z=1) {x = _x; y = _y; z = _z;}
	};

	cl_uint gridSizeLog2[4];
	cl_uint gridSizeShift[4];
	cl_uint gridSize[4];
	cl_uint gridSizeMask[4];

	cl_float voxelSize[4];
	uint numVoxels;
	uint maxVerts;
	uint activeVoxels;
	uint totalVerts;

	cl_float radius;
	float isoValue;
	float dIsoValue;

	// device data
	GLuint posVbo, normalVbo;

	GLint  gl_Shader;

	cl_mem d_pos;
	cl_mem d_normal;

	cl_mem d_volumeData;
	cl_uint pointCnt;

	cl_mem d_voxelVerts;
	cl_mem d_voxelVertsScan;
	cl_mem d_voxelOccupied;
	cl_mem d_voxelOccupiedScan;
	cl_mem d_compVoxelArray;

	// tables
	cl_mem d_numVertsTable;
	cl_mem d_triTable;

	double totalTime;

	// Auto-Verification Code
	int frameCheckNumber;
	unsigned int fpsLimit;        // FPS limit for sampling
	int g_Index;
	unsigned int frameCount;
	unsigned int g_TotalErrors;
	bool g_bNoprompt;
	bool bQATest;
	const char* cpExecutableName;

	vl::ref<vl::DrawArrays> polys;



	void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL)
	{
	    shrLog("\nReset Nbody System...\n\n");

	    /*shrLog("\nnEdges = %d\n", nEdges);

	    // initalize the memory
	    randomizeBodies(&nEdges, config, hPos, hVel, hF, hForces, hColor, activeParams.m_clusterScale,
			            activeParams.m_velocityScale, numBodies);*/

	    shrLog("\nnEdges = %d\n", nEdges);

	    //initialize edges
	    //hEdge = new float[nEdges*3];

	    float tmpNUM = pow(numBodies, 1./3.);
	    float tmpINC = (SCALE)/tmpNUM;
	    int n = 0;
	    int p = 0;
	    bool bX = true, bY = true, bZ = true; // 1
	    bool bZY = true, bZmY = true, bYX = true, bYmX = true, bZX = true, bZmX = true; // 2
	    bool bmXYZ = true, bXYZ = true, bXmYZ = true, bmXmYZ = true; // 3

	    for (int i = 0; i < numBodies; i++) {
	    	for (int j = 0; j < numBodies; j++) {
	    		/*
	    		 * 1 = 3 kusy
	    		 */
	    		if ((hPos[j*4] == (hPos[i*4] + tmpINC)) && bX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4+2] == hPos[i*4+2])) { // x
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC;
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bX = false;
	    		}
	    		else if ((hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && bY && (hPos[j*4] == hPos[i*4]) && (hPos[j*4+2] == hPos[i*4+2])) { // y
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC;
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bY = false;
	    		}
	    		else if ((hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && bZ && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4] == hPos[i*4])) { // z
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC;
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bZ = false;
	    		}

	    		/*
	    		 * 2 = 6 kusov
	    		 */

	    		else if (bZY && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == hPos[i*4])) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(2); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bZY = false;
	    		}
	    		else if (bZmY && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] - tmpINC)) && (hPos[j*4] == hPos[i*4])) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(2); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bZmY = false;
	    		}
	    		else if (bYX && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+2] == hPos[i*4+2])) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(2); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bYX = false;
	    		}
	    		else if (bYmX && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+2] == hPos[i*4+2])) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(2); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bYmX = false;
	    		}
	    		else if (bZX && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+1] == hPos[i*4+1])) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(2); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bZX = false;
	    		}
	    		else if (bZmX && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+1] == hPos[i*4+1])) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(2); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bZmX = false;
	    		}
	    		/*
	    		 * 3 = 4 kusy
	    		 */
	    		else if (bmXYZ && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(3); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bmXYZ = false;
	    		}
	    		else if (bXYZ && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(3); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bXYZ = false;
	    		}
	    		else if (bXmYZ && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] - tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(3); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bXmYZ = false;
	    		}
	    		else if (bmXmYZ && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] - tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;

	    			hEdge[p++] = i;
	    			hEdge[p++] = j;
	    			hEdge[p++] = tmpINC*sqrt(3); // stenova uhlopriecka v kocke
	    			hEdge[p++] = 1.0f;

	    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

	    			bmXmYZ = false;
	    		}
	    	}
	    	//shrLog("\n");
	    	bX = true;
	    	bY = true;
	    	bZ = true;
	    	bZY = true, bZmY = true, bYX = true, bYmX = true, bZX = true, bZmX = true;
	    	bmXYZ = true, bXYZ = true, bXmYZ = true, bmXmYZ = true;
	    }

	    shrLog("\nnEdges created = %d\n", n);

	    for (int i = n; i < nEdges; i++) {
	    	hEdge[i*4] = -1;
	    	hEdge[i*4+1] = -1;
	    	hEdge[i*4+2] = 0;
	    	hEdge[i*4+3] = 0.0f;
	    }

	    for (int i = 0; i < nEdges; i++) {
	    	shrLog("E: %d ==> p1: %d; p2: %d; l0: %f\n", i, (int)hEdge[i*4], (int)hEdge[i*4+1], hEdge[i*4+2]);
	    }

	    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
	    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
	    system->setArray(BodySystem::BODYSYSTEM_F, hF);
	    system->setArray(BodySystem::BODYSYSTEM_FORCES, hForces);
	    system->setArray(BodySystem::BODYSYSTEM_EDGE, hEdge);
	}

	void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
	               int numBodies, int p, int q, bool bUsePBO, bool bDouble, NBodyConfig config)
	{
	    // allocate host memory
	    hPos = new float[numBodies*4];
	    hVel = new float[numBodies*4];
	    hF = new float[numBodies*4];
	    hForces = new float[numBodies*4];
	    hColor = new float[numBodies*4];

	    // initalize the memory
	    randomizeBodies(&nEdges, config, hPos, hVel, hF, hForces, hColor, m_clusterScale,
	    		m_velocityScale, numBodies);

	    float tmpNUM = pow(numBodies, 1./3.);
	    float tmpINC = (SCALE)/tmpNUM;
	    int n = 0;
	    bool bX = true, bY = true, bZ = true; // 1
	    bool bZY = true, bZmY = true, bYX = true, bYmX = true, bZX = true, bZmX = true; // 2
	    bool bmXYZ = true, bXYZ = true, bXmYZ = true, bmXmYZ = true; // 3

	    for (int i = 0; i < numBodies; i++) {
	    	for (int j = 0; j < numBodies; j++) {
	    		/*
	    		 * 1 = 3 kusy
	    		 */
	    		if ((hPos[j*4] == (hPos[i*4] + tmpINC)) && bX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4+2] == hPos[i*4+2])) { // x
	    			n++;
	    			bX = false;
	    		}
	    		else if ((hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && bY && (hPos[j*4] == hPos[i*4]) && (hPos[j*4+2] == hPos[i*4+2])) { // y
	    			n++;
	    			bY = false;
	    		}
	    		else if ((hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && bZ && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4] == hPos[i*4])) { // z
	    			n++;
	    			bZ = false;
	    		}

	    		/*
	    		 * 2 = 6 kusov
	    		 */

	    		else if (bZY && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == hPos[i*4])) {
	    			n++;
	    			bZY = false;
	    		}
	    		else if (bZmY && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] - tmpINC)) && (hPos[j*4] == hPos[i*4])) {
	    			n++;
	    			bZmY = false;
	    		}
	    		else if (bYX && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+2] == hPos[i*4+2])) {
	    			n++;
	    			bYX = false;
	    		}
	    		else if (bYmX && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+2] == hPos[i*4+2])) {
	    			n++;
	    			bYmX = false;
	    		}
	    		else if (bZX && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+1] == hPos[i*4+1])) {
	    			n++;
	    			bZX = false;
	    		}
	    		else if (bZmX && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+1] == hPos[i*4+1])) {
	    			n++;
	    			bZmX = false;
	    		}
	    		/*
	    		 * 3 = 4 kusy
	    		 */
	    		else if (bmXYZ && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;
	    			bmXYZ = false;
	    		}
	    		else if (bXYZ && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;
	    			bXYZ = false;
	    		}
	    		else if (bXmYZ && (hPos[j*4] == (hPos[i*4] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] - tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;
	    			bXmYZ = false;
	    		}
	    		else if (bmXmYZ && (hPos[j*4] == (hPos[i*4] - tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] - tmpINC)) && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC))) {
	    			n++;
	    			bmXmYZ = false;
	    		}
	    	}
	    	//shrLog("\n");
	    	bX = true;
	    	bY = true;
	    	bZ = true;
	    	bZY = true, bZmY = true, bYX = true, bYmX = true, bZX = true, bZmX = true;
	    	bmXYZ = true, bXYZ = true, bXmYZ = true, bmXmYZ = true;
	    }

	    if (n <= 128) {
	    	nEdges = n;
	    } else {
	    	if (n <= 256)
	    		nEdges = 256;
	    	else if (n <= 512)
	    		nEdges= 512;
	    	else if (n <= 1024)
	    		nEdges = 1024;
	    	else if (n <= 2048)
	    		nEdges = 2048;
	    	else if (n <= 4096)
	    		nEdges = 4096;
	    	else if (n <= 8192)
	    		nEdges = 8192;
	    	else if (n <= 16384)
	    		nEdges = 16384;
	    	else if(n <= 32768)
	    		nEdges = 32768;
	    	else if (n <= 65536)
	    		nEdges = 65536;
	    }

	 //   nEdges = n;

	    hEdge = new float[nEdges*4];

	    // New nbody system for Device/GPU computations
	    nbody = new BodySystemOpenCL(numBodies, nEdges, dev, ctx, cmdq, p, q, bUsePBO, bDouble);

	    // Set sim parameters
	    nbody->setSoftening(m_softening);
	    nbody->setDamping(m_damping);
	}

	void randomizeBodies(int *numEdges, NBodyConfig config, float* pos, float* vel, float* force, float *forces, float* color, float clusterScale, float velocityScale, int numBodies)
	{
		*numEdges = 0;

		shrLog("\nRandomize Bodies...\n\n");

		shrLog("\nNBODY_CONFIG_SHELL...\n\n");

		float i, j, k;
		//float tmpNUM = sqrt(sqrt(numBodies));
		float tmpNUM = pow(numBodies, 1./3.);
		float tmpINC = (SCALE)/tmpNUM;
		int p=0, v=0;

		//shrLog("tmpNUM: %f, tmpINC: %f, sucin: %f\n\n", tmpNUM, tmpINC, tmpNUM*tmpINC);

		for (i = 10; i < ((tmpNUM*tmpINC)-0.001)+10; i += tmpINC) {
			for (j = 10; j < ((tmpNUM*tmpINC)-0.001)+10; j += tmpINC) {
				for (k = 10 ; k < ((tmpNUM*tmpINC)-0.001)+10; k += tmpINC) {
					shrLog("BOD - %d - x: %f; y: %f; z: %f\n", int(p/4), i, j, k);
					force[p] = 0.0f;
					forces[p] = 0.0f;
					pos[p++] = i;
					force[p] = 0.0f;
					forces[p] = 0.0f;
					pos[p++] = j;
					force[p] = 0.0f;
					forces[p] = 0.0f;
					pos[p++] = k;
					force[p] = 1.0f;
					forces[p] = 0.0f;
					pos[p++] = 1.0f;

					vel[v++] = 0.0f;
					vel[v++] = 0.0f;
					vel[v++] = 0.0f;
					vel[v++] = 1.0f;

					*numEdges += 3;

					if ((j+tmpINC) >= ((tmpNUM*tmpINC)-0.001)+10) {
										force[p-3] = 200.0f;
									}
				}
				*numEdges -= 1;

			}
			*numEdges -= (int)tmpNUM;
		}
		*numEdges -= ((int)tmpNUM*(int)tmpNUM);

		fprintf(stdout, "\nnumEdges in randomize bodies: %d\n", *numEdges);



	//	force[(numBodies-1)*4+1] = 200.0f;


		//		force[7*4+1] = 20.0f;
//		force[8*4+1] = 20.0f;
//		force[15*4+1] = 20.0f;
//		force[16*4+1] = 20.0f;
//		force[17*4+1] = 20.0f;
//		force[24*4+1] = 20.0f;
//		force[25*4+1] = 20.0f;
//		force[26*4+1] = 20.0f;

	}

	void
	openclScan(cl_mem d_voxelOccupiedScan, cl_mem d_voxelOccupied, int numVoxels) {
		scanExclusiveLarge(
						   cqCommandQueue,
						   d_voxelOccupiedScan,
						   d_voxelOccupied,
						   1,
						   numVoxels);
	}

	void launch_calcFieldValue(dim3 grid, dim3 threads, cl_mem volumeData, cl_mem points, cl_uint pointCnt, cl_float radius, cl_uint gridSizeShift[4], cl_uint gridSizeMask[4]) {
		clSetKernelArg(calcFieldValueKernel, 0, sizeof(cl_mem), &volumeData);
		clSetKernelArg(calcFieldValueKernel, 1, sizeof(cl_mem), &points);
		clSetKernelArg(calcFieldValueKernel, 2, sizeof(cl_uint), &pointCnt);
		clSetKernelArg(calcFieldValueKernel, 3, sizeof(cl_float), &radius);
		clSetKernelArg(calcFieldValueKernel, 4, 4 * sizeof(cl_uint), gridSizeShift);
		clSetKernelArg(calcFieldValueKernel, 5, 4 * sizeof(cl_uint), gridSizeMask);

		grid.x *= threads.x;
		ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, calcFieldValueKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
	}

	void
	launch_classifyVoxel( dim3 grid, dim3 threads, cl_mem voxelVerts, cl_mem voxelOccupied, cl_mem volumeData,
						  cl_uint gridSize[4], cl_uint gridSizeShift[4], cl_uint gridSizeMask[4], uint numVoxels,
						  cl_float voxelSize[4], float isoValue)
	{
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 0, sizeof(cl_mem), &voxelVerts);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 1, sizeof(cl_mem), &voxelOccupied);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 2, sizeof(cl_mem), &volumeData);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 3, 4 * sizeof(cl_uint), gridSize);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 4, 4 * sizeof(cl_uint), gridSizeShift);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 5, 4 * sizeof(cl_uint), gridSizeMask);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 6, sizeof(uint), &numVoxels);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 7, 4 * sizeof(cl_float), voxelSize);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 8, sizeof(float), &isoValue);
		ciErrNum = clSetKernelArg(classifyVoxelKernel, 9, sizeof(cl_mem), &d_numVertsTable);

		grid.x *= threads.x;
		ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, classifyVoxelKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
	}

	void
	launch_compactVoxels(dim3 grid, dim3 threads, cl_mem compVoxelArray, cl_mem voxelOccupied, cl_mem voxelOccupiedScan, uint numVoxels)
	{
		ciErrNum = clSetKernelArg(compactVoxelsKernel, 0, sizeof(cl_mem), &compVoxelArray);
		ciErrNum = clSetKernelArg(compactVoxelsKernel, 1, sizeof(cl_mem), &voxelOccupied);
		ciErrNum = clSetKernelArg(compactVoxelsKernel, 2, sizeof(cl_mem), &voxelOccupiedScan);
		ciErrNum = clSetKernelArg(compactVoxelsKernel, 3, sizeof(cl_uint), &numVoxels);

		grid.x *= threads.x;
		ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, compactVoxelsKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
	}

	void
	launch_generateTriangles2(dim3 grid, dim3 threads,
							  cl_mem pos, cl_mem norm, cl_mem compactedVoxelArray, cl_mem numVertsScanned, cl_mem volumeData,
							  cl_uint gridSize[4], cl_uint gridSizeShift[4], cl_uint gridSizeMask[4],
							  cl_float voxelSize[4], float isoValue, uint activeVoxels, uint maxVerts)
	{
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 0, sizeof(cl_mem), &pos);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 1, sizeof(cl_mem), &norm);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 2, sizeof(cl_mem), &compactedVoxelArray);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 3, sizeof(cl_mem), &numVertsScanned);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 4, sizeof(cl_mem), &volumeData);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 5, 4 * sizeof(cl_uint), gridSize);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 6, 4 * sizeof(cl_uint), gridSizeShift);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 7, 4 * sizeof(cl_uint), gridSizeMask);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 8, 4 * sizeof(cl_float), voxelSize);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 9, sizeof(float), &isoValue);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 10, sizeof(uint), &activeVoxels);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 11, sizeof(uint), &maxVerts);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 12, sizeof(cl_mem), &d_numVertsTable);
		ciErrNum = clSetKernelArg(generateTriangles2Kernel, 13, sizeof(cl_mem), &d_triTable);

		grid.x *= threads.x;
		ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, generateTriangles2Kernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
	}

	void allocateTextures(	cl_mem *d_triTable, cl_mem* d_numVertsTable )
		{
			cl_image_format imageFormat;
			imageFormat.image_channel_order = CL_R;
			imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;


			*d_triTable = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										  &imageFormat,
										  16,256,0, (void*) triTable, &ciErrNum );


			*d_numVertsTable = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											   &imageFormat,
											   256,1,0, (void*) numVertsTable, &ciErrNum );

		}

	////////////////////////////////////////////////////////////////////////////////
	// initialize marching cubes
	////////////////////////////////////////////////////////////////////////////////
	void
	initMC()
	{
		gridSize[0] = 1<<gridSizeLog2[0];
		gridSize[1] = 1<<gridSizeLog2[1];
		gridSize[2] = 1<<gridSizeLog2[2];


		gridSizeMask[0] = gridSize[0]-1;
		gridSizeMask[1] = gridSize[1]-1;
		gridSizeMask[2] = gridSize[2]-1;

		gridSizeShift[0] = 0;
		gridSizeShift[1] = gridSizeLog2[0];
		gridSizeShift[2] = gridSizeLog2[0]+gridSizeLog2[1];

		numVoxels = gridSize[0]*gridSize[1]*gridSize[2];


		voxelSize[0] = 2.0f / gridSize[0];
		voxelSize[1] = 2.0f / gridSize[1];
		voxelSize[2] = 2.0f / gridSize[2];

		maxVerts = gridSize[0]*gridSize[1]*100;

		printf("grid: %d x %d x %d = %d voxels\n", gridSize[0], gridSize[1], gridSize[2], numVoxels);
		printf("max verts = %d\n", maxVerts);

		//pointCnt=729;
		//cl_int points[729][4];

		/*int i,j,k, pointIndex=0;

		for(i=12;i<21;i+=1) {
			for(j=12;j<21;j+=1) {
				for(k=12;k<21;k+=1) {
					points[pointIndex][0]=i;
					points[pointIndex][1]=j;
					points[pointIndex++][2]=k;
				}
			}
		}*/

		// create VBOs
		if( !bQATest) {
			createVBO(&posVbo, maxVerts*sizeof(float)*4, d_pos);
			createVBO(&normalVbo, maxVerts*sizeof(float)*3, d_normal);
		}

		// allocate textures
		allocateTextures(&d_triTable, &d_numVertsTable );

		// allocate device memory
		unsigned int memSize = sizeof(uint) * numVoxels;
		d_voxelVerts = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
		d_voxelVertsScan = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
		d_voxelOccupied = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
		d_voxelOccupiedScan = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
		d_compVoxelArray = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
		d_volumeData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*numVoxels, 0, &ciErrNum);
	}

	void initCL() {
		//Get the NVIDIA platform
		ciErrNum = oclGetPlatformID(&cpPlatform);

		// Get the number of GPU devices available to the platform
		ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);

		// Create the device list
		cdDevices = new cl_device_id [uiDevCount];
		ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);

		// Get device requested on command line, if any
		uiDeviceUsed = 0;
		unsigned int uiEndDev = uiDevCount - 1;

		// Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
		if(g_glInterop)
		{
			bool bSharingSupported = false;
			for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i)
			{
				size_t extensionSize;
				ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
				if(extensionSize > 0)
				{
					char* extensions = (char*)malloc(extensionSize);
					ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
					std::string stdDevString(extensions);
					free(extensions);

					size_t szOldPos = 0;
					size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
					while (szSpacePos != stdDevString.npos)
					{
						if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 )
						{
							// Device supports context sharing with OpenGL
							uiDeviceUsed = i;
							bSharingSupported = true;
							break;
						}
						do
						{
							szOldPos = szSpacePos + 1;
							szSpacePos = stdDevString.find(' ', szOldPos);
						}
						while (szSpacePos == szOldPos);
					}
				}
			}

			shrLog("%s...\n\n", bSharingSupported ? "Using CL-GL Interop" : "No device found that supports CL/GL context sharing");

			// Define OS-specific context properties and create the OpenCL context

			cl_context_properties props[] =
			{
				CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
				CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
				CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
				0
			};
			cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);

		}
		else
		{
			// No GL interop
			cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
			cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);

			g_glInterop = false;
		}

		oclPrintDevInfo(LOGBOTH, cdDevices[uiDeviceUsed]);

		// create a command-queue
		cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);

		// Program Setup
		size_t program_length;
		cPathAndName = shrFindFilePath("marchingCubes_kernel.cl",NULL);
		cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);

		// create the program
		cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
						  (const char **)&cSourceCL, &program_length, &ciErrNum);

		// build the program
		std::string buildOpts = "-cl-mad-enable";
		ciErrNum = clBuildProgram(cpProgram, 0, NULL, buildOpts.c_str(), NULL, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			// write out standard error, Build Log and PTX, then cleanup and return error
			shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
			oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
			oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMarchinCubes.ptx");
			Cleanup(EXIT_FAILURE);
		}

		// create the kernel
		calcFieldValueKernel = clCreateKernel(cpProgram, "calcFieldValue", &ciErrNum);

		classifyVoxelKernel = clCreateKernel(cpProgram, "classifyVoxel", &ciErrNum);

		compactVoxelsKernel = clCreateKernel(cpProgram, "compactVoxels", &ciErrNum);

		generateTriangles2Kernel = clCreateKernel(cpProgram, "generateTriangles2", &ciErrNum);

		// Setup Scan
		initScan(cxGPUContext, cqCommandQueue);
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Run the Cuda part of the computation
	////////////////////////////////////////////////////////////////////////////////
	void
	computeIsosurface()
	{
		int threads = 128;
		dim3 grid(numVoxels / threads, 1, 1);
		// get around maximum grid size of 65535 in each dimension
		if (grid.x > 65535) {
			grid.y = grid.x / 32768;
			grid.x = 32768;
		}

		launch_calcFieldValue(grid, threads,
				d_volumeData, nbody->getPos(), pointCnt, radius, gridSizeShift, gridSizeMask);

		// calculate number of vertices need per voxel
		launch_classifyVoxel(grid, threads,

							d_voxelVerts, d_voxelOccupied, d_volumeData,//nbody->getPos(),

							gridSize, gridSizeShift, gridSizeMask,
							 numVoxels, voxelSize, isoValue);

		// scan voxel occupied array
		openclScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

		// read back values to calculate total number of non-empty voxels
		// since we are using an exclusive scan, the total is the last value of
		// the scan result plus the last value in the input array
		{
			uint lastElement, lastScanElement;

			clEnqueueReadBuffer(cqCommandQueue, d_voxelOccupied,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
			clEnqueueReadBuffer(cqCommandQueue, d_voxelOccupiedScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

			activeVoxels = lastElement + lastScanElement;
		}
		printf("activeVoxels = %d\n", activeVoxels);

		if (activeVoxels==0) {
			// return if there are no full voxels
			totalVerts = 0;
			return;
		}


		// compact voxel index array
		launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);


		// scan voxel vertex count array
		openclScan(d_voxelVertsScan, d_voxelVerts, numVoxels);

		// readback total number of vertices
		{
			uint lastElement, lastScanElement;
			clEnqueueReadBuffer(cqCommandQueue, d_voxelVerts,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
			clEnqueueReadBuffer(cqCommandQueue, d_voxelVertsScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

			totalVerts = lastElement + lastScanElement;
		}

		//printf("totalVerts = %d\n", totalVerts);


		cl_mem interopBuffers[] = {d_pos, d_normal};

		// generate triangles, writing to vertex buffers
		if( g_glInterop ) {
			// Acquire PBO for OpenCL writing
			glFlush();
			ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 2, interopBuffers, 0, 0, 0);

			//fprintf(stderr, "%d\n", ciErrNum);
		}

		dim3 grid2((int) ceil(activeVoxels / (float) NTHREADS), 1, 1);

		while(grid2.x > 65535) {
			grid2.x/=2;
			grid2.y*=2;
		}
		launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal,
												d_compVoxelArray,

												d_voxelVertsScan, d_volumeData,

												gridSize, gridSizeShift, gridSizeMask,
												voxelSize, isoValue, activeVoxels,
								  maxVerts);

		if( g_glInterop ) {
			// Transfer ownership of buffer back from CL to GL
			ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 2, interopBuffers, 0, 0, 0);
			clFinish( cqCommandQueue );
		}

	}

		////////////////////////////////////////////////////////////////////////////////
		//! Create VBO
		////////////////////////////////////////////////////////////////////////////////
		void
		createVBO(GLuint* vbo, unsigned int size, cl_mem &vbo_cl)
		{
			// create buffer object
			vl::glGenBuffers(1, vbo);
			vl::glBindBuffer(GL_ARRAY_BUFFER, *vbo);

			// initialize buffer object
			vl::glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
			vl::glBindBuffer(GL_ARRAY_BUFFER, 0);

			vbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, *vbo, &ciErrNum);
		}

		////////////////////////////////////////////////////////////////////////////////
		//! Delete VBO
		////////////////////////////////////////////////////////////////////////////////
		void
		deleteVBO(GLuint* vbo, cl_mem vbo_cl)
		{
			if( vbo_cl) clReleaseMemObject(vbo_cl);

			if( *vbo ) {
				vl::glBindBuffer(1, *vbo);
				vl::glDeleteBuffers(1, vbo);

				*vbo = 0;
			}
		}

		void Cleanup(int iExitCode)
		{
			deleteVBO(&posVbo, d_pos);
			deleteVBO(&normalVbo, d_normal);

			if( d_triTable ) clReleaseMemObject(d_triTable);
			if( d_numVertsTable ) clReleaseMemObject(d_numVertsTable);

			if( d_voxelVerts) clReleaseMemObject(d_voxelVerts);
			if( d_voxelVertsScan) clReleaseMemObject(d_voxelVertsScan);
			if( d_voxelOccupied) clReleaseMemObject(d_voxelOccupied);
			if( d_voxelOccupiedScan) clReleaseMemObject(d_voxelOccupiedScan);
			if( d_compVoxelArray) clReleaseMemObject(d_compVoxelArray);
			if( d_volumeData) clReleaseMemObject(d_volumeData);

			closeScan();

			if(calcFieldValueKernel)clReleaseKernel(calcFieldValueKernel);
			if(compactVoxelsKernel)clReleaseKernel(compactVoxelsKernel);
			if(compactVoxelsKernel)clReleaseKernel(generateTriangles2Kernel);
			if(compactVoxelsKernel)clReleaseKernel(classifyVoxelKernel);
			if(cpProgram)clReleaseProgram(cpProgram);

			if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
			if(cxGPUContext)clReleaseContext(cxGPUContext);

			if ((g_bNoprompt)||(bQATest))
			{
				shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cpExecutableName);
			}
			else
			{
				shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cpExecutableName);
				#ifdef WIN32
					getchar();
				#endif
			}
			exit (iExitCode);
		}
};


class MCApplet: public vl::Applet
{
public:

  // called once after the OpenGL window has been opened
	void initEvent()
	{
		// allocate the Transform
		transform = new vl::Transform;
		// bind the Transform with the transform tree of the rendring pipeline
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );


		// setup the effect to be used to render the cube
		vl::ref<vl::Effect> effect = new vl::Effect;
		// enable depth test and lighting
		effect->shader()->enable(vl::EN_DEPTH_TEST);

		effect->shader()->enable(vl::EN_NORMALIZE);
		// add a Light to the scene, since no Transform is associated to the Light it will follow the camera
		effect->shader()->setRenderState( new vl::Light, 0 );
		// enable the standard OpenGL lighting
		effect->shader()->enable(vl::EN_LIGHTING);

		vl::ref<vl::GLSLVertexShader> perpixellight_vs = new vl::GLSLVertexShader("./glsl/perpixellight.vs");

		vl::ref<vl::GLSLProgram> glsl = new vl::GLSLProgram;
		glsl->attachShader( perpixellight_vs.get() );
		glsl->attachShader( new vl::GLSLFragmentShader("./glsl/perpixellight_interlaced.fs") );

		effect->shader()->setRenderState(glsl.get());

		// install our scene manager, we use the SceneManagerActorTree which is the most generic
		vl::ref<vl::SceneManagerActorTree> scene_manager = new vl::SceneManagerActorTree;
		rendering()->as<vl::Rendering>()->sceneManagers()->push_back(scene_manager.get());

		MCAnimator *animator = new MCAnimator();

		vl::ref<vl::Actor> actor=scene_manager->tree()->addActor( animator->getNewGeometry(), effect.get(), transform.get());
		actor->actorEventCallbacks()->push_back(animator);
	}

	// called every frame
	virtual void updateScene()
	{
		printf("%f\n",fps());
	}



protected:
	vl::ref<vl::Transform> transform;
};

#endif
