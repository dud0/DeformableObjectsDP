#ifndef MCApplet_INCLUDE_ONCE
#define MCApplet_INCLUDE_ONCE

#include "ConfigurationData.hpp"

#include <string>
#include <iostream>
#include <string.h>   // Required by strcpy()
#include <stdlib.h>   // Required by malloc()

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
#include <vlGraphics/DepthSortCallback.hpp>

#include <math.h>
#include <memory>
#include <cassert>

#include "defines.h"

#include <CL/cl_gl.h>

#include <GL/glx.h>

#include "CLManager.hpp"

#include "Perlin/perlin.c"

#define SCALE 5

#define REFRESH_DELAY	  10 //ms

using namespace std;

class EdgesAnimator: public vl::ActorEventCallback
{
public:
	EdgesAnimator(CLManager *clManager, BodySystemOpenCL *nBody, cl_uint edgeCnt, cl_uint edgeOffset)
	{
		this->clManager = clManager;
		this->nBody = nBody;

		objectDisplayMode = EDGE;
		d_edgePos = 0;

		this->edgeCnt = edgeCnt;
		this->edgeOffset = edgeOffset;

		frameCheckNumber=4;

		// Auto-Verification Code

		fpsLimit = 100;        // FPS limit for sampling
		g_Index = 0;
		frameCount = 0;
		g_TotalErrors = 0;
		g_bNoprompt = false;
		bQATest = false;

		createVBO(&edgePosVbo, edgeCnt*2*sizeof(float)*4, d_edgePos);
		createVBO(&edgeColorVbo, edgeCnt*2*sizeof(float)*4, d_edgeColor);
	}

	vl::Geometry * getNewEdgesGeometry() {
		vl::Geometry *geom = new vl::Geometry;

		vl::ref<vl::ArrayFloat4> vert4 = new vl::ArrayFloat4;

		vert4->setBufferObjectDirty(false);
		vert4->bufferObject()->setHandle(edgePosVbo);

		geom->setVertexArray(vert4.get());

		vl::ref<vl::ArrayFloat4> colors = new vl::ArrayFloat4();

		colors->setBufferObjectDirty(false);
		colors->bufferObject()->setHandle(edgeColorVbo);

		geom->setColorArray(colors.get());

		drawMode = new vl::DrawArrays(vl::PT_LINES, 0, edgeCnt*2);

		printf("\nEDGECOUNT: %d\n", edgeCnt);

		geom->drawCalls()->push_back(drawMode.get());

		return geom;
	}

	virtual void onActorRenderStarted(vl::Actor*, vl::real frame_clock, const vl::Camera*, vl::Renderable* renderable, const vl::Shader*, int pass)
	{

		if (pass>0)
			return;

		cl_mem interopBuffers[] = {d_edgePos, d_edgeColor};

		if( clManager->g_glInterop ) {
			glFlush();
			clEnqueueAcquireGLObjects(clManager->cqCommandQueue, 2, interopBuffers, 0, 0, 0);
		}

		//if (objectDisplayMode==EDGE) {
		clManager->launch_generateLines(16384, 128, d_edgePos, d_edgeColor, nBody->getEdges(), nBody->getPos(), edgeCnt, edgeOffset);
		//}
		//printf("\nEDGEOFFSET: %d\n", edgeOffset);
		//printf("\nEDGECOUNT: %d\n", edgeCnt);

		if( clManager->g_glInterop ) {
			clEnqueueReleaseGLObjects(clManager->cqCommandQueue, 2, interopBuffers, 0, 0, 0);
			clFinish( clManager->cqCommandQueue );
		}

		vl::ref<vl::Geometry> geom = vl::cast<vl::Geometry>( renderable );

		geom->setBoundsDirty(true);
	}

	void Cleanup(int iExitCode)
	{
		deleteVBO(&edgePosVbo, d_edgePos);
		deleteVBO(&edgeColorVbo, d_edgeColor);
	}

	virtual void onActorDelete(vl::Actor*) {
		Cleanup(0);
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

		vbo_cl = clCreateFromGLBuffer(clManager->cxGPUContext,CL_MEM_WRITE_ONLY, *vbo, &(clManager->ciErrNum));
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

	cl_uint global_work_size;

	// device data
	GLuint edgePosVbo;
	GLuint edgeColorVbo;

	cl_mem d_edgePos;
	cl_mem d_edgeColor;

	displayMode objectDisplayMode;

	cl_uint edgeCnt;
	cl_uint edgeOffset;

	// Auto-Verification Code
	int frameCheckNumber;
	unsigned int fpsLimit;        // FPS limit for sampling
	int g_Index;
	unsigned int frameCount;
	unsigned int g_TotalErrors;
	bool g_bNoprompt;
	bool bQATest;
	const char* cpExecutableName;

	vl::ref<vl::DrawArrays> drawMode;


	BodySystemOpenCL *nBody;
	CLManager *clManager;
};

class ObjectAnimator: public vl::ActorEventCallback
{
public:
	ObjectAnimator(CLManager *clManager, BodySystemOpenCL *nBody, cl_uint pointCnt, cl_uint pointOffset, float * pointPos, cl_uint edgeCnt, cl_uint edgeOffset)
	{
		this->clManager = clManager;
		this->nBody = nBody;

		objectDisplayMode = NORMAL;

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


		radius = 1.0f;

		isoValue = 0.5f;

		d_volumeData = 0;
		d_colorIntensities = 0;

		d_pos = 0;
		d_normal = 0;
		d_color = 0;

		global_work_size = 128;
		if (pointCnt<=128) {
			;
		} else {
			if (pointCnt <= 256)
				global_work_size = 256;
			else if (pointCnt <= 512)
				global_work_size= 512;
			else if (pointCnt <= 1024)
				global_work_size = 1024;
			else if (pointCnt <= 2048)
				global_work_size = 2048;
			else if (pointCnt <= 4096)
				global_work_size = 4096;
			else if (pointCnt <= 8192)
				global_work_size = 8192;
			else if (pointCnt <= 16384)
				global_work_size = 16384;
			else if(pointCnt <= 32768)
				global_work_size = 32768;
			else if (pointCnt <= 65536)
				global_work_size = 65536;
		}

		this->pointCnt = pointCnt;
		this->pointOffset = pointOffset;

		this->edgeCnt = edgeCnt;
		this->edgeOffset = edgeOffset;

		d_voxelVerts = 0;
		d_voxelVertsScan = 0;
		d_voxelOccupied = 0;
		d_voxelOccupiedScan = 0;

		totalTime = 0;

		frameCheckNumber=4;

		// Auto-Verification Code

		fpsLimit = 100;        // FPS limit for sampling
		g_Index = 0;
		frameCount = 0;
		g_TotalErrors = 0;
		g_bNoprompt = false;
		bQATest = false;

		initMC();

		colorIntensities=(float *)malloc(sizeof(float)*pointCnt);
		generateColorIntensities(pointPos);

		uploadColorIntensities();
	}

	void uploadColorIntensities() {
		clEnqueueWriteBuffer(clManager->cqCommandQueue, d_colorIntensities, CL_TRUE, 0, sizeof(float)*pointCnt, colorIntensities, 0, NULL, NULL);
	}

	void generateColorIntensities(float * pointPos) {
		for (int i=0; i<pointCnt;i++) {
			colorIntensities[i]=(PerlinNoise3D(pointPos[(i+pointOffset)*4],pointPos[(i+pointOffset)*4 + 1],pointPos[(i+pointOffset)*4 + 2],1.01f,2.0f,3)+1.0f)/2.0f;
			//colorIntensities[i]=0.5f;
		}
	}

	vl::Geometry * getNewSurfaceGeometry() {
		vl::Geometry *geom = new vl::Geometry;

		vl::ref<vl::ArrayFloat4> vert4 = new vl::ArrayFloat4;

		vert4->setBufferObjectDirty(false);
		vert4->bufferObject()->setHandle(posVbo);

		geom->setVertexArray(vert4.get());

		vl::ref<vl::ArrayFloat3> norm3 = new vl::ArrayFloat3;

		norm3->setBufferObjectDirty(false);
		norm3->bufferObject()->setHandle(normalVbo);

		geom->setNormalArray(norm3.get());

		vl::ref<vl::ArrayFloat4> color = new vl::ArrayFloat4;

		color->setBufferObjectDirty(false);
		color->bufferObject()->setHandle(colorVbo);

		geom->setColorArray(color.get());

		drawMode = new vl::DrawArrays(vl::PT_TRIANGLES, 0, 0);

		geom->drawCalls()->push_back(drawMode.get());

		return geom;
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

		vbo_cl = clCreateFromGLBuffer(clManager->cxGPUContext,CL_MEM_WRITE_ONLY, *vbo, &(clManager->ciErrNum));
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
		deleteVBO(&normalVbo, d_color);

		if( d_voxelVerts) clReleaseMemObject(d_voxelVerts);
		if( d_voxelVertsScan) clReleaseMemObject(d_voxelVertsScan);
		if( d_voxelOccupied) clReleaseMemObject(d_voxelOccupied);
		if( d_voxelOccupiedScan) clReleaseMemObject(d_voxelOccupiedScan);
		if( d_compVoxelArray) clReleaseMemObject(d_compVoxelArray);
		if( d_volumeData) clReleaseMemObject(d_volumeData);
		if( d_colorIntensities) clReleaseMemObject(d_colorIntensities);

		closeScan();
	}

	virtual void onActorRenderStarted(vl::Actor*, vl::real frame_clock, const vl::Camera*, vl::Renderable* renderable, const vl::Shader*, int pass)
	{

		if (pass>0)
			return;

		if (objectDisplayMode!=EDGE) {
			//clManager->launch_generateLines(8192, 128, d_edgePos, nBody->getEdges(), nBody->getPos(), edgeCnt, edgeOffset);

			// run kernels to generate geometry
			computeIsosurface();

			drawMode->setCount(totalVerts);
		}



		vl::ref<vl::Geometry> geom = vl::cast<vl::Geometry>( renderable );

		geom->setBoundsDirty(true);
	}

	virtual void onActorDelete(vl::Actor*) {
		Cleanup(0);
	}

	cl_float objectColor[4];

	int *pArgc;
	char **pArgv;

	cl_uint global_work_size;

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

	// device data
	GLuint posVbo, normalVbo, colorVbo;

	GLint  gl_Shader;

	cl_mem d_pos;
	cl_mem d_normal;
	cl_mem d_color;

	displayMode objectDisplayMode;
	float *colorIntensities;

	cl_mem d_colorIntensities;
	cl_mem d_volumeData;
	cl_uint pointCnt;
	cl_uint pointOffset;
	cl_uint edgeCnt;
	cl_uint edgeOffset;

	cl_mem d_voxelVerts;
	cl_mem d_voxelVertsScan;
	cl_mem d_voxelOccupied;
	cl_mem d_voxelOccupiedScan;
	cl_mem d_compVoxelArray;

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

	vl::ref<vl::DrawArrays> drawMode;


	BodySystemOpenCL *nBody;
	CLManager *clManager;

	void setRadius(float radius) {
		this->radius=radius;
	}

	void setIsoValue(float isoValue) {
		this->isoValue=isoValue;
	}

	void setObjectColor(float r, float g, float b, float a) {
		objectColor[0] = r;
		objectColor[1] = g;
		objectColor[2] = b;
		objectColor[3] = a;
	}

	void setObjectDisplayMode(displayMode objectDisplayMode) {
		this->objectDisplayMode = objectDisplayMode;
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

		// create VBOs
		if( !bQATest) {
			createVBO(&posVbo, maxVerts*sizeof(float)*4, d_pos);
			createVBO(&normalVbo, maxVerts*sizeof(float)*3, d_normal);
			createVBO(&colorVbo, maxVerts*sizeof(float)*4, d_color);
		}

		// allocate device memory
		unsigned int memSize = sizeof(uint) * numVoxels;
		d_voxelVerts = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &(clManager->ciErrNum));
		d_voxelVertsScan = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &(clManager->ciErrNum));
		d_voxelOccupied = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &(clManager->ciErrNum));
		d_voxelOccupiedScan = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &(clManager->ciErrNum));
		d_compVoxelArray = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &(clManager->ciErrNum));
		d_volumeData = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*numVoxels*2, 0, &(clManager->ciErrNum));
		d_colorIntensities = clCreateBuffer(clManager->cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*pointCnt, 0, &(clManager->ciErrNum));
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

		if(objectDisplayMode == TENSION) {
			clManager->launch_calcColorIntensitiesTension(global_work_size, threads, nBody->getEdges(), nBody->getPos(), d_colorIntensities, nBody->getNumEdges(), pointCnt, pointOffset);
		}

		clManager->launch_calcFieldValue(grid, threads,
				d_volumeData, nBody->getPos(), d_colorIntensities,pointCnt, pointOffset, radius, gridSizeShift, gridSizeMask);

		// calculate number of vertices need per voxel
		clManager->launch_classifyVoxel(grid, threads,
							d_voxelVerts, d_voxelOccupied, d_volumeData,//nbody->getPos(),
							gridSize, gridSizeShift, gridSizeMask,
							 numVoxels, voxelSize, isoValue);

		// scan voxel occupied array
		clManager->openclScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

		// read back values to calculate total number of non-empty voxels
		// since we are using an exclusive scan, the total is the last value of
		// the scan result plus the last value in the input array
		{
			uint lastElement, lastScanElement;

			clEnqueueReadBuffer(clManager->cqCommandQueue, d_voxelOccupied,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
			clEnqueueReadBuffer(clManager->cqCommandQueue, d_voxelOccupiedScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

			activeVoxels = lastElement + lastScanElement;
		}
		//printf("activeVoxels = %d\n", activeVoxels);

		if (activeVoxels==0) {
			// return if there are no full voxels
			totalVerts = 0;
			return;
		}


		// compact voxel index array
		clManager->launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);


		// scan voxel vertex count array
		clManager->openclScan(d_voxelVertsScan, d_voxelVerts, numVoxels);

		// readback total number of vertices
		{
			uint lastElement, lastScanElement;
			clEnqueueReadBuffer(clManager->cqCommandQueue, d_voxelVerts,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
			clEnqueueReadBuffer(clManager->cqCommandQueue, d_voxelVertsScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

			totalVerts = lastElement + lastScanElement;
		}

		//printf("totalVerts = %d\n", totalVerts);


		cl_mem interopBuffers[] = {d_pos, d_normal, d_color};

		// generate triangles, writing to vertex buffers
		if( clManager->g_glInterop ) {
			// Acquire PBO for OpenCL writing
			glFlush();
			clEnqueueAcquireGLObjects(clManager->cqCommandQueue, 3, interopBuffers, 0, 0, 0);

			//fprintf(stderr, "%d\n", ciErrNum);
		}

		dim3 grid2((int) ceil(activeVoxels / (float) NTHREADS), 1, 1);

		while(grid2.x > 65535) {
			grid2.x/=2;
			grid2.y*=2;
		}

		//printf("\nr: %f g: %f b:%f\n", objectColor[0], objectColor[1],objectColor[2]);

		clManager->launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, d_color,
												d_compVoxelArray,

												d_voxelVertsScan, d_volumeData,

												gridSize, gridSizeShift, gridSizeMask,
												voxelSize, isoValue, activeVoxels,
								  maxVerts, objectColor);

		if( clManager->g_glInterop ) {
			// Transfer ownership of buffer back from CL to GL
			clEnqueueReleaseGLObjects(clManager->cqCommandQueue, 3, interopBuffers, 0, 0, 0);
			clFinish( clManager->cqCommandQueue );
		}

	}


};

class VisualizationObject: public vl::Object
{
public:
	VisualizationObject(ObjectAnimator *objectAnimator, EdgesAnimator *edgesAnimator, vl::ref<vl::Actor> objectActor, vl::ref<vl::Actor> edgesActor) {
		this->objectAnimator=objectAnimator;
		this->edgesAnimator=edgesAnimator;
		this->objectActor=objectActor;
		this->edgesActor=edgesActor;
	}

	void setPointRadius(float radius) {
		objectAnimator->setRadius(radius);
	}

	float getPointRadius() {
		return objectAnimator->radius;
	}

	float getMCIsoValue() {
		return objectAnimator->isoValue;
	}

	void setMCIsoValue(float isoValue) {
		objectAnimator->setIsoValue(isoValue);
	}

	void setObjectColor(float r, float g, float b, float a) {
		objectAnimator->setObjectColor(r, g, b,a);
	}

//protected:
	ObjectAnimator *objectAnimator;
	EdgesAnimator *edgesAnimator;
	vl::ref<vl::Actor> objectActor;
	vl::ref<vl::Actor> edgesActor;
};

class MCApplet: public vl::Applet
{
public:

	MCApplet(ConfigurationData *configData):vl::Applet() {
		this->configData=configData;
	}

  // called once after the OpenGL window has been opened
	void initEvent()
	{
		objects = new vl::Collection<VisualizationObject>;
		clManager = new CLManager();

		// install our scene manager, we use the SceneManagerActorTree which is the most generic
		scene_manager = new vl::SceneManagerActorTree;
		rendering()->as<vl::Rendering>()->sceneManagers()->push_back(scene_manager.get());

		addBorders();

		simulationInit();
	}

	/*void setControlsWindow(ControlsWidget *window) {
		controlsWindow=window;
	}*/

	void addVisualizationObject(int pointCnt, int pointOffset, float * pos, int edgeCnt, int edgeOffset) {
		// allocate the Transform
		transform = new vl::Transform;
		// bind the Transform with the transform tree of the rendring pipeline
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );

		vl::Light *light = new vl::Light;

		// setup the effect to be used to render the cube
		vl::ref<vl::Effect> objectEffect = new vl::Effect;
		// enable depth test and lighting
		objectEffect->shader()->enable(vl::EN_DEPTH_TEST);
		//effect->shader()->gocDepthMask()->set(true);

		objectEffect->shader()->enable(vl::EN_NORMALIZE);
		// add a Light to the scene, since no Transform is associated to the Light it will follow the camera
		objectEffect->shader()->setRenderState( light, 0 );
		// enable the standard OpenGL lighting
		objectEffect->shader()->enable(vl::EN_LIGHTING);

		vl::ref<vl::GLSLProgram> glsl = new vl::GLSLProgram;
		glsl->attachShader( new vl::GLSLVertexShader("./glsl/perpixellight.vs") );
		glsl->attachShader( new vl::GLSLFragmentShader("./glsl/perpixellight_interlaced.fs") );

		objectEffect->shader()->setRenderState(glsl.get());

		ObjectAnimator *animator = new ObjectAnimator(clManager, nbody, pointCnt, pointOffset, pos, edgeCnt, edgeOffset);

		vl::ref<vl::Actor> actor=scene_manager->tree()->addActor( animator->getNewSurfaceGeometry(), objectEffect.get(), transform.get());
		actor->actorEventCallbacks()->push_back(animator);

		//EDGES

		EdgesAnimator *edgesAnimator = new EdgesAnimator(clManager, nbody, edgeCnt, edgeOffset);

		vl::ref<vl::Effect> edgesEffect = new vl::Effect;
		edgesEffect->shader()->enable(vl::EN_DEPTH_TEST);
		//edgesEffect->shader()->enable(vl::EN_LIGHTING);
		//edgesEffect->shader()->gocDepthFunc()->set(vl::FU_NEVER);
		//edgesEffect->shader()->setRenderState( light, 0 );
		//edgesEffect->shader()->gocMaterial()->setDiffuse( vl::green);

		transform = new vl::Transform;
		// bind the Transform with the transform tree of the rendring pipeline
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );

		vl::ref<vl::Actor> edgesActor = new vl::Actor(edgesAnimator->getNewEdgesGeometry(), edgesEffect.get(), transform.get());
		edgesActor->actorEventCallbacks()->push_back(edgesAnimator);

		VisualizationObject *object = new VisualizationObject(animator, edgesAnimator, actor, edgesActor);
		objects->push_back(object);
	}

	void addBorders()
	{
		transform = new vl::Transform;
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );

		vl::ref<vl::Geometry> floor = vl::makeGrid(vl::vec3(0.0f,-1.0f,0.0f), 4, 4, 20, 20);
		floor->computeNormals();

		vl::ref<vl::Effect> effect = new vl::Effect;
		effect->shader()->enable(vl::EN_DEPTH_TEST);
		effect->shader()->setRenderState( new vl::Light, 0 );
		effect->shader()->enable(vl::EN_LIGHTING);
		effect->shader()->gocMaterial()->setAmbient( vl::skyblue);
		effect->shader()->gocLightModel()->setTwoSide(true);
		//effect->shader()->enable(vl::EN_BLEND);

		bordersActor = scene_manager->tree()->addActor( floor.get(), effect.get(), transform.get()  );

		//walls

		effect = new vl::Effect;
		effect->shader()->enable(vl::EN_DEPTH_TEST);
		effect->shader()->setRenderState( new vl::Light, 0 );
		effect->shader()->enable(vl::EN_LIGHTING);
		//effect->shader()->gocMaterial()->setDiffuse( vl::fvec4(0.7f,0.7f,0.7f,1.0f) );
		effect->shader()->gocMaterial()->setTransparency(0.2);
		effect->shader()->gocLightModel()->setTwoSide(true);
		effect->shader()->enable(vl::EN_BLEND);

		//-----------wall--------------
		transform = new vl::Transform;
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );
		transform->rotate(90, 1,0,0);

		vl::ref<vl::Geometry> wall = vl::makeGrid(vl::vec3(0.0f,-1.0f,0.0f), 2, 2, 10, 10);
		floor->computeNormals();
		scene_manager->tree()->addActor( wall.get(), effect.get(), transform.get()  );

		//-----------wall--------------
		transform = new vl::Transform;
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );
		transform->rotate(-90, 1,0,0);

		wall = vl::makeGrid(vl::vec3(0.0f,-1.0f,0.0f), 2, 2, 10, 10);
		floor->computeNormals();
		scene_manager->tree()->addActor( wall.get(), effect.get(), transform.get()  );

		//-----------wall--------------
		transform = new vl::Transform;
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );
		transform->rotate(90, 1,0,0);
		transform->rotate(90, 0,1,0);

		wall = vl::makeGrid(vl::vec3(0.0f,-1.0f,0.0f), 2, 2, 10, 10);
		floor->computeNormals();
		scene_manager->tree()->addActor( wall.get(), effect.get(), transform.get()  );

		//-----------wall--------------
		transform = new vl::Transform;
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );
		transform->rotate(-90, 1,0,0);
		transform->rotate(90, 0,1,0);

		wall = vl::makeGrid(vl::vec3(0.0f,-1.0f,0.0f), 2, 2, 10, 10);
		floor->computeNormals();
		scene_manager->tree()->addActor( wall.get(), effect.get(), transform.get()  );

		//-----------wall--------------
		transform = new vl::Transform;
		rendering()->as<vl::Rendering>()->transform()->addChild( transform.get() );
		transform->rotate(180, 1,0,0);

		wall = vl::makeGrid(vl::vec3(0.0f,-1.0f,0.0f), 2, 2, 10, 10);
		floor->computeNormals();
		scene_manager->tree()->addActor( wall.get(), effect.get(), transform.get()  );

		//bordersActor->actorEventCallbacks()->push_back( new vl::DepthSortCallback );

	}

	vl::ref<vl::Geometry> makeFloor(const vl::vec3& origin, vl::real xside, vl::real yside) {
		vl::ref<vl::Geometry> geom = new vl::Geometry;
		geom->setObjectName("Floor");

		vl::ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
		geom->setVertexArray(vert3.get());

		vl::fvec3 a0( (vl::fvec3)(origin) );
		vl::fvec3 a1( (vl::fvec3)(vl::vec3(0,0,xside) + origin) );
		vl::fvec3 a2( (vl::fvec3)(vl::vec3(yside,0,0) + origin) );
		vl::fvec3 a3( (vl::fvec3)(vl::vec3(yside,0,xside) + origin) );



	   vl::fvec3 verts[] = {
		 a0, a1, a2,
		 a1, a3, a2
	   };

	   vl::ref<vl::DrawArrays> polys = new vl::DrawArrays(vl::PT_TRIANGLES, 0, 6);
	   geom->drawCalls()->push_back( polys.get() );
	   vert3->resize( 6 );
	  memcpy(vert3->ptr(), verts, sizeof(verts));
	   return geom;
	}

	// called every frame
	virtual void updateScene()
	{
		for (int i=0; i<objects->size(); i++) {
			objects->at(i)->setPointRadius(configData->objectData[i].radius);
			objects->at(i)->setMCIsoValue(configData->objectData[i].isoValue);
			objects->at(i)->setObjectColor(configData->objectData[i].colorR, configData->objectData[i].colorG, configData->objectData[i].colorB,configData->objectData[i].colorA);
			/*if (configData->objectData[i].change == true) {
				uiSetForces(1, configData->objectData[i].force[0], i+1);
				uiSetForces(2, configData->objectData[i].force[1], i+1);
				uiSetForces(3, configData->objectData[i].force[2], i+1);
				configData->objectData[i].change = false;
			}*/
			displayMode objectDisplayMode = configData->objectData[i].mode;
			VisualizationObject *object = objects->at(i);
			if (object->objectAnimator->objectDisplayMode != objectDisplayMode) {
				object->objectAnimator->setObjectDisplayMode(objectDisplayMode);
				if (objectDisplayMode == NORMAL) {
					scene_manager->tree()->eraseActor(object->edgesActor.get());
					scene_manager->tree()->eraseActor(object->objectActor.get());
					object->objectAnimator->uploadColorIntensities();
					scene_manager->tree()->addActor(object->objectActor.get());
				} else if(objectDisplayMode == EDGE) {
					scene_manager->tree()->eraseActor(object->objectActor.get());
					scene_manager->tree()->addActor(object->edgesActor.get());
				} else if(objectDisplayMode == TENSION) {
					scene_manager->tree()->eraseActor(object->edgesActor.get());
					scene_manager->tree()->eraseActor(object->objectActor.get());
					object->objectAnimator->uploadColorIntensities();
					scene_manager->tree()->addActor(object->objectActor.get());
				}
			}

		}
		//bordersActor->actorEventCallbacks()->push_back( new vl::DepthSortCallback );
		//run kernels to update particle positions
		//nbody->update(m_timestep);
		printf("%f\n",fps());
	}

	void TranslatePositions(int count, float* pos, int pAct, int numObjects) {
		float min = 256;
		float max = 0;
		int tmp[count*4];
		float dielikFirst;
		float dielikSecond;
		float x1 = 0, x2 = 0;

		int offset = 5*numObjects;
		int size = 6;

		for (int p = pAct; p < (count*4)+pAct; p++) {
			if (pos[p] < min)
				min = pos[p];
			if (pos[p] > max)
				max = pos[p];
		}

		max -= min;

		//shrLog("\nMinimum: %f, maximum: %f\n", min, max);

		dielikFirst = max*0.001; // 0....max
		dielikSecond = size*0.001; // 5....20

		//shrLog("\ndieliky: %f a %f\n", dielikFirst, dielikSecond);
		for (int i = (pAct/4), t = 0; i < count+(pAct/4); i++, t++) {
			tmp[t*4] = (int)pos[i*4] - min;
			tmp[t*4+1] = (int)pos[i*4+1] - min;
			tmp[t*4+2] = (int)pos[i*4+2] - min;
			tmp[t*4+3] = (int)pos[i*4+3];

			//shrLog("\nPosunute--> x: %d, y: %d, z: %d\n", tmp[i*4], tmp[i*4+1], tmp[i*4+2]);
		}

		for (int i = (pAct/4), t = 0; i < count+(pAct/4); i++, t++) {
			pos[i*4] = (tmp[t*4]/dielikFirst)*dielikSecond + offset;
			if (x1 == 0)
				x1 = pos[i*4];
			if (x2 == 0) {
				if (x1 != pos[i*4]) {
					if (pos[i*4] > x1) {
						x2 = pos[i*4];
						edgeLength = x2-x1;
					}
					else if (pos[i*4] < x2) {
						x2 = pos[i*4];
						edgeLength = x1-x2;
					}
				}
			}
			pos[i*4+1] = (tmp[t*4+1]/dielikFirst)*dielikSecond + offset;
			pos[i*4+2] = (tmp[t*4+2]/dielikFirst)*dielikSecond + offset;
			pos[i*4+3] = tmp[t*4+3];

			//shrLog("\nNove--> x: %f, y: %f, z: %f / EdgeLength: %f\n", pos[i*4], pos[i*4+1], pos[i*4+2], edgeLength);
		}
	}

	int ImportFromFile(char *fileName, float* pos, float* vel, float* force, float *forces) {
		string str;
		ifstream infileImp(fileName);

		int numObjects = 0;
		int count;
		float x, y, z;
		int p = 0;
		int pStart = 0;

		if (!infileImp) {
			cout << "There was a problem opening file "
					<< fileName
					<< " for reading."
					<< endl;
		}
		cout << "Opened " << fileName << " for import." << endl;

		while (infileImp >> str) {
			pStart = p;
			numObjects++;

			string s;
			ifstream infile((char*)str.c_str());

			if (!infile) {
				cout << "There was a problem opening file "
						<< str
						<< " for reading."
						<< endl;
				return -1;
			}
			cout << "Opened " << str << " for reading." << endl;

			while (infile >> s) {
				if (s.compare("$Nodes") == 0)
					break;
			}

			infile >> s;
			count = atoi(s.c_str());

			for (int i = 0; i < count; i++) {
				infile >> s;
				if (s.compare("$EndNodes") == 0)
					break;
				infile >> s;
				x = atof(s.c_str());

				vel[p] = 0;
				force[p] = 0;
				forces[p] = 0;
				pos[p++] = x;

				if (s.compare("$EndNodes") == 0)
					break;
				infile >> s;
				y = atof(s.c_str());

				vel[p] = 0;
				force[p] = 0;
				if(numObjects == 2) {
					if (y > 100)
						force[p] = 20;
				}
				forces[p] = 0;
				pos[p++] = y;


				if (s.compare("$EndNodes") == 0)
					break;
				infile >> s;
				z = atof(s.c_str());

				vel[p] = 0;
				force[p] = 0;
				forces[p] = 0;
				pos[p++] = z;

				vel[p] = 1;
				force[p] = 1;
				forces[p] = 1;
				pos[p++] = numObjects;

				if (s.compare("$EndNodes") == 0)
					break;
			}

			shrLog("--> %d <--\n", pStart);

			TranslatePositions(count, pos, pStart, numObjects);
		}

		shrLog("%d-%d\n", pointCnt, numBodies);

		for (int i = pointCnt; i < numBodies; i++) {
			vel[p] = 0;
			force[p] = 0;
			forces[p] = 0;
			pos[p++] = 0;
			vel[p] = 0;
			force[p] = 0;
			forces[p] = 0;
			pos[p++] = 0;
			vel[p] = 0;
			force[p] = 0;
			forces[p] = 0;
			pos[p++] = 0;
			vel[p] = 0;
			force[p] = 0;
			forces[p] = 0;
			pos[p++] = 0;
		}

//		for(int i = 0; i < numBodies; i++) {
//			shrLog("P[%d]: %f - %f - %f - %f\n", i, pos[i*4], pos[i*4+1], pos[i*4+2], pos[i*4+3]);
//		}

		return count;
	}

	int getNumBodies(char *fileName) {
		string s;
		ifstream infile(fileName);
		int count;

		if (!infile) {
			cout << "There was a problem opening file "
					<< fileName
					<< " for reading."
					<< endl;
			return -1;
		}
		cout << "Opened " << fileName << " for reading." << endl;

		while (infile >> s) {
			if (s.compare("$Nodes") == 0)
				break;
		}

		infile >> s;
		count = atoi(s.c_str());

		return count;
	}

	bool InInterval(float middle, float odchylka, float number) {

		if ((number > (middle-odchylka)) && (number < (middle+odchylka)))
			return true;
		else
			return false;

	}

	void uiSetForces(int direct /*1-x, 2-y, 3-z*/, float force, int object) {
		float *tmpF = new float[numBodies*4];
		tmpF = nbody->getArray(BodySystem::BODYSYSTEM_F);

		for (int i = 0; i < numBodies; i++) {
			if (hPos[i*4+3] == object) {
				switch(direct) {
				case 1:
					tmpF[i*4] = force;
					break;
				case 2:
					tmpF[i*4+1] = force;
					break;
				case 3:
					tmpF[i*4+2] = force;
					break;
				default:
					break;
				}
			}
		}

		nbody->setArray(BodySystem::BODYSYSTEM_F, tmpF);
	}

	void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL)
		{
		    shrLog("\nReset Nbody System...\n\n");

		    /*shrLog("\nnEdges = %d\n", nEdges);

		    // initalize the memory
		    randomizeBodies(&nEdges, config, hPos, hVel, hF, hForces, hColor, activeParams.m_clusterScale,
				            activeParams.m_velocityScale, numBodies);*/

		    shrLog("\nnEdges = %d\n", nEdges);

		    int k = 0;

		    //initialize edges
		    //hEdge = new float[nEdges*3];

		    float pole[3];
		    float vectorLength;

		    float tmpNUM = pow(numBodies, 1./3.);
		    float tmpINC = (SCALE)/tmpNUM;
		    int n = 0;
		    int p = 0;
		    bool bX = true, bY = true, bZ = true; // 1
		    bool bZY = true, bZmY = true, bYX = true, bYmX = true, bZX = true, bZmX = true; // 2
		    bool bmXYZ = true, bXYZ = true, bXmYZ = true, bmXmYZ = true; // 3

		    tmpINC = edgeLength;

		    for (int i = 0; i < numBodies; i++) {
		    	for (int j = 0; j < numBodies; j++) {
		    		/*
		    		 * 1 = 3 kusy
		    		 */
		    		if (InInterval(hPos[i*4] + tmpINC, 0.1, hPos[j*4]) && bX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4+2] == hPos[i*4+2])) { // x
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bX = false;
		    		}
		    		else if (InInterval(hPos[i*4+1] + tmpINC, 0.1, hPos[j*4+1]) && bY && (hPos[j*4] == hPos[i*4]) && (hPos[j*4+2] == hPos[i*4+2])) { // y
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bY = false;
		    		}
		    		else if (InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2]) && bZ && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4] == hPos[i*4])) { // z
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bZ = false;
		    		}

		    		/*
		    		 * 2 = 6 kusov
		    		 */

		    	/*	else if (bZY && (hPos[j*4+2] == (hPos[i*4+2] + tmpINC)) && (hPos[j*4+1] == (hPos[i*4+1] + tmpINC)) && (hPos[j*4] == hPos[i*4])) {
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
		    		}*/
		    		else if (InInterval(hPos[i*4] + (tmpINC*2), 0.1, hPos[j*4]) && bZX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4+2] == hPos[i*4+2])) { // x
		    			n++;
	k++;
		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bZX = false;
		    		}
		    		else if (InInterval(hPos[i*4+1] + (tmpINC*2), 0.1, hPos[j*4+1]) && bYmX && (hPos[j*4] == hPos[i*4]) && (hPos[j*4+2] == hPos[i*4+2])) { // y
		    			n++;
	k++;
		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bYmX = false;
		    		}
		    		else if (InInterval(hPos[i*4+2] + (tmpINC*2), 0.1, hPos[j*4+2]) && bZmX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4] == hPos[i*4])) { // z
		    			n++;
	k++;
		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bZmX = false;
		    		}
		    		/*
		    		 * 3 = 4 kusy
		    		 */
		    		else if (bmXYZ && InInterval(hPos[i*4] - tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] + tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bmXYZ = false;
		    		}
		    		else if (bXYZ && InInterval(hPos[i*4] + tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] + tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bXYZ = false;
		    		}
		    		else if (bXmYZ && bXmYZ && InInterval(hPos[i*4] + tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] - tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
		    			hEdge[p++] = 1.0f;

		    			//shrLog("E: %d => P1: %d; P2: %d; l0: %f; Ks: %d; Kd: %d <-->", p-5, i, j, tmpInc, (int)koefKs, (int)koefKd);

		    			bXmYZ = false;
		    		}
		    		else if (bmXmYZ && InInterval(hPos[i*4] - tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] - tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;

		    			hEdge[p++] = i;
		    			hEdge[p++] = j;

		    			pole[0] = hPos[i*4] - hPos[j*4];
		    			pole[1] = hPos[i*4+1] - hPos[j*4+1];
		    			pole[2] = hPos[i*4+2] - hPos[j*4+2];

		    			vectorLength = sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2]);

		    			hEdge[p++] = vectorLength;
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

		    shrLog("\nEdges nove = %d\n", k);

		    for (int i = n; i < nEdges; i++) {
		    	hEdge[i*4] = -1;
		    	hEdge[i*4+1] = -1;
		    	hEdge[i*4+2] = 0;
		    	hEdge[i*4+3] = 0.0f;
		    }

	//	    for (int i = 0; i < nEdges; i++) {
	//	    	shrLog("E: %d ==> p1: %d; p2: %d; l0: %f\n", i, (int)hEdge[i*4], (int)hEdge[i*4+1], hEdge[i*4+2]);
	//	    }

		    shrLog("1\n");
		    system->setArray(BodySystem::BODYSYSTEM_OLD_POSITION, hPos);
		    shrLog("2\n");
		    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
		    shrLog("3\n");
		    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
		    shrLog("4\n");
		    system->setArray(BodySystem::BODYSYSTEM_F, hF);
		    shrLog("5\n");
		    system->setArray(BodySystem::BODYSYSTEM_FORCES, hForces);
		    shrLog("6\n");
		    system->setArray(BodySystem::BODYSYSTEM_EDGE, hEdge);
		    shrLog("7\n");
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
		 //   randomizeBodies(&nEdges, config, hPos, hVel, hF, hForces, hColor, m_clusterScale, m_velocityScale, numBodies);

		    int tmp;
		    tmp = ImportFromFile("import2.txt", hPos, hVel, hF, hForces);

		    float tmpNUM = pow(numBodies, 1./3.);
		    float tmpINC = (SCALE)/tmpNUM;
		    int n = 0;
		    bool bX = true, bY = true, bZ = true; // 1
		    bool bZY = true, bZmY = true, bYX = true, bYmX = true, bZX = true, bZmX = true; // 2
		    bool bmXYZ = true, bXYZ = true, bXmYZ = true, bmXmYZ = true; // 3

		    tmpINC = edgeLength;

		    int currentObject=0;
		    int pointCountCheckpoint=objectsPointCounts[0];
		    int totalEdges=0;

		    for (int i = 0; i < numBodies; i++) {

		    	if (i==pointCountCheckpoint) {

			    	objectsEdgeCounts[currentObject]=n-totalEdges;
			    	totalEdges=n;
		    		currentObject++;
		    		pointCountCheckpoint+=objectsPointCounts[currentObject];
		    	}

		    	for (int j = 0; j < numBodies; j++) {
		    		/*
		    		 * 1 = 3 kusy
		    		 */
		    		if (InInterval(hPos[i*4] + tmpINC, 0.1, hPos[j*4]) && bX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4+2] == hPos[i*4+2])) { // x
		    			n++;
		    			bX = false;
		    		}
		    		else if (InInterval(hPos[i*4+1] + tmpINC, 0.1, hPos[j*4+1]) && bY && (hPos[j*4] == hPos[i*4]) && (hPos[j*4+2] == hPos[i*4+2])) { // y
		    			n++;
		    			bY = false;
		    		}
		    		else if (InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2]) && bZ && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4] == hPos[i*4])) { // z
		    			n++;
		    			bZ = false;
		    		}

		    		/*
		    		 * 2 = 6 kusov
		    		 */
		    		else if (InInterval(hPos[i*4] + (tmpINC*2), 0.1, hPos[j*4]) && bZX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4+2] == hPos[i*4+2])) { // x
		    			n++;

		    			bZX = false;
		    		}
		    		else if (InInterval(hPos[i*4+1] + (tmpINC*2), 0.1, hPos[j*4+1]) && bYmX && (hPos[j*4] == hPos[i*4]) && (hPos[j*4+2] == hPos[i*4+2])) { // y
		    			n++;

		    			bYmX = false;
		    		}
		    		else if (InInterval(hPos[i*4+2] + (tmpINC*2), 0.1, hPos[j*4+2]) && bZmX && (hPos[j*4+1] == hPos[i*4+1]) && (hPos[j*4] == hPos[i*4])) { // z
		    			n++;

		    			bZmX = false;
		    		}
		    		/*
		    		 * 3 = 4 kusy
		    		 */
		    		else if (bmXYZ && InInterval(hPos[i*4] - tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] + tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;
		    			bmXYZ = false;
		    		}
		    		else if (bXYZ && InInterval(hPos[i*4] + tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] + tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;
		    			bXYZ = false;
		    		}
		    		else if (bXmYZ && InInterval(hPos[i*4] + tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] - tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
		    			n++;
		    			bXmYZ = false;
		    		}
		    		else if (bmXYZ && InInterval(hPos[i*4] - tmpINC, 0.1, hPos[j*4]) && InInterval(hPos[i*4+1] - tmpINC, 0.1, hPos[j*4+1]) && InInterval(hPos[i*4+2] + tmpINC, 0.1, hPos[j*4+2])) {
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

		    /*	KONTROLNY VYPIS
		     * for (int asdasd=0; asdasd<objectCount;asdasd++) {
		    	printf("\nOBJEKT %d: %d\n", asdasd, objectsEdgeCounts[asdasd]);
		    }*/

		    shrLog("\nedges: %d", n);

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

			srand ( time(NULL) );

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
						//pos[p++] = (rand() % 100+1) /100.0f;
						pos[p++] = (PerlinNoise3D(i,j,k,1.1f,2.0f,1)+1.0f)/2.0f;
						printf("random: %f\n", pos[p-1]);

						vel[v++] = 0.0f;
						vel[v++] = 0.0f;
						vel[v++] = 0.0f;
						vel[v++] = 1.0f;

						*numEdges += 3;

						if ((j+tmpINC) >= ((tmpNUM*tmpINC)-0.001)+10) {
											force[p-3] = 100.0f;
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

	vl::Collection<VisualizationObject>* getObjects() {
		return objects;
	}

	int getFps() {
		return fps();
	}


protected:
	ConfigurationData *configData;

	vl::ref<vl::Transform> transform;
	CLManager *clManager;
	vl::ref<vl::SceneManagerActorTree> scene_manager;
	vl::ref<vl::Actor> bordersActor;
	vl::Collection<VisualizationObject>* objects;

	//simulation fields
	cl_uint pointCnt;
	float edgeLength;
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

	int objectsPointCounts[20];
	int objectsEdgeCounts[20];
	int objectCount;

	//--

	void simulationInit() {
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
		edgeLength = 0;

		m_timestep=0.01f;
		m_clusterScale=1.54f;
		m_velocityScale=1.0f;
		m_softening=0.1f;
		m_damping=0.95f;
		m_pointSize=5.0f;
		m_x=0;
		m_y=-2;
		m_z=-100;

		string s;
		ifstream infile("import2.txt");

		if (!infile) {
			cout << "There was a problem opening file "
					<< "import.txt"
					<< " for reading."
					<< endl;
		}
		cout << "Opened " << "import2.txt" << " for import." << endl;

		numBodies = 0;
		int newBodiesCount;
		objectCount = 0;

		while (infile >> s) {
			newBodiesCount = getNumBodies((char*)s.c_str());
			objectsPointCounts[objectCount++] = newBodiesCount;
			numBodies += newBodiesCount;
		}

	//	numBodies = getNumBodies("ducky_3.msh");
		pointCnt = numBodies;

		shrLog("--> %d\n", pointCnt);

		if (numBodies <= 128) {

		} else {
			if (numBodies <= 256)
				numBodies = 256;
			else if (numBodies <= 512)
				numBodies= 512;
			else if (numBodies <= 1024)
				numBodies = 1024;
			else if (numBodies <= 2048)
				numBodies = 2048;
			else if (numBodies <= 4096)
				numBodies = 4096;
			else if (numBodies <= 8192)
				numBodies = 8192;
			else if (numBodies <= 16384)
				numBodies = 16384;
			else if(numBodies <= 32768)
				numBodies = 32768;
			else if (numBodies <= 65536)
				numBodies = 65536;
		}

		if ((q * p) > 256)
		{
			p = 256 / q;
		}

		if ((q == 1) && (numBodies < p))
		{
			p = numBodies;
		}

		for (int i=0;i<20;i++) {
			objectsEdgeCounts[i]=0;
		}

		InitNbody(clManager->cdDevices[clManager->uiDeviceUsed], clManager->cxGPUContext, clManager->cqCommandQueue, numBodies, p, q, false, false, NBODY_CONFIG_SHELL);
		ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, false);

		int totalPoints = 0, totalEdges = 0;
		for (int i=0;i<objectCount;i++) {
			addVisualizationObject(objectsPointCounts[i], totalPoints, hPos, objectsEdgeCounts[i], totalEdges);
			totalPoints+=objectsPointCounts[i];
			totalEdges+=objectsEdgeCounts[i];
		}
	}

	void keyPressEvent(unsigned short ch, vl::EKey key)
	{
		vl::Applet::keyPressEvent(ch,key);
		/*if (key== vl::Key_V) {
			objects->at(0)->setPointRadius(objects->at(0)->getPointRadius()+1);
		}*/
	}
};



#endif
