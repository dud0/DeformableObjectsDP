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


// OpenGL Graphics Includes
#include <GL/glew.h>
#ifdef UNIX
    #include <GL/glxew.h>
#endif
#if defined (_WIN32)
    #include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif

#include "oclUtils.h"
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "oclBodySystemOpenclLaunch.h"

BodySystemOpenCL::BodySystemOpenCL(int numBodies, int numEdges, cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
                                   unsigned int p, unsigned int q, bool usePBO, bool bDouble)
: BodySystem(numBodies),
  device(dev),
  cxContext(ctx),
  cqCommandQueue(cmdq),
  m_hPos(0),
  m_hVel(0),
  m_hF(0),
  m_hForces(0),
  m_hEdge(0),
  m_bUsePBO(usePBO),
  m_currentRead(0),
  m_currentWrite(1),
  m_p(p),
  m_q(q),
  m_bDouble(bDouble)
{
    m_dPos[0] = m_dPos[1] = 0;
    m_dVel[0] = m_dVel[1] = 0;
    m_dF[0] = m_dF[1] = 0;
    m_dEdge[0] = m_dEdge[1] = 0;
    m_dForces[0] = m_dForces[1] = 0;

    _initialize(numBodies, numEdges);

    // create non multithreaded program and kernel
    shrLog("\nCreateProgramAndKernel _noMT... ");  
    if (CreateProgramAndKernel(ctx, &dev, "integrateBodies_noMT", &noMT_kernel, m_bDouble)) 
    {
        exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "CreateProgramAndKernel _noMT ", STDERROR)); 
    }

    // create multithreaded program and kernel
    shrLog("\nCreateProgramAndKernel _MT... ");
    if (CreateProgramAndKernel(ctx, &dev, "integrateBodies_MT", &MT_kernel, m_bDouble)) 
    {
        exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "CreateProgramAndKernel _MT ", STDERROR)); 
    }

    setSoftening(0.00125f);
    setDamping(0.995f);   
}

BodySystemOpenCL::~BodySystemOpenCL()
{
    _finalize();
    m_numBodies = 0;
}

void BodySystemOpenCL::_initialize(int numBodies, int numEdges)
{
    oclCheckError(m_bInitialized, shrFALSE);

    m_numBodies = numBodies;
    m_numEdges = numEdges;

    m_hPos = new float[m_numBodies*4];
    m_hVel = new float[m_numBodies*4];
    m_hF = new float[m_numBodies*4];
    m_hEdge = new float[m_numEdges*3];
    m_hForces = new float[m_numBodies*4];

    memset(m_hPos, 0, m_numBodies*4*sizeof(float));
    memset(m_hVel, 0, m_numBodies*4*sizeof(float));
    memset(m_hF, 0, m_numBodies*4*sizeof(float));
    memset(m_hEdge, 0, m_numEdges*3*sizeof(float));
    memset(m_hForces, 0, m_numBodies*4*sizeof(float));

    if (m_bUsePBO)
    {
        // create the position pixel buffer objects for rendering
        // we will actually compute directly from this memory in OpenCL too
        glGenBuffers(2, (GLuint*)m_pboGL);  
        shrLog("Allocating Pixel Buffers\n"); 
        for (int i = 0; i < 2; ++i)
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_pboGL[i]);
            glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * m_numBodies, m_hPos, GL_DYNAMIC_DRAW);

            int size = 0;
            glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size); 
            if ((unsigned)size != 4 * (sizeof(float) * m_numBodies))
            {
                shrLogEx(LOGBOTH, -1, "WARNING: Pixel Buffer Object allocation failed!\n"); 
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            m_pboCL[i] = RegisterGLBufferObject(cxContext, m_pboGL[i]);
        }
    }
    else
    {
        AllocateNBodyArrays(cxContext, m_dPos, m_numBodies, m_bDouble);
        shrLog("\nAllocateNBodyArrays m_dPos\n"); 
    }
    
    AllocateNBodyArrays(cxContext, m_dVel, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dVel\n"); 

    AllocateNBodyArrays(cxContext, m_dF, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dF\n");

    AllocateNBodyArrays(cxContext, m_dForces, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dF\n");

    AllocateNBodyArrays(cxContext, m_dEdge, m_numEdges, 3);
    shrLog("\nAllocateNBodyArrays m_dEdge\n");

    m_bInitialized = true;
}

void BodySystemOpenCL::_finalize()
{
    oclCheckError(m_bInitialized, shrTRUE);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hForces;
    delete [] m_hF;
    delete [] m_hEdge;

	clReleaseKernel(MT_kernel);
	clReleaseKernel(noMT_kernel);

    DeleteNBodyArrays(m_dVel);
    DeleteNBodyArrays(m_dF);
    DeleteNBodyArrays(m_dEdge);
    DeleteNBodyArrays(m_dForces);
    if (m_bUsePBO)
    {
        UnregisterGLBufferObject(m_pboCL[0]);
        UnregisterGLBufferObject(m_pboCL[1]);
        glDeleteBuffers(2, (const GLuint*)m_pboGL);
    }
    else
    {
        DeleteNBodyArrays(m_dPos);
    }
}

void BodySystemOpenCL::setSoftening(float softening)
{
    m_softeningSq = softening * softening;
}

void BodySystemOpenCL::setDamping(float damping)
{
    m_damping = damping;
}

void BodySystemOpenCL::update(float deltaTime)
{
    oclCheckError(m_bInitialized, shrTRUE);
    
    IntegrateNbodySystem(cqCommandQueue,
                         MT_kernel, noMT_kernel,
                         m_dPos[m_currentWrite], m_dVel[m_currentWrite], m_dF[m_currentWrite], m_dEdge[m_currentWrite], m_dForces[m_currentWrite],
                         m_dPos[m_currentRead], m_dVel[m_currentRead], m_dF[m_currentRead], m_dEdge[m_currentRead], m_dForces[m_currentRead],
                         m_pboCL[m_currentRead], m_pboCL[m_currentWrite],
                         deltaTime, m_damping, m_softeningSq,
                         m_numBodies, m_numEdges, m_p, m_q,
                         (m_bUsePBO ? 1 : 0),
						 m_bDouble);

    //float* V;
    //V = getArray(BODYSYSTEM_VELOCITY);
    //shrLog("x: %f - y: %f - z: %f\n", V[0], V[1], V[2]);

    //float* E;
    //E = getArray(BODYSYSTEM_EDGE);
    //for(int i = 0; i < m_numEdges; i++) {
    //	shrLog("%d:       %d s %d ---> l0: %f\n", i, (int)E[i*3+0], (int)E[i*3+1], E[i*3+2]);
    //}
    //shrLog("0:       %d s %d ---> l0: %f\n",  (int)E[0*3+0], (int)E[0*3+1], E[0*3+2]);

    float* F;
    F = getArray(BODYSYSTEM_FORCES);
    shrLog("x: %f - y: %f - z:%f\n", F[0], F[1], F[2]);
    shrLog("x: %f - y: %f - z:%f\n\n", F[4], F[5], F[6]);


    std::swap(m_currentRead, m_currentWrite);
}

float* BodySystemOpenCL::getArray(BodyArray array)
{
    oclCheckError(m_bInitialized, shrTRUE);
 
    float *hdata = 0;
    cl_mem ddata = 0;
    cl_mem pbo = 0;
    int size = 0;
    int nB = 0;

    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
            hdata = m_hPos;
            ddata = m_dPos[m_currentRead];
            if (m_bUsePBO)
            {
                pbo = m_pboCL[m_currentRead];
            }
            nB = m_numBodies;
            break;
        case BODYSYSTEM_VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel[m_currentRead];
            nB = m_numBodies;
            break;
        case BODYSYSTEM_F:
        	hdata = m_hF;
        	ddata = m_dF[m_currentRead];
        	nB = m_numBodies;
        	break;
        case BODYSYSTEM_FORCES:
        	hdata = m_hForces;
        	ddata = m_dForces[m_currentRead];
        	nB = m_numBodies;
        	break;
        case BODYSYSTEM_EDGE:
        	size = 3;
        	nB = m_numEdges;
        	hdata = m_hEdge;
        	ddata = m_dEdge[m_currentRead];
        	break;
    }

    CopyArrayFromDevice(size, cqCommandQueue, hdata, ddata, pbo, nB, m_bDouble);

    return hdata;
}

void BodySystemOpenCL::setArray(BodyArray array, const float* data)
{
    oclCheckError(m_bInitialized, shrTRUE);
 
    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
        {
            if (m_bUsePBO)
            {
                UnregisterGLBufferObject(m_pboCL[m_currentRead]);
                glBindBuffer(GL_ARRAY_BUFFER, m_pboGL[m_currentRead]);
                glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * m_numBodies, data, GL_DYNAMIC_DRAW);
            
                int size = 0;
                glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size); 
                if ((unsigned)size != 4 * (sizeof(float) * m_numBodies))
                {
                    shrLogEx(LOGBOTH, -1, "WARNING: Pixel Buffer Object download failed!\n"); 
                }
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                m_pboCL[m_currentRead] = RegisterGLBufferObject(cxContext, m_pboGL[m_currentRead]);
            }
            else
            {
                CopyArrayToDevice(0, cqCommandQueue, m_dPos[m_currentRead], data, m_numBodies, m_bDouble);
            }
        }
            break;
        case BODYSYSTEM_VELOCITY:
            CopyArrayToDevice(0, cqCommandQueue, m_dVel[m_currentRead], data, m_numBodies, m_bDouble);
            break;

        case BODYSYSTEM_F:
        	CopyArrayToDevice(0, cqCommandQueue, m_dF[m_currentRead], data, m_numBodies, m_bDouble);
        	break;

        case BODYSYSTEM_FORCES:
        	CopyArrayToDevice(0, cqCommandQueue, m_dForces[m_currentRead], data, m_numBodies, m_bDouble);
        	break;

        case BODYSYSTEM_EDGE:
        	CopyArrayToDevice(3, cqCommandQueue, m_dEdge[m_currentRead], data, m_numEdges, m_bDouble);
        	break;
    }       
}

void BodySystemOpenCL::synchronizeThreads() const
{
    ThreadSync(cqCommandQueue);
}


