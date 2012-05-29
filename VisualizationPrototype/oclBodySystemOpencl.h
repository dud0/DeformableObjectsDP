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

#ifndef __CL_BODYSYSTEMOPENCL_H__
#define __CL_BODYSYSTEMOPENCL_H__

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 
#include "oclBodySystem.h"

// OpenCL BodySystem: runs on the GPU
class BodySystemOpenCL : public BodySystem
{
    public:
        BodySystemOpenCL(int numBodies, int numEdges, cl_device_id dev, cl_context ctx, cl_command_queue cmdq, unsigned int p, unsigned int q, bool usePBO, bool bDouble);
        virtual ~BodySystemOpenCL();

        virtual void update(float deltaTime);

        virtual void setSoftening(float softening);
        virtual void setDamping(float damping);

        virtual float* getArray(BodyArray array);
        virtual void   setArray(BodyArray array, const float* data);

        virtual size_t getCurrentReadBuffer() const 
        {
            if (m_bUsePBO) 
            {
                return m_pboGL[m_currentRead]; 
            } 
            else 
            {
                return (size_t) m_hPos;
            }
        }

        virtual void synchronizeThreads() const;

        cl_mem getPos();
        cl_mem getEdges();

    protected: // methods
        BodySystemOpenCL() {}

        virtual void _initialize(int numBodies, int numEdges);
        virtual void _finalize();
        
    protected: // data
        cl_device_id device;
        cl_context cxContext;
        cl_command_queue cqCommandQueue;

        cl_kernel MT_kernel;
        cl_kernel noMT_kernel;

        cl_kernel extFor_kernel;
        cl_kernel sprFor_kernel;
        cl_kernel intBod_kernel;
        cl_kernel intVer_kernel;

        cl_kernel calcHash_kernel;
        cl_kernel memSet_kernel;
        cl_kernel findBound_kernel;
        cl_kernel collide_kernel;

        cl_kernel bitLoc_kernel;
        cl_kernel bitGlo_kernel;

        // CPU data
        float* m_hOldPos;
        float* m_hPos;
        float* m_hVel;
        float* m_hF;
        float* m_hEdge;
        float* m_hForces;
        float* m_hHash;
        float* m_hIndex;

        float* tmpF;
        float* tmpFF;

        long t;

        // GPU data
        cl_mem m_dPos[2];
        cl_mem m_dOldPos[2];
        cl_mem m_dVel[2];
        cl_mem m_dF[2];
        cl_mem m_dEdge[2];
        cl_mem m_dForces[2];
        // kolizie
        cl_mem m_dHash[2];
        cl_mem m_dIndex[2];
        cl_mem m_dCellStart[2];
        cl_mem m_dCellEnd[2];
        cl_mem m_dReorderedPos[2];
        cl_mem m_dReorderedVel[2];
        cl_mem m_dReorderedForce[2];

        bool m_bUsePBO;

        float m_softeningSq;
        float m_damping;

        unsigned int m_pboGL[2];
        cl_mem       m_pboCL[2];
        unsigned int m_currentRead;
        unsigned int m_currentWrite;

        unsigned int m_p;
        unsigned int m_q;

		//for double precision
		bool m_bDouble;
};

#endif // __CLH_BODYSYSTEMOPENCL_H__
