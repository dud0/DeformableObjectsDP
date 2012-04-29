#include "oclUtils.h"
#include <memory.h>
#include <vlGraphics/Rendering.hpp>


#include "oclBodySystemOpenclLaunch.h"

BodySystemOpenCL::BodySystemOpenCL(int numBodies, int numEdges, cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
                                   unsigned int p, unsigned int q, bool usePBO, bool bDouble)
: BodySystem(numBodies),
  device(dev),
  cxContext(ctx),
  cqCommandQueue(cmdq),
  m_hPos(0),
  m_hOldPos(0),
  m_hVel(0),
  m_hF(0),
  m_hForces(0),
  m_hEdge(0),
  m_hHash(0),
  m_hIndex(0),
  m_bUsePBO(usePBO),
  m_currentRead(0),
  m_currentWrite(1),
  m_p(p),
  m_q(q),
  m_bDouble(bDouble)
{
	t=0;
    m_dPos[0] = m_dPos[1] = 0;
    m_dOldPos[0] = m_dOldPos[1] = 0;
    m_dVel[0] = m_dVel[1] = 0;
    m_dF[0] = m_dF[1] = 0;
    m_dEdge[0] = m_dEdge[1] = 0;
    m_dForces[0] = m_dForces[1] = 0;

    _initialize(numBodies, numEdges);

    // **************************************************************************************

    shrLog("\nCreateProgramAndKernel EXTERN FORCES... ");
    if (CreateProgramAndKernel(ctx, &dev, "externForces", &extFor_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "externForces ", STDERROR));
    }

    shrLog("\nCreateProgramAndKernel SPRINGS FORCES... ");
    if (CreateProgramAndKernel(ctx, &dev, "springsForces", &sprFor_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "springsForces ", STDERROR));
    }

    shrLog("\nCreateProgramAndKernel INTEGRATE SYSTEM... ");
    if (CreateProgramAndKernel(ctx, &dev, "integrateBodies", &intBod_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "integrateBodies ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel INTEGRATE VERLET...");
    if (CreateProgramAndKernel(ctx, &dev, "integrateVerlet", &intVer_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "integrateVerlet ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel CALC HASH...");
    if (CreateProgramAndKernel(ctx, &dev, "calcHash", &calcHash_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "calcHash ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel MEM SET...");
    if (CreateProgramAndKernel(ctx, &dev, "Memset", &memSet_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "Memset ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel FIND BOUNDARIES...");
    if (CreateProgramAndKernel(ctx, &dev, "findCellBoundsAndReorder", &findBound_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "findCellBoundsAndReorder ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel COLLIDE...");
    if (CreateProgramAndKernel(ctx, &dev, "collide", &collide_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "collide ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel BITONIC LOCAL...");
    if (CreateProgramAndKernel(ctx, &dev, "bitonicSortLocal1", &bitLoc_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "bitLocal ", STDERROR));
    }

    shrLog("\nCreateProgramAndkernel BITONIC GLOBAL...");
    if (CreateProgramAndKernel(ctx, &dev, "bitonicMergeGlobal", &bitGlo_kernel, m_bDouble))
    {
    	exit(shrLogEx(LOGBOTH | CLOSELOG, -1, "bitGlobal ", STDERROR));
    }

    setSoftening(0.00125f);
    setDamping(0.995f);   
}

BodySystemOpenCL::~BodySystemOpenCL()
{
    _finalize();
    m_numBodies = 0;
    m_numEdges = 0;
}

void BodySystemOpenCL::_initialize(int numBodies, int numEdges)
{
    oclCheckError(m_bInitialized, shrFALSE);

    m_numBodies = numBodies;
    m_numEdges = numEdges;

    m_hOldPos = new float[m_numBodies*4];
    m_hPos = new float[m_numBodies*4];
    m_hVel = new float[m_numBodies*4];
    m_hF = new float[m_numBodies*4];
    m_hEdge = new float[m_numEdges*4];
    m_hForces = new float[m_numBodies*4];
    m_hHash = new float[m_numBodies*4];
    m_hIndex = new float[m_numBodies*4];

    memset(m_hOldPos, 0, m_numBodies*4*sizeof(float));
    memset(m_hPos, 0, m_numBodies*4*sizeof(float));
    memset(m_hVel, 0, m_numBodies*4*sizeof(float));
    memset(m_hF, 0, m_numBodies*4*sizeof(float));
    memset(m_hEdge, 0, m_numEdges*4*sizeof(float));
    memset(m_hForces, 0, m_numBodies*4*sizeof(float));
    memset(m_hIndex, 0, m_numBodies*4*sizeof(float));
    memset(m_hHash, 0, m_numBodies*4*sizeof(float));

    if (m_bUsePBO)
    {
        // create the position pixel buffer objects for rendering
        // we will actually compute directly from this memory in OpenCL too
        vl::glGenBuffers(2, (GLuint*)m_pboGL);
        shrLog("Allocating Pixel Buffers\n"); 
        for (int i = 0; i < 2; ++i)
        {
            vl::glBindBuffer(GL_ARRAY_BUFFER, m_pboGL[i]);
            vl::glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * m_numBodies, m_hPos, GL_DYNAMIC_DRAW);

            int size = 0;
            vl::glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);
            if ((unsigned)size != 4 * (sizeof(float) * m_numBodies))
            {
                shrLogEx(LOGBOTH, -1, "WARNING: Pixel Buffer Object allocation failed!\n"); 
            }
            vl::glBindBuffer(GL_ARRAY_BUFFER, 0);
            m_pboCL[i] = RegisterGLBufferObject(cxContext, m_pboGL[i]);
        }
    }
    else
    {
        AllocateNBodyArrays(cxContext, m_dPos, m_numBodies, m_bDouble);
        shrLog("\nAllocateNBodyArrays m_dPos\n"); 
    }
    
    AllocateNBodyArrays(cxContext, m_dOldPos, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dOldPos\n");

    AllocateNBodyArrays(cxContext, m_dVel, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dVel\n"); 

    AllocateNBodyArrays(cxContext, m_dF, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dF\n");

    AllocateNBodyArrays(cxContext, m_dForces, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dF\n");

    AllocateNBodyArrays(cxContext, m_dEdge, m_numEdges, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dEdge\n");

    AllocateNBodyArrays(cxContext, m_dHash, m_numBodies, m_bDouble);
    shrLog("\nAllocateArray m_dHash\n");

    AllocateNBodyArrays(cxContext, m_dIndex, m_numBodies, m_bDouble);
    shrLog("\nAllocateArray m_dIndex\n");

    AllocateNBodyArrays(cxContext, m_dCellStart, 64*64*64, m_bDouble);
    shrLog("\nAllocateArray m_dCellStart\n");

    AllocateNBodyArrays(cxContext, m_dCellEnd, 64*64*64, m_bDouble);
    shrLog("\nAllocateArray m_dCellEnd\n");

    AllocateNBodyArrays(cxContext, m_dReorderedPos, m_numBodies, m_bDouble);
    shrLog("\nAllocateArray m_dReorderedPos\n");

    AllocateNBodyArrays(cxContext, m_dReorderedVel, m_numBodies, m_bDouble);
    shrLog("\nAllocateArray m_dReorderedVel\n");

    m_bInitialized = true;
}

void BodySystemOpenCL::_finalize()
{
    oclCheckError(m_bInitialized, shrTRUE);

    delete [] m_hPos;
    delete [] m_hOldPos;
    delete [] m_hVel;
    delete [] m_hForces;
    delete [] m_hF;
    delete [] m_hEdge;
    delete [] m_hHash;
    delete [] m_hIndex;

	clReleaseKernel(extFor_kernel);
	clReleaseKernel(sprFor_kernel);
	clReleaseKernel(intBod_kernel);
	clReleaseKernel(calcHash_kernel);
	clReleaseKernel(memSet_kernel);
	clReleaseKernel(findBound_kernel);
	clReleaseKernel(bitLoc_kernel);
	clReleaseKernel(bitGlo_kernel);

    DeleteNBodyArrays(m_dVel);
    DeleteNBodyArrays(m_dF);
    DeleteNBodyArrays(m_dEdge);
    DeleteNBodyArrays(m_dForces);
    DeleteNBodyArrays(m_dOldPos);
    DeleteNBodyArrays(m_dHash);
    DeleteNBodyArrays(m_dCellStart);
    DeleteNBodyArrays(m_dCellEnd);
    DeleteNBodyArrays(m_dIndex);
    DeleteNBodyArrays(m_dReorderedPos);
    DeleteNBodyArrays(m_dReorderedVel);

    if (m_bUsePBO)
    {
        UnregisterGLBufferObject(m_pboCL[0]);
        UnregisterGLBufferObject(m_pboCL[1]);
        vl::glDeleteBuffers(2, (const GLuint*)m_pboGL);
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

    computeExternalForces(cqCommandQueue,
        		extFor_kernel,
        		m_dForces[m_currentWrite],
        		m_dF[m_currentWrite],
        		m_dForces[m_currentRead],
        		m_dF[m_currentRead],
        		m_dVel[m_currentRead],
        		m_numBodies, m_p, m_q,
        		1);

    computeSpringsForces(cqCommandQueue,
        		sprFor_kernel,
        		m_dForces[m_currentWrite],
        		m_dEdge[m_currentWrite],
        		m_dPos[m_currentWrite],
        		m_dForces[m_currentRead],
        		m_dPos[m_currentRead],
        		m_dEdge[m_currentRead],
        		m_numEdges, m_p, m_q,
        		1);

/*    integrateSystemVerlet(cqCommandQueue,
        			intVer_kernel,
        			m_dPos[m_currentWrite],
        			m_dPos[m_currentRead],
        			m_dOldPos[m_currentWrite],
        			m_dOldPos[m_currentRead],
        			m_dForces[m_currentRead],
        			deltaTime,
        			m_numBodies, m_p, m_q,
       			1);
*/
    	integrateSystem(cqCommandQueue,
    			intBod_kernel,
    			m_dPos[m_currentWrite],
    			m_dVel[m_currentWrite],
    			m_dEdge[m_currentWrite],
    			m_dForces[m_currentWrite],
    			m_dPos[m_currentRead],
    			m_dVel[m_currentRead],
    			m_dEdge[m_currentRead],
    			m_dForces[m_currentRead],
    			deltaTime, m_damping,
    			m_numBodies, m_p, m_q,
    			1);

    	calcHash(cqCommandQueue,
    			calcHash_kernel,
    	        m_dHash[0],
    	        m_dIndex[0],
    	        m_dPos[m_currentWrite],
    	        m_numBodies
    	    );

    	bitonicSort(cqCommandQueue, bitGlo_kernel, bitLoc_kernel, m_dHash[0], m_dIndex[0], m_dHash[0], m_dIndex[0], 1, m_numBodies, 0);

    	//Find start and end of each cell and
    	//Reorder particle data for better cache coherency
    	findCellBoundsAndReorder(cqCommandQueue,
    			findBound_kernel,
    			memSet_kernel,
    			m_dCellStart[0],
    			m_dCellEnd[0],
    			m_dReorderedPos[0],
    			m_dReorderedVel[0],
    			m_dHash[0],
    			m_dIndex[0],
    			m_dPos[m_currentWrite],
    			m_dVel[m_currentWrite],
    			m_numBodies,
    			64*64*64
    	);

    	collide(cqCommandQueue,
    			collide_kernel,
    	        m_dVel[m_currentWrite],
    	        m_dReorderedPos[0],
    	        m_dReorderedVel[0],
    	        m_dIndex[0],
    	        m_dCellStart[0],
    	        m_dCellEnd[0],
    	        m_numBodies,
    	        64*64*64
    	    );

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
        case BODYSYSTEM_OLD_POSITION:
        	hdata = m_hOldPos;
        	ddata = m_dOldPos[m_currentRead];
        	nB = m_numBodies;
        	break;
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
        case BODYSYSTEM_F: //externe
        	hdata = m_hF;
        	ddata = m_dF[m_currentRead];
        	nB = m_numBodies;
        	break;
        case BODYSYSTEM_FORCES: //interne
        	hdata = m_hForces;
        	ddata = m_dForces[m_currentRead];
        	nB = m_numBodies;
        	break;
        case BODYSYSTEM_FORCES_WRITE:
        	hdata = m_hForces;
        	ddata = m_dForces[m_currentWrite];
        	nB = m_numBodies;
        	break;
        case BODYSYSTEM_EDGE:
        	hdata = m_hEdge;
        	ddata = m_dEdge[m_currentRead];
        	nB = m_numEdges;
        	shrLog("\nBODYSYSTEM_EDGE %d\n", nB);
        	break;
        case COL_HASH:
        	hdata = m_hHash;
        	ddata = m_dHash[0];
        	nB = m_numBodies;
        	break;
        case COL_INDEX:
        	hdata = m_hIndex;
        	ddata = m_dIndex[0];
        	nB = m_numBodies;
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
                vl::glBindBuffer(GL_ARRAY_BUFFER, m_pboGL[m_currentRead]);
                vl::glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * m_numBodies, data, GL_DYNAMIC_DRAW);
            
                int size = 0;
                vl::glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);
                if ((unsigned)size != 4 * (sizeof(float) * m_numBodies))
                {
                    shrLogEx(LOGBOTH, -1, "WARNING: Pixel Buffer Object download failed!\n"); 
                }
                vl::glBindBuffer(GL_ARRAY_BUFFER, 0);
                m_pboCL[m_currentRead] = RegisterGLBufferObject(cxContext, m_pboGL[m_currentRead]);
            }
            else
            {
                CopyArrayToDevice(0, cqCommandQueue, m_dPos[m_currentRead], data, m_numBodies, m_bDouble);
            }
        }
            break;
        case BODYSYSTEM_OLD_POSITION:
        	CopyArrayToDevice(0, cqCommandQueue, m_dOldPos[m_currentRead], data, m_numBodies, m_bDouble);
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

        case BODYSYSTEM_FORCES_WRITE:
        	CopyArrayToDevice(0, cqCommandQueue, m_dForces[m_currentWrite], data, m_numBodies, m_bDouble);
        	break;

        case BODYSYSTEM_EDGE:
        	CopyArrayToDevice(0, cqCommandQueue, m_dEdge[m_currentRead], data, m_numEdges, m_bDouble);
        	break;
    }       
}

void BodySystemOpenCL::synchronizeThreads() const
{
    ThreadSync(cqCommandQueue);
}

cl_mem BodySystemOpenCL::getPos() {
	return m_dPos[m_currentRead];
}

cl_mem BodySystemOpenCL::getEdges() {
	return m_dEdge[m_currentRead];
}


