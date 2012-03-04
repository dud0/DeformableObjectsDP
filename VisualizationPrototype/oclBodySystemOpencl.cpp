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
	t=0;
    m_dPos[0] = m_dPos[1] = 0;
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

    m_hPos = new float[m_numBodies*4];
    m_hVel = new float[m_numBodies*4];
    m_hF = new float[m_numBodies*4];
    m_hEdge = new float[m_numEdges*4];
    m_hForces = new float[m_numBodies*4];

    memset(m_hPos, 0, m_numBodies*4*sizeof(float));
    memset(m_hVel, 0, m_numBodies*4*sizeof(float));
    memset(m_hF, 0, m_numBodies*4*sizeof(float));
    memset(m_hEdge, 0, m_numEdges*4*sizeof(float));
    memset(m_hForces, 0, m_numBodies*4*sizeof(float));

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
    
    AllocateNBodyArrays(cxContext, m_dVel, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dVel\n"); 

    AllocateNBodyArrays(cxContext, m_dF, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dF\n");

    AllocateNBodyArrays(cxContext, m_dForces, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dF\n");

    AllocateNBodyArrays(cxContext, m_dEdge, m_numEdges, m_bDouble);
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

	clReleaseKernel(extFor_kernel);
	clReleaseKernel(sprFor_kernel);
	clReleaseKernel(intBod_kernel);

    DeleteNBodyArrays(m_dVel);
    DeleteNBodyArrays(m_dF);
    DeleteNBodyArrays(m_dEdge);
    DeleteNBodyArrays(m_dForces);
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
	t++;
	/*bool cont = true;

	if (t == 200) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 400) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		tmpF[0*4+1] = -30.0f;
		tmpF[1*4+1] = -30.0f;
		tmpF[2*4+1] = -30.0f;
		tmpF[9*4+1] = -30.0f;
		tmpF[10*4+1] = -30.0f;
		tmpF[11*4+1] = -30.0f;
		tmpF[18*4+1] = -30.0f;
		tmpF[19*4+1] = -30.0f;
		tmpF[20*4+1] = -30.0f;
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 700) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 900) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		tmpF[6*4+1] = 30.0f;
		tmpF[7*4+1] = 30.0f;
		tmpF[8*4+1] = 30.0f;
		tmpF[15*4+1] = 30.0f;
		tmpF[16*4+1] = 30.0f;
		tmpF[17*4+1] = 30.0f;
		tmpF[24*4+1] = 30.0f;
		tmpF[25*4+1] = 30.0f;
		tmpF[26*4+1] = 30.0f;
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 1100) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 1250) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		tmpF[18*4] = 30.0f;
		tmpF[19*4] = 30.0f;
		tmpF[20*4] = 30.0f;
		tmpF[21*4] = 30.0f;
		tmpF[22*4] = 30.0f;
		tmpF[23*4] = 30.0f;
		tmpF[24*4] = 30.0f;
		tmpF[25*4] = 30.0f;
		tmpF[26*4] = 30.0f;
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 1500) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 1650) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		tmpF[0*4] = -30.0f;
		tmpF[1*4] = -30.0f;
		tmpF[2*4] = -30.0f;
		tmpF[3*4] = -30.0f;
		tmpF[4*4] = -30.0f;
		tmpF[5*4] = -30.0f;
		tmpF[6*4] = -30.0f;
		tmpF[7*4] = -30.0f;
		tmpF[8*4] = -30.0f;
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 1850) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		setArray(BODYSYSTEM_F, tmpF);
	}

	else if (t == 2000) {
		tmpF = getArray(BODYSYSTEM_F);
		for(int i=0; i<m_numBodies*4; i++) {
			tmpF[i] = 0;
		}
		tmpF[24*4] = 30.0f;
		tmpF[25*4] = 30.0f;
		tmpF[26*4] = 30.0f;
		tmpF[24*4+1] = 30.0f;
		tmpF[25*4+1] = 30.0f;
		tmpF[26*4+1] = 30.0f;
		setArray(BODYSYSTEM_F, tmpF);
	}
*/
	fprintf(stdout, "%d\n",t);

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

 /*   tmpF = getArray(BODYSYSTEM_FORCES_WRITE);
    fprintf(stdout,"\n\n#####################################################\n");
    fprintf(stdout,"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    for(int i = 0; i < m_numBodies; i++) {
    	fprintf(stdout, "%d--> x: %f, y: %f, z: %f\n", i, tmpF[i*4], tmpF[i*4+1], tmpF[i*4+2]);
    }


    tmpF = getArray(BODYSYSTEM_FORCES);
    setArray(BODYSYSTEM_FORCES_WRITE, tmpF);
*/
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


  /*  fprintf(stdout,"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    tmpF = getArray(BODYSYSTEM_FORCES_WRITE);
    for(int i = 0; i < m_numBodies; i++) {
    	fprintf(stdout, "%d--> x: %f, y: %f, z: %f\n", i, tmpF[i*4], tmpF[i*4+1], tmpF[i*4+2]);
    }
    fprintf(stdout,"#####################################################\n");

    tmpF = getArray(BODYSYSTEM_FORCES_WRITE);
    setArray(BODYSYSTEM_FORCES, tmpF);
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


