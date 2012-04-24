/*
 * CLManager.hpp
 *
 *  Created on: Mar 18, 2012
 *      Author: dud0
 */

#ifndef CLMANAGER_HPP_
#define CLMANAGER_HPP_

#include "tables.h"

#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"

#include "oclScan_common.h"

class CLManager {
public:
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
	cl_kernel calcColorIntensitiesTensionKernel;
	cl_int ciErrNum;

	cl_bool g_glInterop;

	CLManager() {
		g_glInterop = true;

		initCL();

		// tables
		d_numVertsTable = 0;
		d_triTable = 0;

		// allocate textures
		allocateTextures(&d_triTable, &d_numVertsTable );

		cPathAndName = NULL;
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

		calcColorIntensitiesTensionKernel = clCreateKernel(cpProgram, "calcColorIntensitiesTension", &ciErrNum);

		// Setup Scan
		initScan(cxGPUContext, cqCommandQueue);
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

	void Cleanup(int iExitCode) {
		if(calcColorIntensitiesTensionKernel)clReleaseKernel(calcColorIntensitiesTensionKernel);
		if(calcFieldValueKernel)clReleaseKernel(calcFieldValueKernel);
		if(compactVoxelsKernel)clReleaseKernel(compactVoxelsKernel);
		if(compactVoxelsKernel)clReleaseKernel(generateTriangles2Kernel);
		if(compactVoxelsKernel)clReleaseKernel(classifyVoxelKernel);
		if(cpProgram)clReleaseProgram(cpProgram);

		if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
		if(cxGPUContext)clReleaseContext(cxGPUContext);

		if( d_triTable ) clReleaseMemObject(d_triTable);
		if( d_numVertsTable ) clReleaseMemObject(d_numVertsTable);

		/*if ((g_bNoprompt)||(bQATest))
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
		exit (iExitCode);*/
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

	void launch_calcColorIntensitiesTension(size_t global_work_size, size_t local_work_size, cl_mem edges, cl_mem points, cl_mem colorIntensities, cl_uint edgeCnt, cl_uint pointCnt, cl_uint offset) {
		clSetKernelArg(calcColorIntensitiesTensionKernel, 0, sizeof(cl_mem), &edges);
		clSetKernelArg(calcColorIntensitiesTensionKernel, 1, sizeof(cl_mem), &points);
		clSetKernelArg(calcColorIntensitiesTensionKernel, 2, sizeof(cl_mem), &colorIntensities);
		clSetKernelArg(calcColorIntensitiesTensionKernel, 3, sizeof(cl_uint), &edgeCnt);
		clSetKernelArg(calcColorIntensitiesTensionKernel, 4, sizeof(cl_uint), &pointCnt);
		clSetKernelArg(calcColorIntensitiesTensionKernel, 5, sizeof(cl_uint), &offset);
		ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, calcColorIntensitiesTensionKernel, 1, NULL, &global_work_size, &local_work_size, 0, 0, 0);

		//printf("\nERROR: %s\n", oclErrorString(ciErrNum));
	}

	void launch_calcFieldValue(dim3 grid, dim3 threads, cl_mem volumeData, cl_mem points, cl_mem colorIntensities, cl_uint pointCnt, cl_uint offset, cl_float radius, cl_uint gridSizeShift[4], cl_uint gridSizeMask[4]) {
		clSetKernelArg(calcFieldValueKernel, 0, sizeof(cl_mem), &volumeData);
		clSetKernelArg(calcFieldValueKernel, 1, sizeof(cl_mem), &points);
		clSetKernelArg(calcFieldValueKernel, 2, sizeof(cl_mem), &colorIntensities);
		clSetKernelArg(calcFieldValueKernel, 3, sizeof(cl_uint), &pointCnt);
		clSetKernelArg(calcFieldValueKernel, 4, sizeof(cl_uint), &offset);
		clSetKernelArg(calcFieldValueKernel, 5, sizeof(cl_float), &radius);
		clSetKernelArg(calcFieldValueKernel, 6, 4 * sizeof(cl_uint), gridSizeShift);
		clSetKernelArg(calcFieldValueKernel, 7, 4 * sizeof(cl_uint), gridSizeMask);

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
protected:

	// tables
	cl_mem d_numVertsTable;
	cl_mem d_triTable;

	char* cPathAndName;          		// var for full paths to data, src, etc.
	char* cSourceCL;                    // Buffer to hold source for compilation
};


#endif /* CLMANAGER_HPP_ */
