#include <stdlib.h>

#include "gpuR/dynEigenVec.hpp"

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

// [[Rcpp::export]]
void clfft_test(SEXP ptrA_)
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_event event = NULL;
	
    Rcpp::XPtr<dynEigenVec<float> > ptrA(ptrA_);
	
	// move data to device
	ptrA->to_device(0);
	
	// get device pointer
	viennacl::vector_base<float> *vcl_A = ptrA->getDevicePtr();
	
	// number of elements
	const size_t N = vcl_A->internal_size();

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[1] = {N};

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	
    // Create a command queue and use the first device
    // CommandQueue queue = CommandQueue(context, devices[0], 0, &err);
	queue = vcl_A->handle().opencl_handle().context().get_queue().handle().get();

    /* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);


    /* Prepare OpenCL memory objects and place matrices inside them. */
    // Get memory buffers
    cl_mem *bufX = &vcl_A->handle().opencl_handle().get();

	/* Create a default plan for a complex FFT. */
	err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

	/* Set plan parameters. */
	err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /* Bake the plan. */
	err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

	/* Execute the plan. */
	err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, bufX, NULL, NULL);

	/* Wait for calculations to be finished. */
	err = clFinish(queue);

	// Copy back to host
	ptrA->to_host();
					  

	/* Release the plan. */
	err = clfftDestroyPlan( &planHandle );

    /* Release clFFT library. */
    clfftTeardown( );

    /* Release OpenCL working objects. */
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return;
}
