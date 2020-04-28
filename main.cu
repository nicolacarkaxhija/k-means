#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cfloat>
#include <cmath>
#include "cuda.h"

using namespace std;

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err
			<< ") at " << file << ":" << line << endl;
	exit(1);
}

void loadArray(const char * fileName, float * data, int size) {
	ifstream inputFile(fileName);
	if (!inputFile) {
		cout << "Error: unable to open file" << endl;
		exit(1);
	}

	string line;
	int i = 0;
	while (inputFile.good() && i < size) {
		getline(inputFile, line);
		data[i] = strtof(line.c_str(), NULL);
		i++;
	}
	inputFile.close();
}

void storeArray(const char * fileName, int * data, int size) {
	ofstream outputFile(fileName);
	if (!outputFile) {
		cout << "Error: unable to open file" << endl;
		exit(1);
	}
	for (int i = 0; i < size; i++)
		outputFile << data[i] << endl;
	outputFile.close();
}

bool stoppingCriterion(float *oldClusters, float *newClusters, int length, float tolerance) {

	for (int i = 0; i < length; i++) {
		float difference = oldClusters[i] - newClusters[i];
		if (abs(difference) > tolerance)
			return false;
	}

	return true;
}

void sequentialKMeans(float *points, int *assignments, float *centroids, float *oldCentroids, int *counters, const int N, const int P, const int K, const int ITERATIONS) {
	// Initialize centroids
	for (int i = 0; i < K; i++)
		for (int j = 0; j < P; j++)
			centroids[i * P + j] = points[i * P + j];

	bool converged = false;

	int count = 0;
	while (!converged) {

		// Reset counters
		for (int i = 0; i < K; i++)
			counters[i] = 0;

		// Compute nearest cluster
		for (int i = 0; i < N; i++) {
			float minDistance = FLT_MAX;
			short int minIndex = -1;
			for (int j = 0; j < K; j++) {
				float distance = 0.0;
				for (int l = 0; l < P; l++)
					// The square root has not influence for the purpose of the results
					distance += pow(points[i * P + l] - centroids[j * P + l], 2);
				if (distance < minDistance) {
					minDistance = distance;
					minIndex = j;
				}
			}
			assignments[i] = minIndex;
			counters[minIndex]++;
		}

		// Store old centroids
		for (int i = 0; i < K * P; i++)
			oldCentroids[i] = centroids[i];

		// Reset centroids
		for (int i = 0; i < K * P; i++) {
			centroids[i] = 0;
		}

		// Update centrois
		for (int i = 0; i < N; i++) {
			int clusterId = assignments[i];
			for (int j = 0; j < P; j++)
				centroids[clusterId * P + j] += points[i * P + j] / counters[clusterId];
		}

		// Stopping criterion
		if (count == ITERATIONS)
			converged = true;
		count++;

	}
}

__global__ void computeNearestCluster(float* points, float *centroids,
		int* assignments, int* counter, int n, int p, int k) {
	short int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		float minDistance = FLT_MAX;
		short int minIndex = -1;
		for (int i = 0; i < k; i++) {
			float distance = 0.0;
			for (int j = 0; j < p; j++)
				// The square root has not influence for the purpose of the results
				distance += pow(points[index * p + j] - centroids[i * p + j],
						2);

			bool compare = (minDistance <= distance);
			minDistance = compare * minDistance + (1 - compare) * distance;
			minIndex = compare * minIndex + (1 - compare) * i;
		}
		assignments[index] = minIndex;
		atomicAdd(&(counter[minIndex]), 1);
	}
}

__global__ void computeMean(float* points, float* devCentroids,
		int* devAssignments, int* counter, int n, int p) {
	short int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		short int clusterIndex = devAssignments[index];
		for (int i = 0; i < p; i++)
			atomicAdd(&(devCentroids[clusterIndex * p + i]),
					points[index * p + i] / counter[clusterIndex]);
	}
}

void parallelKMeans(float *hostData, int *hostAssignments, float *hostCentroids, float *hostOldCentroids, int *counter, const int N, const int P, const int K, const int ITERATIONS) {

	// Allocate device (GPU) memories
	float *devData, *devCentroids, *devOldCentroids;
	int *devAssignments, *devCounter;

	CUDA_CHECK_RETURN(cudaMalloc((void** )&devData, N * P * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devCentroids, P * K * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devOldCentroids, P * K * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devAssignments, N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devCounter, K * sizeof(int)));

	// Copy data from CPU memory to GPU memory
	CUDA_CHECK_RETURN(cudaMemcpy(devData, hostData, N * P * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(devCentroids, hostData, K * P * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemset(devAssignments, 0, N * sizeof(int)));
	cudaDeviceSynchronize();

	// Invoke the kernels

	dim3 DimBlock(1024);
	dim3 DimGrid(N / 1024 + 1);

	int count = 0;
	bool converged = false;

	while (!converged) {

		CUDA_CHECK_RETURN(cudaMemcpy(devOldCentroids, devCentroids, P * K * sizeof(float), cudaMemcpyDeviceToDevice));
		CUDA_CHECK_RETURN(cudaMemset(devCounter, 0, K * sizeof(int)));
		cudaDeviceSynchronize();

		computeNearestCluster<<<DimGrid, DimBlock>>>(devData, devCentroids, devAssignments, devCounter, N, P, K);
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMemset(devCentroids, 0, P * K * sizeof(float)));
		cudaDeviceSynchronize();

		computeMean<<<DimGrid, DimBlock>>>(devData, devCentroids, devAssignments, devCounter, N, P);
		cudaDeviceSynchronize();

		if (count == ITERATIONS)
			converged = true;
		count++;

	}

	// Copy data back from GPU memory to CPU memory
	CUDA_CHECK_RETURN(cudaMemcpy(hostAssignments, devAssignments, N * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(hostCentroids, devCentroids, P * K * sizeof(float), cudaMemcpyDeviceToHost));

	// Destroy GPU memories
	CUDA_CHECK_RETURN(cudaFree(devAssignments));
	CUDA_CHECK_RETURN(cudaFree(devCentroids));
	CUDA_CHECK_RETURN(cudaFree(devOldCentroids));
	CUDA_CHECK_RETURN(cudaFree(devData));
	CUDA_CHECK_RETURN(cudaFree(devCounter));

	/*
	// Store assignments
	storeArray("assignments.txt", hostAssignments, N);

	// Round centroids (the function storeArray takes int)
	int *intCentroids = (int*)malloc(P * K * sizeof(int));
	for(int i = 0; i < K * P; i++)
	    intCentroids[i] = int(hostCentroids[i]);

	// Store centroids
	storeArray("centroids.txt", intCentroids, K * P);
    */
}

int main() {

	const int N = 200000;    // No. of points
	const int P = 2;         // Features per point
	const int K = 1024;	     // Number of clusters
	const int MAX_ITERATIONS = 200;  // Used as stopping criterion

	float *points = (float*) malloc(N * P * sizeof(float));
	// Generate points
	for (int i = 0; i < N * P; i++)
		points[i] = rand();

	int *assignments = (int*) malloc(N * sizeof(int));
	float *centroids = (float*) malloc(K * P * sizeof(float));
	float *oldCentroids = (float*) malloc(K * P * sizeof(float));
	int *counter = (int*) malloc(K * sizeof(int));

	clock_t start, end;
	double cpu_clocks;
	double seqTime, parTime;

	cout << "Running k-means..." << endl;

	// Averaging results over "iterations" runs
	const int iterations = 1;
	for (int i = 0; i < iterations; i++) {

		cout << "Iteration " << i + 1 << " of " << iterations << endl;

		// Sequential version
		start = clock();
		sequentialKMeans(points, assignments, centroids, oldCentroids, counter,	N, P, K, MAX_ITERATIONS);
		end = clock();
		cpu_clocks = end - start;
		seqTime += cpu_clocks / CLOCKS_PER_SEC;

		// Parallel version
		start = clock();
		parallelKMeans(points, assignments, centroids, oldCentroids, counter, N, P, K, MAX_ITERATIONS);
		end = clock();
		cpu_clocks = end - start;
		parTime += cpu_clocks / CLOCKS_PER_SEC;
	}

	/*
	// Store results
	char fileName[50];
	cout << "File name: ";
	cin >> fileName;
	storeArray(fileName, assignments, N);
	*/

	free(points);
	free(centroids);
	free(oldCentroids);
	free(assignments);
	free(counter);

	cout << "Average sequential execution time: " << seqTime / iterations << "s" << endl;
	cout << "Average parallel execution time: " << parTime / iterations << "s" << endl;
	cout << "Speed-up: " << seqTime / parTime << endl;

	return 0;
}
