#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

using namespace std;

#define NUM_THREADS 256

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

extern double size;
//
//  benchmarking program
//

double s;

void init(int n) {
    s = sqrt(density * n);
}

int getRows() {
    return s / cutoff + 1;
}

__device__ int getLocation(double cord) {
    double unit = cutoff;
    int loc = cord / unit;
    return loc;
}

__device__ int get_bin_index(particle_t p, int rows){
    double x = p.x;
    double y = p.y;
    int column = getLocation(x);
    int row = getLocation(y);
    int location = row * rows + column;
    return location;
}


__global__ void compute_bin_size(particle_t* &particles, int n, int * offsets, int rows){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
    int location = get_bin_index(particles[i],rows);
    atomicAdd(offsets + location,1);
  }
}

__global__ void push_in_bins(particle_t* particles, int* particles_index, int n, int * offsets, int rows){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
    int location = get_bin_index(particles[i],rows);
    int old_upper_border = atomicSub(offsets + location,1); // It returns the old value;
    particles_index[old_upper_border-1] = i;
  }
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}


__constant__ const int a[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__constant__ const int b[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

__global__ void compute_forces(particle_t* particles, int* particles_index, int n, int * offsets, int rows){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
    double x = particles[particles_index[i]].x;
    double y = particles[particles_index[i]].y;
    particles[particles_index[i]].ax = 0;
    particles[particles_index[i]].ay = 0;
    int currentX = getLocation(x);
    int currentY = getLocation(y);

    for (int j = 0; j < 9; j++) {
      int newX = currentX + a[j];
      int newY = currentY + b[j];
      if (newX >= 0 && newX < rows && newY >= 0 && newY < rows) {
        int location = newY * rows + newX;
        int start = (location==0)?0:offsets[location-1];
        int end = offsets[location];
        for(int k = start; k<end; k++){
          if(k==i) continue;
          apply_force_gpu(particles[particles_index[i]], particles[particles_index[k]]);
        }
      }
    }
  }
}




__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}



int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * particles_gpu;
    cudaMalloc((void **) &particles_gpu, n * sizeof(particle_t));

    init( n );
    set_size( n );
    int rows = getRows();
    int binNum = rows * rows;

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(particles_gpu, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;


    int * particles_index;
    cudaMalloc((void **) &particles_index,n * sizeof(int));

    int * partition_offsets;
    cudaMalloc((void **) &partition_offsets,binNum * sizeof(int));
    // a number  p = partition_offsets[i] means the i-th bin ends at particle p.

    int partition_offsets_cpu[binNum];


    //  simulate a number of time steps
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

    for( int step = 0; step < NSTEPS; step++ )
    {

      // Compute size of each bin
      cudaMemset(partition_offsets, 0, binNum * sizeof(int));
      compute_bin_size <<< blks, NUM_THREADS >>> (particles_gpu, n, partition_offsets, rows);
      cudaThreadSynchronize();

      // Prefix-sum could be computed faster in parallel, but probably very complicated >.< refer to the followsing links
      // http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
      // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

      // Prefix-sum to compute offsets.
      cudaMemcpy(partition_offsets_cpu, partition_offsets, binNum * sizeof(int), cudaMemcpyDeviceToHost);
      for(int i=1;i<binNum;i++){
        partition_offsets_cpu[i] += partition_offsets_cpu[i-1];
      }
      cudaMemcpy(partition_offsets, partition_offsets_cpu, binNum * sizeof(int), cudaMemcpyHostToDevice);


      // Push particles in bins
      push_in_bins <<< blks, NUM_THREADS >>> (particles_gpu, particles_index, n, partition_offsets, rows);
      cudaThreadSynchronize();
      // The offsets are all 0 after this function, so we have to restore it.
      cudaMemcpy(partition_offsets, partition_offsets_cpu, binNum * sizeof(int), cudaMemcpyHostToDevice);


      //  compute forces
      compute_forces <<< blks, NUM_THREADS >>> (particles_gpu, particles_index, n, partition_offsets, rows);
      cudaThreadSynchronize();


      //  move particles (don't need to change this)
      move_gpu <<< blks, NUM_THREADS >>> (particles_gpu, n, size);
      cudaThreadSynchronize();
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, particles_gpu, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(particles_gpu);
    cudaFree(particles_index);
    cudaFree(partition_offsets);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
