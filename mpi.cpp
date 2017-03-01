#include <mpi.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "common.h"

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

using namespace std;

double s;

void init(int n){
    s=sqrt(density * n);
}

int getRows(){
    return s/cutoff+1;
}

int getLocation( double cord) {
    double unit = cutoff;
    int loc = cord / unit;
    return loc;
}

std::vector<int> getNeighbors(int rows, int currentX, int currentY) {
    int a[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int b[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    std::vector<int> result;
    for (int i = 0; i <9; i++) {
        int newX = currentX + a[i];
        int newY = currentY + b[i];
        if (newX >= 0 && newX < rows && newY >= 0 && newY < rows) {
            int location = newY * rows + newX;
            result.push_back(location);
        }
    }
    return result;
}

int get_bin_index(particle_t p, int rows){
    double x = p.x;
    double y = p.y;
    int column = getLocation(x);
    int row = getLocation(y);
    int location = row * rows + column;
    return location;
}

int main( int argc, char **argv )
{
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;
    
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //cout<<rank<<" "<<n_proc<<endl;
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;
    /*    
    cout<<"start malloc for particles"<<endl;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    cout<<"start malloc for particles"<<endl;
    */

    init(n);
    set_size(n);
    int rows = getRows();
    int binNum = rows * rows;
    int rows_per_proc = rows/n_proc; // was +1
    int bin_per_proc = rows_per_proc * rows;
    
    
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );


    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ ){
        if(i==n_proc){
            partition_offsets[i] = binNum;
        }
        else partition_offsets[i] = i * bin_per_proc;
        //cout<<partition_offsets[i]<<" ";
    }
    //cout<<endl;
    
   
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
    // Should consider the case that if later processes have no bins at all
    // e.g. binNum = 70, n_proc = 32, 3 for each, but 9 will be idle
    
    //
    //  allocate storage for local partition
    //
    int binNum_local = partition_sizes[rank];
    //cout<<binNum_local<<" in "<<rank<<endl;
    //std::vector<std::vector<particle_t> > particle_bin(binNum_local, std::vector<particle_t>());
    std::vector<std::vector<particle_t> > particle_bin;
    for (int i = 0; i < binNum_local; i++) {
        particle_bin.push_back(std::vector<particle_t>());
    }
    
    
    if( rank == 0 )
        init_particles( n, particles );
    //MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
    //int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )
    MPI_Bcast( particles, n, PARTICLE, 0, MPI_COMM_WORLD ); //Want to put all particles to all local copy, to push in local bins
    
    for(int i = 0; i < n; i++){
        int location = get_bin_index(particles[i], rows);
        int local_location = location - partition_offsets[rank];
        if(local_location >= 0 && local_location < binNum_local) // This particle is in my control
            particle_bin[local_location].push_back(particles[i]);
    }
    
    std::vector<particle_t> up_border;
    std::vector<particle_t> down_border;
    
    std::vector<particle_t> from_up_border_recv(n);
    std::vector<particle_t> from_down_border_recv(n);
    
    std::vector<std::vector<particle_t> > to_send_after_move(n_proc,std::vector<particle_t>());
    
    std::vector<std::vector<particle_t> > recv_from_proc(n_proc, std::vector<particle_t>(n));
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        //if(rank==n_proc-1) cout<<"step"<<step<<endl;
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        //
        //  collect all global data locally (not good idea to do)
        //
        //MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        
        // Send a vector MPI_Send(&data.front(), data.size(), MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        
        // Before computing forces, send all my side ghost zones.
        if(rank != 0){ //Care about up_border
            if(up_border.size()!=0)
                up_border.clear();
           
            for(int i=0;i<rows;i++){
                for(int j=0;j<particle_bin[i].size();j++){
                    up_border.push_back(particle_bin[i][j]);
                }
            }
            //send up_border
            int tag = 1; // Denoting sending an up_border?
            MPI_Request request;
            MPI_Isend(&up_border.front(), up_border.size(), PARTICLE, rank-1, tag, MPI_COMM_WORLD, &request);
            // Or replace &up_border.front() by &up_border[0]
        }
        if(rank != n_proc-1){ //Care about down_border
            down_border.clear();
            for(int i = binNum_local - rows; i < binNum_local; i++){
                for(int j=0;j<particle_bin[i].size();j++){
                    down_border.push_back(particle_bin[i][j]);
                }
            }
            //send down_border
            int tag = 2; // Denoting sending an down_border?
            MPI_Request request;
            MPI_Isend(&down_border.front(), down_border.size(), PARTICLE, rank+1, tag, MPI_COMM_WORLD, &request);
        }
        
        std::vector<std::vector<particle_t> > from_up_bin(rows, vector<particle_t>());
        std::vector<std::vector<particle_t> > from_down_bin(rows, vector<particle_t>());
        /*
         for (int i = 0; i < rows; i++) {
         from_up_bin.push_back(std::vector<particle_t>());
         from_down_bin.push_back(std::vector<particle_t>());
         }
         */
        
        int num_in_from_up;
        int num_in_from_down;
        
        if(rank != 0){ // consider the info received from the process above (from_up)
            MPI_Status status;
            //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
            MPI_Recv(&from_up_border_recv.front(), n, PARTICLE, rank-1, 2, MPI_COMM_WORLD, &status);
            //int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)
            MPI_Get_count(&status, PARTICLE, &num_in_from_up);
            
            for(int i=0; i < num_in_from_up ; i++){
                int location = get_bin_index(from_up_border_recv[i],rows);
                int local_location = location % rows;
                from_up_bin[local_location].push_back(from_up_border_recv[i]);
            }
        }
        if(rank != n_proc-1){
            MPI_Status status;
            MPI_Recv(&from_down_border_recv.front(), n, PARTICLE, rank+1, 1, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &num_in_from_down);
            
            for(int i=0; i < num_in_from_down; i++){
                int location = get_bin_index(from_down_border_recv[i],rows);
                int local_location = location % rows;
                from_down_bin[local_location].push_back(from_down_border_recv[i]);
            }
        }
        
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );
        
        //
        //  compute all forces
        //
        for(int i=0; i < binNum_local; i++){
            for(int j=0; j < particle_bin[i].size(); j++){
                double x = particle_bin[i][j].x;
                double y = particle_bin[i][j].y;
                particle_bin[i][j].ax = 0;
                particle_bin[i][j].ay = 0;
                int currentX = getLocation(x);
                int currentY = getLocation(y);
                
                std::vector<int> neighbors = getNeighbors(rows, currentX, currentY);
                // neighbors contain the index of neighbors in the original coordination;
                
                for(int k=0; k < neighbors.size() ;k++){
                    int currentIndex = neighbors[k] - partition_offsets[rank];
                    
                    // Want to use pointer to simplify, not sure whether it works, if not, change back to complicated code
                    //(std::vector<particle_t>)* pointer;
                    std::vector<particle_t>::iterator it;
                    std::vector<particle_t>::iterator it_end;
                    
                    if(currentIndex < 0){
                        //pointer = &(from_up_bin[currentIndex+rows]);
                        it = from_up_bin[currentIndex+rows].begin();
                        it_end = from_up_bin[currentIndex+rows].end();
                    }
                    else if(currentIndex < partition_sizes[rank]){
                        //pointer = &(particle_bin[currentIndex]);
                        it = particle_bin[currentIndex].begin();
                        it_end = particle_bin[currentIndex].end();
                    }
                    else{
                        //pointer = &(from_down_bin[currentIndex - binNum_local]);
                        it = from_down_bin[currentIndex - binNum_local].begin();
                        it_end = from_down_bin[currentIndex - binNum_local].end();
                    }
                    
                    // for (int q = 0; q<(*pointer).size(); q++) {
                    //     apply_force(particle_bin[i][j], (*pointer)[q], &dmin, &davg, &navg);
                    // }
                    for(;it!=it_end;it++){
                        apply_force(particle_bin[i][j],*it, &dmin, &davg, &navg);
                    }
                }
            }
        }
        

        
        
        if( find_option( argc, argv, "-no" ) == -1 )
        {
            
            MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
            
            
            if (rank == 0){
                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg +=  rdavg/rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
            }
        }
        
        
        //
        //  move particles
        //
        
        for(int i=0;i<n_proc;i++) to_send_after_move[i].clear();
        
        for(int i=0; i < binNum_local; i++){
            for(int j=0; j < particle_bin[i].size(); j++){
                move(particle_bin[i][j]);
                int location = get_bin_index(particle_bin[i][j],rows);
                int dest_proc = location / bin_per_proc;
                if(dest_proc>n_proc-1){
                    dest_proc = n_proc-1;
                }
                to_send_after_move[dest_proc].push_back(particle_bin[i][j]);
            }
            particle_bin[i].clear();
        }
        
        
        for(int i=0; i < n_proc; i++){
            if(i!=rank){
                //send moved particle;
                MPI_Request request;
                int tag = 3;
                MPI_Isend(&(to_send_after_move[i].front()), to_send_after_move[i].size(), PARTICLE,
                          i, tag, MPI_COMM_WORLD, &request);
            }
        }
        
        // Receiving all the info.
        int num_from_each_proc[n_proc];
        for(int i=0;i<n_proc-1;i++){
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
            int sender = status.MPI_SOURCE;
            MPI_Get_count(&status, PARTICLE, &(num_from_each_proc[sender]));
            MPI_Recv(&(recv_from_proc[sender].front()), n, PARTICLE, sender, 3, MPI_COMM_WORLD, &status);
        }
        
        
        
        // Inserting in all the particles
        for(int i=0;i<n_proc;i++){
            if(i!=rank){
                for(int j=0;j<num_from_each_proc[i];j++){
                    int location = get_bin_index(recv_from_proc[i][j],rows);
                    int local_location = location - partition_offsets[rank];
                    particle_bin[local_location].push_back(recv_from_proc[i][j]);
                }
            }
            else{
                for(int j=0;j<to_send_after_move[i].size();j++){
                    int location = get_bin_index(to_send_after_move[i][j],rows);
                    int local_location = location - partition_offsets[rank];
                    particle_bin[local_location].push_back(to_send_after_move[i][j]);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // End of main loop
    }
    simulation_time = read_timer( ) - simulation_time;
    
    if (rank == 0) {
        printf( "n = %d, simulation time = %g seconds", n, simulation_time);
        
        if( find_option( argc, argv, "-no" ) == -1 )
        {
            if (nabsavg) absavg /= nabsavg;
            //
            //  -The minimum distance absmin between 2 particles during the run of the simulation
            //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
            //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
            //
            //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
            //
            printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
            if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
            if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
        printf("\n");     
        
        //  
        // Printing summary data
        //  
        if( fsum)
            fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
    
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    
    free( partition_offsets );
    free( partition_sizes );
    free( particles );
    //cout<<"Fucking finishing in rank "<<rank<<endl;
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    //cout<<"Fucking ennding in rank "<<rank<<endl;
    return 0;
}

