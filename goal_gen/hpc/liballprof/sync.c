#include "sync.h"
#include <stdio.h>

#define MAX_DOUBLE 1e100
#define NUMBER_SMALLER 100
static double *diffs=NULL; /* global array of all diffs to all ranks - only
                  completely valid on rank 0 */
static double gdiff;

double sync_peer(int client, int peer, MPI_Comm comm) {
    const double ABORT_VAL = 9999999.0;
    int notsmaller = 0; /* count number of RTTs that are *not* smaller than
                    the current smallest one */
    int server=0;
    double tstart, /* local start time */
           tend, /* local end time */
           trem, /* remote time */
           tmpdiff, /* temporary difference to remote clock */
           diff; /* difference to remote clock */
    int res, r;
  res = PMPI_Comm_rank(comm, &r);

    if(!client) server = 1;

    double smallest = MAX_DOUBLE; /* the current smallest time */
    do {
      /* the client sends a ping to the server and waits for a pong (and
       * takes the RTT time). It repeats this procedure until the last
       * NUMBER_SMALLER RTTs have not been smaller than the smallest
       * (tries to find the smallest RTT). When the smallest RTT is
       * found, it sends a special flag (0d) to the server that it knows
       * that the benchmark is finished. The client computes the diff
       * with this smallest RTT with the scheme described in the paper.
       * */
      if(client) {
        tstart = PMPI_Wtime();
        res = PMPI_Send(&tstart, 1, MPI_DOUBLE, peer, 0, comm);
        res = PMPI_Recv(&trem, 1, MPI_DOUBLE, peer, 0, comm, MPI_STATUS_IGNORE);
        tend = PMPI_Wtime();
        tmpdiff = tstart + (tend-tstart)/2 - trem;
        
        if(tend-tstart < smallest) {
          smallest = tend-tstart;
          notsmaller = 0;
          diff = tmpdiff; /* save new smallest diff-time */
        } else {
          if(++notsmaller == NUMBER_SMALLER) {
            /* send abort flag to client */
            trem = ABORT_VAL;
            res = PMPI_Send(&trem, 1, MPI_DOUBLE, peer, 0, comm);
            /*printf("[%i] diff to %i: %lf\n", r, peer, diff*1e6);*/
            break;
          }
        }
        /*printf("[%i] notsmaller: %i\n", r, notsmaller);*/
      }

      /* The server just replies with the local time to the client
       * requests and aborts the benchmark if the abort flag (0d) is
       * received in any of the requests. */
      if(server) {
        /* printf("[%i] server: waiting for ping from %i\n", r, peer); */
        res = PMPI_Recv(&tstart, 1, MPI_DOUBLE, peer, 0, comm, MPI_STATUS_IGNORE);
        if(tstart == ABORT_VAL) {break;} /* this is the signal from the client to stop */
        trem = PMPI_Wtime(); /* fill in local time on server */
        /* printf("[%i] server: got ping from %i (%lf) \n", r, peer, tstart); */
        res = PMPI_Send(&trem, 1, MPI_DOUBLE, peer, 0, comm);
      }
      /* this loop is only left with a break */
    } while(1);
    return diff;
}


/* tree-based synchronization mechanism 
 * - */
double sync_tree(MPI_Comm comm) {
  int p, r, res, dist, round;
  int power; /* biggest power of two value that is smaller or equal to p */
  int peer; /* synchronization peer */
  double diff;

  res = PMPI_Comm_rank(comm, &r);
  res = PMPI_Comm_size(comm, &p);
  
  /* reallocate tha diffs array with the right size */
  if(diffs != NULL) free(diffs);
  diffs = (double*)calloc(1, p*sizeof(double));
  
  /* check if p is power of 2 
  { int i=1;
    while((i = i << 1) < p) {};
    if(i != p) { 
      printf("communicator size (%i) must be power of 2 (%i)!\n", p, i);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }*/

  { /* get the maximum power of 2 that is smaller than p */
    int num=1;
    do {
      num *= 2;
    } while(num*2 <= p);
    power = num;
  }

  /* if I am in the powers-of two group? */ 
  if(r < power) { 
    dist = 1; /* this gets left-shifted (<<) every round and is after 
                     $\lceil log_2(p) \rceil$ rounds >= p */
    round = 1; /* fun and printf round counter - not really needed */
    do {
      int client, server;

      client = 0; server = 0;
      client = ((r % (dist << 1)) == 0);
      server = ((r % (dist << 1)) == dist);
      
      if(server) {
        peer = r - dist;
        if(peer < 0) server = 0; /* disable yourself if there is no peer*/
        /*if(server) printf("(%i) %i <- %i\n", round, r, peer);*/
      }
      if(client) {
        peer = r + dist;
        if(peer >= p) client = 0; /* disable yourself if there is no peer*/
        /*if(client) printf("(%i) %i -> %i\n", round, peer, r);*/
      }
      if(!client && !server) break; /* TODO: leave loop if no peer left -
                                       works only for power of two process
                                       groups */
      
	diff = sync_peer(client, peer, comm);
      
      /* diff is the time difference between client and server. This is
       * only valid on the client, and is derived with the following
       * formula: diff = tstart + (tend-tstart)/2 - trem;
       * example:
       *     Client                       Server
       *     tstart = 100                 200 (those are local times, but at the same moment)
       *     send message (L=10)
       *     110                          trem = 210
       *                                  send message back (L=10)
       *     tend = 120                   220
       *
       *     diff = 100 + (120-100)/2 - 210
       *          = 100 + 10 - 210 = 100
       *
       *  now, to get the local time on a server on a client:
       *        t_s = r_c - diff
       */
      
      /* the client measured the time difference to his peer-server of the
       * current round. Since rank 0 is the global synchronization point,
       * rank 0's array has to be up to date and the other clients have to
       * communicate all their knowledge to rank 0 as described in the
       * paper. */
      
      if(client) {
        /* all clients just measured the time difference to node r + diff
         * (=peer) */
        diffs[peer] = diff;

        /* we are a client - we need to receive all the knowledge
         * (differences) that the server we just synchronized with holds!
         * Our server has been "round-1" times client and measures
         * "round-1" diffs */
        if(round > 1) {
          double *recvbuf; /* receive the server's data */
          int items, i;

          items = (1 << (round-1))-1;
          recvbuf = (double*)malloc(items*sizeof(double));
          
          res = PMPI_Recv(recvbuf, items, MPI_DOUBLE, peer, 0, comm, MPI_STATUS_IGNORE);
          
          /*printf("[%i] round: %i, client merges %i items\n", r, round, items);*/
          /* merge data into my own field */
          for(i=0; i<items; i++) {
            diffs[peer+i+1] =  diffs[peer] /* diff to server */ + 
                            recvbuf[i] /* received time */; 
          }
          free(recvbuf);
        }
      }

      if(server) {
        /* we are a server, we need to send all our knowledge (time
         * differences to our client */
      
        /* we have measured "round-1" nodes at the end of round "round"
         * and hold $2^(round-1)-1$ diffs at this time*/
        if(round > 1) {
          int i, tmpdist, tmppeer, items;
          double *sendbuf;
          
          items = (1 << (round-1))-1;
          sendbuf = (double*)malloc(items*sizeof(double));

          /*printf("[%i] round: %i, server sends %i items\n", r, round, items);*/

          /* fill buffer - every server holds the $2^(round-1)-1$ next
           * diffs */
          for(i=0; i<items; i++) {
            sendbuf[i] = diffs[r+i+1];
          }
          res = PMPI_Send(sendbuf, items, MPI_DOUBLE, peer, 0, comm);
          free(sendbuf);
        }
      }
      
      dist = dist << 1;
      round++;
    } while(dist < power);
  }
  /* all first power-of two nodes have their time difference now and the
   * others have to synchronize with the first powers of two nodes ... 
   * example p=6 -> power=4
   * rank 0..3 are synched at this stage and rank 4 and 5 have to sync
   * with 0 and 1 respectively */
  if(r < power) {
    /* check if I have a partner in the non power group */
    if(p - power > r) { /* I have a partner */
      peer = power + r; /* that's my partner */
      /*printf("[%i] server for %i\n", r, peer);*/
      sync_peer(0, peer, comm); /* I am the server */
    }
  } else {
    peer = r - power; /* that's my partner */
    /*printf("[%i] client for %i\n", r, peer);*/
    diff = sync_peer(1, peer, comm); /* I am the client */
    res = PMPI_Send(&diff, 1, MPI_DOUBLE, 0, 1, comm);
  }

  if(0 == r) {
    int syncpeer;
    MPI_Request *reqs;
    double *tmpdiffs;
    
    reqs = (MPI_Request*)malloc((p-power)*sizeof(MPI_Request));
    tmpdiffs = (double*)malloc((p-power)*sizeof(double));
   
    /* pre-post all recv-request to speed it up a bit */
    for(peer = power; peer < p; peer++) {
      res = PMPI_Irecv(&tmpdiffs[peer-power], 1, MPI_DOUBLE, peer, 1, comm, &reqs[peer-power]);
    }
    
    PMPI_Waitall(p-power,reqs,MPI_STATUSES_IGNORE);
    
    for(peer = power; peer < p; peer++) {
      syncpeer = peer-power; /* the rank that 'peer' synchronized with */
      diffs[peer] = diffs[peer-power] - tmpdiffs[peer-power];
    }

    free(reqs);
    free(tmpdiffs);
  }

  /* scatter all the time diffs to the processes */
  PMPI_Scatter(diffs, 1, MPI_DOUBLE, &gdiff, 1, MPI_DOUBLE, 0, comm);
  /*printf("[%i] diff_tree: %lf usec\n", r, gdiff*1e6);*/
  return gdiff*1e6;
}

/* linear synchronization mechanism 
 * - */
double sync_lin(MPI_Comm comm) {
  int p, r, res, peer=0;

  res = PMPI_Comm_rank(comm, &r);
  res = PMPI_Comm_size(comm, &p);
  
  /* reallocate tha diffs array with the right size */
  if(diffs != NULL) free(diffs);
  diffs = (double*)calloc(1, p*sizeof(double));
  
  if(r == 0) {
    for(peer = 1; peer < p; peer++) {
      diffs[peer] = sync_peer(0, peer, comm);
    }
  } else {
    sync_peer(1, peer, comm);
  }

  /* scatter all the time diffs to the processes */
  PMPI_Scatter(diffs, 1, MPI_DOUBLE, &gdiff, 1, MPI_DOUBLE, 0, comm);
  /*printf("[%i] diff_lin: %lf usec\n", r, gdiff*1e6);*/
  return gdiff*1e6;
}


