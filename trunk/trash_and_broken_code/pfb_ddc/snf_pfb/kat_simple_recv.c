/*
 * SPEAD payload capture using Myricom Sniffer API.
 *
 * Portions from Myricom Sniffer example code.
 */

#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <signal.h>
#include <assert.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "snf.h"

#define UDP_HDRLEN 42           // ethernet + ip + udp header size
#define PACKETS_TO_BUFFER 32    // seems to be the sweet spot for software raid
#define PAYLOAD_SIZE 4096       // hardcoded for now

uint64_t num_pkts = 0;
uint64_t missed_heaps  = 0;
uint64_t spead_id_zero = 0;
unsigned int max_received_tv_delta = 0;
uint64_t num_bytes = 0;
uint64_t number_writes = 0;
int write_size = PAYLOAD_SIZE * PACKETS_TO_BUFFER;
snf_ring_t hring;

#define TOGGLE(i) ((i+ 1) & 1)
#define TV_TO_US(tv) ((tv)->tv_sec * 1000000 + (tv)->tv_usec)
unsigned int itvl_idx = 0;
struct itvl_stats {
  struct timeval tv;
  uint64_t usecs;
  uint64_t num_pkts;
  uint64_t num_bytes;
} itvl[2];

float secs_delta;

void
stats()
{
  struct snf_ring_stats stats;
  uint64_t nbytes;
  int rc;
  if ((rc = snf_ring_getstats(hring, &stats))) {
    perror("nic stats failed");
  }

  fprintf(stderr,"\n");
  if (num_pkts == stats.ring_pkt_recv) {
    fprintf(stderr,"Packets received   in HW:        %" PRIu64 "\n", num_pkts);
  } else {
    fprintf(stderr,"Packets received,    app:        %" PRIu64 ", ring: %" PRIu64 "\n",
           num_pkts, stats.ring_pkt_recv);
  }

  fprintf(stderr,"Total bytes received,  app:        %" PRIu64 " (%" PRIu64 " MB)\n",
          num_bytes, num_bytes / 1024 / 1024);
  nbytes = stats.nic_bytes_recv;
  nbytes -= (8 /* HW header */ + 4 /* CRC */) * stats.nic_pkt_recv;
  fprintf(stderr,"Total bytes received + HW aligned: %" PRIu64 " (%" PRIu64 " MB)\n",
          nbytes, nbytes / 1024 / 1024);
  if (num_pkts > 0) {
    fprintf(stderr,"Average Packet Length:    %" PRIu64 " bytes\n",
          num_bytes / num_pkts);
  }

  fprintf(stderr,"Dropped, NIC overflow:    %" PRIu64 "\n", stats.nic_pkt_overflow);
  fprintf(stderr,"Dropped, ring overflow:   %" PRIu64 "\n", stats.ring_pkt_overflow);
  fprintf(stderr,"Dropped, bad:             %" PRIu64 "\n\n", stats.nic_pkt_bad);
  fprintf(stderr,"SPEAD ID of packet 0:     %" PRIu64 "\n", spead_id_zero);
  fprintf(stderr,"Missed SPEAD Packets:     %" PRIu64 "\n", missed_heaps);
  fprintf(stderr,"Number of writes:         %" PRIu64 " (%i bytes per write)\n\n", number_writes, write_size);
}

void
print_periodic_stats(void)
{
  struct itvl_stats *this_itvl = &itvl[itvl_idx];
  struct itvl_stats *last_itvl = &itvl[TOGGLE(itvl_idx)];
  float delta_secs;
  uint64_t delta_pkts;
  uint64_t delta_bytes;
  uint32_t pps;
  float gbps;
  float bps;

  gettimeofday(&this_itvl->tv, NULL);
  this_itvl->usecs = TV_TO_US(&this_itvl->tv);
  this_itvl->num_pkts = num_pkts;
  this_itvl->num_bytes = num_bytes;
  delta_secs = (this_itvl->usecs - last_itvl->usecs) / 1000000.0;
  delta_pkts = this_itvl->num_pkts - last_itvl->num_pkts;
  delta_bytes = this_itvl->num_bytes - last_itvl->num_bytes;

  if (delta_pkts != 0) {
    pps = delta_pkts / delta_secs;
    bps = ((float) delta_bytes * 8) / delta_secs;
    gbps = bps / 1000000000.0;

    fprintf
      (stderr,"%" PRIu64 " pkts (%" PRIu64 "B) in %.3f secs (%u pps), Avg Pkt: %"
       PRIu64 ", BW (Gbps): %6.3f\n", delta_pkts, delta_bytes, delta_secs,
       pps, delta_bytes / delta_pkts, gbps);
    fflush(stderr);
  }

  itvl_idx = TOGGLE(itvl_idx);
}

void
sigexit(int sig)
{
  stats();
  exit(0);
}

void
sigalrm(int sig)
{
  print_periodic_stats();
  alarm(1);
}

void
usage(void)
{
 fprintf(stderr,"Usage: kat_simple-recv [-d <ring sz>] [-t] [-b <board number>] [-f <output_filename>] [-p] [-n <num pkts>]\n");
 fprintf(stderr,"                  -t: print periodic statistics\n");
 fprintf(stderr,"                  -b <board number>: Myri10G board number to use.\n");
 fprintf(stderr,"                  -p: poll for packets instead of blocking\n");
 fprintf(stderr,"                  -n <num pkts>: number of packet to receive (default: 0 - infinite)\n");
 fprintf(stderr,"                  -d <ring sz>: size if the recieve ring in bytes\n");
 fprintf(stderr,"                  -f: redirect ot file instead of standard out\n");
 exit(1);
}

uint64_t
host_nsecs(void)
{
  uint64_t nsecs;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  nsecs = (uint64_t) ts.tv_sec * 1000000000 + ts.tv_nsec;
  return nsecs;
}

int
main(int argc, char **argv)
{
  int rc;
  snf_handle_t hsnf;
  struct snf_recv_req recv_req;
  char c;
  int periodic_stats = 0;
  int decode = 1;
  int boardnum = 0;
  uint64_t pkts_expected = 0xffffffffffffffffULL;
  int open_flags = 0;
  uint64_t dataring_sz = 0;
  int timeout_ms = -1;
// SPEAD counters
  unsigned short spead_header_size = 0;
  uint64_t heap_cnt = 0;
  uint64_t old_heap_cnt = 0;
  unsigned char spead_buffer[PACKETS_TO_BUFFER * PAYLOAD_SIZE];
  int outfh;
  char *output_filename = NULL;

  /* get args */
  while ((c = getopt(argc, argv, "vtb:f:pn:d:S:")) != -1) {
    if (c == 't') {
      periodic_stats = 1; 
    } else if (c == 'b') {
      boardnum = strtoul(optarg, NULL, 0);
    } else if (c == 'p') {
      timeout_ms = 0;
    } else if (c == 'f') {
      output_filename = optarg;
    } else if (c == 'n') {
      pkts_expected = strtoul(optarg, NULL, 0);
    } else if (c == 'd') {
      dataring_sz = strtoull(optarg, NULL, 0);
    } else {
      fprintf(stderr,"Unknown option: %c\n", c);
      usage();
    }
  }

  snf_init(SNF_VERSION_API);
  rc = snf_open(boardnum, 1, NULL, dataring_sz, open_flags, &hsnf);
  if (rc) {
    errno = rc;
    perror("Can't open snf for sniffing");
    return -1;
  }
  rc = snf_ring_open(hsnf, &hring);
  if (rc) {
    errno = rc;
    perror("Can't open a receive ring for sniffing");
    return -1;
  }
  rc = snf_start(hsnf);
  if (rc) {
    errno = rc;
    perror("Can't start packet capture for sniffing");
    return -1;
  }

  if (output_filename != NULL) { 
   fprintf(stderr,"Writing output data to :%s:\n", output_filename);
   outfh = open(output_filename, O_RDWR | O_CREAT, S_IRWXU);
   if (outfh < 0) {
    perror("Failed to open data file");
    exit(1);
   }
  }
  else {
   if (isatty(1)) { 
    fprintf(stderr,"You have requested data be sent to stdout which has not been redirected. This is a bad idea :). Quitting...\n");
    exit(1);
   }
   outfh = 1;
  }

  fprintf(stderr,"snf_recv ready to receive\n");

  if (SIG_ERR == signal(SIGINT, sigexit))
    exit(1);
  if (SIG_ERR == signal(SIGTERM, sigexit))
    exit(1);

  if (periodic_stats) {
    if (SIG_ERR == signal(SIGALRM, sigalrm))
      exit(1);
    itvl[itvl_idx].num_pkts = 0;
    itvl[itvl_idx].num_bytes = 0;
    gettimeofday(&itvl[itvl_idx].tv, NULL);
    itvl[itvl_idx].usecs = 0;
    itvl_idx = TOGGLE(itvl_idx);
    alarm(1);
  }

  while (num_pkts < pkts_expected) {
    rc = snf_ring_recv(hring, timeout_ms, &recv_req);
    if (rc == EAGAIN || rc == EINTR)
      continue;
    else if (rc == 0) {
      if ((unsigned short)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN] == 83) {
       spead_header_size = ((unsigned short)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN + 7] + 1) * 8; 
       heap_cnt = (uint64_t)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+3] * 256 * 256 * 256 * 256 + (uint64_t)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+4] * 256 * 256 * 256 + ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+5] * 256 * 256  + ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+6] * 256 + ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+7];
       if (num_pkts > 0 && heap_cnt > old_heap_cnt + 1 ) {
        missed_heaps += (heap_cnt - old_heap_cnt) - 1;
        fprintf(stderr,"Heap cnt diff:%lu for heap:%lu in packet: %lu (header: %i)\n",heap_cnt - old_heap_cnt, heap_cnt, num_pkts, ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN]);
        //sigexit(0);
       }
       old_heap_cnt = heap_cnt;
       memcpy(spead_buffer + ((num_pkts % PACKETS_TO_BUFFER) * PAYLOAD_SIZE), recv_req.pkt_addr+UDP_HDRLEN+spead_header_size, recv_req.length - UDP_HDRLEN - spead_header_size);

       if (num_pkts % 1000000 == 0) { 
	if (num_pkts == 0) spead_id_zero = heap_cnt;
        fprintf(stderr,"Packet: %" PRIu64 ", Heap count: %" PRIu64 ", Differential (heap_cnt - first_heap_cnt - num_pkts): %li .\n",num_pkts, heap_cnt, (long)(heap_cnt - spead_id_zero) - num_pkts);
       }
       if (num_pkts % PACKETS_TO_BUFFER == 0 && num_pkts > 0) { 
        write(1, (unsigned char *)spead_buffer, PAYLOAD_SIZE * PACKETS_TO_BUFFER); 
        //write(1, (unsigned char*)(recv_req.pkt_addr+UDP_HDRLEN+spead_header_size), recv_req.length - UDP_HDRLEN - spead_header_size);
       }
      num_pkts++;
      num_bytes += recv_req.length;
      } // valid SPEAD packet
    }
    else {
      fprintf(stderr, "error: snf_recv = %d (%s)\n",
                 rc, strerror(rc));
      break;
    }
  }

  rc = snf_ring_close(hring);
  if (rc) {
    errno = rc;
    perror("Can't close receive ring");
    return -1;
  }
  rc = snf_close(hsnf);
  if (rc) {
    errno = rc;
    perror("Can't close sniffer device");
    return -1;
  }
  stats();
  return 0;
}

struct pkt_hdr {
  uint32_t length;
  uint32_t ofst;
};
typedef struct pkt_hdr pkt_hdr_t;
