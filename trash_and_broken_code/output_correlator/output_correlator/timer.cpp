/*
 * timer.cpp
 *
 *  Created on: 13 May 2013
 *      Author: benjamin
 */

#include "timer.h"
timespec timer::start;
timespec timer::stop;
void timer::tic(){
	clock_gettime( CLOCK_REALTIME, &start);
}
double timer::toc(){
	clock_gettime( CLOCK_REALTIME, &stop);
	return ( stop.tv_sec - start.tv_sec )
            + (double)( stop.tv_nsec - start.tv_nsec )
              / (double)(1000000000L);
}
