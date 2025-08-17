// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "compute_event.h"
#include "ecn.h"
#include "queue.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <regex>
#include <stdio.h>
#include <utility>

#define timeInf 0

ComputeEvent::ComputeEvent(EventList &eventList)
        : EventSource(eventList, "compute_event") {} 

void ComputeEvent::doNextEvent() {
    /* printf("ComputeEventOver at %lu ps\n", eventlist().now());
    fflush(stdout);  */
    if (f_compute_over_hook) {
        f_compute_over_hook(1);
    }

    return;
}

void ComputeEvent::setCompute(simtime_picosec computation_time) {
/*     printf("ComputeEvent::setCompute at %lu ps at %lu\n", computation_time * 1000, eventlist().now());
 */    eventlist().sourceIsPendingRel(*this, computation_time * 1000); // ns to ps
    // std::cout << "[DEBUG] ComputeEvent::setCompute at " << computation_time * 1000 << std::endl;
    // std::cout << "[DEBUG] evenlist size: " << eventlist().length() << std::endl;
    // eventlist().doNextEvent();
}

void ComputeEvent::startComputations() { 
    printf("ComputeEvent::startComputations at %lu ps\n", eventlist().now());
    eventlist().doNextEvent(); 
}
