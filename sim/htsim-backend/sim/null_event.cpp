// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "null_event.h"
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

NullEvent::NullEvent(EventList &eventList)
        : EventSource(eventList, "compute_event") {} 

void NullEvent::doNextEvent() {
    /* printf("NullEventOver at %lu ps\n", eventlist().now());
    fflush(stdout);  */
    if (f_null_over_hook) {
        f_null_over_hook(1);
    }

    return;
}

void NullEvent::setCompute(simtime_picosec computation_time) {
    eventlist().sourceIsPendingRel(*this, computation_time * 1000 - eventlist().now()); // ns to ps
}

void NullEvent::startComputations() { eventlist().doNextEvent(); }
