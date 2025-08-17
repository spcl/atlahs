// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-

#ifndef COMPUTEEVENT_H
#define COMPUTEEVENT_H

/*
 * A UEC source and sink
 */
#include "config.h"
#include "eventlist.h"
#include "network.h"
#include "trigger.h"
#include "eqdspacket.h"
#include <functional>
#include <list>
#include <map>

class ComputeEvent : public EventSource {

  public:
    ComputeEvent(EventList &eventList);

    virtual void doNextEvent() override;

    void set_compute_over_hook(std::function<void(int)> hook) {
        f_compute_over_hook = hook;
    }

    void setCompute(simtime_picosec computation_time);
    void startComputations();
    
    std::function<void(int)> f_compute_over_hook;

  private:
};

#endif
