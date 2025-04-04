// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-

#ifndef NULLEVENT_H
#define NULLEVENT_H

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

class NullEvent : public EventSource {

  public:
    NullEvent(EventList &eventList);

    virtual void doNextEvent() override;

    void set_null_over_hook(std::function<void(int)> hook) {
      f_null_over_hook = hook;
    }

    void setCompute(simtime_picosec computation_time);
    void startComputations();
    
    std::function<void(int)> f_null_over_hook;

  private:
};

#endif
