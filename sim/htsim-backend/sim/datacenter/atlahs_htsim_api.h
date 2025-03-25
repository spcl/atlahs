#ifndef ATLAHS_HTSIM_API_H
#define ATLAHS_HTSIM_API_H

#include "atlahs_api.h"
#include <iostream>
#include <functional>
#include "../compute_event.h"
#include "atlahs_event.h"


// Forward declarations
class EventList;
class UecRtxTimerScanner;
class FatTreeTopology;
class LogSimInterface;
class EventOver;
//class ComputeEvent;

class AtlahsHtsimApi : public AtlahsApi {
public:
    AtlahsHtsimApi() = default;
    virtual ~AtlahsHtsimApi() = default;
    
    virtual void Send(const SendEvent &event) override;
    virtual void Recv(const RecvEvent &event) override;
    virtual void Calc(const ComputeAtlahsEvent &event) override;
    virtual void Setup() override;
    virtual void EventFinished(const EventOver &event) override;

    // Getter and setter for EventList
    void setEventList(EventList* eventlist) { _eventlist = eventlist; }
    EventList* getEventList() const { return _eventlist; }
    
    // Getter and setter for UecRtxTimerScanner
    void setUecRtxScanner(UecRtxTimerScanner* scanner) { _uecRtxScanner = scanner; }
    UecRtxTimerScanner* getUecRtxScanner() const { return _uecRtxScanner; }
    
    // Getter and setter for FatTreeTopology
    void setTopology(FatTreeTopology* topo) { _topo = topo; }
    FatTreeTopology* getTopology() const { return _topo; }

    // Getter and setter for LogSimInterface
    void setLogSimInterface(LogSimInterface* logsim_interface) { _logsim_interface = logsim_interface; }
    LogSimInterface* getLogSimInterface() const { return _logsim_interface; }

    // Getter and setter for ComputeEvent
    void setComputeEvent(ComputeEvent* compute_event) { 
        compute_events_handler = compute_event; 
        compute_events_handler->set_compute_over_hook(
            std::bind(&AtlahsHtsimApi::compute_over_intermediate, this, std::placeholders::_1));
    }
    ComputeEvent* getComputeEvent() const { return compute_events_handler; }


    void compute_over_intermediate(int i) {
        EventOver event;
        event.event_type = AtlahsEventType::COMPUTE_EVENT_OVER;
        this->EventFinished(event);
        return;
    }

    simtime_picosec getGlobalTimePs() const { return _eventlist->now(); }
    simtime_picosec getGlobalTimeNs() const { return _eventlist->now() * 1000; }
    
private:
    EventList* _eventlist = nullptr;
    UecRtxTimerScanner* _uecRtxScanner = nullptr;
    FatTreeTopology* _topo = nullptr;
    LogSimInterface* _logsim_interface = nullptr;
    ComputeEvent *compute_events_handler = nullptr;
};

#endif // ATLAHS_HTSIM_API_H