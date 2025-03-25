#ifndef ATLAHS_EVENT_H
#define ATLAHS_EVENT_H

#include <cstdint>

enum class AtlahsEventType {
    SEND_EVENT_OVER,
    RECV_EVENT_OVER,
    COMPUTE_EVENT_OVER,
};

class AtlahsEvent {
public:
    virtual ~AtlahsEvent() = default;
};

class SendEvent : public AtlahsEvent {
public:
    int from;
    int to;
    int size_bytes;
    int tag;
    uint64_t start_time_event;
    Packet *pkt;
    
    SendEvent(int from, int to, int size, int tag, uint64_t start_time_event)
        : from(from), to(to), size_bytes(size), tag(tag), start_time_event(start_time_event)
    { }

    int getFrom() const { return from; }
    int getTo() const { return to; }
    int getSizeBytes() const { return size_bytes; }
    int getTag() const { return tag; }
    uint64_t getStartTimeEvent() const { return start_time_event; }
    Packet* getPacket() const { return pkt; }
    void setPacket(Packet* packet) { pkt = packet; }
};

class RecvEvent : public AtlahsEvent {
    // Implementation details...
};

class ComputeAtlahsEvent : public AtlahsEvent {
public:
    uint64_t start_time_event;

    ComputeAtlahsEvent(uint64_t start_time_event)
        : start_time_event(start_time_event)
    { }
};

class EventOver : public AtlahsEvent {
public:
    int from;
    int to;
    int size_bytes;
    int tag;
    uint64_t start_time_event;
    Packet *pkt;
    AtlahsEventType event_type; // New field

    // Default constructor
    EventOver()
        : from(0), to(0), size_bytes(0), tag(0), start_time_event(0),
          pkt(nullptr), event_type(AtlahsEventType::SEND_EVENT_OVER)
    { }
        
    EventOver(int from, int to, int size, int tag, uint64_t start_time_event, AtlahsEventType event_type)
        : from(from), to(to), size_bytes(size), tag(tag), 
          start_time_event(start_time_event), pkt(nullptr), event_type(event_type)
    { }

    int getFrom() const { return from; }
    int getTo() const { return to; }
    int getSizeBytes() const { return size_bytes; }
    int getTag() const { return tag; }
    uint64_t getStartTimeEvent() const { return start_time_event; }
    Packet* getPacket() const { return pkt; }
    void setPacket(Packet* packet) { pkt = packet; }
    
    AtlahsEventType getEventType() const { return event_type; }
    void setEventType(AtlahsEventType newEventType) { event_type = newEventType; }
};

#endif // ATLAHS_EVENT_H