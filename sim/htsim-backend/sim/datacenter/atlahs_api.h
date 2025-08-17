#ifndef ATLAHS_API_H
#define ATLAHS_API_H

class AtlahsEvent;
class SendEvent;
class RecvEvent;
class ComputeAtlahsEvent;
class EventOver;
class graph_node_properties;

class AtlahsApi {
public:
    virtual ~AtlahsApi() = default;

    virtual void Send(const SendEvent &event, graph_node_properties node) = 0;
    virtual void Recv(const RecvEvent &event) = 0;
    virtual void Calc(const ComputeAtlahsEvent &event) = 0;
    virtual void Setup() = 0;
    virtual void EventFinished(const EventOver &event) = 0;
};

#endif // ATLAHS_API_H