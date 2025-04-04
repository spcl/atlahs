// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        
#include "fairpullqueue.h"
#include "ndppacket.h"
#include <cstdio>
#include <cassert>
#include <memory>

template<class PullPkt>
BasePullQueue<PullPkt>::BasePullQueue() : _pull_count(0), _preferred_flow(-1) {
}

//template<class PullPkt>
//BasePullQueue<PullPkt>::~BasePullQueue() {
//}


template<class PullPkt>
FifoPullQueue<PullPkt>::FifoPullQueue() {
}

template<class PullPkt>
void
FifoPullQueue<PullPkt>::enqueue(PullPkt& pkt, int /*priority*/) {
    if (this->_preferred_flow >= 0 && pkt.flow_id() == (packetid_t)this->_preferred_flow) {
        //cout << "Got a pkt from the preffered flow " << pull_pkt->flow_id()<<endl;
        typename std::list<PullPkt*>::iterator it = _pull_queue.begin();

        while (it != _pull_queue.end() && ((*it)->flow_id() == (packetid_t)this->_preferred_flow))
            it++;

        _pull_queue.insert(it, &pkt);
    } else {
        printf("Enqueue");
        _pull_queue.push_front(&pkt);
    }
    this->_pull_count++;
    assert((size_t)this->_pull_count == _pull_queue.size());
}

template<class PullPkt>
PullPkt*
FifoPullQueue<PullPkt>::dequeue() {
    if (this->_pull_count == 0) {
        printf("Dequeue empty");
        return 0;
    }
    printf("Dequeue");
    PullPkt* packet = _pull_queue.back();
    _pull_queue.pop_back();
    this->_pull_count--;
    assert(this->_pull_count >= 0 && (size_t)this->_pull_count == _pull_queue.size());
    return packet;
}

template<class PullPkt>
void
FifoPullQueue<PullPkt>::flush_flow(flowid_t flow_id, int /*priority*/) {
    typename std::list<PullPkt*>::iterator it = _pull_queue.begin();
    while (it != _pull_queue.end()) {
        PullPkt* pull = *it;
        if (pull->flow_id() == flow_id && pull->type() == NDPPULL) {
            pull->free();
            it = _pull_queue.erase(it);
            this->_pull_count--;
        } else {
            it++;
        }
    }
    assert((size_t)this->_pull_count == _pull_queue.size());
}

template<class PullPkt>
FairPullQueue<PullPkt>::FairPullQueue() {
    _current_queue = _queue_map.begin();
}


template<class PullPkt>
void
FairPullQueue<PullPkt>::enqueue(PullPkt& pkt, int /*priority*/) {
    CircularBuffer<PullPkt*>* pull_queue;
    if (queue_exists(pkt)) {
            pull_queue = find_queue(pkt);
    }  else {
            pull_queue = create_queue(pkt);
    }
    // we add packets to the front, remove them from the back
    PullPkt* pkt_p = &pkt;
    pull_queue->push(pkt_p);
    this->_pull_count++;
}

template<class PullPkt>
PullPkt* FairPullQueue<PullPkt>::dequeue() {
    if (this->_pull_count == 0)
        return nullptr;

    // Debug: print the flow id of the current queue (if applicable)
    //printf("Entering While loop - for %d\n", _current_queue->first);

    while (true) {
        if (_current_queue == _queue_map.end())
            _current_queue = _queue_map.begin();

        // Use .get() to retrieve the raw pointer from the unique_ptr.
        CircularBuffer<PullPkt*>* pull_queue = _current_queue->second.get();

        // Fix the format string: print the size of the current queue.
        //printf("We are inside this loop - size: %d\n", pull_queue->size());

        // If the queue is empty, remove it from the map and update _current_queue.
        if (pull_queue->empty()) {
            _current_queue = _queue_map.erase(_current_queue);
            continue;
        }

        // Save the current iterator and then advance _current_queue for round-robin.
        auto current_it = _current_queue;
        ++_current_queue;

        // Pop a packet from the non-empty queue.
        PullPkt* packet = pull_queue->pop();
        this->_pull_count--;

        // If the queue becomes empty after the pop, remove it.
        if (pull_queue->empty()) {
            _queue_map.erase(current_it);
        }

        return packet;
    }
}

template<class PullPkt>
void
FairPullQueue<PullPkt>::flush_flow(flowid_t flow_id, int /*priority*/) {
    typename std::map<flowid_t, std::unique_ptr<CircularBuffer<PullPkt*>>>::iterator i;
    i = _queue_map.find(flow_id);
    if (i == _queue_map.end())
        return;
    // Retrieve the raw pointer from the unique_ptr.
    CircularBuffer<PullPkt*>* pull_queue = i->second.get();
    while (!pull_queue->empty()) {
            PullPkt* packet = pull_queue->pop();
            packet->free();
            this->_pull_count--;
    }
    if (_current_queue == i)
        _current_queue++;
    _queue_map.erase(i);
}

template<class PullPkt>
bool 
FairPullQueue<PullPkt>::queue_exists(const PullPkt& pkt) {
    typename std::map<flowid_t, std::unique_ptr<CircularBuffer<PullPkt*>>>::iterator i;
    i = _queue_map.find(pkt.flow_id());
    if (i == _queue_map.end())
        return false;
    return true;
}

template<class PullPkt>
CircularBuffer<PullPkt*>* 
FairPullQueue<PullPkt>::find_queue(const PullPkt& pkt) {
    typename std::map<flowid_t, std::unique_ptr<CircularBuffer<PullPkt*>>>::iterator i;
    i = _queue_map.find(pkt.flow_id());
    if (i == _queue_map.end())
        return 0;
    return i->second.get();
}

template<class PullPkt>
CircularBuffer<PullPkt*>* 
FairPullQueue<PullPkt>::create_queue(const PullPkt& pkt) {
    std::unique_ptr<CircularBuffer<PullPkt*>> new_queue(new CircularBuffer<PullPkt*>);
    CircularBuffer<PullPkt*>* raw_ptr = new_queue.get();
    _queue_map.insert(std::pair<flowid_t, std::unique_ptr<CircularBuffer<PullPkt*>>>(pkt.flow_id(), std::move(new_queue)));
    return raw_ptr;
}

template class BasePullQueue<NdpPull>;
template class FifoPullQueue<NdpPull>;
template class FairPullQueue<NdpPull>;

template class BasePullQueue<NdpRTS>;
template class FifoPullQueue<NdpRTS>;
template class FairPullQueue<NdpRTS>;

template class BasePullQueue<Packet>;
template class FairPullQueue<Packet>;
