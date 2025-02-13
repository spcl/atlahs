#include <algorithm>

class KeyRankPair {
  int key;
  int rank;

public:
  KeyRankPair(int key, int rank) {
    this->key = key;
    this->rank = rank;
  }
  bool operator<(const KeyRankPair &rhs) const {
    return ((key == rhs.key) && (rank < rhs.rank)) || (key < rhs.key);
  }
};

class Comm {

  /* The GOAL base API assumes only a single communicator, aka MPI_COMM_WORLD.
   * This class provides communicator support */

private:
  Comm *base_comm; // pointer to root of the comm tree
  int id; // unique ID of this communicator, id=0 means this is MPI_COMM_WORLD
  int color; // if this comm was created by comm_split, this is his color
  std::vector<KeyRankPair> key2rank; // key, world_rank, pos is new rank
  std::set<Comm *> children;
  int next_free_id; // only used at base comm for now

  Comm *find_comm_rec(int comm_id) {
    if (this->id == comm_id)
      return this;
    for (auto c : this->children) {
      Comm *r = c->find_comm_rec(comm_id);
      if (r != NULL)
        return r;
    }
    return NULL;
  }

public:
  Comm() {
    this->base_comm = this;
    this->id = 0;
    this->next_free_id = 1;
  }

  Comm *find_comm(int comm_id) {
    auto r = this->base_comm->find_comm_rec(comm_id);
    if (r == NULL)
      fprintf(stderr, "Did not find comm %i\n", comm_id);
    return r;
  }

  int getId(void) { return this->id; }

  int nextId() { return this->base_comm->next_free_id++; }

  Comm *find_or_create_child_comm(int color) {
    for (auto c : this->children) {
      if (c->color == color)
        return c;
    }
    Comm *c = new Comm;
    c->base_comm = this->base_comm;
    c->id = this->base_comm->nextId();
    c->color = color;
    return c;
  }

  void add_rank_key(int world_rank, int key) {
    auto p = KeyRankPair(key, world_rank);
    this->key2rank.push_back(p);
    std::sort(
        this->key2rank.begin(),
        this->key2rank
            .end()); // we could add a "close_comm" method and sort only once
  }
};