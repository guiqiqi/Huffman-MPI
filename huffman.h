#ifndef MPI_HUFFMAN_H
#define MPI_HUFFMAN_H

#include "common.h"
#include "heap.h"
#include "bits.h"
#include "utils.h"

// Required element type T: Default constructor - T t;

namespace Huffman {

    enum NodePathType {
        Left = true,
        Right = false
    };

    template<typename T>
    class Node {
    public:
        using DataType = T;
        using Self = Node<T>;

    public:
        DataType data;
        float probability = 0.;
        Self *left = nullptr;
        Self *right = nullptr;

        // Default constructor for leaf node
        Node(DataType d, float p) : data(d), probability(p) {}

        // Merge two nodes as a new one with sum probabilities,
        // make the less probability one as left child node
        Node(Self *a, Self *b) {
            this->probability = a->probability + b->probability;
            if (a->probability <= b->probability) {
                this->left = a;
                this->right = b;
            } else {
                this->left = b;
                this->right = a;
            }
        }

        [[nodiscard]] bool leaf() const {
            if (this->left == nullptr && this->right == nullptr)
                return true;
            return false;
        }

        // Comparator
        bool operator<(const Self &other) const {
            return this->probability < other.probability;
        }

        bool operator>(const Self &other) const {
            return this->probability > other.probability;
        }

        bool operator<=(const Self &other) const {
            return this->probability <= other.probability;
        }
    };

    // Make a Huffman tree with analysed alphabet frequency mapping
    template<typename T>
    class Tree {
    public:
        using NodeT = Node<T>;
        NodeT *root = nullptr;
        std::vector<NodeT *> nodes;

    public:
        explicit Tree(const std::map<T, float> &dict) {
            Heap::MinHeap<NodeT> heap;
            for (auto &kv: dict)
                heap.append(NodeT(kv.first, kv.second));

            // Merge until heap size equals to 1
            while (heap.size() != 1) {
                auto first = new NodeT(heap.pop());
                auto second = new NodeT(heap.pop());
                this->nodes.push_back(first);
                this->nodes.push_back(second);
                heap.append(NodeT(first, second));
            }

            // Set root node
            this->root = new NodeT(heap.pop());
            this->nodes.push_back(this->root);
        }

        ~Tree() {
            for (auto &node: this->nodes)
                delete node;
        }

        [[nodiscard]] std::map<T, std::vector<bool>> traverse() const {
            if (this->root == nullptr)
                throw std::invalid_argument("tree is not ready");

            std::map<T, std::vector<bool>> result;

            // Set unvisited nodes vector as pair made by node and its bit
            std::vector<std::pair<NodeT *, std::vector<bool>>> unvisited;
            unvisited.push_back(std::make_pair(this->root, std::vector<bool>()));

            // DFS traverse and make result
            while (!unvisited.empty()) {
                auto pair = unvisited.back();
                unvisited.pop_back();
                auto node = pair.first;
                auto bits = pair.second;
                if (node->leaf()) {
                    result[node->data] = bits;
                    continue;
                }
                if (node->left) {
                    std::vector<bool> copied;
                    copied.assign(bits.begin(), bits.end());
                    copied.push_back(Left);
                    unvisited.push_back(std::make_pair(node->left, copied));
                }
                if (node->right) {
                    std::vector<bool> copied;
                    copied.assign(bits.begin(), bits.end());
                    copied.push_back(Right);
                    unvisited.push_back(std::make_pair(node->right, copied));
                }
            }

            return result;
        }
    };

    // Split this counting function out for make MPI concurrency easier
    template<class Iterator, typename T = typename std::iterator_traits<Iterator>::value_type>
    std::map<T, size_t> statistic(Iterator begin, Iterator end) {
        std::map<T, size_t> stats;
        while (begin != end) {
            const T &item = *begin;
            if (stats.find(item) == stats.end())
                stats[item] = 1;
            else
                stats[item] += 1;
            begin++;
        }
        return stats;
    }

    template<class Iterator, typename T = typename std::iterator_traits<Iterator>::value_type>
    std::map<T, size_t> MPI_Statistic(Iterator begin, Iterator end) {
        // Get world info
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Decide start end iterator
        size_t n = std::distance(begin, end);
        int offset = static_cast<int>(n) / world_size;
        auto start = begin + world_rank * offset;
        auto stop = (world_rank == world_size - 1) ? end : start + offset;
        std::map<T, size_t> part = statistic(start, stop);

        // Send part to manager process
        if (world_rank == 0) {
            for (int source = 1; source < world_size; ++source) {
                std::map<T, size_t> received = MPI_Receive_map<T, size_t>(source, 0);
                for (const auto &[key, value]: received) {
                    if (part.find(key) == part.end())
                        part[key] = value;
                    else
                        part[key] += value;
                }
            }
        } else {
            MPI_Send_map(part, 0, 0);
        }

        // Now stats dict stored on manager (which world_rank equals to 0) variable part
        // Synchronizing stats dict
        std::map<T, size_t> stats;
        if (world_rank == 0) {
            for (int dest = 1; dest < world_size; ++dest) {
                MPI_Send_map(part, dest, 3);
            }
            stats = part;
        } else {
            stats = MPI_Receive_map<T, size_t>(0, 3);
        }
        return stats;
    }

    template<typename T>
    class Encoder {
    private:
        std::map<T, float> frequency;
        std::vector<T> data;
        Tree<T> *tree;

    public:
        template<class Iterator>
        Encoder(Iterator begin, Iterator end, bool mpi = false) {
            // Check iterator value type during compiling
            static_assert(
                    std::is_same<typename std::iterator_traits<Iterator>::value_type, T>::value,
                    "iterator value type should as same as data type");

            this->data.assign(begin, end);
            std::map<T, size_t> stats;
            if (!mpi)
                stats = statistic(begin, end);
            else
                stats = MPI_Statistic(begin, end);
            auto total = static_cast<float>(this->data.size());
            for (const auto &pair: stats)
                this->frequency[pair.first] = static_cast<float>(pair.second) / total;
            this->tree = new Tree<T>(this->frequency);
        }

        // Encode dict into std::vector<bool> so it could be appended into head of file
        // The format of encoded dict is:
        //   count: size_t, first_element: T, first_frequency: float, ..., nth_element: T, nth_frequency: float
        [[nodiscard]] std::vector<bool> dict() const {
            std::vector<bool> encoded;
            auto count = static_cast<size_t>(this->frequency.size());
            for (const auto &bit: Bits::serialize<size_t>(count))
                encoded.push_back(bit);
            for (const auto &pair: this->frequency) {
                for (const auto &bit: Bits::serialize<T>(pair.first))
                    encoded.push_back(bit);
                for (const auto &bit: Bits::serialize<float>(pair.second))
                    encoded.push_back(bit);
            }
            return encoded;
        }

        // Encode data using iterator for selecting part of data from container
        // When encoding using MPI, it requests different part of container
        template<class Iterator>
        std::vector<bool> encode(Iterator begin, Iterator end) const {
            auto dict = this->tree->traverse();
            std::vector<bool> encoded;
            while (begin != end) {
                for (const auto &bit: dict[*begin])
                    encoded.push_back(bit);
                begin++;
            }
            return encoded;
        }

        // Encode tree building data by default
        [[nodiscard]] std::vector<bool> encode() const {
            return this->encode(this->data.begin(), this->data.end());
        }

        ~Encoder() {
            if (this->tree)
                delete this->tree;
        }

        // Calculating encoding price:
        //   sum([encoding_length] * [frequency])
        [[nodiscard]] float price() const {
            float result = 0.;
            for (const auto &pair: this->tree->traverse())
                result += this->frequency.at(pair.first) * pair.second.size();
            return result;
        }

        // Rewrite encode function for using all nodes parallel
        template<class Iterator>
        std::vector<bool> MPI_Encode(Iterator begin, Iterator end) const {
            // Get world info
            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            // Decide start end iterator
            size_t n = std::distance(begin, end);
            int offset = static_cast<int>(n) / world_size;
            auto start = begin + world_rank * offset;
            auto stop = (world_rank == world_size - 1) ? end : start + offset;
            auto part = this->encode(start, stop);

            // Send parts to manager process
            std::vector<bool> encoded;
            if (world_rank == 0) {
                encoded.assign(part.begin(), part.end());
                for (int source = 1; source < world_size; ++source) {
                    std::vector<bool> received = MPI_Receive_vector<bool>(source, 0);
                    encoded.insert(encoded.end(), received.begin(), received.end());
                }
            } else {
                MPI_Send_vector(part, 0, 0);
            }

            // Synchronizing from manager node to worker nodes
            if (world_rank == 0) {
                for (int dest = 1; dest < world_size; ++dest)
                    MPI_Send_vector(encoded, dest, 2);
            } else {
                encoded = MPI_Receive_vector<bool>(0, 2);
            }
            return encoded;
        }
    };

    template<typename T>
    class Decoder {
    private:
        Tree<T> *tree;
        std::map<T, float> frequency;
        std::vector<bool> data;

    public:
        explicit Decoder(const std::vector<bool> &bits) {
            std::vector<bool> part;
            auto iterator = bits.begin();

            // Get total element type count
            part.assign(iterator, iterator + sizeof(size_t) * 8);
            iterator += sizeof(size_t) * 8;
            auto count = Bits::deserialize<size_t>(part);
            part.clear();

            // Recover saved frequency mapping
            for (size_t index = 0; index < count; ++index) {
                part.assign(iterator, iterator + sizeof(T) * 8);
                iterator += sizeof(T) * 8;
                auto key = Bits::deserialize<T>(part);
                part.clear();
                part.assign(iterator, iterator + sizeof(float) * 8);
                iterator += sizeof(float) * 8;
                auto value = Bits::deserialize<float>(part);
                part.clear();
                this->frequency[key] = value;
            }

            // Rebuild Huffman tree
            this->tree = new Tree<T>(this->frequency);

            // Save encoded data
            this->data.assign(iterator, bits.end());
        }

        // Decode data using Huffman tree
        template<class Inserter>
        void decode(const std::vector<bool> &bits, Inserter inserter) const {
            Node<T> *current = this->tree->root;
            for (const auto &bit: bits) {
                if (bit == Left)
                    current = current->left;
                if (bit == Right)
                    current = current->right;
                if (current->leaf()) {
                    inserter = current->data;
                    current = this->tree->root;
                }
            }
        }

        // Decode given data by default
        template<class Inserter>
        void decode(Inserter inserter) const {
            return this->decode(this->data, inserter);
        }

        ~Decoder() {
            if (this->tree)
                delete this->tree;
        }

    };
}

#endif //MPI_HUFFMAN_H
