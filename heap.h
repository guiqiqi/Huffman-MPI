#ifndef MPI_HEAP_H
#define MPI_HEAP_H

#include "common.h"

namespace Heap {
    template<typename T>
    class MinHeap {
    private:
        std::vector<T> pool;

    public:
        T &at(size_t index) {
            if (index < this->pool.size())
                return this->pool.at(index);
            throw std::out_of_range("invalid index");
        }

        [[nodiscard]] size_t size() const {
            return this->pool.size();
        }

    private:
        void float_(size_t index) {
            if (index == 0)
                return;
            size_t current = index;
            size_t father = (index - 1) / 2;
            if (this->at(current) < this->at(father)) {
                std::swap(this->at(current), this->at(father));
                this->float_(father);
            }
        }

        void sink(size_t index) {
            size_t current = index;
            size_t left = index * 2 + 1;
            size_t right = left + 1;

            // Check if only one child or no child of current node has
            if (right >= this->pool.size())
                right = left;
            if (left >= this->pool.size())
                return;

            size_t minimal = this->at(left) <= this->at(right) ? left : right;
            if (this->at(minimal) < this->at(current)) {
                std::swap(this->at(minimal), this->at(current));
                this->sink(minimal);
            }
        }

    public:
        void append(const T &element) {
            this->pool.push_back(element);
            this->float_(this->pool.size() - 1);
        }

        T pop() {
            if (this->pool.size() == 0)
                throw std::out_of_range("invalid index");
            std::swap(this->at(0), this->at(this->pool.size() - 1));
            const T element = this->pool.back();
            this->pool.pop_back();
            this->sink(0);
            return element;
        }
    };
}

#endif //MPI_HEAP_H
