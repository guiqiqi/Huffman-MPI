#ifndef MPI_RLE_H
#define MPI_RLE_H

#include "common.h"
#include "utils.h"

namespace RLE {
    template<typename Iterator, typename Inserter>
    void encode(Iterator begin, Iterator end, Inserter inserter) {
        uint8_t count = 0;
        auto previous = *begin;
        while (begin != end) {
            auto current = *begin;
            if (current == previous && count < 255) {
                count++;
                begin++;
            } else {
                inserter = count;
                inserter = previous;
                previous = current;
                count = 0;
            }
        }
        inserter = count;
        inserter = previous;
    }

    template<typename Iterator, typename Inserter>
    void decode(Iterator begin, Iterator end, Inserter inserter) {
        if (std::distance(begin, end) % 2 != 0)
            throw std::length_error("invalid encoded size");
        while (begin != end) {
            auto count = (uint8_t) *begin;
            begin++;
            for (uint8_t _ = 0; _ < count; ++_)
                inserter = *(begin);
            begin++;
        }
    }

    template<typename Iterator, typename Inserter>
    void MPI_Decode(Iterator begin, Iterator end, Inserter inserter) {
        // Ensure iterator generates POD type
        using DataType = typename std::iterator_traits<Iterator>::value_type;
        static_assert(std::is_pod<DataType>::value, "T is not a POD type");

        // Ensure container size
        if (std::distance(begin, end) % 2 != 0)
            throw std::length_error("invalid encoded size");

        // Getting world rank and size info
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Each slot process n/size elements
        // Util the last one, which process n/size+n%size elements
        size_t n = std::distance(begin, end) / 2;
        std::vector<DataType> pool;
        int offset = static_cast<int>(n) / world_size;
        auto start = begin + world_rank * offset * 2;
        auto stop = (world_rank == world_size - 1) ? end : start + offset * 2;
        decode(start, stop, std::back_inserter(pool));

        // Send proceed elements in processes except manager one to manager
        if (world_rank == 0) {
            for (int source = 1; source < world_size; ++source) {
                std::vector<DataType> part = MPI_Receive_vector<DataType>(source, 0);
                pool.insert(pool.end(), part.begin(), part.end());
            }
        } else {
            MPI_Send_vector<DataType>(pool, 0, 0);
        }

        // Synchronizing from manager to workers
        if (world_rank == 0) {
            for (int dest = 1; dest < world_size; ++dest) {
                MPI_Send_vector<DataType>(pool, dest, 2);
            }
        } else {
            std::vector<DataType> received = MPI_Receive_vector<DataType>(0, 2);
            pool.clear();
            pool.assign(received.begin(), received.end());
        }

        // Write back to result
        for (const auto &item: pool)
            inserter = item;
    }

    template<typename Iterator, typename Inserter>
    void MPI_Encode(Iterator begin, Iterator end, Inserter inserter) {
        // Ensure iterator generates POD type
        using DataType = typename std::iterator_traits<Iterator>::value_type;
        static_assert(std::is_pod<DataType>::value, "T is not a POD type");

        // Getting world rank and size info
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Each slot process n/size elements
        // Util the last one, which process n/size+n%size elements
        size_t n = std::distance(begin, end);
        std::vector<DataType> pool;
        int offset = static_cast<int>(n) / world_size;
        auto start = begin + world_rank * offset;
        auto stop = (world_rank == world_size - 1) ? end : start + offset;
        encode(start, stop, std::back_inserter(pool));

        // Send proceed elements in processes except manager one to manager
        if (world_rank == 0) {
            for (int source = 1; source < world_size; ++source) {
                std::vector<DataType> part = MPI_Receive_vector<DataType>(source, 0);
                pool.insert(pool.end(), part.begin(), part.end());
            }
        } else {
            MPI_Send_vector<DataType>(pool, 0, 0);
        }

        // Synchronizing from manager to workers
        if (world_rank == 0) {
            for (int dest = 1; dest < world_size; ++dest) {
                MPI_Send_vector<DataType>(pool, dest, 2);
            }
        } else {
            std::vector<DataType> received = MPI_Receive_vector<DataType>(0, 2);
            pool.clear();
            pool.assign(received.begin(), received.end());
        }

        // Write back to result
        for (const auto &item: pool)
            inserter = item;
    }
}

#endif //MPI_RLE_H
