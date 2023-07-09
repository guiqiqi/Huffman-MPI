#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "common.h"

template<typename T, class Inserter>
void choices(size_t n, const std::vector<T> &form, Inserter inserter) {
    for (size_t index = 0; index < n; ++index)
        std::sample(form.begin(), form.end(), inserter, 1,
                    std::mt19937{std::random_device{}()});
}

template<typename KT, typename VT>
void MPI_Send_map(const std::map<KT, VT> &map, int destination, int message_no) {
    // Check if T is a POD type
    static_assert(std::is_pod<KT>::value, "KT is not a POD type");
    static_assert(std::is_pod<VT>::value, "VT is not a POD type");

    size_t size = map.size();
    std::vector<KT> keys;
    std::vector<VT> values;
    for (const auto &[key, value]: map) {
        keys.push_back(key);
        values.push_back(value);
    }
    MPI_Send(&size, 1, MPI_UNSIGNED_LONG, destination, message_no, MPI_COMM_WORLD);
    MPI_Send(keys.data(), size * sizeof(KT), MPI_BYTE, destination, message_no + 1, MPI_COMM_WORLD);
    MPI_Send(values.data(), size * sizeof(VT), MPI_BYTE, destination, message_no + 2, MPI_COMM_WORLD);
}

template<typename KT, typename VT>
std::map<KT, VT> MPI_Receive_map(int source, int message_no) {
    // Check if T is a POD type
    static_assert(std::is_pod<KT>::value, "KT is not a POD type");
    static_assert(std::is_pod<VT>::value, "VT is not a POD type");

    // Receive keys and values
    size_t size;
    MPI_Recv(&size, 1, MPI_UNSIGNED_LONG, source, message_no, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<KT> keys(size);
    std::vector<VT> values(size);
    MPI_Recv(keys.data(), size * sizeof(KT), MPI_BYTE, source, message_no + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(values.data(), size * sizeof(VT), MPI_BYTE, source, message_no + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Convert to map
    std::map<KT, VT> map;
    for (size_t index = 0; index < size; ++index)
        map[keys[index]] = values[index];
    return map;
}

template<typename T>
void MPI_Send_vector(const std::vector<T> &items, int destination, int message_no) {
    // Check if T is a POD type
    static_assert(std::is_pod<T>::value, "T is not a POD type");

    size_t size = items.size();
    MPI_Send(&size, 1, MPI_UNSIGNED_LONG, destination, message_no, MPI_COMM_WORLD);
    MPI_Send(items.data(), size * sizeof(T), MPI_BYTE, destination, message_no + 1, MPI_COMM_WORLD);
}

template<typename T>
std::vector<T> MPI_Receive_vector(int source, int message_no) {
    // Check if T is a POD type
    static_assert(std::is_pod<T>::value, "T is not a POD type");

    size_t size;
    MPI_Recv(&size, 1, MPI_UNSIGNED_LONG, source, message_no, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<T> receiver(size);
    MPI_Recv(receiver.data(), size * sizeof(T), MPI_BYTE, source, message_no + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return receiver;
}

template<>
void MPI_Send_vector(const std::vector<bool> &items, int destination, int message_no) {
    std::vector<char> converted(items.begin(), items.end());
    return MPI_Send_vector<char>(converted, destination, message_no);
}

template<>
std::vector<bool> MPI_Receive_vector(int source, int message_no) {
    std::vector<char> received = MPI_Receive_vector<char>(source, message_no);
    std::vector<bool> converted(received.begin(), received.end());
    return converted;
}

template<typename T, class Inserter>
void MPI_Choices(size_t n, const std::vector<T> &form, Inserter inserter) {
    // Check if T is a POD type
    static_assert(std::is_pod<T>::value, "T is not a POD type");

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // For manager, generate n/size + n%size times and receive from other processes
    // for other processes, generate n/size times and send it
    std::vector<T> pool;
    if (world_rank == 0) {
        choices(n / world_size + n % world_size, form, std::back_inserter(pool));
        for (int source = 1; source < world_size; ++source) {
            std::vector<T> received = MPI_Receive_vector<T>(source, 0);
            pool.insert(pool.end(), received.begin(), received.end());
        }
    } else {
        choices(n / world_size, form, std::back_inserter(pool));
        MPI_Send_vector<T>(pool, 0, 0);
    }

    // Synchronizing from manager node to worker nodes
    if (world_rank == 0) {
        for (int dest = 1; dest < world_size; ++dest)
            MPI_Send(pool.data(), n * sizeof(T), MPI_BYTE, dest, 2, MPI_COMM_WORLD);
    } else {
        pool.clear();
        pool.resize(n);
        MPI_Recv(pool.data(), n * sizeof(T), MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Write back to result
    for (const auto &item: pool)
        inserter = item;
}

#endif //MPI_UTILS_H
