#ifndef MPI_BITS_H
#define MPI_BITS_H

#include "common.h"

namespace Bits {
    // Convert given object with type T to bits vector
    // For example:
    //   assert (serialize<char>('A') == std::vector({0, 1, 0, 0, 0, 0, 0, 1}));
    // Note: endian not concerned here, data must be restored using same endian
    // Also, if some data stored in heap but not in stack, it will just copy a pointer to nowhere
    template<typename T>
    std::vector<bool> serialize(T &&object) {
        std::vector<bool> result;
        char buffer[sizeof(T)];
        memcpy(buffer, &object, sizeof(T));
        auto index = 8;
        for (size_t offset = 0; offset < sizeof(T); ++offset) {
            while (index-- > 0)
                result.push_back(buffer[offset] & (1 << index));
            index = 8;
        }
        return result;
    }

    template<typename T>
    std::vector<bool> serialize(const T &object) {
        return serialize(std::move(object));
    }

    // Convert given bits vector to given type of object
    // For example;
    //   assert(deserialize<char>(std::vector({0, 1, 0, 0, 0, 0, 0, 1})) == 'A');
    template<typename T>
    T deserialize(const std::vector<bool> &bits) {
        assert(bits.size() / 8 == sizeof(T) && bits.size() % 8 == 0);
        char buffer[sizeof(T)];
        memset(buffer, 0, sizeof(T));
        size_t index = 0;
        for (const auto &bit: bits) {
            if (bit)
                buffer[index / 8] |= 1 << (7 - (index % 8));
            index++;
        }
        T deserialized;
        memcpy(&deserialized, buffer, sizeof(T));
        return deserialized;
    }

    class BitArray {
    private:
        size_t length;
        std::vector<uint8_t> data;

    public:
        BitArray() : length(0) {}

        explicit BitArray(const std::vector<bool> &sequence) {
            this->length = sequence.size();

            uint8_t current = 0;
            size_t count = 0;
            for (auto const &value: sequence) {
                count++;
                if (value)
                    current |= 1 << (8 - count);
                if (count == 8) {
                    data.push_back(current);
                    current = 0;
                    count = 0;
                }
            }

            // I won't forget the last part of bits
            if (count != 0)
                data.push_back(current);
        }

        // Convert bitarray to binary string
        [[nodiscard]] std::string stringify() const {
            auto buffer = new char[this->length + 1];
            for (size_t index = 0; index < this->length; ++index)
                buffer[index] = (this->at(index)) ? '1' : '0';
            buffer[this->length] = 0;
            std::string result(buffer);
            delete[] buffer;
            return result;
        }

        // Convert bitarray to raw std::vector<bool>
        [[nodiscard]] std::vector<bool> vectorize() const {
            std::vector<bool> result;
            for (size_t index = 0; index < this->length; ++index)
                result.emplace_back(this->at(index));
            return result;
        }

        // Check given index status in bits
        [[nodiscard]] bool at(size_t index) const {
            if (index >= this->length)
                throw std::range_error("invalid index");
            return this->data[index / 8] & 1 << (7 - index % 8);
        }

        // Write to output stream
        friend std::ostream &operator<<(std::ostream &output, const BitArray &bits) {
            char buffer[sizeof(size_t)];
            memcpy(buffer, &(bits.length), sizeof(size_t));
            output.write(buffer, sizeof(size_t));
            for (const auto &value: bits.data)
                output << value;
            return output;
        }

        // Recover from input stream
        friend std::istream &operator>>(std::istream &input, BitArray &bits) {
            bits.length = 0;
            bits.data.clear();
            char buffer[sizeof(size_t)];
            input.read(buffer, sizeof(size_t));
            memcpy(&(bits.length), buffer, sizeof(size_t));
            char current;
            while (input) {
                input.get(current);
                bits.data.push_back(current);
            }
            return input;
        }

        [[nodiscard]] size_t size() const {
            return this->length;
        }
    };
}

#endif //MPI_BITS_H
