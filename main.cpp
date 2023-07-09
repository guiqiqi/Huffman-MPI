#include <iostream>

#include "common.h"

#include "huffman.h"
#include "bits.h"
#include "rle.h"
#include "utils.h"

static const size_t RandomStringLength = 100;

#define MPI
#ifdef MPI

int main() {
    MPI_Init(nullptr, nullptr);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Generate random string using MPI concurrently
    std::string source;
    std::vector<char> form = {'A', 'B', 'C', 'D'};
    MPI_Choices(RandomStringLength, form, std::back_inserter(source));

    // RLE encoding source string using MPI concurrently
    std::string rle_encoded;
    std::string rle_decoded;
    RLE::MPI_Encode(source.begin(), source.end(), std::back_inserter(rle_encoded));
    RLE::MPI_Decode(rle_encoded.begin(), rle_encoded.end(), std::back_inserter(rle_decoded));

    // Huffman encoding
    Huffman::Encoder<char> encoder(source.begin(), source.end(), true);
    auto dict = encoder.dict();
    auto content = encoder.MPI_Encode(source.begin(), source.end());
    if (world_rank == 0) {
        std::cout << "RLE encoded string size: " << rle_encoded.size() * 8 << std::endl;
        std::cout << "Source string size: " << source.size() * 8 << std::endl;
        std::cout << "Huffman encoding price: " << encoder.price() << std::endl;
        std::cout << "Huffman encoding dict size: " << dict.size() << std::endl;
        std::cout << "Huffman encoded string size: " << content.size() << std::endl;
    }

    std::vector<bool> encoded;
    encoded.assign(dict.begin(), dict.end());
    encoded.insert(encoded.end(), content.begin(), content.end());
    Huffman::Decoder<char> decoder(encoded);
    std::string decoded;
    decoder.decode(std::back_inserter(decoded));

    // Show result
    if (world_rank == 0) {
        if (source == decoded && source == rle_decoded)
            std::cout << "\nDecoded string equals to source one." << std::endl;
        else
            std::cout << "\nFailed." << std::endl;
        std::cout << "Source string:  " << source << std::endl;
    }

    MPI_Finalize();
    return 0;
}

#endif

#ifndef MPI
int main() {
    static const char* SavingToFile = "encoded";

    // Generate random string
    std::string source;
    std::vector<char> form = {'A', 'B', 'C', 'D'};
    choices(RandomStringLength, form, std::back_inserter(source));

    // RLE encoding
    std::string rle_encoded;
    std::string rle_decoded;
    RLE::encode(source.begin(), source.end(), std::back_inserter(rle_encoded));
    RLE::decode(rle_encoded.begin(), rle_encoded.end(), std::back_inserter(rle_decoded));
    std::cout << "RLE encoded string size: " << rle_encoded.size() * 8 << std::endl;

    // Huffman encoding
    Huffman::Encoder<char> encoder(source.begin(), source.end());
    auto dict = encoder.dict();
    auto content = encoder.encode();
    std::cout << "Source string size: " << source.size() * 8 << std::endl;
    std::cout << "Huffman encoding price: " << encoder.price() << std::endl;
    std::cout << "Huffman encoding dict size: " << dict.size() << std::endl;
    std::cout << "Huffman encoded string size: " << content.size() << std::endl;

    // Saving encoded string to file
    std::vector<bool> encoded;
    encoded.insert(encoded.end(), dict.begin(), dict.end());
    encoded.insert(encoded.end(), content.begin(), content.end());
    Bits::BitArray bits(encoded);
    std::ofstream writer;
    writer.open(SavingToFile, std::ios::out | std::ios::trunc);
    writer << bits;
    writer.close();
    std::cout << "Saved encoded string to file: " << SavingToFile << std::endl;
    std::cout << "Total size: " << bits.size() + sizeof(size_t) * 8 << std::endl;

    // Recovering from saved file
    std::ifstream reader;
    reader.open(SavingToFile, std::ios::in);
    reader >> bits;
    Huffman::Decoder<char> decoder(bits.vectorize());
    std::string decoded;
    decoder.decode(std::back_inserter(decoded));
    std::cout << "Recovered from file: " << SavingToFile << std::endl;

    // Check if recovered string equals to source one
    if (source == decoded && source == rle_decoded)
        std::cout << "\nDecoded string equals to source one." << std::endl;
    else
        std::cout << "\nFailed." << std::endl;
    std::cout << "Source string:  " << source << std::endl;
    std::cout << "Decoded string: " << decoded << std::endl;

    return 0;
}
#endif