#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    std::string inpath;

    int power = 1;

    // Parse inputs
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);

        if (arg.rfind("-i=", 0) != std::string::npos) { // Input path
            inpath = arg.substr(3);
        } else { // Power
            try {
                power = std::stoi(arg);
            } catch (const std::exception& exc) {
                continue;
            }
        }
    }

    // Read input
    std::ifstream infile(inpath);
    if (!infile.is_open()) {
        std::cerr << "Could not open input file!\n" << inpath << std::endl;
        return 2;
    }

    bool skipped_header = false;
    for (std::string line; std::getline(infile, line);) {
        // Skip  the header
        if (!skipped_header) {
            skipped_header = true;
            continue;
        }

        // Get values
        auto idx = line.find(":");

        float x1 = std::stof(line.substr(0, idx));
        float x2 = std::stof(line.substr(idx + 1));

        // Compute result
        std::cout << (std::pow(x1, power) + x2) << std::endl;
    }
    infile.close();

    return 0;
}