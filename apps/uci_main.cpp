/*
 * Pufferfish Chess Engine
 * UCI Main Executable
 */

#include "../src/uci.h"
#include <iostream>

using namespace pufferfish;

int main()
{
    try
    {
        UCIHandler uci;
        uci.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
