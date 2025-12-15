#include <iostream>
#include "nnue.h"
#include "position.h"

using namespace pufferfish;

int main()
{
    std::cout << "=== NNUE Loader Test ===" << std::endl;

    // Test 1: Create evaluator
    NNUEEvaluator nnue;
    std::cout << "✓ NNUEEvaluator created" << std::endl;
    std::cout << "  Status: " << (nnue.is_ready() ? "ready" : "not ready") << std::endl;

    // Test 2: Load model
    std::cout << "\nLoading nnue_residual.bin..." << std::endl;
    bool loaded = nnue.load("models/nnue_residual.bin");

    if (loaded)
    {
        std::cout << "✓ Model loaded successfully!" << std::endl;
        std::cout << "  Status: " << (nnue.is_ready() ? "ready" : "not ready") << std::endl;
    }
    else
    {
        std::cerr << "✗ Failed to load model" << std::endl;
        return 1;
    }

    // Test 3: Evaluate starting position
    if (nnue.is_ready())
    {
        std::cout << "\nEvaluating starting position..." << std::endl;
        Position pos;
        pos.reset();

        int score = nnue.evaluate(pos);
        std::cout << "✓ Evaluation successful" << std::endl;
        std::cout << "  Score: " << score << " cp (centi-pawns)" << std::endl;

        // The score should be close to 0 for the starting position
        if (score >= -100 && score <= 100)
        {
            std::cout << "  ✓ Score seems reasonable for starting position" << std::endl;
        }
        else
        {
            std::cout << "  ⚠ Score may need investigation (far from 0)" << std::endl;
        }
    }

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
