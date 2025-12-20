/*
 * Pufferfish Chess Engine
 * NNUE loader + accumulator smoke test
 */

#include <iostream>
#include "nnue.h"
#include "position.h"

using namespace pufferfish;

int main()
{
    std::cout << "=== NNUE Loader Test ===" << std::endl;

    NNUEEvaluator nnue;
    std::cout << "NNUEEvaluator created" << std::endl;
    std::cout << "  Status: " << (nnue.is_ready() ? "ready" : "not ready") << std::endl;

    std::cout << "\nLoading nnue_residual.bin..." << std::endl;
    if (!nnue.load("models/nnue_residual.bin"))
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::cout << "Model loaded successfully" << std::endl;
    std::cout << "  Status: " << (nnue.is_ready() ? "ready" : "not ready") << std::endl;

    Position pos;
    pos.reset();

    std::cout << "\nEvaluating starting position..." << std::endl;
    int score_start = nnue.evaluate(pos);
    std::cout << "  Score: " << score_start << " cp" << std::endl;

    std::cout << "\nTesting incremental update (e2e4)..." << std::endl;
    Move m(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);
    Undo undo;
    pos.make_move(m, undo);
    nnue.update_after_move(pos, m, undo);
    int score_after = nnue.evaluate(pos);
    std::cout << "  Score after e2e4: " << score_after << " cp" << std::endl;

    pos.unmake_move(m, undo);
    int score_restore = nnue.evaluate(pos);
    std::cout << "  Score after unmake: " << score_restore << " cp" << std::endl;

    if (score_restore != score_start)
    {
        std::cerr << "Accumulator mismatch after unmake" << std::endl;
        return 1;
    }

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
