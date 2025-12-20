/*
 * Pufferfish Chess Engine
 * UCI Protocol Implementation
 */

#include "uci.h"
#include "movegen.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace pufferfish
{

    // =============================================================================
    // UCI Handler Implementation
    // =============================================================================

    UCIHandler::UCIHandler()
        : search_(16) // 16 MB TT
    {
        position_.set_fen(STARTPOS_FEN);
    }

    void UCIHandler::run()
    {
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (!process_command(line))
                break;
        }
    }

    bool UCIHandler::process_command(const std::string &line)
    {
        if (line.empty())
            return true;

        // Parse command
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "uci")
        {
            handle_uci();
        }
        else if (cmd == "isready")
        {
            handle_isready();
        }
        else if (cmd == "position")
        {
            // Get remaining args after "position"
            std::string args;
            if (line.length() > 9)
                args = line.substr(9); // Skip "position "
            handle_position(args);
        }
        else if (cmd == "go")
        {
            // Get remaining args after "go"
            std::string args;
            if (line.length() > 3)
                args = line.substr(3); // Skip "go "
            handle_go(args);
        }
        else if (cmd == "stop")
        {
            handle_stop();
        }
        else if (cmd == "quit")
        {
            handle_quit();
            return false;
        }
        else if (cmd == "setoption")
        {
            // TODO: Implement option handling (e.g., hash size)
        }
        // Silently ignore unknown commands

        return running_;
    }

    // =============================================================================
    // Command Handlers
    // =============================================================================

    void UCIHandler::handle_uci()
    {
        std::cout << "id name Pufferfish" << std::endl;
        std::cout << "id author Development Team" << std::endl;

        // Check NNUE status
        if (search_.nnue_ready())
        {
            std::cout << "info string NNUE loaded successfully" << std::endl;
        }
        else
        {
            std::cout << "info string NNUE not loaded - using material evaluation" << std::endl;
        }

        // TODO: Add options (e.g., hash size, threads)
        std::cout << "uciok" << std::endl;
    }

    void UCIHandler::handle_isready()
    {
        std::cout << "readyok" << std::endl;
    }

    void UCIHandler::handle_position(const std::string &args)
    {
        PositionArgs pos_args = parse_position_args(args);

        if (pos_args.use_fen)
        {
            position_.set_fen(pos_args.fen);
        }
        else
        {
            position_.set_fen(STARTPOS_FEN);
        }

        // Apply moves
        for (const auto &move_str : pos_args.moves)
        {
            std::vector<Move> legal_moves;
            generate_legal_moves(position_, legal_moves);

            // Find matching move
            bool found = false;
            for (const auto &m : legal_moves)
            {
                if (m.to_uci() == move_str)
                {
                    Undo undo;
                    position_.make_move(m, undo);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                // Invalid move - ignore for now
            }
        }
    }

    void UCIHandler::handle_go(const std::string &args)
    {
        GoArgs go_args = parse_go_args(args);
        SearchTimeManager time_mgr = get_time_manager(go_args);

        // Find best move
        Move best_move;
        if (time_mgr.mode == SearchTimeManager::FIXED_DEPTH)
        {
            best_move = search_.find_best_move(position_, go_args.depth);
        }
        else
        {
            best_move = search_.find_best_move_iterative(position_, time_mgr);
        }

        // Output result
        if (best_move != Move())
        {
            std::cout << "bestmove " << best_move.to_uci() << std::endl;
        }
        else
        {
            // No legal moves (stalemate/checkmate)
            std::cout << "bestmove 0000" << std::endl;
        }
    }

    void UCIHandler::handle_stop()
    {
        // TODO: Implement search stopping (would need threading)
        // For now, just acknowledge
    }

    void UCIHandler::handle_quit()
    {
        running_ = false;
    }

    // =============================================================================
    // Helper Functions
    // =============================================================================

    std::vector<std::string> UCIHandler::split(const std::string &s)
    {
        std::vector<std::string> result;
        std::istringstream iss(s);
        std::string word;
        while (iss >> word)
        {
            result.push_back(word);
        }
        return result;
    }

    UCIHandler::PositionArgs UCIHandler::parse_position_args(const std::string &args)
    {
        PositionArgs result;
        std::vector<std::string> tokens = split(args);

        if (tokens.empty())
            return result;

        size_t idx = 0;

        if (tokens[idx] == "startpos")
        {
            result.use_fen = false;
            ++idx;
        }
        else if (tokens[idx] == "fen")
        {
            result.use_fen = true;
            ++idx;

            // Collect FEN tokens (up to 6 parts)
            std::string fen;
            int fen_parts = 0;
            while (idx < tokens.size() && fen_parts < 6 && tokens[idx] != "moves")
            {
                if (!fen.empty())
                    fen += " ";
                fen += tokens[idx];
                ++fen_parts;
                ++idx;
            }
            result.fen = fen;
        }

        // Parse moves
        if (idx < tokens.size() && tokens[idx] == "moves")
        {
            ++idx;
            while (idx < tokens.size())
            {
                result.moves.push_back(tokens[idx]);
                ++idx;
            }
        }

        return result;
    }

    UCIHandler::GoArgs UCIHandler::parse_go_args(const std::string &args)
    {
        GoArgs result;
        std::vector<std::string> tokens = split(args);

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (tokens[i] == "depth" && i + 1 < tokens.size())
            {
                result.depth = std::stoi(tokens[i + 1]);
                result.mode = SearchTimeManager::FIXED_DEPTH;
                ++i;
            }
            else if (tokens[i] == "movetime" && i + 1 < tokens.size())
            {
                result.movetime = std::stoull(tokens[i + 1]);
                result.mode = SearchTimeManager::FIXED_TIME;
                ++i;
            }
            else if (tokens[i] == "movestogo" && i + 1 < tokens.size())
            {
                result.movestogo = std::stoull(tokens[i + 1]);
                ++i;
            }
            else if (tokens[i] == "wtime" && i + 1 < tokens.size())
            {
                result.wtime = std::stoull(tokens[i + 1]);
                ++i;
            }
            else if (tokens[i] == "btime" && i + 1 < tokens.size())
            {
                result.btime = std::stoull(tokens[i + 1]);
                ++i;
            }
            else if (tokens[i] == "winc" && i + 1 < tokens.size())
            {
                result.winc = std::stoull(tokens[i + 1]);
                ++i;
            }
            else if (tokens[i] == "binc" && i + 1 < tokens.size())
            {
                result.binc = std::stoull(tokens[i + 1]);
                ++i;
            }
        }

        return result;
    }

    SearchTimeManager UCIHandler::get_time_manager(const GoArgs &args)
    {
        SearchTimeManager mgr;

        if (args.movetime > 0)
        {
            mgr.mode = SearchTimeManager::FIXED_TIME;
            mgr.time_ms = args.movetime;
        }
        else if (args.wtime > 0 || args.btime > 0)
        {
            // Temporary: fixed 2s per move regardless of clock (for testing).
            mgr.mode = SearchTimeManager::FIXED_TIME;
            mgr.time_ms = 2000;
        }
        else
        {
            // Default: fixed depth
            mgr.mode = SearchTimeManager::FIXED_DEPTH;
            mgr.depth = 4;
        }

        return mgr;
    }

} // namespace pufferfish
