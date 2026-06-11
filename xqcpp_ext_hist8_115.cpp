
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>

namespace py = pybind11;

// ----------------------------
// Xiangqi C++ core (Board + MCTS)
// Coordinate system:
//   - 9x10 board
//   - square index = y*9 + x, y=0 (top/black side) .. 9 (bottom/red side)
// Action encoding:
//   action_id = from_sq*90 + to_sq, 0..8099
// Side encoding:
//   0 = RED (w), 1 = BLACK (b)
// ----------------------------

static inline int sq(int x, int y) { return y * 9 + x; }
static inline void xy(int s, int &x, int &y) { y = s / 9; x = s - y * 9; }

enum Side : int { RED = 0, BLACK = 1 };
enum TerminalCode : int {
    TERMINAL_ONGOING = -1,
    TERMINAL_CHECKMATE_OR_STALEMATE = 0,
    TERMINAL_MAX_PLIES_DRAW = 1,
    TERMINAL_REPETITION_DRAW = 2,
    TERMINAL_NO_CAPTURE_DRAW = 3,
    TERMINAL_PERPETUAL_CHECK_LOSS = 4,
};

// Piece codes: abs value indicates type
// 1=K,2=A,3=B,4=N,5=R,6=C,7=P ; sign indicates side (+red, -black)
static inline int piece_type(int8_t p) { return (p >= 0) ? (int)p : (int)(-p); }
static inline int piece_side(int8_t p) { return (p > 0) ? RED : BLACK; }
static inline bool is_empty(int8_t p) { return p == 0; }
static inline bool same_side(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return false;
    return (a > 0) == (b > 0);
}
static inline bool opp_side(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return false;
    return (a > 0) != (b > 0);
}

// Zobrist hashing
static bool ZOB_INIT = false;
static uint64_t ZOB[14][90];
static uint64_t ZOB_TURN;

static inline int zob_piece_index(int8_t p) {
    // p in {-7..-1,1..7}
    if (p > 0) return (int)p - 1;          // 0..6
    return 7 + ((int)(-p) - 1);            // 7..13
}

static inline int canonical_square(int s, bool stm_black) {
    return stm_black ? (89 - s) : s;
}

static inline int canonical_action(int action_id, bool stm_black) {
    if (!stm_black) return action_id;
    int from = action_id / 90;
    int to = action_id % 90;
    return canonical_square(from, true) * 90 + canonical_square(to, true);
}

static inline int canonical_piece_plane_index(int8_t p, bool stm_black) {
    if (!stm_black) return zob_piece_index(p);
    int t = piece_type(p) - 1;
    return (p < 0) ? t : (7 + t);
}

static void init_zobrist() {
    if (ZOB_INIT) return;
    std::mt19937_64 rng(0xC0FFEE123456789ULL);
    auto rand64 = [&]() -> uint64_t { return rng(); };
    for (int i = 0; i < 14; ++i) {
        for (int s = 0; s < 90; ++s) ZOB[i][s] = rand64();
    }
    ZOB_TURN = rand64();
    ZOB_INIT = true;
}


// Enable expensive check-validation (fast in_check vs slow generator-based check)
// Set env var XQCPP_VERIFY_CHECK=1 to enable.
static inline bool xqcpp_verify_check_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char* v = std::getenv("XQCPP_VERIFY_CHECK");
        enabled = (v && v[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

// Utility: palace / river constraints
static inline bool in_palace(int side, int x, int y) {
    if (x < 3 || x > 5) return false;
    if (side == RED)  return (y >= 7 && y <= 9);
    return (y >= 0 && y <= 2);
}
static inline bool on_own_side_elephant(int side, int y) {
    // Elephant cannot cross river.
    // River is between y=4 and y=5.
    if (side == RED)  return (y >= 5);
    return (y <= 4);
}
static inline bool pawn_crossed_river(int side, int y) {
    // Red pawns move upward (toward smaller y) and cross river when y <= 4
    // Black pawns move downward and cross river when y >= 5
    if (side == RED) return (y <= 4);
    return (y >= 5);
}

struct Undo {
    int from;
    int to;
    int8_t moved;
    int8_t captured;

    int old_turn;
    uint64_t old_hash;
    int old_king_red;
    int old_king_black;

    // For history-stack features (8 frames)
    int old_no_capture;
    int old_plies_played;
    int old_hist_head;
    int old_hist_len;

    int overwritten_idx;                       // which slot was overwritten on push
    std::array<int8_t, 90> overwritten_board;  // previous content of that slot
    uint64_t overwritten_key;
};

class Board {
public:
    Board() {
        init_zobrist();
        reset();
    }

    void reset() {
        // Starting position
        // Top (black): r n b a k a b n r
        // Cannons at (1,2),(7,2); pawns at (0,3),(2,3),(4,3),(6,3),(8,3)
        // Bottom (red): R N B A K A B N R
        b.fill(0);
        auto setp = [&](int x, int y, int8_t p) { b[sq(x,y)] = p; };

        // black main pieces
        setp(0,0, -5); setp(1,0, -4); setp(2,0, -3); setp(3,0, -2); setp(4,0, -1);
        setp(5,0, -2); setp(6,0, -3); setp(7,0, -4); setp(8,0, -5);
        // black cannons
        setp(1,2, -6); setp(7,2, -6);
        // black pawns
        setp(0,3, -7); setp(2,3, -7); setp(4,3, -7); setp(6,3, -7); setp(8,3, -7);

        // red main pieces
        setp(0,9, +5); setp(1,9, +4); setp(2,9, +3); setp(3,9, +2); setp(4,9, +1);
        setp(5,9, +2); setp(6,9, +3); setp(7,9, +4); setp(8,9, +5);
        // red cannons
        setp(1,7, +6); setp(7,7, +6);
        // red pawns
        setp(0,6, +7); setp(2,6, +7); setp(4,6, +7); setp(6,6, +7); setp(8,6, +7);

        turn = RED;

        king_red = sq(4,9);
        king_black = sq(4,0);

        history.clear();
        recompute_hash();

        // init 8-frame history ring (only current position valid)
        no_capture = 0;
        plies_played = 0;
        hist_head = 0;
        hist_len = 1;
        hist_board[0] = b;
        hist_key[0] = hash;
        repetition_counts.clear();
        repetition_counts[hash] = 1;
        key_history.clear();
        key_history.push_back(hash);
        move_gave_check_history.clear();
    }

    int get_turn() const { return turn; }
    int get_plies_played() const { return plies_played; }
    int get_no_capture_count() const { return no_capture; }
    int get_current_repetition_count() const {
        auto it = repetition_counts.find(hash);
        return (it == repetition_counts.end()) ? 0 : it->second;
    }

    uint64_t key() const { return hash; }

    float material_score() const {
        // Red advantage (positive = red better)
        // Values aligned to your Python mapping.
        auto val = [](int t) -> float {
            switch(t){
                case 1: return 0.0f;  // king
                case 2: return 2.0f;  // advisor
                case 3: return 2.0f;  // elephant
                case 4: return 4.0f;  // horse
                case 5: return 9.0f;  // rook
                case 6: return 4.5f;  // cannon
                case 7: return 1.0f;  // pawn
                default: return 0.0f;
            }
        };
        float s = 0.0f;
        for (int i = 0; i < 90; ++i) {
            int8_t p = b[i];
            if (p == 0) continue;
            float v = val(piece_type(p));
            if (p > 0) s += v;
            else s -= v;
        }
        return s;
    }

    torch::Tensor to_tensor() const {
        // [1,115,10,9] float32
        // 8 frames * 14 piece planes = 112
        // + 3 extra planes:
        //   112: side-to-move (black=1)
        //   113: repetition count within last-8 (0,0.5,1)
        //   114: no-capture count (tanh normalized 0..1)
        auto t = torch::zeros({115,10,9}, torch::TensorOptions().dtype(torch::kFloat32));
        float* data = t.data_ptr<float>();

        // 8 frames: frame0=current, frame1=prev1, ... frame7=prev7
        for (int f = 0; f < 8; ++f) {
            if (f >= hist_len) break;
            int idx = (hist_head - f + 8) % 8;
            const auto &bb = hist_board[idx];
            int base = f * 14;
            for (int s = 0; s < 90; ++s) {
                int8_t p = bb[s];
                if (p == 0) continue;
                int pi = base + zob_piece_index(p); // 0..13 -> [base, base+13]
                int y = s / 9;
                int x = s - y*9;
                data[pi*90 + y*9 + x] = 1.0f;
            }
        }

        // side-to-move plane (current)
        if (turn == BLACK) {
            float* p112 = data + 112*90;
            for (int i = 0; i < 90; ++i) p112[i] = 1.0f;
        }

        // repetition count within last 8 frames (exclude current) -> {0, 0.5, 1}
        int rep = 0;
        for (int f = 0; f < hist_len; ++f) {
            int idx = (hist_head - f + 8) % 8;
            if (hist_key[idx] == hash) rep++;
        }
        rep = std::max(0, rep - 1);
        if (rep > 2) rep = 2;
        float rep_norm = 0.5f * (float)rep;
        if (rep_norm != 0.0f) {
            float* p113 = data + 113*90;
            for (int i = 0; i < 90; ++i) p113[i] = rep_norm;
        }

        // no-capture count tanh normalized
        float nc_norm = std::tanh((float)no_capture / 30.0f);
        if (nc_norm != 0.0f) {
            float* p114 = data + 114*90;
            for (int i = 0; i < 90; ++i) p114[i] = nc_norm;
        }

        return t.unsqueeze(0);
    }

    torch::Tensor to_tensor_canonical() const {
        // Current-side perspective:
        // - when BLACK to move, rotate 180 degrees and swap side planes so the
        //   current player always occupies planes 0..6.
        // - side-to-move plane stays zero because the tensor is already canonicalized.
        auto t = torch::zeros({115,10,9}, torch::TensorOptions().dtype(torch::kFloat32));
        float* data = t.data_ptr<float>();
        bool stm_black = (turn == BLACK);

        for (int f = 0; f < 8; ++f) {
            if (f >= hist_len) break;
            int idx = (hist_head - f + 8) % 8;
            const auto &bb = hist_board[idx];
            int base = f * 14;
            for (int s = 0; s < 90; ++s) {
                int8_t p = bb[s];
                if (p == 0) continue;
                int pi = base + canonical_piece_plane_index(p, stm_black);
                int cs = canonical_square(s, stm_black);
                int y = cs / 9;
                int x = cs - y * 9;
                data[pi * 90 + y * 9 + x] = 1.0f;
            }
        }

        int rep = 0;
        for (int f = 0; f < hist_len; ++f) {
            int idx = (hist_head - f + 8) % 8;
            if (hist_key[idx] == hash) rep++;
        }
        rep = std::max(0, rep - 1);
        if (rep > 2) rep = 2;
        float rep_norm = 0.5f * (float)rep;
        if (rep_norm != 0.0f) {
            float* p113 = data + 113 * 90;
            for (int i = 0; i < 90; ++i) p113[i] = rep_norm;
        }

        float nc_norm = std::tanh((float)no_capture / 30.0f);
        if (nc_norm != 0.0f) {
            float* p114 = data + 114 * 90;
            for (int i = 0; i < 90; ++i) p114[i] = nc_norm;
        }

        return t.unsqueeze(0);
    }
    // Apply a move, action_id = from*90 + to
    void push(int action_id) {
        int from = action_id / 90;
        int to = action_id - from*90;

        int8_t moved = b[from];
        int8_t captured = b[to];

        Undo u;
        u.from = from;
        u.to = to;
        u.moved = moved;
        u.captured = captured;
        u.old_turn = turn;
        u.old_hash = hash;
        u.old_king_red = king_red;
        u.old_king_black = king_black;

        // history-ring + counters (before push)
        u.old_no_capture = no_capture;
        u.old_plies_played = plies_played;
        u.old_hist_head = hist_head;
        u.old_hist_len = hist_len;

        int new_head = (hist_head + 1) % 8; // 0..7
        u.overwritten_idx = new_head;
        u.overwritten_board = hist_board[new_head];
        u.overwritten_key = hist_key[new_head];

        history.push_back(u);

        // update hash: remove moved from, remove captured, add moved to, flip turn
        if (moved != 0) hash ^= ZOB[zob_piece_index(moved)][from];
        if (captured != 0) hash ^= ZOB[zob_piece_index(captured)][to];
        // move pieces
        b[from] = 0;
        b[to] = moved;
        if (moved != 0) hash ^= ZOB[zob_piece_index(moved)][to];

        // update king positions if needed
        if (moved != 0 && piece_type(moved) == 1) {
            if (moved > 0) king_red = to;
            else king_black = to;
        }
        if (captured != 0 && piece_type(captured) == 1) {
            if (captured > 0) king_red = -1;
            else king_black = -1;
        }

        // flip turn
        turn = 1 - turn;
        hash ^= ZOB_TURN;

        // update no-capture counter (plies since last capture)
        if (captured != 0) no_capture = 0;
        else no_capture += 1;
        plies_played += 1;

        // update 8-frame history ring with new position
        hist_head = new_head;
        if (hist_len < 8) hist_len += 1;
        hist_board[new_head] = b;
        hist_key[new_head] = hash;
        repetition_counts[hash] += 1;
        key_history.push_back(hash);
        move_gave_check_history.push_back(in_check(turn));
    }

    void pop() {
        if (history.empty()) return;
        Undo u = history.back();
        history.pop_back();

        uint64_t current_hash = hash;
        auto rep_it = repetition_counts.find(current_hash);
        if (rep_it != repetition_counts.end()) {
            rep_it->second -= 1;
            if (rep_it->second <= 0) repetition_counts.erase(rep_it);
        }
        if (!key_history.empty()) key_history.pop_back();
        if (!move_gave_check_history.empty()) move_gave_check_history.pop_back();

        // restore board
        b[u.from] = u.moved;
        b[u.to] = u.captured;
        turn = u.old_turn;
        hash = u.old_hash;
        king_red = u.old_king_red;
        king_black = u.old_king_black;

        // restore history ring + counters
        no_capture = u.old_no_capture;
        plies_played = u.old_plies_played;
        hist_head = u.old_hist_head;
        hist_len = u.old_hist_len;
        hist_board[u.overwritten_idx] = u.overwritten_board;
        hist_key[u.overwritten_idx] = u.overwritten_key;
    }

    void set_search_context(int plies, int no_capture_count, int repetition_count_hint = 1) {
        history.clear();
        plies_played = std::max(0, plies);
        no_capture = std::max(0, no_capture_count);
        int rep_hint = std::max(1, repetition_count_hint);

        hist_head = 0;
        hist_len = std::min(rep_hint, 8);
        for (int i = 0; i < 8; ++i) {
            hist_board[i] = b;
            hist_key[i] = hash;
        }

        repetition_counts.clear();
        repetition_counts[hash] = rep_hint;
        key_history.clear();
        key_history.push_back(hash);
        move_gave_check_history.clear();
    }

    int result_red_view() {
        // 1 = red win, -1 = black win, 0 = ongoing
        if (king_red < 0) return -1;
        if (king_black < 0) return 1;

        // If side-to-move has no legal moves: in Xiangqi it's a loss (checkmate or stalemate)
        auto lm = legal_moves();
        if (lm.empty()) {
            return (turn == RED) ? -1 : 1;
        }
        return 0;
    }

    int terminal_result_red_view(int terminal_code) {
        switch (terminal_code) {
            case TERMINAL_CHECKMATE_OR_STALEMATE:
                return result_red_view();
            case TERMINAL_PERPETUAL_CHECK_LOSS:
            {
                int loser = perpetual_check_loser_side_for_current_repetition();
                if (loser == RED) return -1;
                if (loser == BLACK) return 1;
                // Legacy fallback: if the repeated terminal position itself is
                // in check, the checking side loses and the side-to-move wins.
                return (turn == RED) ? 1 : -1;
            }
            case TERMINAL_MAX_PLIES_DRAW:
            case TERMINAL_REPETITION_DRAW:
            case TERMINAL_NO_CAPTURE_DRAW:
            case TERMINAL_ONGOING:
            default:
                return 0;
        }
    }

    int terminal_code(
        int max_plies = 0,
        int repeat_limit = 0,
        int repeat_min_ply = 0,
        int no_capture_limit = 0
    ) {
        if (king_red < 0 || king_black < 0) {
            return TERMINAL_CHECKMATE_OR_STALEMATE;
        }
        auto lm = legal_moves();
        if (lm.empty()) {
            return TERMINAL_CHECKMATE_OR_STALEMATE;
        }
        if (max_plies > 0 && plies_played >= max_plies) {
            return TERMINAL_MAX_PLIES_DRAW;
        }
        if (repeat_limit > 0 && plies_played >= repeat_min_ply && get_current_repetition_count() >= repeat_limit) {
            if (perpetual_check_loser_side_for_current_repetition() >= 0 || in_check(turn)) {
                return TERMINAL_PERPETUAL_CHECK_LOSS;
            }
            return TERMINAL_REPETITION_DRAW;
        }
        if (no_capture_limit > 0 && no_capture >= no_capture_limit) {
            return TERMINAL_NO_CAPTURE_DRAW;
        }
        return TERMINAL_ONGOING;
    }

    bool is_game_over() {
        return result_red_view() != 0;
    }

    std::string fen() const {
        // Xiangqi-style piece placement + " " + turn ('w' red, 'b' black)
        auto pc = [&](int8_t p)->char{
            if (p==0) return '0';
            int t = piece_type(p);
            bool red = (p>0);
            switch(t){
                case 1: return red ? 'K':'k';
                case 2: return red ? 'A':'a';
                case 3: return red ? 'B':'b';
                case 4: return red ? 'N':'n';
                case 5: return red ? 'R':'r';
                case 6: return red ? 'C':'c';
                case 7: return red ? 'P':'p';
                default: return '?';
            }
        };
        std::ostringstream oss;
        for (int y=0;y<10;++y){
            int empt=0;
            for (int x=0;x<9;++x){
                int8_t p=b[sq(x,y)];
                if(p==0){ empt++; continue; }
                if(empt>0){ oss<<empt; empt=0; }
                oss<<pc(p);
            }
            if(empt>0) oss<<empt;
            if(y!=9) oss<<"/";
        }
        oss<<" "<<(turn==RED?'w':'b');
        return oss.str();
    }

    // Optional: set from fen (only placement + turn)
    void set_fen(const std::string& fen) {
        // Minimal parser: "rows turn"
        // rows: 10 ranks separated by '/'
        // pieces: KABNRC P for red, lowercase for black, digits for empties
        // turn: 'w' or 'b'
        std::istringstream iss(fen);
        std::string rows;
        std::string tstr;
        if(!(iss>>rows)) throw std::runtime_error("bad fen");
        if(!(iss>>tstr)) tstr="w";

        std::vector<std::string> ranks;
        {
            std::string cur;
            for(char c: rows){
                if(c=='/'){ ranks.push_back(cur); cur.clear(); }
                else cur.push_back(c);
            }
            ranks.push_back(cur);
        }
        if(ranks.size()!=10) throw std::runtime_error("fen ranks!=10");
        b.fill(0);
        king_red = -1;
        king_black = -1;

        auto decode_piece = [&](char c)->int8_t{
            bool red = (c>='A' && c<='Z');
            char lc = (char)std::tolower((unsigned char)c);
            int8_t t=0;
            switch(lc){
                case 'k': t=1; break;
                case 'a': t=2; break;
                case 'b': t=3; break;
                case 'n': t=4; break;
                case 'r': t=5; break;
                case 'c': t=6; break;
                case 'p': t=7; break;
                default: t=0; break;
            }
            if(t==0) return 0;
            return red ? t : (int8_t)(-t);
        };

        for(int y=0;y<10;++y){
            int x=0;
            for(char c: ranks[y]){
                if(std::isdigit((unsigned char)c)){
                    int n=c-'0';
                    x += n;
                }else{
                    if(x>=9) throw std::runtime_error("fen file overflow");
                    int8_t p=decode_piece(c);
                    b[sq(x,y)] = p;
                    if(p!=0 && piece_type(p)==1){
                        if(p>0) king_red = sq(x,y);
                        else king_black = sq(x,y);
                    }
                    x++;
                }
            }
            if(x!=9) throw std::runtime_error("fen rank not 9 files");
        }

        turn = (tstr.size()>0 && (tstr[0]=='b' || tstr[0]=='B')) ? BLACK : RED;
        history.clear();
        recompute_hash();

        // init 8-frame history ring (only current position valid)
        no_capture = 0;
        plies_played = 0;
        hist_head = 0;
        hist_len = 1;
        hist_board[0] = b;
        hist_key[0] = hash;
        repetition_counts.clear();
        repetition_counts[hash] = 1;
        key_history.clear();
        key_history.push_back(hash);
        move_gave_check_history.clear();
    }

    int perpetual_check_loser_side_for_current_repetition() {
        if (key_history.size() < 2) return -1;
        size_t last_pos = key_history.size() - 1;
        if (history.size() < last_pos || move_gave_check_history.size() < last_pos) return -1;

        uint64_t current_key = key_history[last_pos];
        int prev_pos = -1;
        for (int i = (int)last_pos - 1; i >= 0; --i) {
            if (key_history[(size_t)i] == current_key) {
                prev_pos = i;
                break;
            }
        }
        if (prev_pos < 0 || (size_t)prev_pos >= last_pos) return -1;

        int moves_by_side[2] = {0, 0};
        int checks_by_side[2] = {0, 0};
        for (size_t move_i = (size_t)prev_pos; move_i < last_pos; ++move_i) {
            if (move_i >= history.size() || move_i >= move_gave_check_history.size()) return -1;
            int mover = history[move_i].old_turn;
            if (mover != RED && mover != BLACK) return -1;
            moves_by_side[mover] += 1;
            if (move_gave_check_history[move_i]) checks_by_side[mover] += 1;
        }

        for (int side = RED; side <= BLACK; ++side) {
            int other = 1 - side;
            if (
                moves_by_side[side] >= 2 &&
                checks_by_side[side] == moves_by_side[side] &&
                checks_by_side[other] == 0
            ) {
                return side;
            }
        }
        return -1;
    }

    // Get a piece code at square (for Python-side debugging)
    int8_t piece_at(int square) const {
        if(square<0 || square>=90) return 0;
        return b[square];
    }

    bool is_capture(int action_id) const {
        int from = action_id / 90;
        int to = action_id - from*90;
        int8_t a = b[from];
        int8_t c = b[to];
        return (a!=0 && c!=0 && opp_side(a,c));
    }

    std::vector<int> legal_moves() {
        std::vector<int> pseudo;
        pseudo.reserve(64);

        for(int from=0; from<90; ++from){
            int8_t p = b[from];
            if(p==0) continue;
            if(piece_side(p)!=turn) continue;

            int t = piece_type(p);
            switch(t){
                case 1: gen_king(from, p, pseudo); break;
                case 2: gen_advisor(from, p, pseudo); break;
                case 3: gen_elephant(from, p, pseudo); break;
                case 4: gen_horse(from, p, pseudo); break;
                case 5: gen_rook(from, p, pseudo); break;
                case 6: gen_cannon(from, p, pseudo); break;
                case 7: gen_pawn(from, p, pseudo); break;
                default: break;
            }
        }

        // Filter illegal (leaving king in check)
        std::vector<int> legal;
        legal.reserve(pseudo.size());

        int side_before = turn;
        size_t base_hist = history.size();

        for(int mv: pseudo){
            push(mv);
            bool chk = in_check(side_before);
            if(xqcpp_verify_check_enabled()) {
                bool slow = in_check_slow(side_before);
                if(slow != chk) chk = slow;
            }
            bool ok = !chk;
            pop();
            // ensure we restored
            // (pop already restores fully; base_hist not used)
            if(ok) legal.push_back(mv);
        }

        return legal;
    }


    // --- Debug / safety helpers (public) ---
    // NOTE: push() accepts any action_id without validation. For external GUI integration,
    // use is_legal() / push_legal() to ensure moves obey Xiangqi rules (including responding to check).

    bool in_check_turn() {
        return in_check(turn);
    }

    bool in_check_side(int side) {
        return in_check(side);
    }

    // Pseudo-legal check: movement rules + destination not occupied by own piece.
    // Does NOT check self-check.
    bool is_pseudo_legal(int action_id) {
        int from = action_id / 90;
        int to = action_id % 90;
        if(from < 0 || from >= 90 || to < 0 || to >= 90) return false;
        int8_t p = b[from];
        if(p == 0) return false;
        if(piece_side(p) != turn) return false;
        int8_t dst = b[to];
        if(dst != 0 && same_side(p, dst)) return false;

        std::vector<int> out;
        out.reserve(32);
        int t = piece_type(p);
        switch(t){
            case 1: gen_king(from, p, out); break;
            case 2: gen_advisor(from, p, out); break;
            case 3: gen_elephant(from, p, out); break;
            case 4: gen_horse(from, p, out); break;
            case 5: gen_rook(from, p, out); break;
            case 6: gen_cannon(from, p, out); break;
            case 7: gen_pawn(from, p, out); break;
            default: return false;
        }
        for(int mv : out){
            if(mv == action_id) return true;
        }
        return false;
    }

    // Full legality check: must be pseudo-legal AND must not leave own king in check.
    bool is_legal(int action_id) {
        if(!is_pseudo_legal(action_id)) return false;
        int side_before = turn;
        push(action_id);
        bool ok = !in_check(side_before);
        pop();
        return ok;
    }

    // Safer push for GUI / external integration.
    void push_legal(int action_id) {
        if(!is_legal(action_id)) {
            std::ostringstream oss;
            int from = action_id / 90;
            int to = action_id % 90;
            oss << "illegal move action_id=" << action_id << " from=" << from << " to=" << to;
            throw std::runtime_error(oss.str());
        }
        push(action_id);
    }

    // Slow check detector for debugging: generate opponent pseudo moves and see if any attacks the king.
    // Useful to diagnose missed patterns in in_check().
    bool in_check_slow(int side) {
        int ks = (side==RED) ? king_red : king_black;
        if(ks < 0) return true;
        int kx, ky; xy(ks, kx, ky);
        int opp_king = (side==RED) ? king_black : king_red;
        if(opp_king >= 0){
            int ox, oy; xy(opp_king, ox, oy);
            if(ox == kx){
                int step = (oy > ky) ? 1 : -1;
                bool clear = true;
                for(int y = ky + step; y != oy; y += step){
                    if(b[sq(kx, y)] != 0){ clear = false; break; }
                }
                if(clear) return true; // flying general check
            }
        }
        int opp = 1 - side;
        for(int from=0; from<90; ++from){
            int8_t p = b[from];
            if(p==0) continue;
            if(piece_side(p)!=opp) continue;
            std::vector<int> out;
            out.reserve(32);
            int t = piece_type(p);
            switch(t){
                case 1: gen_king(from, p, out); break;
                case 2: gen_advisor(from, p, out); break;
                case 3: gen_elephant(from, p, out); break;
                case 4: gen_horse(from, p, out); break;
                case 5: gen_rook(from, p, out); break;
                case 6: gen_cannon(from, p, out); break;
                case 7: gen_pawn(from, p, out); break;
                default: break;
            }
            for(int mv : out){
                int to = mv % 90;
                if(to == ks) return true;
            }
        }
        return false;
    }

    bool in_check_slow_turn() {
        return in_check_slow(turn);
    }

private:
    std::array<int8_t, 90> b{};
    int turn = RED;
    uint64_t hash = 0;
    int king_red = -1;
    int king_black = -1;
    std::vector<Undo> history;

    // 8-frame history for NN input (112 piece planes) + 3 extra planes
    int no_capture = 0;
    int plies_played = 0;
    int hist_head = 0; // index of current position in ring buffer
    int hist_len = 0;  // number of valid frames in ring buffer (<=8)
    std::array<std::array<int8_t, 90>, 8> hist_board{};
    std::array<uint64_t, 8> hist_key{};
    std::unordered_map<uint64_t, int> repetition_counts{};
    std::vector<uint64_t> key_history{};
    std::vector<uint8_t> move_gave_check_history{};

    void recompute_hash() {
        hash = 0;
        for(int s=0;s<90;++s){
            int8_t p = b[s];
            if(p==0) continue;
            hash ^= ZOB[zob_piece_index(p)][s];
        }
        if(turn==BLACK) hash ^= ZOB_TURN;
    }

    bool in_check(int side) {
        int ks = (side==RED) ? king_red : king_black;
        if(ks < 0) return true; // captured king => illegal
        int kx, ky; xy(ks, kx, ky);

        int opp = 1 - side;

        // 1) Flying general / king line
        {
            // up
            for(int y=ky-1; y>=0; --y){
                int s = sq(kx,y);
                int8_t p = b[s];
                if(p==0) continue;
                if(piece_type(p)==1 && piece_side(p)==opp) return true;
                break;
            }
            // down
            for(int y=ky+1; y<=9; ++y){
                int s = sq(kx,y);
                int8_t p = b[s];
                if(p==0) continue;
                if(piece_type(p)==1 && piece_side(p)==opp) return true;
                break;
            }
        }

        // 2) Rook / cannon line attacks
        const int dx4[4] = {1,-1,0,0};
        const int dy4[4] = {0,0,1,-1};
        for(int dir=0; dir<4; ++dir){
            int x=kx, y=ky;
            int screens=0;
            while(true){
                x += dx4[dir]; y += dy4[dir];
                if(x<0||x>=9||y<0||y>=10) break;
                int s = sq(x,y);
                int8_t p = b[s];
                if(p==0) continue;
                if(screens==0){
                    // first piece: rook check
                    if(piece_side(p)==opp){
                        int t = piece_type(p);
                        if(t==5) return true;          // rook
                        // also king already handled
                    }
                    screens=1;
                }else{
                    // second piece after screen: cannon check
                    if(piece_side(p)==opp && piece_type(p)==6) return true;
                    break;
                }
            }
        }

        // 3) Horse attacks (with leg)
        struct HAtk { int dx, dy, lx, ly; };
        // Reverse horse attack mapping (from king square to potential attacker square).
        // For check detection, the required horse leg is diagonal-adjacent to king.
        static const HAtk HATK[8] = {
            {+2,+1, +1,+1}, {+2,-1, +1,-1}, {-2,+1, -1,+1}, {-2,-1, -1,-1},
            {+1,+2, +1,+1}, {-1,+2, -1,+1}, {+1,-2, +1,-1}, {-1,-2, -1,-1},
        };
        for(const auto& h : HATK){
            int hx = kx + h.dx;
            int hy = ky + h.dy;
            int lx = kx + h.lx;
            int ly = ky + h.ly;
            if(hx<0||hx>=9||hy<0||hy>=10) continue;
            if(lx<0||lx>=9||ly<0||ly>=10) continue;
            if(b[sq(lx,ly)] != 0) continue; // leg blocked
            int8_t p = b[sq(hx,hy)];
            if(p!=0 && piece_side(p)==opp && piece_type(p)==4) return true;
        }

        // 4) Pawn attacks
        if(opp==RED){
            // red pawn attacks upward (toward y-1). So it attacks king from (kx,ky+1)
            if(ky+1 <= 9){
                int8_t p = b[sq(kx,ky+1)];
                if(p!=0 && piece_side(p)==RED && piece_type(p)==7) return true;
            }
            // sideways after crossing river (pawn y <=4)
            if(kx-1 >= 0){
                int s = sq(kx-1,ky);
                int8_t p = b[s];
                if(p!=0 && piece_side(p)==RED && piece_type(p)==7){
                    int py; int px; xy(s,px,py);
                    if(pawn_crossed_river(RED, py)) return true;
                }
            }
            if(kx+1 <= 8){
                int s = sq(kx+1,ky);
                int8_t p = b[s];
                if(p!=0 && piece_side(p)==RED && piece_type(p)==7){
                    int py; int px; xy(s,px,py);
                    if(pawn_crossed_river(RED, py)) return true;
                }
            }
        }else{
            // black pawn attacks downward (toward y+1). So it attacks king from (kx,ky-1)
            if(ky-1 >= 0){
                int8_t p = b[sq(kx,ky-1)];
                if(p!=0 && piece_side(p)==BLACK && piece_type(p)==7) return true;
            }
            // sideways after crossing river (pawn y >=5)
            if(kx-1 >= 0){
                int s = sq(kx-1,ky);
                int8_t p = b[s];
                if(p!=0 && piece_side(p)==BLACK && piece_type(p)==7){
                    int py; int px; xy(s,px,py);
                    if(pawn_crossed_river(BLACK, py)) return true;
                }
            }
            if(kx+1 <= 8){
                int s = sq(kx+1,ky);
                int8_t p = b[s];
                if(p!=0 && piece_side(p)==BLACK && piece_type(p)==7){
                    int py; int px; xy(s,px,py);
                    if(pawn_crossed_river(BLACK, py)) return true;
                }
            }
        }

        // 5) Advisor attacks (diagonal adjacent)
        const int dx2[4] = {+1,+1,-1,-1};
        const int dy2[4] = {+1,-1,+1,-1};
        for(int i=0;i<4;++i){
            int ax = kx + dx2[i];
            int ay = ky + dy2[i];
            if(ax<0||ax>=9||ay<0||ay>=10) continue;
            int8_t p = b[sq(ax,ay)];
            if(p!=0 && piece_side(p)==opp && piece_type(p)==2) return true;
        }

        // 6) Elephant attacks (2 diag with eye empty)
        const int dxE[4] = {+2,+2,-2,-2};
        const int dyE[4] = {+2,-2,+2,-2};
        for(int i=0;i<4;++i){
            int ex = kx + dxE[i];
            int ey = ky + dyE[i];
            int mx = kx + dxE[i]/2;
            int my = ky + dyE[i]/2;
            if(ex<0||ex>=9||ey<0||ey>=10) continue;
            if(mx<0||mx>=9||my<0||my>=10) continue;
            if(b[sq(mx,my)]!=0) continue;
            int8_t p = b[sq(ex,ey)];
            if(p!=0 && piece_side(p)==opp && piece_type(p)==3){
                // elephant must be on its own side and destination (king square) also must be reachable
                if(on_own_side_elephant(opp, ey) && on_own_side_elephant(opp, ky)) return true;
            }
        }

        return false;
    }

    // Pseudo move generators (do not check for self-check)
    void push_move_if_ok(int from, int to, int8_t p, std::vector<int>& out) {
        if(to<0||to>=90) return;
        int8_t dst = b[to];
        if(dst==0 || opp_side(p, dst)){
            out.push_back(from*90 + to);
        }
    }

    void gen_king(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        int side = piece_side(p);
        static const int dx[4]={1,-1,0,0};
        static const int dy[4]={0,0,1,-1};
        for(int i=0;i<4;++i){
            int nx=x+dx[i], ny=y+dy[i];
            if(nx<0||nx>=9||ny<0||ny>=10) continue;
            if(!in_palace(side,nx,ny)) continue;
            push_move_if_ok(from, sq(nx,ny), p, out);
        }
        // Flying general capture along an open file.
        int step = (side == RED) ? -1 : +1;
        for(int ny = y + step; ny >= 0 && ny < 10; ny += step){
            int to = sq(x, ny);
            int8_t dst = b[to];
            if(dst == 0) continue;
            if(opp_side(p, dst) && piece_type(dst) == 1){
                out.push_back(from * 90 + to);
            }
            break;
        }
    }

    void gen_advisor(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        int side = piece_side(p);
        static const int dx[4]={1,1,-1,-1};
        static const int dy[4]={1,-1,1,-1};
        for(int i=0;i<4;++i){
            int nx=x+dx[i], ny=y+dy[i];
            if(nx<0||nx>=9||ny<0||ny>=10) continue;
            if(!in_palace(side,nx,ny)) continue;
            push_move_if_ok(from, sq(nx,ny), p, out);
        }
    }

    void gen_elephant(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        int side = piece_side(p);
        static const int dx[4]={2,2,-2,-2};
        static const int dy[4]={2,-2,2,-2};
        for(int i=0;i<4;++i){
            int nx=x+dx[i], ny=y+dy[i];
            int mx=x+dx[i]/2, my=y+dy[i]/2;
            if(nx<0||nx>=9||ny<0||ny>=10) continue;
            if(mx<0||mx>=9||my<0||my>=10) continue;
            if(!on_own_side_elephant(side, ny)) continue;
            if(b[sq(mx,my)]!=0) continue; // eye blocked
            push_move_if_ok(from, sq(nx,ny), p, out);
        }
    }

    void gen_horse(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        struct H { int dx,dy,lx,ly; };
        static const H HMV[8]={
            {+2,+1, +1,0}, {+2,-1, +1,0}, {-2,+1, -1,0}, {-2,-1, -1,0},
            {+1,+2, 0,+1}, {-1,+2, 0,+1}, {+1,-2, 0,-1}, {-1,-2, 0,-1},
        };
        for(const auto& h: HMV){
            int nx=x+h.dx, ny=y+h.dy;
            int lx=x+h.lx, ly=y+h.ly;
            if(nx<0||nx>=9||ny<0||ny>=10) continue;
            if(lx<0||lx>=9||ly<0||ly>=10) continue;
            if(b[sq(lx,ly)]!=0) continue; // leg blocked
            push_move_if_ok(from, sq(nx,ny), p, out);
        }
    }

    void gen_rook(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        const int dx[4]={1,-1,0,0};
        const int dy[4]={0,0,1,-1};
        for(int dir=0; dir<4; ++dir){
            int nx=x, ny=y;
            while(true){
                nx += dx[dir]; ny += dy[dir];
                if(nx<0||nx>=9||ny<0||ny>=10) break;
                int to = sq(nx,ny);
                int8_t dst = b[to];
                if(dst==0){
                    out.push_back(from*90 + to);
                    continue;
                }
                if(opp_side(p,dst)) out.push_back(from*90 + to);
                break;
            }
        }
    }

    void gen_cannon(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        const int dx[4]={1,-1,0,0};
        const int dy[4]={0,0,1,-1};
        for(int dir=0; dir<4; ++dir){
            int nx=x, ny=y;
            bool screened=false;
            while(true){
                nx += dx[dir]; ny += dy[dir];
                if(nx<0||nx>=9||ny<0||ny>=10) break;
                int to = sq(nx,ny);
                int8_t dst = b[to];
                if(!screened){
                    if(dst==0){
                        out.push_back(from*90 + to); // non-capture
                        continue;
                    }else{
                        screened=true;
                        continue;
                    }
                }else{
                    if(dst==0) continue;
                    if(opp_side(p,dst)) out.push_back(from*90 + to); // capture over one screen
                    break;
                }
            }
        }
    }

    void gen_pawn(int from, int8_t p, std::vector<int>& out){
        int x,y; xy(from,x,y);
        int side = piece_side(p);
        int fdy = (side==RED) ? -1 : +1;
        int ny = y + fdy;
        if(ny>=0 && ny<=9){
            push_move_if_ok(from, sq(x,ny), p, out);
        }
        if(pawn_crossed_river(side, y)){
            // sideways
            if(x-1>=0) push_move_if_ok(from, sq(x-1,y), p, out);
            if(x+1<=8) push_move_if_ok(from, sq(x+1,y), p, out);
        }
    }
};

// ----------------------------
// MCTS
// ----------------------------

struct Node {
    int parent = -1;
    float prior = 0.0f;       // prior from parent
    int visit = 0;
    float value_sum = 0.0f;   // from this node's side-to-move perspective
    bool expanded = false;
    bool terminal = false;
    float terminal_value = 0.0f; // from this node's side-to-move perspective
    std::vector<int> children;    // indices of child nodes
    std::vector<int> moves;       // action_id for each child
};

static inline float node_value(const Node& n) {
    return (n.visit > 0) ? (n.value_sum / (float)n.visit) : 0.0f;
}

static inline float effective_cpuct(const Node& parent, float c_puct, float c_puct_base, float c_puct_factor) {
    if(std::abs(c_puct_factor) <= 1e-12f) return c_puct;
    float base = std::max(c_puct_base, 1e-6f);
    return c_puct + c_puct_factor * std::log(((float)parent.visit + base + 1.0f) / base);
}

static inline float ucb_score(
    const Node& parent,
    const Node& child,
    float c_puct,
    float q_weight,
    float q_clip,
    float c_puct_base,
    float c_puct_factor,
    float fpu_reduction
) {
    float cpuct_eff = effective_cpuct(parent, c_puct, c_puct_base, c_puct_factor);
    float u = cpuct_eff * child.prior * std::sqrt((float)parent.visit + 1.0f) / (1.0f + (float)child.visit);
    float q_raw = 0.0f;
    if(child.visit > 0) {
        q_raw = -node_value(child);
    } else if(fpu_reduction >= 0.0f) {
        q_raw = node_value(parent) - fpu_reduction;
    }
    float q = std::clamp(q_raw, -q_clip, q_clip) * q_weight;
    return q + u;
}

static std::vector<float> dirichlet_noise(int n, float alpha, std::mt19937 &rng) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> x(n);
    float sum = 0.0f;
    for(int i=0;i<n;++i){
        float v = gamma(rng);
        x[i]=v;
        sum += v;
    }
    if(sum <= 1e-12f){
        float p = 1.0f / std::max(1,n);
        std::fill(x.begin(), x.end(), p);
        return x;
    }
    for(auto &v: x) v /= sum;
    return x;
}

static inline float value_from_wdl_logits(const torch::Tensor& wdl_logits_row) {
    // wdl_logits_row: [3]
    auto probs = torch::softmax(wdl_logits_row, 0);
    float pw = probs[0].item<float>();
    float pl = probs[2].item<float>();
    return pw - pl;
}

static py::dict require_net_output_dict(const py::object& out) {
    if (!py::isinstance<py::dict>(out)) {
        throw std::runtime_error(
            "net(batch) must return a dict with at least 'policy_logits' and 'value_scalar'"
        );
    }
    return out.cast<py::dict>();
}

static torch::Tensor require_output_tensor(
    const py::dict& out,
    const char* key,
    int64_t expected_batch,
    int64_t expected_last_dim
) {
    py::str py_key(key);
    if (!out.contains(py_key)) {
        throw std::runtime_error(std::string("net(batch) is missing required key '") + key + "'");
    }

    torch::Tensor t = out[py_key].cast<torch::Tensor>().contiguous();
    if (!t.defined()) {
        throw std::runtime_error(std::string("net(batch)['") + key + "'] returned an undefined tensor");
    }
    if (!t.device().is_cpu()) {
        throw std::runtime_error(std::string("net(batch)['") + key + "'] must be a CPU tensor");
    }
    if (t.scalar_type() != torch::kFloat32) {
        throw std::runtime_error(std::string("net(batch)['") + key + "'] must be float32");
    }
    if (t.dim() != 2 || t.size(0) != expected_batch || t.size(1) != expected_last_dim) {
        std::ostringstream oss;
        oss << "net(batch)['" << key << "'] must have shape ["
            << expected_batch << "," << expected_last_dim << "], got " << t.sizes();
        throw std::runtime_error(oss.str());
    }
    return t;
}

// Expand a node with network output (policy logits + value), using legal-softmax priors.
static void expand_node(
    std::deque<Node> &nodes,
    int node_idx,
    const torch::Tensor &policy_logits_row,  // [8100]
    float value_stm,                          // scalar, from current side-to-move
    bool canonical_policy,
    bool stm_black,
    const std::vector<int> &legal_moves       // action ids at this position
) {
    // Use deque for the node pool so tail growth does not relocate existing nodes.
    Node &node = nodes[node_idx];
    if (node.expanded) return;

    node.expanded = true;
    node.terminal = false;
    node.terminal_value = 0.0f;
    node.children.clear();
    node.moves.clear();

    if(legal_moves.empty()){
        node.terminal = true;
        node.terminal_value = value_stm;
        return;
    }

    // Compute priors on legal moves only: softmax(logits[legal])
    torch::Tensor logits_row = policy_logits_row.contiguous();
    auto logits_acc = logits_row.accessor<float,1>();
    float maxlog = -1e30f;
    for(int mv: legal_moves){
        int logit_idx = canonical_policy ? canonical_action(mv, stm_black) : mv;
        float l = logits_acc[logit_idx];
        if(l > maxlog) maxlog = l;
    }
    std::vector<float> expv;
    expv.reserve(legal_moves.size());
    float sum = 0.0f;
    for(int mv: legal_moves){
        int logit_idx = canonical_policy ? canonical_action(mv, stm_black) : mv;
        float e = std::exp(logits_acc[logit_idx] - maxlog);
        expv.push_back(e);
        sum += e;
    }
    if(sum <= 1e-12f){
        sum = (float)legal_moves.size();
        std::fill(expv.begin(), expv.end(), 1.0f);
    }

    node.children.reserve(legal_moves.size());
    node.moves.reserve(legal_moves.size());

    for(size_t i=0;i<legal_moves.size();++i){
        int mv = legal_moves[i];
        float pr = expv[i] / sum;
        int child_idx = (int)nodes.size();
        Node child;
        child.parent = node_idx;
        child.prior = pr;
        nodes.push_back(std::move(child));
        node.children.push_back(child_idx);
        node.moves.push_back(mv);
    }
}

// Terminal value from side-to-move perspective.
static float terminal_value_stm(Board &board) {
    int res = board.result_red_view(); // 1 red win, -1 black win, 0 ongoing
    if(res==0) return 0.0f;
    int winner = (res==1) ? RED : BLACK;
    return (board.get_turn() == winner) ? 1.0f : -1.0f;
}

static inline bool is_draw_terminal_code(int terminal_code) {
    return terminal_code == TERMINAL_MAX_PLIES_DRAW
        || terminal_code == TERMINAL_REPETITION_DRAW
        || terminal_code == TERMINAL_NO_CAPTURE_DRAW;
}

static float terminal_value_from_code(Board &board, int terminal_code) {
    int res = board.terminal_result_red_view(terminal_code);
    if (res == 0) return 0.0f;
    int winner = (res > 0) ? RED : BLACK;
    return (board.get_turn() == winner) ? 1.0f : -1.0f;
}

static bool has_immediate_winning_move(
    Board &board,
    const std::vector<int> &legal_moves,
    int max_plies,
    int repeat_limit,
    int repeat_min_ply,
    int no_capture_limit
) {
    // Conservative symbolic probe: only override neural eval when there is a
    // one-ply terminal win. Draw terminals are ignored.
    for (int mv : legal_moves) {
        board.push(mv);
        int code = board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit);
        bool winning = false;
        if (code != TERMINAL_ONGOING && !is_draw_terminal_code(code)) {
            // After pushing mv, the opponent is side-to-move. A negative
            // terminal value from that perspective means the mover just won.
            winning = terminal_value_from_code(board, code) < -0.5f;
        }
        board.pop();
        if (winning) return true;
    }
    return false;
}

static bool terminal_is_win_for_side_to_move(Board &board, int terminal_code) {
    if (terminal_code == TERMINAL_ONGOING || is_draw_terminal_code(terminal_code)) {
        return false;
    }
    return terminal_value_from_code(board, terminal_code) > 0.5f;
}

static bool terminal_is_win_for_previous_mover(Board &board, int terminal_code) {
    if (terminal_code == TERMINAL_ONGOING || is_draw_terminal_code(terminal_code)) {
        return false;
    }
    // After pushing a move, the opponent is side-to-move. A negative terminal
    // value from that perspective means the previous mover just won.
    return terminal_value_from_code(board, terminal_code) < -0.5f;
}

static bool has_check_forced_mate_in_two(
    Board &board,
    const std::vector<int> &legal_moves,
    int max_plies,
    int repeat_limit,
    int repeat_min_ply,
    int no_capture_limit
) {
    // Conservative symbolic probe: only count mate-in-2 when the first move is
    // a checking move and every legal reply still allows an immediate terminal
    // win. Quiet mate nets are intentionally ignored to keep this narrow.
    for (int mv : legal_moves) {
        board.push(mv);
        int code_after_first = board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit);
        if (terminal_is_win_for_previous_mover(board, code_after_first)) {
            board.pop();
            return true;
        }
        if (code_after_first != TERMINAL_ONGOING) {
            board.pop();
            continue;
        }
        if (!board.in_check_turn()) {
            board.pop();
            continue;
        }

        std::vector<int> replies = board.legal_moves();
        if (replies.empty()) {
            board.pop();
            continue;
        }

        bool forced = true;
        for (int reply : replies) {
            board.push(reply);
            int code_after_reply = board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit);
            bool reply_ok = false;
            if (code_after_reply != TERMINAL_ONGOING) {
                reply_ok = terminal_is_win_for_side_to_move(board, code_after_reply);
            } else {
                std::vector<int> followups = board.legal_moves();
                reply_ok = has_immediate_winning_move(
                    board, followups, max_plies, repeat_limit, repeat_min_ply, no_capture_limit);
            }
            board.pop();
            if (!reply_ok) {
                forced = false;
                break;
            }
        }
        board.pop();
        if (forced) return true;
    }
    return false;
}

struct PendingSim {
    std::vector<int> path;          // node indices from root..leaf
    int leaf = -1;
    std::vector<int> legal_moves;   // legal moves at leaf
    torch::Tensor state;            // [1,115,10,9]
    bool stm_black = false;
};

// Main MCTS search (batched eval).
static py::tuple mcts_search_cpp_impl(
    Board &board,
    py::object net,
    int num_simulations,
    float c_puct,
    float q_weight,
    float q_clip,
    bool add_root_noise,
    float dirichlet_alpha,
    float dirichlet_eps,
    float temperature_move,
    float temperature_target,
    int eval_batch_size,
    int seed,
    bool canonical_input,
    bool canonical_policy,
    int max_plies,
    int repeat_limit,
    int repeat_min_ply,
    int no_capture_limit,
    bool tactical_mate1_extension,
    bool tactical_mate2_extension,
    float c_puct_base,
    float c_puct_factor,
    float fpu_reduction_root,
    float fpu_reduction_tree,
    bool include_root_stats
) {
    if(eval_batch_size < 1) eval_batch_size = 1;

    std::mt19937 rng(seed ? (uint32_t)seed : (uint32_t)std::random_device{}());

    std::deque<Node> nodes;
    nodes.push_back(Node{}); // root at 0

    // Evaluate & expand root
    {
        int root_terminal = board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit);
        if(root_terminal != TERMINAL_ONGOING){
            Node &r = nodes[0];
            r.expanded = true;
            r.terminal = true;
            r.terminal_value = terminal_value_from_code(board, root_terminal);
        }else{
            std::vector<int> legal = board.legal_moves();
            auto st = canonical_input ? board.to_tensor_canonical() : board.to_tensor();
            py::dict out = require_net_output_dict(net(st));
            torch::Tensor p_logits = require_output_tensor(out, "policy_logits", 1, 8100);
            torch::Tensor value_tensor = require_output_tensor(out, "value_scalar", 1, 1);
            if (out.contains("wdl_logits")) {
                torch::Tensor wdl_tensor = require_output_tensor(out, "wdl_logits", 1, 3);
                float wdl_debug = value_from_wdl_logits(wdl_tensor[0]);
                if (!std::isfinite(wdl_debug)) {
                    throw std::runtime_error("net(batch)['wdl_logits'] produced a non-finite debug value");
                }
            }
            float v = value_tensor[0][0].item<float>();
            if (!std::isfinite(v)) {
                throw std::runtime_error("net(batch)['value_scalar'] produced a non-finite value");
            }
            expand_node(nodes, 0, p_logits[0].contiguous(), v, canonical_policy, board.get_turn() == BLACK, legal);
        }
    }

    // Root dirichlet noise
    if(add_root_noise && nodes[0].expanded && !nodes[0].terminal && !nodes[0].children.empty()){
        auto noise = dirichlet_noise((int)nodes[0].children.size(), dirichlet_alpha, rng);
        for(size_t i=0;i<nodes[0].children.size();++i){
            int ci = nodes[0].children[i];
            Node &ch = nodes[ci];
            ch.prior = (1.0f - dirichlet_eps) * ch.prior + dirichlet_eps * noise[i];
        }
    }

    // If root terminal, return no move.
    if(nodes[0].terminal){
        // policy empty
        auto idxs = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
        auto probs = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32));
        if (include_root_stats) {
            py::list root_stats;
            return py::make_tuple(-1, idxs, probs, nodes[0].terminal_value, root_stats);
        }
        return py::make_tuple(-1, idxs, probs, nodes[0].terminal_value);
    }

    // Simulations
    std::vector<PendingSim> pending;
    pending.reserve((size_t)eval_batch_size);

    auto flush_pending = [&](){
        if(pending.empty()) return;

        // stack states
        std::vector<torch::Tensor> st_list;
        st_list.reserve(pending.size());
        for(auto &ps : pending){
            st_list.push_back(ps.state);
        }
        torch::Tensor batch = torch::cat(st_list, 0); // [B,115,10,9]
        py::dict out = require_net_output_dict(net(batch));
        int64_t batch_size = (int64_t)pending.size();
        torch::Tensor p_logits = require_output_tensor(out, "policy_logits", batch_size, 8100);
        torch::Tensor value_tensor = require_output_tensor(out, "value_scalar", batch_size, 1);
        torch::Tensor wdl_tensor;
        bool has_wdl = out.contains("wdl_logits");
        if (has_wdl) {
            wdl_tensor = require_output_tensor(out, "wdl_logits", batch_size, 3);
        }

        for(size_t i=0;i<pending.size();++i){
            auto &ps = pending[i];
            if (has_wdl) {
                float wdl_debug = value_from_wdl_logits(wdl_tensor[(int)i]);
                if (!std::isfinite(wdl_debug)) {
                    throw std::runtime_error("net(batch)['wdl_logits'] produced a non-finite debug value");
                }
            }
            float v = value_tensor[(int)i][0].item<float>();
            if (!std::isfinite(v)) {
                throw std::runtime_error("net(batch)['value_scalar'] produced a non-finite value");
            }
            // expand leaf (will be a no-op if already expanded by a duplicate)
            expand_node(nodes, ps.leaf, p_logits[(int)i], v, canonical_policy, ps.stm_black, ps.legal_moves);
            
            // backprop value_sum. We added a virtual loss of 1.0 at the leaf
            // during traversal, so we substitute it by adding (v - 1.0f).
            float cur_diff = v - 1.0f;
            for(int pi=(int)ps.path.size()-1; pi>=0; --pi){
                int ni = ps.path[pi];
                nodes[ni].value_sum += cur_diff;
                cur_diff = -cur_diff;
            }
        }
        pending.clear();
    };

    size_t root_depth = 0; // board history depth at root (we always restore by pops)

    for(int sim=0; sim<num_simulations; ++sim){
        int node_idx = 0;
        std::vector<int> path;
        path.reserve(64);
        path.push_back(0);

        // Traverse
        while(nodes[node_idx].expanded && !nodes[node_idx].terminal && !nodes[node_idx].children.empty()){
            Node &n = nodes[node_idx];
            float best = -1e30f;
            int best_k = 0;
            for(size_t k=0;k<n.children.size();++k){
                int ci = n.children[k];
                float fpu_reduction = (node_idx == 0) ? fpu_reduction_root : fpu_reduction_tree;
                float s = ucb_score(
                    n,
                    nodes[ci],
                    c_puct,
                    q_weight,
                    q_clip,
                    c_puct_base,
                    c_puct_factor,
                    fpu_reduction
                );
                if(s > best){
                    best = s;
                    best_k = (int)k;
                }
            }
            int mv = n.moves[best_k];
            int child_idx = n.children[best_k];
            board.push(mv);
            node_idx = child_idx;
            path.push_back(node_idx);
        }

        // Leaf handling
        Node &leaf = nodes[node_idx];

        if(!leaf.expanded){
            int leaf_terminal = board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit);
            if(leaf_terminal != TERMINAL_ONGOING){
                leaf.expanded = true;
                leaf.terminal = true;
                leaf.terminal_value = terminal_value_from_code(board, leaf_terminal);

                float v = leaf.terminal_value;
                float cur = v;
                for(int pi=(int)path.size()-1; pi>=0; --pi){
                    int ni = path[pi];
                    nodes[ni].visit += 1;
                    nodes[ni].value_sum += cur;
                    cur = -cur;
                }
                for(size_t k=1; k<path.size(); ++k) board.pop();
                continue;
            }

            // Generate legal moves once after terminal handling.
            // DO NOT set leaf.expanded = true yet! (Let expand_node do it when value arrives).
            std::vector<int> legal = board.legal_moves();
            bool tactical_forced_win = false;
            if (tactical_mate1_extension && has_immediate_winning_move(
                    board, legal, max_plies, repeat_limit, repeat_min_ply, no_capture_limit)) {
                tactical_forced_win = true;
            } else if (tactical_mate2_extension && has_check_forced_mate_in_two(
                    board, legal, max_plies, repeat_limit, repeat_min_ply, no_capture_limit)) {
                tactical_forced_win = true;
            }
            if (tactical_forced_win) {
                leaf.expanded = true;
                leaf.terminal = true;
                leaf.terminal_value = 1.0f;

                float cur = leaf.terminal_value;
                for(int pi=(int)path.size()-1; pi>=0; --pi){
                    int ni = path[pi];
                    nodes[ni].visit += 1;
                    nodes[ni].value_sum += cur;
                    cur = -cur;
                }
                for(size_t k=1; k<path.size(); ++k) board.pop();
                continue;
            }

            // Virtual visit + virtual loss increment now (to diversify within batch)
            // A virtual loss means the parent will see this choice as losing (-1.0).
            // We add 1.0 to the leaf's value_sum, which alternates to -1.0 at the parent.
            float cur_virtual = 1.0f;
            for(int pi=(int)path.size()-1; pi>=0; --pi){
                int ni = path[pi];
                nodes[ni].visit += 1;
                nodes[ni].value_sum += cur_virtual;
                cur_virtual = -cur_virtual;
            }

            PendingSim ps;
            ps.path = std::move(path);
            ps.leaf = node_idx;
            ps.legal_moves = std::move(legal);
            ps.state = canonical_input ? board.to_tensor_canonical() : board.to_tensor();
            ps.stm_black = (board.get_turn() == BLACK);
            pending.push_back(std::move(ps));

            // undo to root
            for(size_t k=1; k<pending.back().path.size(); ++k) board.pop();

            if((int)pending.size() >= eval_batch_size){
                flush_pending();
            }
            continue;
        }

        if(leaf.terminal){
            float v = leaf.terminal_value;
            float cur = v;
            for(int pi=(int)path.size()-1; pi>=0; --pi){
                int ni = path[pi];
                nodes[ni].visit += 1;
                nodes[ni].value_sum += cur;
                cur = -cur;
            }
            for(size_t k=1; k<path.size(); ++k) board.pop();
            continue;
        }
// Expanded but not terminal (should have children), we need to backup? In standard MCTS, reaching expanded leaf means
        // we continue until unexpanded; here it shouldn't happen because loop stops at node with children, but if children empty
        // and expanded (rare), treat as terminal-like with stored value.
        if(leaf.terminal){
            float v = leaf.terminal_value;
            float cur = v;
            for(int pi=(int)path.size()-1; pi>=0; --pi){
                int ni = path[pi];
                nodes[ni].visit += 1;
                nodes[ni].value_sum += cur;
                cur = -cur;
            }
            for(size_t k=1; k<path.size(); ++k) board.pop();
            continue;
        }

        // If we reach here, leaf is expanded and has children but we stopped due to while condition? Actually while stops only when leaf has no children.
        // So this path shouldn't happen. Still, safe:
        {
            float v = node_value(leaf);
            float cur = v;
            for(int pi=(int)path.size()-1; pi>=0; --pi){
                int ni = path[pi];
                nodes[ni].visit += 1;
                nodes[ni].value_sum += cur;
                cur = -cur;
            }
            for(size_t k=1; k<path.size(); ++k) board.pop();
        }
    }

    // flush leftovers
    flush_pending();

    // Build root policy (target) from visit counts
    Node &root = nodes[0];
    int n = (int)root.children.size();
    std::vector<int64_t> idxs_vec;
    std::vector<float> probs_vec;
    idxs_vec.reserve(n);
    probs_vec.reserve(n);

    std::vector<double> visits;
    visits.reserve(n);
    for(int k=0;k<n;++k){
        int ci = root.children[k];
        visits.push_back((double)nodes[ci].visit);
        int mv = root.moves[k];
        idxs_vec.push_back((int64_t)(canonical_policy ? canonical_action(mv, board.get_turn() == BLACK) : mv));
    }

    // policy target distribution
    std::vector<double> pi_target(n, 0.0);
    if(temperature_target <= 1e-6f){
        int best = (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
        pi_target[best] = 1.0;
    }else{
        double invT = 1.0 / (double)temperature_target;
        double sum = 0.0;
        for(int i=0;i<n;++i){
            double x = std::pow(visits[i] + 1e-12, invT);
            pi_target[i] = x;
            sum += x;
        }
        if(sum <= 1e-18) sum = 1.0;
        for(int i=0;i<n;++i) pi_target[i] /= sum;
    }
    for(int i=0;i<n;++i) probs_vec.push_back((float)pi_target[i]);

    // move selection distribution (can differ)
    int chosen = 0;
    if(temperature_move <= 1e-6f){
        chosen = (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
    }else{
        double invT = 1.0 / (double)temperature_move;
        std::vector<double> p(n,0.0);
        double sum = 0.0;
        for(int i=0;i<n;++i){
            double x = std::pow(visits[i] + 1e-12, invT);
            p[i]=x;
            sum += x;
        }
        if(sum <= 1e-18) sum = 1.0;
        for(int i=0;i<n;++i) p[i] /= sum;
        // sample
        std::uniform_real_distribution<double> uni(0.0,1.0);
        double r = uni(rng);
        double c = 0.0;
        for(int i=0;i<n;++i){
            c += p[i];
            if(r <= c){ chosen=i; break; }
        }
    }

    int best_move = (chosen>=0 && chosen<n) ? root.moves[chosen] : -1;

    auto idxs = torch::from_blob(idxs_vec.data(), {(int64_t)idxs_vec.size()}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto probs = torch::from_blob(probs_vec.data(), {(int64_t)probs_vec.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

    // Root value estimate: from root side-to-move perspective (red starts)
    float root_v = node_value(root);

    if (include_root_stats) {
        py::list root_stats;
        double visit_sum = 0.0;
        for (double v : visits) visit_sum += v;
        for (int k = 0; k < n; ++k) {
            int ci = root.children[k];
            const Node &child = nodes[ci];
            int mv = root.moves[k];
            int64_t canonical_idx = (int64_t)(canonical_policy ? canonical_action(mv, board.get_turn() == BLACK) : mv);
            float q_child = node_value(child);
            float q_root = -q_child;
            float final_ucb = ucb_score(
                root,
                child,
                c_puct,
                q_weight,
                q_clip,
                c_puct_base,
                c_puct_factor,
                fpu_reduction_root
            );
            py::dict row;
            row["move_raw"] = mv;
            row["canonical_idx"] = canonical_idx;
            row["visit_count"] = child.visit;
            row["visit_prob"] = (visit_sum > 0.0) ? (double)child.visit / visit_sum : 0.0;
            row["target_prob"] = (k >= 0 && k < (int)probs_vec.size()) ? probs_vec[k] : 0.0f;
            row["prior"] = child.prior;
            row["q_root_pov"] = q_root;
            row["q_child_pov"] = q_child;
            row["ucb_score"] = final_ucb;
            row["selected"] = (k == chosen);
            root_stats.append(row);
        }
        return py::make_tuple(best_move, idxs, probs, root_v, root_stats);
    }

    return py::make_tuple(best_move, idxs, probs, root_v);
}

static py::tuple mcts_search_cpp(
    Board &board,
    py::object net,
    int num_simulations,
    float c_puct,
    float q_weight,
    float q_clip,
    bool add_root_noise,
    float dirichlet_alpha,
    float dirichlet_eps,
    float temperature_move,
    float temperature_target,
    int eval_batch_size,
    int seed,
    bool canonical_input,
    bool canonical_policy,
    int max_plies,
    int repeat_limit,
    int repeat_min_ply,
    int no_capture_limit,
    bool tactical_mate1_extension,
    bool tactical_mate2_extension,
    float c_puct_base,
    float c_puct_factor,
    float fpu_reduction_root,
    float fpu_reduction_tree
) {
    return mcts_search_cpp_impl(
        board,
        net,
        num_simulations,
        c_puct,
        q_weight,
        q_clip,
        add_root_noise,
        dirichlet_alpha,
        dirichlet_eps,
        temperature_move,
        temperature_target,
        eval_batch_size,
        seed,
        canonical_input,
        canonical_policy,
        max_plies,
        repeat_limit,
        repeat_min_ply,
        no_capture_limit,
        tactical_mate1_extension,
        tactical_mate2_extension,
        c_puct_base,
        c_puct_factor,
        fpu_reduction_root,
        fpu_reduction_tree,
        false
    );
}

static py::tuple mcts_search_with_root_stats_cpp(
    Board &board,
    py::object net,
    int num_simulations,
    float c_puct,
    float q_weight,
    float q_clip,
    bool add_root_noise,
    float dirichlet_alpha,
    float dirichlet_eps,
    float temperature_move,
    float temperature_target,
    int eval_batch_size,
    int seed,
    bool canonical_input,
    bool canonical_policy,
    int max_plies,
    int repeat_limit,
    int repeat_min_ply,
    int no_capture_limit,
    bool tactical_mate1_extension,
    bool tactical_mate2_extension,
    float c_puct_base,
    float c_puct_factor,
    float fpu_reduction_root,
    float fpu_reduction_tree
) {
    return mcts_search_cpp_impl(
        board,
        net,
        num_simulations,
        c_puct,
        q_weight,
        q_clip,
        add_root_noise,
        dirichlet_alpha,
        dirichlet_eps,
        temperature_move,
        temperature_target,
        eval_batch_size,
        seed,
        canonical_input,
        canonical_policy,
        max_plies,
        repeat_limit,
        repeat_min_ply,
        no_capture_limit,
        tactical_mate1_extension,
        tactical_mate2_extension,
        c_puct_base,
        c_puct_factor,
        fpu_reduction_root,
        fpu_reduction_tree,
        true
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("reset", &Board::reset)
        .def("turn", &Board::get_turn)
        .def("key", &Board::key)
        .def("fen", &Board::fen)
        .def("set_fen", &Board::set_fen)
        .def("plies_played", &Board::get_plies_played)
        .def("no_capture_count", &Board::get_no_capture_count)
        .def("current_repetition_count", &Board::get_current_repetition_count)
        .def("perpetual_check_loser_side_for_current_repetition", &Board::perpetual_check_loser_side_for_current_repetition)
        .def(
            "set_search_context",
            &Board::set_search_context,
            py::arg("plies"),
            py::arg("no_capture_count"),
            py::arg("repetition_count_hint") = 1
        )
        .def(
            "terminal_code",
            &Board::terminal_code,
            py::arg("max_plies") = 0,
            py::arg("repeat_limit") = 0,
            py::arg("repeat_min_ply") = 0,
            py::arg("no_capture_limit") = 0
        )
        .def("terminal_result_red_view", &Board::terminal_result_red_view)
        .def("to_tensor", &Board::to_tensor)
        .def("to_tensor_canonical", &Board::to_tensor_canonical)
        .def("push", &Board::push)
        .def("pop", &Board::pop)
        .def("legal_moves", &Board::legal_moves)
        .def("is_game_over", &Board::is_game_over)
        .def("result_red_view", &Board::result_red_view)
        .def("piece_at", [](const Board& self, int square) { return int(self.piece_at(square)); })
        .def("is_capture", &Board::is_capture)
        .def("material_score", &Board::material_score)
        .def("in_check_turn", &Board::in_check_turn)
        .def("in_check_side", &Board::in_check_side)
        .def("in_check_slow", &Board::in_check_slow)
        .def("in_check_slow_turn", &Board::in_check_slow_turn)
        .def("is_pseudo_legal", &Board::is_pseudo_legal)
        .def("is_legal", &Board::is_legal)
        .def("push_legal", &Board::push_legal)

        ;

    m.def("canonical_square", [](int square, bool stm_black) {
        return canonical_square(square, stm_black);
    });
    m.def("canonical_action", [](int action_id, bool stm_black) {
        return canonical_action(action_id, stm_black);
    });

    m.def("mcts_search", &mcts_search_cpp,
          py::arg("board"),
          py::arg("net"),
          py::arg("num_simulations"),
          py::arg("c_puct"),
          py::arg("q_weight") = 1.0f,
          py::arg("q_clip") = 1.0f,
          py::arg("add_root_noise"),
          py::arg("dirichlet_alpha"),
          py::arg("dirichlet_eps"),
          py::arg("temperature_move"),
          py::arg("temperature_target"),
          py::arg("eval_batch_size") = 16,
          py::arg("seed") = 0,
          py::arg("canonical_input") = true,
          py::arg("canonical_policy") = true,
          py::arg("max_plies") = 0,
          py::arg("repeat_limit") = 0,
          py::arg("repeat_min_ply") = 0,
          py::arg("no_capture_limit") = 0,
          py::arg("tactical_mate1_extension") = false,
          py::arg("tactical_mate2_extension") = false,
          py::arg("c_puct_base") = 1.0f,
          py::arg("c_puct_factor") = 0.0f,
          py::arg("fpu_reduction_root") = -1.0f,
          py::arg("fpu_reduction_tree") = -1.0f,
          R"doc(
Run batched MCTS search.

Returns:
  (best_move_action_id, policy_idxs[int64], policy_probs[float32], root_value_estimate)

Notes:
- net(batch) must return a dict with:
  - policy_logits [B,8100] float32 CPU tensor
  - value_scalar [B,1] float32 CPU tensor
- optional wdl_logits [B,3] is accepted for debug validation only.
- tactical_mate1_extension replaces neural leaf eval with +1 when the
  side-to-move has an immediate terminal win.
- tactical_mate2_extension additionally treats check-only forced mate-in-2
  leaves as +1. Quiet mate nets are intentionally ignored.
- c_puct_base/c_puct_factor enable log-scaled PUCT when factor is non-zero:
  c_eff = c_puct + c_puct_factor * log((N + base + 1) / base).
- fpu_reduction_root/tree use parent_value - reduction for unvisited children
  when >= 0. Negative values keep legacy first-play urgency at Q=0.
- policy is sparse over root legal moves only.
- policy_probs uses temperature_target shaping; best_move sampling uses temperature_move.
- draw-aware terminal handling supports max_plies / repetition / no-capture rules.
)doc");
    m.def("mcts_search_with_root_stats", &mcts_search_with_root_stats_cpp,
          py::arg("board"),
          py::arg("net"),
          py::arg("num_simulations"),
          py::arg("c_puct"),
          py::arg("q_weight") = 1.0f,
          py::arg("q_clip") = 1.0f,
          py::arg("add_root_noise"),
          py::arg("dirichlet_alpha"),
          py::arg("dirichlet_eps"),
          py::arg("temperature_move"),
          py::arg("temperature_target"),
          py::arg("eval_batch_size") = 16,
          py::arg("seed") = 0,
          py::arg("canonical_input") = true,
          py::arg("canonical_policy") = true,
          py::arg("max_plies") = 0,
          py::arg("repeat_limit") = 0,
          py::arg("repeat_min_ply") = 0,
          py::arg("no_capture_limit") = 0,
          py::arg("tactical_mate1_extension") = false,
          py::arg("tactical_mate2_extension") = false,
          py::arg("c_puct_base") = 1.0f,
          py::arg("c_puct_factor") = 0.0f,
          py::arg("fpu_reduction_root") = -1.0f,
          py::arg("fpu_reduction_tree") = -1.0f,
          R"doc(
Run batched MCTS search and include root child diagnostics.

Returns:
  (best_move_action_id, policy_idxs[int64], policy_probs[float32],
   root_value_estimate, root_stats[list[dict]])

Each root_stats row contains:
  move_raw, canonical_idx, visit_count, visit_prob, target_prob, prior,
  q_root_pov, q_child_pov, ucb_score, selected.
)doc");
}
