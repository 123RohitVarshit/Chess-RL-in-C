// chess.c
#include "chess.h"
#include <stdlib.h> // for abs()

int board[BOARD_SIZE];

void init_board() {
    for(int i = 0; i < 64; i++) board[i] = 0;
    board[3] = -2;                  // Black King
    for(int i = 8; i < 16; i++) board[i] = -1;  // Black Pawns
    for(int i = 48; i < 56; i++) board[i] = 1;  // White Pawns
    board[59] = 2;                  // White King
}

void board_to_tensor(float* tensor) {
    for(int i = 0; i < 64; i++) {
        if(board[i] > 0) tensor[i] = 1.0f;
        else if(board[i] < 0) tensor[i] = -1.0f;
        else tensor[i] = 0.0f;
    }
}

int get_legal_moves(int player, int* moves) {
    int count = 0;
    int is_white = (player == 1);
    
    for(int i = 0; i < 64; i++) {
        int piece = board[i];
        if((is_white && piece <= 0) || (!is_white && piece >= 0)) continue;
        
        int col = i % 8;

        if(abs(piece) == 1) { // --- PAWN LOGIC ---
            // White (1) is at bottom, moves UP (decreases index) -> -8
            // Black (-1) is at top, moves DOWN (increases index) -> +8
            int dir = is_white ? -8 : 8; 
            int target = i + dir;
            
            // 1. Push forward 1 square
            if(target >= 0 && target < 64 && board[target] == 0) {
                moves[count++] = (i * 64) + target;
            }
            // 2. Capture diagonally (Left)
            if(col > 0) { 
                int cap = i + dir - 1; 
                if(cap >= 0 && cap < 64 && board[cap] != 0 && ((is_white && board[cap] < 0) || (!is_white && board[cap] > 0))) {
                    moves[count++] = (i * 64) + cap;
                }
            }
            // 3. Capture diagonally (Right)
            if(col < 7) { 
                int cap = i + dir + 1; 
                if(cap >= 0 && cap < 64 && board[cap] != 0 && ((is_white && board[cap] < 0) || (!is_white && board[cap] > 0))) {
                    moves[count++] = (i * 64) + cap;
                }
            }
        } 
        else if(abs(piece) == 2) { // --- KING LOGIC ---
            int dirs[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
            for(int d = 0; d < 8; d++) {
                int target = i + dirs[d];
                // Check bounds AND ensure we didn't wrap around the board edges
                if(target >= 0 && target < 64 && abs((target % 8) - col) <= 1) {
                    if(board[target] == 0 || ((is_white && board[target] < 0) || (!is_white && board[target] > 0))) {
                        moves[count++] = (i * 64) + target;
                    }
                }
            }
        }
    }
    return count;
}

int make_move(int from, int to) {
    int captured = board[to];
    board[to] = board[from];
    board[from] = 0;
    if(abs(board[to]) == 1 && (to < 8 || to > 55)) board[to] = (board[to] > 0) ? 2 : -2;
    return captured;
}

int check_winner() {
    int white_king = 0, black_king = 0;
    for(int i = 0; i < 64; i++) {
        if(board[i] == 2) white_king = 1;
        if(board[i] == -2) black_king = 1;
    }
    if(!white_king) return -1;
    if(!black_king) return 1;
    return 0;
}